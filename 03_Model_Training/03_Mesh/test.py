import os
import csv
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim

from dataset import WificamDataset, NUM_SUBCARRIERS
from vae import VAE


# =========================
# 기본 설정
# =========================
num_workers = 2
torch.set_num_threads(4)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    accelerator = "mps"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    accelerator = "gpu"
else:
    device = torch.device("cpu")
    accelerator = "cpu"


# =========================
# Metric 함수들
# =========================
def compute_psnr(gt_img, pred_img):
    gt = gt_img.astype(np.float32)
    pred = pred_img.astype(np.float32)

    mse = np.mean((gt - pred) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(gt_img, pred_img):
    """
    gt_img, pred_img: uint8, BGR, HWC
    grayscale 기준 SSIM 계산
    """
    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)

    score = ssim(
        gt_gray,
        pred_gray,
        data_range=255,
    )

    return score


def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    if union == 0:
        return 1.0

    return intersection / union


def compute_dice(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    total = gt_mask.sum() + pred_mask.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection) / total


def get_bbox_from_mask(mask):
    """
    mask: bool, HW
    return: (x1, y1, x2, y2) or None
    """
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return x1, y1, x2, y2


def compute_bbox_iou_from_bboxes(gt_bbox, pred_bbox):
    """
    gt_bbox, pred_bbox: (x1, y1, x2, y2) or None
    """
    if gt_bbox is None and pred_bbox is None:
        return 1.0

    if gt_bbox is None or pred_bbox is None:
        return 0.0

    gx1, gy1, gx2, gy2 = gt_bbox
    px1, py1, px2, py2 = pred_bbox

    inter_x1 = max(gx1, px1)
    inter_y1 = max(gy1, py1)
    inter_x2 = min(gx2, px2)
    inter_y2 = min(gy2, py2)

    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h

    gt_area = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    pred_area = (px2 - px1 + 1) * (py2 - py1 + 1)

    union_area = gt_area + pred_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def safe_mean(values):
    values = [v for v in values if not np.isnan(v)]

    if len(values) == 0:
        return float("nan")

    return float(np.mean(values))


def safe_std(values):
    values = [v for v in values if not np.isnan(v)]

    if len(values) == 0:
        return float("nan")

    return float(np.std(values))


def to_fid_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(torch.uint8)


# =========================
# Mask 생성 함수
# =========================
def keep_largest_component(mask_uint8):
    """
    mask_uint8: 0 또는 255 값을 갖는 uint8 mask
    가장 큰 connected component만 남김
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    # label 0은 배경
    if num_labels <= 1:
        return mask_uint8

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    largest_mask = np.zeros_like(mask_uint8)
    largest_mask[labels == largest_label] = 255

    return largest_mask


def make_binary_mask(img_bgr, threshold=10, use_morphology=False, keep_largest=False):
    """
    img_bgr: uint8, HWC, BGR
    return: bool mask, HW
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if use_morphology:
        kernel = np.ones((3, 3), np.uint8)

        # 작은 노이즈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 작은 구멍 메우기
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if keep_largest:
        mask = keep_largest_component(mask)

    return mask > 0


# =========================
# bbox 시각화 함수
# =========================
def draw_bbox(img, bbox, label=None):
    """
    img: uint8, BGR, HWC
    bbox: (x1, y1, x2, y2) or None
    """
    out = img.copy()

    if bbox is None:
        return out

    x1, y1, x2, y2 = bbox

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if label is not None:
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def make_bbox_visualization(gt_img, pred_img, gt_mask, pred_mask):
    """
    GT/PRED 이미지 위에 bbox를 그리고 좌우로 붙임
    """
    gt_bbox = get_bbox_from_mask(gt_mask)
    pred_bbox = get_bbox_from_mask(pred_mask)

    gt_vis = draw_bbox(gt_img, gt_bbox, label="GT")
    pred_vis = draw_bbox(pred_img, pred_bbox, label="Pred")

    bbox_compare = np.concatenate([gt_vis, pred_vis], axis=1)

    return bbox_compare, gt_bbox, pred_bbox


# =========================
# 이미지 파일 정렬용
# =========================
def natural_key(filename):
    name = os.path.splitext(filename)[0]

    # 예: 123_bbox.png 같은 경우도 숫자 기준으로 정렬
    name = name.replace("_bbox", "")

    try:
        return int(name)
    except ValueError:
        return name


# =========================
# 이미지 폴더 → mp4 변환
# =========================
def make_video_from_image_folder(image_folder, output_video_path, fps=10):
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    image_files = sorted(image_files, key=natural_key)

    if len(image_files) == 0:
        print("영상으로 만들 이미지가 없습니다:", image_folder)
        return

    first_path = os.path.join(image_folder, image_files[0])
    first_frame = cv2.imread(first_path)

    if first_frame is None:
        print("첫 프레임을 읽을 수 없습니다:", first_path)
        return

    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for filename in tqdm(image_files, desc=f"영상 생성 중: {os.path.basename(image_folder)}"):
        img_path = os.path.join(image_folder, filename)
        frame = cv2.imread(img_path)

        if frame is None:
            print("이미지 못읽음:", img_path)
            continue

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        out.write(frame)

    out.release()
    print("영상 저장 완료:", output_video_path)


# =========================
# Main
# =========================
def main(args):
    current_file_path = Path(__file__).resolve()
    current_folder = current_file_path.parent
    project_root = current_folder.parent

    test_dir = args.test_dir
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path

    image_dir = os.path.join(output_dir, "images")
    gt_dir = os.path.join(image_dir, "gt")
    pred_dir = os.path.join(image_dir, "pred")
    compare_dir = os.path.join(image_dir, "compare")
    bbox_compare_dir = os.path.join(image_dir, "bbox_compare")

    mask_dir = os.path.join(image_dir, "mask")
    mask_gt_dir = os.path.join(mask_dir, "gt")
    mask_pred_dir = os.path.join(mask_dir, "pred")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)
    os.makedirs(bbox_compare_dir, exist_ok=True)

    if args.save_masks:
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(mask_gt_dir, exist_ok=True)
        os.makedirs(mask_pred_dir, exist_ok=True)

    compare_video_file = os.path.join(output_dir, "compare.mp4")
    bbox_compare_video_file = os.path.join(output_dir, "bbox_compare.mp4")

    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    summary_csv_path = os.path.join(output_dir, "metrics_summary.csv")

    print("device:", device)
    print("accelerator:", accelerator)
    print("project_root:", project_root)
    print("test_dir:", test_dir)
    print("output_dir:", output_dir)
    print("checkpoint_path:", checkpoint_path)
    print("test_dir exists:", os.path.exists(test_dir))
    print("output_dir exists:", os.path.exists(output_dir))
    print("checkpoint exists:", os.path.exists(checkpoint_path))
    print("save_images:", args.save_images)
    print("save_video:", args.save_video)
    print("save_bbox_images:", args.save_bbox_images)
    print("save_bbox_video:", args.save_bbox_video)
    print("save_masks:", args.save_masks)
    print("compute_fid:", args.fid)
    print("mask_threshold:", args.mask_threshold)
    print("use_morphology:", args.use_morphology)
    print("keep_largest_gt:", args.keep_largest_gt)
    print("keep_largest_pred:", args.keep_largest_pred)

    dataset_test = WificamDataset(test_dir, args.window_size)

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size * 2,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
    )

    model = VAE.load_from_checkpoint(
        checkpoint_path,
        window_size=args.window_size,
        num_subcarriers=NUM_SUBCARRIERS,
    )

    model.to(device)
    model.eval()

    # =========================
    # FID 준비
    # =========================
    fid_metric = None

    if args.fid:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            fid_metric = FrechetInceptionDistance(feature=2048, normalize=False)
            fid_metric = fid_metric.to(device)
            print("FID metric loaded.")
        except Exception as e:
            print("FID metric load 실패. FID는 계산하지 않습니다.")
            print("에러:", e)
            fid_metric = None

    rows = []

    psnr_values = []
    ssim_values = []
    iou_values = []
    dice_values = []
    bbox_iou_values = []

    saved_count = 0

    # =========================
    # Inference Loop
    # =========================
    for batch in tqdm(dataloader_test, desc="추론 및 이미지 저장 중"):
        spectrogram, image, image_path = batch

        spectrogram = spectrogram.to(device)
        image = image.to(device)

        with torch.no_grad():
            qz_x, skip = model.encode(spectrogram)
            reconstruction = model.decode(qz_x, skip)

        image = image.permute(0, 2, 3, 1).cpu().numpy()
        reconstruction = reconstruction.permute(0, 2, 3, 1).cpu().numpy()

        for i in range(len(reconstruction)):
            # GT / Pred 자체만 사용
            # 배경 합성 안 함
            data_content = (np.clip(image[i][..., ::-1], 0, 1) * 255).astype(np.uint8)
            pred_content = (np.clip(reconstruction[i][..., ::-1], 0, 1) * 255).astype(np.uint8)

            filename = os.path.basename(image_path[i])
            name = os.path.splitext(filename)[0]

            h, w = data_content.shape[:2]
            if pred_content.shape[:2] != (h, w):
                pred_content = cv2.resize(pred_content, (w, h))

            # =========================
            # Mask 생성
            # =========================
            mask_gt = make_binary_mask(
                data_content,
                threshold=args.mask_threshold,
                use_morphology=args.use_morphology,
                keep_largest=args.keep_largest_gt,
            )

            mask_pred = make_binary_mask(
                pred_content,
                threshold=args.mask_threshold,
                use_morphology=args.use_morphology,
                keep_largest=args.keep_largest_pred,
            )

            # =========================
            # bbox 생성
            # =========================
            gt_bbox = get_bbox_from_mask(mask_gt)
            pred_bbox = get_bbox_from_mask(mask_pred)

            # =========================
            # Compare
            # 왼쪽: GT 원본 / 오른쪽: Pred 원본
            # 배경 합성 없음
            # =========================
            compare = np.concatenate([data_content, pred_content], axis=1)

            # =========================
            # bbox Compare
            # 왼쪽: GT + bbox / 오른쪽: Pred + bbox
            # =========================
            bbox_compare, gt_bbox_vis, pred_bbox_vis = make_bbox_visualization(
                data_content,
                pred_content,
                mask_gt,
                mask_pred,
            )

            # =========================
            # Metrics 계산
            # =========================
            psnr = compute_psnr(data_content, pred_content)
            ssim_score = compute_ssim(data_content, pred_content)
            iou = compute_iou(mask_gt, mask_pred)
            dice = compute_dice(mask_gt, mask_pred)
            bbox_iou = compute_bbox_iou_from_bboxes(gt_bbox, pred_bbox)

            psnr_values.append(psnr if np.isfinite(psnr) else np.nan)
            ssim_values.append(ssim_score)
            iou_values.append(iou)
            dice_values.append(dice)
            bbox_iou_values.append(bbox_iou)

            rows.append(
                {
                    "filename": filename,
                    "psnr": psnr,
                    "ssim": ssim_score,
                    "iou": iou,
                    "dice": dice,
                    "bbox_iou": bbox_iou,
                    "gt_bbox": gt_bbox,
                    "pred_bbox": pred_bbox,
                }
            )

            # =========================
            # FID 업데이트
            # =========================
            if fid_metric is not None:
                with torch.no_grad():
                    real_tensor = to_fid_tensor(data_content).to(device)
                    fake_tensor = to_fid_tensor(pred_content).to(device)

                    fid_metric.update(real_tensor, real=True)
                    fid_metric.update(fake_tensor, real=False)

            # =========================
            # 이미지 저장
            # =========================
            if args.save_images:
                cv2.imwrite(os.path.join(gt_dir, f"{name}.png"), data_content)
                cv2.imwrite(os.path.join(pred_dir, f"{name}.png"), pred_content)
                cv2.imwrite(os.path.join(compare_dir, f"{name}.png"), compare)

            # =========================
            # mask 저장
            # =========================
            if args.save_masks:
                cv2.imwrite(
                    os.path.join(mask_gt_dir, f"{name}.png"),
                    (mask_gt.astype(np.uint8) * 255),
                )

                cv2.imwrite(
                    os.path.join(mask_pred_dir, f"{name}.png"),
                    (mask_pred.astype(np.uint8) * 255),
                )

            # =========================
            # bbox 이미지 저장
            # =========================
            if args.save_bbox_images:
                cv2.imwrite(
                    os.path.join(bbox_compare_dir, f"{name}_bbox.png"),
                    bbox_compare,
                )

            saved_count += 1

    # =========================
    # compare 폴더 이미지들로 영상 생성
    # =========================
    if args.save_video:
        make_video_from_image_folder(
            image_folder=compare_dir,
            output_video_path=compare_video_file,
            fps=args.fps,
        )

    # =========================
    # bbox compare 폴더 이미지들로 영상 생성
    # =========================
    if args.save_bbox_video:
        make_video_from_image_folder(
            image_folder=bbox_compare_dir,
            output_video_path=bbox_compare_video_file,
            fps=args.fps,
        )

    # =========================
    # FID 최종 계산
    # =========================
    fid_score = None

    if fid_metric is not None:
        try:
            fid_score = float(fid_metric.compute().item())
        except Exception as e:
            print("FID 계산 실패:", e)
            fid_score = None

    # =========================
    # metrics.csv 저장
    # =========================
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename",
            "psnr",
            "ssim",
            "iou",
            "dice",
            "bbox_iou",
            "gt_bbox",
            "pred_bbox",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    # =========================
    # metrics_summary.csv 저장
    # =========================
    summary_rows = [
        {
            "metric": "PSNR",
            "mean": safe_mean(psnr_values),
            "std": safe_std(psnr_values),
        },
        {
            "metric": "SSIM",
            "mean": safe_mean(ssim_values),
            "std": safe_std(ssim_values),
        },
        {
            "metric": "IoU",
            "mean": safe_mean(iou_values),
            "std": safe_std(iou_values),
        },
        {
            "metric": "Dice",
            "mean": safe_mean(dice_values),
            "std": safe_std(dice_values),
        },
        {
            "metric": "bbox_IoU",
            "mean": safe_mean(bbox_iou_values),
            "std": safe_std(bbox_iou_values),
        },
    ]

    if fid_score is not None:
        summary_rows.append(
            {
                "metric": "FID",
                "mean": fid_score,
                "std": "",
            }
        )

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "mean", "std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    print("\n저장 완료!")
    print("처리된 이미지 개수:", saved_count)

    print("\n[평균 지표]")
    print("PSNR:", safe_mean(psnr_values))
    print("SSIM:", safe_mean(ssim_values))
    print("IoU:", safe_mean(iou_values))
    print("Dice:", safe_mean(dice_values))
    print("bbox IoU:", safe_mean(bbox_iou_values))

    if fid_score is not None:
        print("FID:", fid_score)
    else:
        print("FID: 계산 안 함")

    print("\n[저장 경로]")
    print("metrics.csv:", metrics_csv_path)
    print("metrics_summary.csv:", summary_csv_path)

    if args.save_images:
        print("GT 이미지 폴더:", gt_dir)
        print("Pred 이미지 폴더:", pred_dir)
        print("Compare 이미지 폴더:", compare_dir)

    if args.save_bbox_images:
        print("bbox Compare 이미지 폴더:", bbox_compare_dir)

    if args.save_masks:
        print("GT mask 폴더:", mask_gt_dir)
        print("Pred mask 폴더:", mask_pred_dir)

    if args.save_video:
        print("Compare 영상:", compare_video_file)

    if args.save_bbox_video:
        print("bbox Compare 영상:", bbox_compare_video_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_dir",
        type=str,
        default=r"C:\Users\user\Desktop\3d\CSI-to-image-reconstruction\03_Model_Training\data\rx1_test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\user\Desktop\3d\CSI-to-image-reconstruction\03_Model_Training\03_Mesh\outputs\0708_outputs_rx1_dice_iou_loss",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=r"C:\Users\user\Desktop\3d\CSI-to-image-reconstruction\03_Model_Training\03_Mesh\outputs\0708_outputs_rx1_dice_iou_loss\latest-epoch=199-val_loss=0.2015403.ckpt",
    )

    parser.add_argument("--window_size", type=int, default=151)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fps", type=int, default=10)

    # 저장 옵션
    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--no_save_images", dest="save_images", action="store_false")

    parser.add_argument("--save_video", action="store_true", default=True)
    parser.add_argument("--no_save_video", dest="save_video", action="store_false")

    parser.add_argument("--save_bbox_images", action="store_true", default=True)
    parser.add_argument("--no_save_bbox_images", dest="save_bbox_images", action="store_false")

    parser.add_argument("--save_bbox_video", action="store_true", default=True)
    parser.add_argument("--no_save_bbox_video", dest="save_bbox_video", action="store_false")

    parser.add_argument("--save_masks", action="store_true", default=False)

    # metric / mask 옵션
    parser.add_argument("--mask_threshold", type=int, default=10)
    parser.add_argument("--use_morphology", action="store_true")

    parser.add_argument("--keep_largest_gt", action="store_true", default=False)
    parser.add_argument("--keep_largest_pred", action="store_true", default=False)

    # FID 옵션
    parser.add_argument("--fid", action="store_true")

    args = parser.parse_args()

    main(args)
