from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NormalizationStats, WificamDataset, discover_csi_csvs
from vae import VAE


# ============================================================
# 실행 인자
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",type=str,default="../data/rx1_test")
    parser.add_argument("--checkpoint",type=str,default="./outputs/0714_window_mean_vae/checkpoints/epochs/epoch-199-val_loss-0.0178.ckpt" )
    parser.add_argument("--normalization_path", type=str,default="./outputs/0714_window_mean_vae/normalization.json")
    parser.add_argument("--output_dir",type=str,default="./outputs/0714_window_mean_vae")
    parser.add_argument("--window_size", type=int, default=151)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # 0~1 범위 이미지 기준 마스크 임계값
    parser.add_argument("--mask_threshold", type=float, default=0.05)

    # 영상 FPS
    parser.add_argument("--fps", type=int, default=10)

    # 이미지 저장 옵션
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_save_images",
        dest="save_images",
        action="store_false",
    )

    # bbox 이미지 저장 옵션
    parser.add_argument(
        "--save_bbox_images",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_save_bbox_images",
        dest="save_bbox_images",
        action="store_false",
    )

    # 마스크 저장 옵션
    parser.add_argument(
        "--save_masks",
        action="store_true",
        default=True,
    )

    # 일반 비교 영상 저장 옵션
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_save_video",
        dest="save_video",
        action="store_false",
    )

    # bbox 비교 영상 저장 옵션
    parser.add_argument(
        "--save_bbox_video",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_save_bbox_video",
        dest="save_bbox_video",
        action="store_false",
    )

    return parser.parse_args()


# ============================================================
# Metric 함수
# ============================================================
def batch_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    mse = (
        (prediction - target)
        .pow(2)
        .flatten(1)
        .mean(dim=1)
    )

    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


def batch_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    padding = window_size // 2

    mu_pred = F.avg_pool2d(
        prediction,
        window_size,
        stride=1,
        padding=padding,
    )
    mu_target = F.avg_pool2d(
        target,
        window_size,
        stride=1,
        padding=padding,
    )

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred = F.avg_pool2d(
        prediction * prediction,
        window_size,
        stride=1,
        padding=padding,
    ) - mu_pred_sq

    sigma_target = F.avg_pool2d(
        target * target,
        window_size,
        stride=1,
        padding=padding,
    ) - mu_target_sq

    sigma_cross = F.avg_pool2d(
        prediction * target,
        window_size,
        stride=1,
        padding=padding,
    ) - mu_cross

    c1 = 0.01**2
    c2 = 0.03**2

    numerator = (
        (2.0 * mu_cross + c1)
        * (2.0 * sigma_cross + c2)
    )

    denominator = (
        (mu_pred_sq + mu_target_sq + c1)
        * (sigma_pred + sigma_target + c2)
    )

    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map.flatten(1).mean(dim=1)


def masks_from_images(
    images: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    images: B, C, H, W
    return: B, H, W bool mask
    """
    return images.mean(dim=1) > threshold


def batch_iou_dice(
    prediction_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intersection = (
        (prediction_mask & target_mask)
        .flatten(1)
        .sum(dim=1)
        .float()
    )

    union = (
        (prediction_mask | target_mask)
        .flatten(1)
        .sum(dim=1)
        .float()
    )

    pred_area = (
        prediction_mask
        .flatten(1)
        .sum(dim=1)
        .float()
    )

    target_area = (
        target_mask
        .flatten(1)
        .sum(dim=1)
        .float()
    )

    iou = (intersection + 1e-8) / (union + 1e-8)

    dice = (2.0 * intersection + 1e-8) / (
        pred_area + target_area + 1e-8
    )

    return iou, dice


# ============================================================
# Bounding Box 함수
# ============================================================
def bbox_from_mask(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """
    mask: H, W bool 배열

    return:
        (x1, y1, x2, y2)
        마스크가 없으면 None
    """
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def bbox_iou(
    first: tuple[int, int, int, int] | None,
    second: tuple[int, int, int, int] | None,
) -> float:
    if first is None and second is None:
        return 1.0

    if first is None or second is None:
        return 0.0

    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection_width = max(0, x2 - x1 + 1)
    intersection_height = max(0, y2 - y1 + 1)

    intersection = intersection_width * intersection_height

    first_area = (
        (first[2] - first[0] + 1)
        * (first[3] - first[1] + 1)
    )

    second_area = (
        (second[2] - second[0] + 1)
        * (second[3] - second[1] + 1)
    )

    union = first_area + second_area - intersection

    return float(intersection / max(union, 1))


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    label: str,
) -> np.ndarray:
    """
    image: uint8 BGR 이미지
    """
    output = image.copy()

    if bbox is None:
        cv2.putText(
            output,
            f"{label}: No bbox",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        output,
        label,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# 이미지 변환 및 저장 함수
# ============================================================
def tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    """
    image: C, H, W / RGB / 0~1 범위

    return:
        H, W, C / BGR / uint8
    """
    image_np = (
        image.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )

    image_np = (image_np * 255.0).astype(np.uint8)

    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


def save_image(
    output_path: Path,
    image: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"이미지 저장 실패: {output_path}"
        )


def make_bbox_visualization(
    target_bgr: np.ndarray,
    prediction_bgr: np.ndarray,
    target_bbox: tuple[int, int, int, int] | None,
    prediction_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    target_bbox_image = draw_bbox(
        target_bgr,
        target_bbox,
        "GT",
    )

    prediction_bbox_image = draw_bbox(
        prediction_bgr,
        prediction_bbox,
        "Pred",
    )

    return np.concatenate(
        [
            target_bbox_image,
            prediction_bbox_image,
        ],
        axis=1,
    )


# ============================================================
# 이미지 폴더를 MP4 영상으로 변환
# ============================================================
def natural_key(path: Path) -> tuple[int, str]:
    """
    파일 이름의 앞쪽 sample_index를 기준으로 정렬
    예:
        000001_123.png
        000002_124.png
    """
    first_part = path.stem.split("_")[0]

    try:
        return int(first_part), path.name
    except ValueError:
        return 0, path.name


def make_video_from_image_folder(
    image_folder: Path,
    output_video_path: Path,
    fps: int = 10,
) -> None:
    image_files = sorted(
        [
            path
            for path in image_folder.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ],
        key=natural_key,
    )

    if not image_files:
        print(
            "영상으로 만들 이미지가 없습니다:",
            image_folder,
        )
        return

    first_frame = cv2.imread(str(image_files[0]))

    if first_frame is None:
        print(
            "첫 번째 프레임을 읽지 못했습니다:",
            image_files[0],
        )
        return

    height, width = first_frame.shape[:2]

    output_video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"영상 파일을 열 수 없습니다: {output_video_path}"
        )

    for image_path in tqdm(
        image_files,
        desc=f"영상 생성: {output_video_path.name}",
    ):
        frame = cv2.imread(str(image_path))

        if frame is None:
            print("이미지 읽기 실패:", image_path)
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(
                frame,
                (width, height),
            )

        writer.write(frame)

    writer.release()

    print("영상 저장 완료:", output_video_path)


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    # --------------------------------------------------------
    # 저장 폴더 구성
    # --------------------------------------------------------
    output_dir = Path(args.output_dir)

    image_dir = output_dir / "images"

    gt_dir = image_dir / "gt"
    pred_dir = image_dir / "pred"
    compare_dir = image_dir / "compare"
    bbox_compare_dir = image_dir / "bbox_compare"

    mask_dir = image_dir / "mask"
    mask_gt_dir = mask_dir / "gt"
    mask_pred_dir = mask_dir / "pred"

    compare_video_path = output_dir / "compare.mp4"
    bbox_compare_video_path = output_dir / "bbox_compare.mp4"

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "metrics_summary.csv"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.save_images:
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)

    if args.save_bbox_images:
        bbox_compare_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if args.save_masks:
        mask_gt_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        mask_pred_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # 장치 설정
    # --------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)
    print("data_dir:", args.data_dir)
    print("checkpoint:", args.checkpoint)
    print("normalization_path:", args.normalization_path)
    print("output_dir:", output_dir)
    print("mask_threshold:", args.mask_threshold)
    print("save_images:", args.save_images)
    print("save_bbox_images:", args.save_bbox_images)
    print("save_masks:", args.save_masks)
    print("save_video:", args.save_video)
    print("save_bbox_video:", args.save_bbox_video)

    # --------------------------------------------------------
    # 데이터셋
    # --------------------------------------------------------
    stats = NormalizationStats.load(
        args.normalization_path
    )

    csv_paths = discover_csi_csvs(
        args.data_dir
    )

    if not csv_paths:
        raise RuntimeError(
            f"CSI CSV를 찾지 못했습니다: {args.data_dir}"
        )

    dataset = WificamDataset(
        csv_paths=csv_paths,
        window_size=args.window_size,
        normalization_stats=stats,
        is_train=False,
        noise_std=0.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    print("평가 데이터 개수:", len(dataset))

    # --------------------------------------------------------
    # 모델
    # --------------------------------------------------------
    model = VAE.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # 평가용 변수
    # --------------------------------------------------------
    rows: list[dict[str, object]] = []

    sample_index = 0
    saved_image_count = 0
    saved_bbox_count = 0
    saved_mask_count = 0

    # --------------------------------------------------------
    # 추론
    # --------------------------------------------------------
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="추론 및 이미지 저장",
        ):
            csi = batch["csi"].to(
                device,
                non_blocking=True,
            )

            window_mean = batch["window_mean"].to(
                device,
                non_blocking=True,
            )

            target = batch["image"].to(
                device,
                non_blocking=True,
            )

            image_paths = batch["image_path"]

            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_context:
                _, prediction = model(
                    csi,
                    window_mean,
                )

            prediction = prediction.float()
            target = target.float()

            # ------------------------------------------------
            # Metric 계산
            # ------------------------------------------------
            psnr_values = batch_psnr(
                prediction,
                target,
            )

            ssim_values = batch_ssim(
                prediction,
                target,
            )

            prediction_mask = masks_from_images(
                prediction,
                args.mask_threshold,
            )

            target_mask = masks_from_images(
                target,
                args.mask_threshold,
            )

            iou_values, dice_values = batch_iou_dice(
                prediction_mask,
                target_mask,
            )

            prediction_mask_np = (
                prediction_mask
                .detach()
                .cpu()
                .numpy()
            )

            target_mask_np = (
                target_mask
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # 배치 내부 샘플별 저장
            # ------------------------------------------------
            for batch_index in range(prediction.shape[0]):
                current_image_path = str(
                    image_paths[batch_index]
                )

                original_name = Path(
                    current_image_path
                ).stem

                # 같은 파일 이름이 여러 폴더에 존재하더라도
                # 덮어쓰지 않도록 sample_index 추가
                save_name = (
                    f"{sample_index:06d}_"
                    f"{original_name}.png"
                )

                target_bgr = tensor_to_bgr(
                    target[batch_index]
                )

                prediction_bgr = tensor_to_bgr(
                    prediction[batch_index]
                )

                # 출력 크기가 다른 경우 GT 크기에 맞춤
                target_height, target_width = (
                    target_bgr.shape[:2]
                )

                if prediction_bgr.shape[:2] != (
                    target_height,
                    target_width,
                ):
                    prediction_bgr = cv2.resize(
                        prediction_bgr,
                        (target_width, target_height),
                    )

                current_target_mask = (
                    target_mask_np[batch_index]
                )

                current_prediction_mask = (
                    prediction_mask_np[batch_index]
                )

                target_bbox = bbox_from_mask(
                    current_target_mask
                )

                prediction_bbox = bbox_from_mask(
                    current_prediction_mask
                )

                current_bbox_iou = bbox_iou(
                    prediction_bbox,
                    target_bbox,
                )

                window_mean_value = float(
                    window_mean[batch_index]
                    .reshape(-1)[0]
                    .item()
                )

                rows.append(
                    {
                        "sample_index": sample_index,
                        "image_path": current_image_path,
                        "window_mean": window_mean_value,
                        "psnr": float(
                            psnr_values[batch_index].item()
                        ),
                        "ssim": float(
                            ssim_values[batch_index].item()
                        ),
                        "iou": float(
                            iou_values[batch_index].item()
                        ),
                        "dice": float(
                            dice_values[batch_index].item()
                        ),
                        "bbox_iou": current_bbox_iou,
                        "gt_bbox": target_bbox,
                        "pred_bbox": prediction_bbox,
                    }
                )

                # --------------------------------------------
                # GT / Pred / Compare 이미지 저장
                # --------------------------------------------
                if args.save_images:
                    compare_image = np.concatenate(
                        [
                            target_bgr,
                            prediction_bgr,
                        ],
                        axis=1,
                    )

                    save_image(
                        gt_dir / save_name,
                        target_bgr,
                    )

                    save_image(
                        pred_dir / save_name,
                        prediction_bgr,
                    )

                    save_image(
                        compare_dir / save_name,
                        compare_image,
                    )

                    saved_image_count += 1

                # --------------------------------------------
                # bbox 비교 이미지 저장
                # --------------------------------------------
                if args.save_bbox_images:
                    bbox_compare_image = (
                        make_bbox_visualization(
                            target_bgr=target_bgr,
                            prediction_bgr=prediction_bgr,
                            target_bbox=target_bbox,
                            prediction_bbox=prediction_bbox,
                        )
                    )

                    bbox_save_name = (
                        f"{sample_index:06d}_"
                        f"{original_name}_bbox.png"
                    )

                    save_image(
                        bbox_compare_dir / bbox_save_name,
                        bbox_compare_image,
                    )

                    saved_bbox_count += 1

                # --------------------------------------------
                # GT / Pred 마스크 저장
                # --------------------------------------------
                if args.save_masks:
                    target_mask_image = (
                        current_target_mask.astype(
                            np.uint8
                        )
                        * 255
                    )

                    prediction_mask_image = (
                        current_prediction_mask.astype(
                            np.uint8
                        )
                        * 255
                    )

                    save_image(
                        mask_gt_dir / save_name,
                        target_mask_image,
                    )

                    save_image(
                        mask_pred_dir / save_name,
                        prediction_mask_image,
                    )

                    saved_mask_count += 1

                sample_index += 1

    if not rows:
        raise RuntimeError(
            "평가 샘플이 없습니다."
        )

    # --------------------------------------------------------
    # 샘플별 metrics.csv 저장
    # --------------------------------------------------------
    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # 평균 및 표준편차 계산
    # --------------------------------------------------------
    metric_names = [
        "psnr",
        "ssim",
        "iou",
        "dice",
        "bbox_iou",
    ]

    summary: dict[str, object] = {
        "sample_count": len(rows),
        "checkpoint": str(
            Path(args.checkpoint).resolve()
        ),
        "normalization_path": str(
            Path(args.normalization_path).resolve()
        ),
        "mask_threshold": args.mask_threshold,
    }

    summary_csv_rows: list[dict[str, object]] = []

    for metric_name in metric_names:
        values = np.asarray(
            [
                float(row[metric_name])
                for row in rows
            ],
            dtype=np.float64,
        )

        metric_mean = float(
            np.nanmean(values)
        )

        metric_std = float(
            np.nanstd(values)
        )

        summary[metric_name] = {
            "mean": metric_mean,
            "std": metric_std,
        }

        summary_csv_rows.append(
            {
                "metric": metric_name,
                "mean": metric_mean,
                "std": metric_std,
            }
        )

    # --------------------------------------------------------
    # summary.json 저장
    # --------------------------------------------------------
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # metrics_summary.csv 저장
    # --------------------------------------------------------
    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "mean",
                "std",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_csv_rows)

    # --------------------------------------------------------
    # Compare 영상 생성
    # --------------------------------------------------------
    if args.save_video:
        make_video_from_image_folder(
            image_folder=compare_dir,
            output_video_path=compare_video_path,
            fps=args.fps,
        )

    # --------------------------------------------------------
    # bbox Compare 영상 생성
    # --------------------------------------------------------
    if args.save_bbox_video:
        make_video_from_image_folder(
            image_folder=bbox_compare_dir,
            output_video_path=bbox_compare_video_path,
            fps=args.fps,
        )

    # --------------------------------------------------------
    # 결과 출력
    # --------------------------------------------------------
    print("\n평가 완료")
    print("전체 평가 샘플:", len(rows))
    print("저장된 GT/Pred/Compare 이미지:", saved_image_count)
    print("저장된 bbox 이미지:", saved_bbox_count)
    print("저장된 마스크 이미지:", saved_mask_count)

    print("\n[평균 지표]")

    for metric_name in metric_names:
        metric_summary = summary[metric_name]

        print(
            f"{metric_name}: "
            f"{metric_summary['mean']} "
            f"(std: {metric_summary['std']})"
        )

    print("\n[저장 경로]")
    print("Metrics:", metrics_path)
    print("Summary JSON:", summary_path)
    print("Summary CSV:", summary_csv_path)

    if args.save_images:
        print("GT 이미지:", gt_dir)
        print("Pred 이미지:", pred_dir)
        print("Compare 이미지:", compare_dir)

    if args.save_bbox_images:
        print("bbox Compare 이미지:", bbox_compare_dir)

    if args.save_masks:
        print("GT Mask 이미지:", mask_gt_dir)
        print("Pred Mask 이미지:", mask_pred_dir)

    if args.save_video:
        print("Compare 영상:", compare_video_path)

    if args.save_bbox_video:
        print("bbox Compare 영상:", bbox_compare_video_path)


if __name__ == "__main__":
    main()
