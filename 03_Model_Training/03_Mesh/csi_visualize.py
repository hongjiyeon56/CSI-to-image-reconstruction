"""
analyze_csi_patterns.py

목적
----
1) 각 target PNG를 가장 가까운 CSI id와 매칭
2) 해당 시점을 중심으로 151 x 52 amplitude window 생성
3) target 사람 크기(bbox height)를 거리 proxy로 사용
4) 가까운 그룹 / 먼 그룹 CSI 차이 비교
5) 사람 크기와 CSI 통계량의 상관관계 확인
6) 대표 near/far heatmap, 그룹 평균 subcarrier profile, PCA 저장

현재 모델과 동일하게 amplitude만 사용:
raw CSI -> real/imag 분리 -> amplitude -> log -> normalization
-> clipping -> subcarrier smoothing

주의
----
- bbox_height가 클수록 사람을 더 "가까운 것"으로 간주한다.
- target PNG가 검은 화면이면 해당 샘플은 분석에서 제외한다.
- 이 스크립트는 선택한 csi.csv 한 파일 기준으로 정규화한다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSI_VALID_SUBCARRIER_INDEX = (
    list(range(6, 32))
    + list(range(33, 59))
)
NUM_SUBCARRIERS = len(CSI_VALID_SUBCARRIER_INDEX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSI amplitude 패턴 분석"
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="생략하면 csi.csv와 같은 폴더 사용",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./csi_pattern_analysis",
    )
    parser.add_argument("--window_size", type=int, default=151)
    parser.add_argument("--start_image_id", type=int, default=None)
    parser.add_argument("--end_image_id", type=int, default=None)
    parser.add_argument(
        "--image_stride",
        type=int,
        default=1,
        help="1이면 모든 이미지, 5이면 5장마다 1장",
    )
    parser.add_argument(
        "--black_threshold",
        type=int,
        default=10,
        help="사람 영역 판단용 pixel threshold",
    )
    parser.add_argument(
        "--min_nonzero_ratio",
        type=float,
        default=0.0005,
        help="이 비율보다 밝은 pixel이 적으면 검은 이미지로 제외",
    )
    parser.add_argument(
        "--near_quantile",
        type=float,
        default=0.75,
        help="bbox height 상위 quantile을 near 그룹으로 사용",
    )
    parser.add_argument(
        "--far_quantile",
        type=float,
        default=0.25,
        help="bbox height 하위 quantile을 far 그룹으로 사용",
    )
    parser.add_argument(
        "--representative_count",
        type=int,
        default=3,
        help="near/far 대표 샘플 개수",
    )
    return parser.parse_args()


def load_csi(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)

    required = {"id", "data"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {sorted(missing)}")

    df = df.sort_values("id").reset_index(drop=True)

    rows = []
    lengths = []

    for row_index, value in enumerate(df["data"]):
        try:
            arr = np.asarray(json.loads(value), dtype=np.float32)
        except Exception as exc:
            raise ValueError(
                f"{row_index}번째 CSI 행 파싱 실패"
            ) from exc

        rows.append(arr)
        lengths.append(len(arr))

    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        raise ValueError(
            f"CSI 행 길이가 서로 다릅니다: {unique_lengths}"
        )

    return df, np.stack(rows, axis=0)


def extract_amplitude(raw_csi: np.ndarray) -> np.ndarray:
    real_indices = np.asarray(
        [i * 2 for i in CSI_VALID_SUBCARRIER_INDEX],
        dtype=np.int64,
    )
    imag_indices = np.asarray(
        [i * 2 - 1 for i in CSI_VALID_SUBCARRIER_INDEX],
        dtype=np.int64,
    )

    required_index = int(
        max(real_indices.max(), imag_indices.max())
    )

    if raw_csi.shape[1] <= required_index:
        raise ValueError(
            f"CSI row length={raw_csi.shape[1]}, "
            f"필요 index={required_index}"
        )

    real = raw_csi[:, real_indices]
    imag = raw_csi[:, imag_indices]

    amplitude = np.sqrt(
        real.astype(np.float64) ** 2
        + imag.astype(np.float64) ** 2
    )

    return amplitude.astype(np.float32)


def smooth_subcarriers(data: np.ndarray) -> np.ndarray:
    padded = np.pad(
        data,
        pad_width=((0, 0), (1, 1)),
        mode="constant",
        constant_values=0,
    )

    return (
        padded[:, :-2]
        + padded[:, 1:-1]
        + padded[:, 2:]
    ) / 3.0


def preprocess_amplitude(
    amplitude: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    processed = np.log(amplitude.astype(np.float32) + 1e-6)

    mean = float(processed.mean())
    std = float(processed.std())

    processed = (processed - mean) / (std + 1e-6)
    processed = np.clip(processed, -3, 3)
    processed = smooth_subcarriers(processed)
    processed = np.clip(processed, -3, 3)

    return processed.astype(np.float32), mean, std


def list_images(
    image_dir: Path,
    start_image_id: int | None,
    end_image_id: int | None,
    stride: int,
) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []

    for path in image_dir.glob("*.png"):
        try:
            image_id = int(path.stem)
        except ValueError:
            continue

        if start_image_id is not None and image_id < start_image_id:
            continue
        if end_image_id is not None and image_id > end_image_id:
            continue

        items.append((image_id, path))

    items.sort(key=lambda x: x[0])
    return items[::max(1, stride)]


def extract_bbox_metrics(
    image_path: Path,
    threshold: int,
    min_nonzero_ratio: float,
) -> dict[str, float] | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        return None

    if image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mask = gray > threshold
    nonzero_ratio = float(mask.mean())

    if nonzero_ratio < min_nonzero_ratio:
        return None

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    bbox_area = int(bbox_width * bbox_height)
    foreground_area = int(mask.sum())

    return {
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "foreground_area": foreground_area,
        "nonzero_ratio": nonzero_ratio,
    }


def find_nearest_csi_row(
    csi_ids: np.ndarray,
    image_id: int,
) -> int:
    return int(np.abs(csi_ids - image_id).argmin())


def extract_centered_window(
    data: np.ndarray,
    center_row: int,
    window_size: int,
) -> tuple[np.ndarray, int, int]:
    if len(data) < window_size:
        raise ValueError(
            f"전체 패킷 수({len(data)})가 "
            f"window_size({window_size})보다 작습니다."
        )

    start = center_row - window_size // 2
    start = max(0, min(start, len(data) - window_size))
    end = start + window_size

    return data[start:end], start, end


def compute_window_features(window: np.ndarray) -> dict[str, float]:
    temporal_centered = (
        window - window.mean(axis=0, keepdims=True)
    )

    temporal_std_by_subcarrier = window.std(axis=0)

    motion_energy = float(
        np.abs(np.diff(window, axis=0)).mean()
    )

    temporal_std = float(
        temporal_std_by_subcarrier.mean()
    )

    frequency_std = float(
        window.std(axis=1).mean()
    )

    centered_energy = float(
        np.abs(temporal_centered).mean()
    )

    return {
        "window_mean": float(window.mean()),
        "window_std": float(window.std()),
        "temporal_std": temporal_std,
        "motion_energy": motion_energy,
        "frequency_std": frequency_std,
        "centered_energy": centered_energy,
    }


def save_heatmap(
    data: np.ndarray,
    path: Path,
    title: str,
    colorbar_label: str,
    center_zero: bool = False,
) -> None:
    plt.figure(figsize=(10, 7))

    kwargs = {}
    if center_zero:
        limit = float(
            np.percentile(np.abs(data), 99)
        )
        limit = max(limit, 1e-6)
        kwargs["vmin"] = -limit
        kwargs["vmax"] = limit
        kwargs["cmap"] = "RdBu_r"

    image = plt.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        **kwargs,
    )

    positions = np.arange(NUM_SUBCARRIERS)
    labels = np.asarray(CSI_VALID_SUBCARRIER_INDEX)

    plt.xticks(
        positions[::5],
        labels[::5],
    )
    plt.xlabel("Subcarrier index")
    plt.ylabel("Packet within window")
    plt.title(title)

    colorbar = plt.colorbar(image)
    colorbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(x, y)[0, 1])

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.65)

    if len(x) >= 2 and np.std(x) > 0:
        coeff = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coeff[0] * x_line + coeff[1]
        plt.plot(x_line, y_line)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nPearson r = {corr:.4f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    return corr


def save_group_profile(
    near_profile: np.ndarray,
    far_profile: np.ndarray,
    path: Path,
) -> None:
    x = np.asarray(CSI_VALID_SUBCARRIER_INDEX)

    plt.figure(figsize=(10, 5))
    plt.plot(x, near_profile, label="Near group mean")
    plt.plot(x, far_profile, label="Far group mean")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Mean processed amplitude")
    plt.title("Near vs far mean subcarrier profile")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_group_difference(
    difference: np.ndarray,
    path: Path,
) -> None:
    x = np.asarray(CSI_VALID_SUBCARRIER_INDEX)

    plt.figure(figsize=(10, 5))
    plt.plot(x, difference)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Subcarrier index")
    plt.ylabel("Near mean - Far mean")
    plt.title("Subcarrier difference between near and far groups")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_metric_group_comparison(
    result_df: pd.DataFrame,
    near_mask: np.ndarray,
    far_mask: np.ndarray,
    path: Path,
) -> None:
    metric_names = [
        "window_mean",
        "window_std",
        "temporal_std",
        "motion_energy",
        "frequency_std",
        "centered_energy",
    ]

    near_values = [
        result_df.loc[near_mask, name].mean()
        for name in metric_names
    ]
    far_values = [
        result_df.loc[far_mask, name].mean()
        for name in metric_names
    ]

    x = np.arange(len(metric_names))
    width = 0.38

    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, near_values, width, label="Near")
    plt.bar(x + width / 2, far_values, width, label="Far")
    plt.xticks(x, metric_names, rotation=25, ha="right")
    plt.ylabel("Mean metric value")
    plt.title("Near vs far CSI feature comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pca_plot(
    sample_vectors: np.ndarray,
    bbox_heights: np.ndarray,
    path: Path,
) -> None:
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("Warning: scikit-learn이 없어 PCA를 건너뜁니다.")
        return

    x = sample_vectors.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(x)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=bbox_heights,
        alpha=0.75,
    )
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(
        "PCA of CSI windows\n"
        f"explained variance="
        f"{pca.explained_variance_ratio_[0]:.3f}, "
        f"{pca.explained_variance_ratio_[1]:.3f}"
    )
    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Target bbox height")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    image_dir = (
        Path(args.image_dir)
        if args.image_dir is not None
        else csv_path.parent
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    representative_dir = output_dir / "representative_heatmaps"
    representative_dir.mkdir(parents=True, exist_ok=True)

    print("1) CSI 로딩")
    df, raw_csi = load_csi(csv_path)

    print("2) Amplitude 계산 및 전처리")
    amplitude = extract_amplitude(raw_csi)
    processed, global_mean, global_std = preprocess_amplitude(
        amplitude
    )

    csi_ids = df["id"].to_numpy(dtype=np.int64)

    print("3) 이미지 목록 확인")
    image_items = list_images(
        image_dir=image_dir,
        start_image_id=args.start_image_id,
        end_image_id=args.end_image_id,
        stride=args.image_stride,
    )

    if not image_items:
        raise RuntimeError("분석할 PNG 이미지가 없습니다.")

    records: list[dict[str, float | int | str]] = []
    windows: list[np.ndarray] = []
    mean_profiles: list[np.ndarray] = []

    skipped_black = 0
    skipped_read = 0

    print("4) 이미지별 CSI window와 통계 계산")
    for index, (image_id, image_path) in enumerate(image_items):
        bbox_metrics = extract_bbox_metrics(
            image_path,
            threshold=args.black_threshold,
            min_nonzero_ratio=args.min_nonzero_ratio,
        )

        if bbox_metrics is None:
            skipped_black += 1
            continue

        center_row = find_nearest_csi_row(
            csi_ids,
            image_id,
        )
        matched_csi_id = int(csi_ids[center_row])

        window, window_start, window_end = extract_centered_window(
            processed,
            center_row=center_row,
            window_size=args.window_size,
        )

        feature_values = compute_window_features(window)

        record = {
            "image_id": image_id,
            "image_path": str(image_path),
            "matched_csi_id": matched_csi_id,
            "csi_id_difference": abs(matched_csi_id - image_id),
            "center_row": center_row,
            "window_start": window_start,
            "window_end": window_end,
            **bbox_metrics,
            **feature_values,
        }

        records.append(record)
        windows.append(window)
        mean_profiles.append(window.mean(axis=0))

        if index % 100 == 0:
            print(
                f"[{index + 1}/{len(image_items)}] "
                f"image={image_id}, CSI={matched_csi_id}"
            )

    if not records:
        raise RuntimeError(
            "유효한 target 이미지가 없습니다. "
            "black threshold 설정을 확인하세요."
        )

    result_df = pd.DataFrame(records)
    windows_array = np.stack(windows, axis=0)
    profiles_array = np.stack(mean_profiles, axis=0)

    result_csv_path = output_dir / "csi_pattern_features.csv"
    result_df.to_csv(result_csv_path, index=False)

    print(f"유효 샘플 수: {len(result_df)}")
    print(f"검은 이미지 제외 수: {skipped_black}")

    bbox_heights = result_df["bbox_height"].to_numpy(dtype=float)

    far_threshold = float(
        np.quantile(bbox_heights, args.far_quantile)
    )
    near_threshold = float(
        np.quantile(bbox_heights, args.near_quantile)
    )

    far_mask = bbox_heights <= far_threshold
    near_mask = bbox_heights >= near_threshold

    result_df["distance_group"] = "middle"
    result_df.loc[far_mask, "distance_group"] = "far"
    result_df.loc[near_mask, "distance_group"] = "near"
    result_df.to_csv(result_csv_path, index=False)

    print(
        f"Far threshold bbox height <= {far_threshold:.2f}, "
        f"samples={int(far_mask.sum())}"
    )
    print(
        f"Near threshold bbox height >= {near_threshold:.2f}, "
        f"samples={int(near_mask.sum())}"
    )

    # 5) bbox height와 각 CSI 지표 scatter
    metric_names = [
        "window_mean",
        "window_std",
        "temporal_std",
        "motion_energy",
        "frequency_std",
        "centered_energy",
    ]

    correlations: dict[str, float] = {}

    for metric_name in metric_names:
        corr = save_scatter(
            x=bbox_heights,
            y=result_df[metric_name].to_numpy(dtype=float),
            path=output_dir / f"bbox_vs_{metric_name}.png",
            xlabel="Target bbox height (distance proxy)",
            ylabel=metric_name,
            title=f"Bbox height vs {metric_name}",
        )
        correlations[metric_name] = corr

    # 6) near/far 그룹 평균 subcarrier profile
    near_profile = profiles_array[near_mask].mean(axis=0)
    far_profile = profiles_array[far_mask].mean(axis=0)

    save_group_profile(
        near_profile=near_profile,
        far_profile=far_profile,
        path=output_dir / "near_vs_far_mean_profile.png",
    )

    save_group_difference(
        difference=near_profile - far_profile,
        path=output_dir / "near_minus_far_profile.png",
    )

    # 7) near/far 평균 temporal-centered heatmap
    centered_windows = (
        windows_array
        - windows_array.mean(axis=1, keepdims=True)
    )

    near_centered_mean = centered_windows[near_mask].mean(axis=0)
    far_centered_mean = centered_windows[far_mask].mean(axis=0)
    centered_difference = near_centered_mean - far_centered_mean

    save_heatmap(
        near_centered_mean,
        output_dir / "near_group_centered_heatmap.png",
        "Near group mean temporal-centered CSI",
        "Centered amplitude",
        center_zero=True,
    )

    save_heatmap(
        far_centered_mean,
        output_dir / "far_group_centered_heatmap.png",
        "Far group mean temporal-centered CSI",
        "Centered amplitude",
        center_zero=True,
    )

    save_heatmap(
        centered_difference,
        output_dir / "near_minus_far_centered_heatmap.png",
        "Near - Far temporal-centered CSI difference",
        "Centered amplitude difference",
        center_zero=True,
    )

    # 8) near/far feature 평균 비교
    save_metric_group_comparison(
        result_df=result_df,
        near_mask=near_mask,
        far_mask=far_mask,
        path=output_dir / "near_vs_far_feature_means.png",
    )

    # 9) 대표 샘플 heatmap
    representative_count = max(1, args.representative_count)

    far_indices = np.where(far_mask)[0]
    near_indices = np.where(near_mask)[0]

    far_sorted = far_indices[
        np.argsort(bbox_heights[far_indices])
    ]
    near_sorted = near_indices[
        np.argsort(-bbox_heights[near_indices])
    ]

    for group_name, selected_indices in [
        ("far", far_sorted[:representative_count]),
        ("near", near_sorted[:representative_count]),
    ]:
        for rank, sample_index in enumerate(selected_indices, start=1):
            row = result_df.iloc[sample_index]
            window = windows_array[sample_index]
            centered = window - window.mean(axis=0, keepdims=True)

            save_heatmap(
                window,
                representative_dir
                / f"{group_name}_{rank}_processed_"
                  f"image_{int(row['image_id'])}.png",
                (
                    f"{group_name.upper()} sample {rank}: "
                    f"image {int(row['image_id'])}, "
                    f"bbox height {int(row['bbox_height'])}"
                ),
                "Processed amplitude",
                center_zero=False,
            )

            save_heatmap(
                centered,
                representative_dir
                / f"{group_name}_{rank}_centered_"
                  f"image_{int(row['image_id'])}.png",
                (
                    f"{group_name.upper()} centered sample {rank}: "
                    f"image {int(row['image_id'])}, "
                    f"bbox height {int(row['bbox_height'])}"
                ),
                "Centered amplitude",
                center_zero=True,
            )

    # 10) PCA
    flattened_centered = centered_windows.reshape(
        len(centered_windows),
        -1,
    )

    save_pca_plot(
        flattened_centered,
        bbox_heights,
        output_dir / "pca_csi_windows.png",
    )

    # 11) 요약 txt
    summary_lines = [
        f"CSV: {csv_path}",
        f"Image directory: {image_dir}",
        f"Valid samples: {len(result_df)}",
        f"Skipped black images: {skipped_black}",
        f"Window size: {args.window_size}",
        f"Processed global mean: {global_mean:.6f}",
        f"Processed global std: {global_std:.6f}",
        f"Far bbox threshold: {far_threshold:.6f}",
        f"Near bbox threshold: {near_threshold:.6f}",
        f"Far samples: {int(far_mask.sum())}",
        f"Near samples: {int(near_mask.sum())}",
        "",
        "Pearson correlation with bbox height:",
    ]

    for metric_name, corr in correlations.items():
        summary_lines.append(
            f"{metric_name}: {corr:.6f}"
        )

    (output_dir / "analysis_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("분석 완료")
    print(f"결과 폴더: {output_dir.resolve()}")
    print("핵심 확인 파일:")
    print("  csi_pattern_features.csv")
    print("  analysis_summary.txt")
    print("  near_vs_far_mean_profile.png")
    print("  near_minus_far_profile.png")
    print("  near_minus_far_centered_heatmap.png")
    print("  near_vs_far_feature_means.png")
    print("  pca_csi_windows.png")
    print("  representative_heatmaps/")
    print("=" * 70)


if __name__ == "__main__":
    main()
