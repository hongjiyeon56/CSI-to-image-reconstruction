from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CSI_VALID_SUBCARRIER_INDEX = list(range(6, 32)) + list(range(33, 59))
NUM_SUBCARRIERS = len(CSI_VALID_SUBCARRIER_INDEX)


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(mean=float(values["mean"]), std=float(values["std"]))


def discover_csi_csvs(base_dir: str | Path) -> list[str]:
    paths = sorted(
        glob(os.path.join(str(base_dir), "**", "csi.csv"), recursive=True)
    )
    if not paths:
        raise FileNotFoundError(f"csi.csv를 찾지 못했습니다: {base_dir}")
    return paths


def _load_raw_csi(csv_path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    required_columns = {"id", "data"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}에 필요한 열이 없습니다: {sorted(missing)}")

    df = df.sort_values("id").reset_index(drop=True)
    rows: list[np.ndarray] = []

    for row_index, value in enumerate(df["data"].values):
        try:
            row = np.asarray(json.loads(value), dtype=np.float32)
        except Exception as exc:
            raise ValueError(
                f"{csv_path}의 {row_index}번째 CSI 행 파싱 실패"
            ) from exc
        rows.append(row)

    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"{csv_path}의 CSI 행 길이가 서로 다릅니다: {sorted(lengths)}"
        )

    return df, np.stack(rows, axis=0)


def _extract_amplitude(raw_csi: np.ndarray) -> np.ndarray:
    real_indices = np.asarray(
        [index * 2 for index in CSI_VALID_SUBCARRIER_INDEX], dtype=np.int64
    )
    imag_indices = np.asarray(
        [index * 2 - 1 for index in CSI_VALID_SUBCARRIER_INDEX], dtype=np.int64
    )

    required_index = int(max(real_indices.max(), imag_indices.max()))
    if raw_csi.shape[1] <= required_index:
        raise ValueError(
            f"CSI row length={raw_csi.shape[1]}, 필요 index={required_index}"
        )

    real = raw_csi[:, real_indices].astype(np.float64)
    imag = raw_csi[:, imag_indices].astype(np.float64)
    amplitude = np.sqrt(real**2 + imag**2)
    return amplitude.astype(np.float32)


def compute_normalization_stats(
    csv_paths: Iterable[str | Path],
    row_ranges: dict[str, tuple[int, int]] | None = None,
    eps: float = 1e-6,
) -> NormalizationStats:
    """학습 구간의 log-amplitude만 이용해 global mean/std를 계산한다."""
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for csv_path in csv_paths:
        csv_path = str(Path(csv_path).resolve())
        _, raw_csi = _load_raw_csi(csv_path)
        amplitude = _extract_amplitude(raw_csi)

        start, end = 0, len(amplitude)
        if row_ranges and csv_path in row_ranges:
            start, end = row_ranges[csv_path]

        selected = amplitude[start:end]
        if selected.size == 0:
            continue

        log_amplitude = np.log(selected.astype(np.float64) + eps)
        total_sum += float(log_amplitude.sum())
        total_sq_sum += float(np.square(log_amplitude).sum())
        total_count += int(log_amplitude.size)

    if total_count == 0:
        raise RuntimeError("정규화 통계를 계산할 학습 데이터가 없습니다.")

    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean**2, 0.0)
    std = float(np.sqrt(variance))
    return NormalizationStats(mean=float(mean), std=max(std, eps))


class WificamDataset(Dataset):
    """
    반환:
      csi: (T, 52)
      window_mean: (1,)
      image: (3, 480, 640)
      image_path: str
    """

    def __init__(
        self,
        csv_paths: Iterable[str | Path],
        window_size: int,
        normalization_stats: NormalizationStats,
        is_train: bool,
        row_ranges: dict[str, tuple[int, int]] | None = None,
        image_size: tuple[int, int] = (640, 480),
        noise_std: float = 0.003,
        clip_value: float = 3.0,
    ):
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size는 1 이상이어야 합니다.")

        self.csv_paths = [str(Path(path).resolve()) for path in csv_paths]
        self.window_size = int(window_size)
        self.normalization_stats = normalization_stats
        self.is_train = bool(is_train)
        self.row_ranges = row_ranges or {}
        self.image_size = image_size
        self.noise_std = float(noise_std)
        self.clip_value = float(clip_value)

        self.records: list[dict[str, object]] = []
        self.samples: list[tuple[int, int, str]] = []
        self._load_data()

    def _load_data(self) -> None:
        for csv_path in self.csv_paths:
            data_dir = str(Path(csv_path).parent)
            df, raw_csi = _load_raw_csi(csv_path)
            amplitude = _extract_amplitude(raw_csi)
            ids = df["id"].to_numpy(dtype=np.int64)

            readable_id_to_path: dict[int, str] = {}
            for image_path in glob(os.path.join(data_dir, "*.png")):
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue
                try:
                    image_id = int(Path(image_path).stem)
                except ValueError:
                    continue
                readable_id_to_path[image_id] = image_path

            if not readable_id_to_path:
                print(f"Warning: {data_dir}에서 읽을 수 있는 PNG가 없어 건너뜁니다.")
                continue

            image_ids = np.asarray(sorted(readable_id_to_path), dtype=np.int64)
            range_start, range_end = self.row_ranges.get(
                csv_path, (0, len(amplitude))
            )
            range_start = max(0, int(range_start))
            range_end = min(len(amplitude), int(range_end))

            if range_end - range_start < self.window_size:
                print(f"Warning: {csv_path}의 구간이 window_size보다 짧아 건너뜁니다.")
                continue

            record_index = len(self.records)
            self.records.append({"amplitude": amplitude, "ids": ids})

            last_start = range_end - self.window_size
            for start_index in range(range_start, last_start + 1):
                center_index = start_index + self.window_size // 2
                target_id = int(ids[center_index])
                best_image_id = int(
                    image_ids[np.abs(image_ids - target_id).argmin()]
                )
                image_path = readable_id_to_path[best_image_id]
                self.samples.append((record_index, start_index, image_path))

        if not self.samples:
            raise RuntimeError("유효한 CSI-image window를 만들지 못했습니다.")

    def __len__(self) -> int:
        return len(self.samples)

    def _preprocess_window(
        self, amplitude_window: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.normalization_stats

        processed = np.log(amplitude_window.astype(np.float32) + 1e-6)
        processed = (processed - stats.mean) / (stats.std + 1e-6)
        processed = np.clip(processed, -self.clip_value, self.clip_value)

        spectrogram = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
        spectrogram = F.avg_pool2d(
            spectrogram,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )
        spectrogram = spectrogram.squeeze(0).squeeze(0)
        spectrogram = torch.clamp(
            spectrogram, -self.clip_value, self.clip_value
        )

        # 분석 코드와 같은 정의: 전처리된 window 전체의 평균.
        window_mean = spectrogram.mean().reshape(1)

        # window_mean에는 noise를 넣지 않고 CSI branch에만 augmentation 적용.
        if self.is_train and self.noise_std > 0:
            spectrogram = spectrogram + torch.randn_like(spectrogram) * self.noise_std
            spectrogram = torch.clamp(
                spectrogram, -self.clip_value, self.clip_value
            )

        return spectrogram, window_mean

    def __getitem__(self, index: int) -> dict[str, object]:
        record_index, start_index, image_path = self.samples[index]

        amplitude = self.records[record_index]["amplitude"]
        assert isinstance(amplitude, np.ndarray)
        end_index = start_index + self.window_size
        amplitude_window = amplitude[start_index:end_index]

        spectrogram, window_mean = self._preprocess_window(amplitude_window)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽지 못했습니다: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        image_tensor = (
            torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        )

        return {
            "csi": spectrogram,
            "window_mean": window_mean,
            "image": image_tensor,
            "image_path": image_path,
        }
