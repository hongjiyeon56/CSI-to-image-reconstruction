from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import (
    NUM_SUBCARRIERS,
    WificamDataset,
    compute_normalization_stats,
    discover_csi_csvs,
)
from vae import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/rx1_train")
    parser.add_argument("--output_dir", type=str, default="./outputs/0714_window_mean_vae")
    parser.add_argument("--window_size", type=int, default=151)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def build_train_val_split(
    csv_paths: list[str],
    val_ratio: float,
    window_size: int,
    seed: int,
) -> tuple[
    list[str],
    list[str],
    dict[str, tuple[int, int]],
    dict[str, tuple[int, int]],
]:
    """
    csi.csv가 여러 개면 session 단위로 분리한다.
    하나뿐이면 시간축 연속 구간으로 나누어 window 중복을 방지한다.
    """
    resolved_paths = [str(Path(path).resolve()) for path in csv_paths]

    if len(resolved_paths) >= 2:
        train_paths, val_paths = train_test_split(
            resolved_paths,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
        )
        return sorted(train_paths), sorted(val_paths), {}, {}

    only_path = resolved_paths[0]
    row_count = len(pd.read_csv(only_path, usecols=["id"]))
    split_row = int(row_count * (1.0 - val_ratio))
    minimum_rows = window_size + 1

    if split_row < minimum_rows or row_count - split_row < minimum_rows:
        raise ValueError(
            "단일 csi.csv를 train/val로 나누기에는 데이터가 부족합니다. "
            f"rows={row_count}, window_size={window_size}"
        )

    train_ranges = {only_path: (0, split_row)}
    val_ranges = {only_path: (split_row, row_count)}
    return [only_path], [only_path], train_ranges, val_ranges


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio는 0과 1 사이여야 합니다.")

    L.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = discover_csi_csvs(args.data_dir)
    (
        train_csv_paths,
        val_csv_paths,
        train_row_ranges,
        val_row_ranges,
    ) = build_train_val_split(
        csv_paths=csv_paths,
        val_ratio=args.val_ratio,
        window_size=args.window_size,
        seed=args.seed,
    )

    # train 데이터만으로 normalization 통계를 계산한다.
    stats = compute_normalization_stats(
        train_csv_paths,
        row_ranges=train_row_ranges,
    )
    stats.save(output_dir / "normalization.json")

    split_info = {
        "train_csv_paths": train_csv_paths,
        "val_csv_paths": val_csv_paths,
        "train_row_ranges": train_row_ranges,
        "val_row_ranges": val_row_ranges,
        "window_size": args.window_size,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    (output_dir / "split_info.json").write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_dataset = WificamDataset(
        csv_paths=train_csv_paths,
        window_size=args.window_size,
        normalization_stats=stats,
        is_train=True,
        row_ranges=train_row_ranges,
        noise_std=args.noise_std,
    )
    val_dataset = WificamDataset(
        csv_paths=val_csv_paths,
        window_size=args.window_size,
        normalization_stats=stats,
        is_train=False,
        row_ranges=val_row_ranges,
        noise_std=0.0,
    )

    persistent_workers = args.num_workers > 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    model = VAE(
        window_size=args.window_size,
        num_subcarriers=NUM_SUBCARRIERS,
        z_dim=args.z_dim,
        lr=args.lr,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=output_dir / "checkpoints",
        filename="best-window-mean-vae-{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
    )
    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints" / "epochs",
        filename="epoch-{epoch:03d}-val_loss-{val_loss:.4f}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(save_dir=str(output_dir), name="logs")
    # PyTorch Lightning 1.9.x uses integer precision values.
    precision = 16 if torch.cuda.is_available() else 32

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[
            checkpoint_callback,
            epoch_checkpoint_callback,
            lr_monitor,
        ],
        logger=logger,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        deterministic=False,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )

    print("\n학습 완료")
    print("Best checkpoint:", checkpoint_callback.best_model_path)
    print("Normalization:", output_dir / "normalization.json")


if __name__ == "__main__":
    main()
