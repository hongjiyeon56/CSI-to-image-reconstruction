import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

import pytorch_lightning as L
import torch
from torch.utils.data import Subset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from dataset import WificamDataset, NUM_SUBCARRIERS
from vae import VAE


num_workers = 0
torch.set_num_threads(4)

if torch.backends.mps.is_available():
    device = torch.device('mps')
    accelerator = 'mps'
elif torch.cuda.is_available():
    device = torch.device('cuda')
    accelerator = 'gpu'
else:
    device = torch.device('cpu')
    accelerator = 'cpu'


current_file_path = Path(__file__).resolve()
current_folder = current_file_path.parent
project_root = current_folder.parent

data_dir = os.path.join(project_root, 'data', 'rx1_train')

window_size = 151
batch_size = 32
epochs = 200

persistent_workers = True if num_workers > 0 else False


class LossHistory(L.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.history_path = os.path.join(save_dir, "loss_history.json")

        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as f:
                self.history = json.load(f)
            print("loss history 불러옴")
        else:
            self.history = {
                # total loss
                "train_loss": [],
                "val_loss": [],

                # VAE loss
                "train_kl": [],
                "val_kl": [],
                "train_l1": [],
                "val_l1": [],

                # Dice / IoU loss
                "train_dice": [],
                "val_dice": [],
                "train_iou": [],
                "val_iou": [],

                # image structure loss
                "train_grad": [],
                "val_grad": [],
                "train_freq": [],
                "val_freq": [],
            }

    def save_history(self):
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def on_train_epoch_end(self, trainer, pl_module):
        for key in self.history.keys():
            if key.startswith("train"):
                val = trainer.callback_metrics.get(key)
                if val is not None:
                    self.history[key].append(val.detach().cpu().item())

        if len(self.history["train_loss"]) > 0:
            msg = f"[Epoch {trainer.current_epoch}] train_loss: {self.history['train_loss'][-1]:.7f}"

            if len(self.history["train_dice"]) > 0:
                msg += f", train_dice: {self.history['train_dice'][-1]:.7f}"

            if len(self.history["train_iou"]) > 0:
                msg += f", train_iou: {self.history['train_iou'][-1]:.7f}"

            print(msg)

        self.save_history()

    def on_validation_epoch_end(self, trainer, pl_module):
        for key in self.history.keys():
            if key.startswith("val"):
                val = trainer.callback_metrics.get(key)
                if val is not None:
                    self.history[key].append(val.detach().cpu().item())

        self.save_history()


class LatestCheckpoints(L.Callback):
    def __init__(self, dirpath, keep_last=3):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.keep_last = keep_last

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            filename = f"latest-epoch={trainer.current_epoch}.ckpt"
        else:
            filename = f"latest-epoch={trainer.current_epoch}-val_loss={val_loss.detach().cpu().item():.7f}.ckpt"

        ckpt_path = self.dirpath / filename
        trainer.save_checkpoint(str(ckpt_path))
        self._delete_old_checkpoints()

    def _delete_old_checkpoints(self):
        checkpoints = sorted(
            self.dirpath.glob("latest-epoch=*.ckpt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True
        )

        for checkpoint in checkpoints[self.keep_last:]:
            checkpoint.unlink(missing_ok=True)


def find_latest_checkpoint(output_dir):
    output_path = Path(output_dir)
    checkpoints = sorted(
        output_path.glob("latest-epoch=*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )
    if checkpoints:
        return str(checkpoints[0])

    last_checkpoint = output_path / "last.ckpt"
    return str(last_checkpoint) if last_checkpoint.exists() else None


def train():
    # Dice + IoU loss 실험이므로 기존 output과 분리
    output_dir = os.path.join(current_folder, 'outputs/0708_outputs_rx1_dice_iou_loss')
    save_dir = os.path.join(output_dir, 'loss')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    loss_history = LossHistory(save_dir)

    dataset_train = WificamDataset(data_dir, window_size, is_train=True)
    dataset_val = WificamDataset(data_dir, window_size, is_train=False)

    train_idx, val_idx = train_test_split(
        list(range(len(dataset_train))),
        test_size=0.1,
        shuffle=False
    )

    dataset_train = Subset(dataset_train, train_idx)
    dataset_val = Subset(dataset_val, val_idx)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    model = VAE(
        window_size=window_size,
        num_subcarriers=NUM_SUBCARRIERS
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=5,
        save_last=False,
        filename='top-epoch={epoch}-val_loss={val_loss:.7f}',
        dirpath=output_dir,
        auto_insert_metric_name=False,
        verbose=True
    )

    latest_checkpoint_callback = LatestCheckpoints(
        dirpath=output_dir,
        keep_last=3
    )

    callbacks = [
        checkpoint_callback,
        latest_checkpoint_callback,
        loss_history
    ]

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=1.0,
        logger=True,
        callbacks=callbacks,
        max_epochs=epochs
    )

    checkpoint_path = find_latest_checkpoint(output_dir)

    if checkpoint_path is not None:
        print("기존 checkpoint 발견. 이어서 학습합니다:", checkpoint_path)
        trainer.fit(
            model,
            dataloader_train,
            dataloader_val,
            ckpt_path=checkpoint_path
        )
    else:
        print("새로운 학습을 시작합니다.")
        trainer.fit(
            model,
            dataloader_train,
            dataloader_val
        )


if __name__ == '__main__':
    print("project_root:", project_root)
    print("data_dir:", data_dir)
    print("exists:", os.path.exists(data_dir))
    print("accelerator:", accelerator)

    train()
