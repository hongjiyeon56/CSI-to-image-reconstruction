from pathlib import Path
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from dataset import WificamDataset, NUM_SUBCARRIERS
from vae import VAE

current_folder = Path(__file__).resolve().parent
data_dir = current_folder.parent / 'data' / '20260309_train_mesh'


class KeepLastNCheckpoints(Callback):
    def __init__(self, dirpath, filename_prefix, n=3):
        self.dirpath = Path(dirpath)
        self.filename_prefix = filename_prefix
        self.n = n
        self.saved = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            return
        filename = f"{self.filename_prefix}-epoch={epoch}-val_loss={val_loss:.4f}.ckpt"
        filepath = self.dirpath / filename
        trainer.save_checkpoint(str(filepath))
        self.saved.append(filepath)
        while len(self.saved) > self.n:
            old = self.saved.pop(0)
            if old.exists():
                old.unlink()


def train():
    window_size, batch_size, z_dim, lr = 151, 32, 256, 1e-3

    dataset = WificamDataset(str(data_dir), window_size)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=False)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size * 2, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)

    model = VAE(window_size=window_size, num_subcarriers=NUM_SUBCARRIERS, z_dim=z_dim, lr=lr)

    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='outputs',
            filename='best_vae-{epoch}-{val_loss:.4f}',
            save_top_k=1,
        ),
        KeepLastNCheckpoints(dirpath='outputs', filename_prefix='last_vae', n=3),
    ]

    trainer = L.Trainer(accelerator='auto', devices=1, max_epochs=200, callbacks=callbacks, precision=16)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
