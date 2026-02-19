import os
from pathlib import Path
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from dataset import WificamDataset, NUM_SUBCARRIERS
from vae import VAE

current_folder = Path(__file__).resolve().parent
data_dir = current_folder.parent / 'data' / 'data_2026_01_20_mesh'

def train():
    window_size, batch_size, z_dim, lr = 151, 32, 256, 1e-3
    
    dataset = WificamDataset(str(data_dir), window_size)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, shuffle=False)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size * 2, shuffle=False, num_workers=4, drop_last=True, persistent_workers=True)
    
    model = VAE(window_size=window_size, num_subcarriers=NUM_SUBCARRIERS, z_dim=z_dim, lr=lr)
    
    callbacks = [
        ModelCheckpoint(monitor='val_loss', dirpath='outputs', filename='best_vae-{epoch}-{val_loss:.4f}', save_top_k=1),
    ]
    trainer = L.Trainer(accelerator='auto', devices=1, max_epochs=200, callbacks=callbacks, precision=16)
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    train()
