import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.distributions import Normal

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class TransformerEncoder(nn.Module):
    def __init__(self, window_size, num_subcarriers, z_dim, num_heads=8, num_layers=4):
        super().__init__()
        self.embed_dim = 128
        # 1D-Conv로 지역적 특징 추출 (CSI 주파수 특성 강화)
        self.input_conv = nn.Sequential(
            nn.Conv1d(num_subcarriers, self.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU()
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, self.embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=num_heads, dim_feedforward=512,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mu = nn.Linear(self.embed_dim, z_dim)
        self.logvar = nn.Linear(self.embed_dim, z_dim)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.input_conv(x).transpose(1, 2)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling
        return self.mu(x), self.logvar(x)

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, 512 * 4 * 4)
        self.layers = nn.ModuleList([
            self._make_layer(512, 256), # 4x4 -> 8x8
            self._make_layer(256, 128), # 8x8 -> 16x16
            self._make_layer(128, 64),  # 16x16 -> 32x32
            self._make_layer(64, 32),   # 32x32 -> 64x64
            self._make_layer(32, 16)    # 64x64 -> 128x128
        ])
        self.final = nn.Sequential(nn.Conv2d(16, 3, 3, 1, 1), nn.Sigmoid())

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_c)
        )

    def forward(self, z):
        x = self.decoder_input(z).view(-1, 512, 4, 4)
        for layer in self.layers: x = layer(x)
        return self.final(x)

class VAE(L.LightningModule):
    def __init__(self, window_size, num_subcarriers, z_dim=128, lr=1e-3, beta=1.0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = TransformerEncoder(window_size, num_subcarriers, z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample() if self.training else mu
        return self.decoder(z), mu, logvar

    def training_step(self, batch, batch_idx):
        csi, img = batch
        recon, mu, logvar = self(csi)
        recon_loss = nn.functional.mse_loss(recon, img, reduction='sum') / img.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = recon_loss + self.hparams.beta * kl_loss
        self.log_dict({'train_loss': loss, 'train_recon': recon_loss, 'train_kl': kl_loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        csi, img = batch
        recon, mu, logvar = self(csi)
        recon_loss = nn.functional.mse_loss(recon, img, reduction='sum') / img.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = recon_loss + self.hparams.beta * kl_loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return [opt], [sch]
