import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.distributions import Normal
from torch_ema import ExponentialMovingAverage
import numpy as np


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
    def __init__(self, window_size, num_subcarriers, z_dim, num_heads=8, num_layers=4, gru_hidden=128):
        super().__init__()
        self.embed_dim = 128
        self.input_conv = nn.Sequential(
            nn.Conv1d(num_subcarriers, self.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.embed_dim),
            nn.GELU()
        )
        self.bigru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.gru_proj = nn.Linear(gru_hidden * 2, self.embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, window_size, self.embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=num_heads, dim_feedforward=512,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.mu = nn.Linear(self.embed_dim, z_dim)
        self.logvar = nn.Linear(self.embed_dim, z_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_conv(x).transpose(1, 2)
        x, _ = self.bigru(x)
        x = self.gru_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    """
    6x8 → 12x16 → 24x32 → 48x64 → 96x128 → 192x256 → 384x512 → 480x640
    6단계 x2 upsample 후 마지막 Upsample(size=(480,640))으로 정확히 맞춤
    """
    def __init__(self, z_dim, image_height=480, image_width=640):
        super().__init__()
        self.start_h = 6
        self.start_w = 8
        self.image_height = image_height
        self.image_width = image_width

        self.decoder_input = nn.Linear(z_dim, 512 * self.start_h * self.start_w)
        self.layers = nn.ModuleList([
            self._make_layer(512, 256),  # 6x8   -> 12x16
            self._make_layer(256, 128),  # 12x16  -> 24x32
            self._make_layer(128, 64),   # 24x32  -> 48x64
            self._make_layer(64, 32),    # 48x64  -> 96x128
            self._make_layer(32, 16),    # 96x128 -> 192x256
            self._make_layer(16, 8),     # 192x256 -> 384x512
        ])
        self.final = nn.Sequential(
            nn.Upsample(size=(image_height, image_width), mode='bilinear', align_corners=True),  # 384x512 -> 480x640
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_c)
        )

    def forward(self, z):
        x = self.decoder_input(z).view(-1, 512, self.start_h, self.start_w)
        for layer in self.layers:
            x = layer(x)
        return self.final(x)


# ── Loss 함수들 ──────────────────────────────────────────────────────────────

class MultiScaleReconLoss(nn.Module):
    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, recon, target):
        loss = 0.0
        for s in self.scales:
            if s == 1:
                r, t = recon, target
            else:
                r = nn.functional.avg_pool2d(recon,  kernel_size=s, stride=s)
                t = nn.functional.avg_pool2d(target, kernel_size=s, stride=s)
            loss += nn.functional.mse_loss(r, t, reduction='sum') / target.size(0)
        return loss / len(self.scales)


class FrequencyLoss(nn.Module):
    def forward(self, recon, target):
        # cuFFT half precision on CUDA only supports power-of-two spatial sizes.
        # Our images are 480x640, so run the FFT in float32 for stability.
        recon_fft  = torch.fft.fft2(recon.float(),  norm='ortho')
        target_fft = torch.fft.fft2(target.float(), norm='ortho')
        recon_mag  = torch.abs(recon_fft)
        target_mag = torch.abs(target_fft)
        return nn.functional.mse_loss(recon_mag, target_mag)


class GradientLoss(nn.Module):
    def forward(self, recon, target):
        diff_x = (recon[:, :, :, 1:] - recon[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])
        diff_y = (recon[:, :, 1:, :] - recon[:, :, :-1, :]) - (target[:, :, 1:, :] - target[:, :, :-1, :])
        return diff_x.abs().mean() + diff_y.abs().mean()


def temporal_smoothness_loss(z_seq):
    if z_seq.size(0) < 2:
        return z_seq.new_tensor(0.0)
    diff = z_seq[1:] - z_seq[:-1]
    return (diff ** 2).mean()


# ── VAE ─────────────────────────────────────────────────────────────────────

class VAE(L.LightningModule):
    def __init__(self, window_size, num_subcarriers, z_dim=128, lr=1e-3,
                 beta=1.0, lambda_smooth=0.1, lambda_freq=0.1, lambda_grad=0.1,
                 beta_warmup_epochs=20, ema_decay=0.999,
                 image_height=480, image_width=640, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder   = TransformerEncoder(window_size, num_subcarriers, z_dim)
        self.decoder   = Decoder(z_dim, image_height=image_height, image_width=image_width)
        self.ms_loss   = MultiScaleReconLoss(scales=(1, 2, 4))
        self.freq_loss = FrequencyLoss()
        self.grad_loss = GradientLoss()
        self.ema       = None

    def on_fit_start(self):
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.hparams.ema_decay)

    def _current_beta(self):
        if self.hparams.beta_warmup_epochs <= 0:
            return self.hparams.beta
        progress = min(self.current_epoch / self.hparams.beta_warmup_epochs, 1.0)
        return self.hparams.beta * progress

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample() if self.training else mu
        return self.decoder(z), mu, logvar, z

    def training_step(self, batch, batch_idx):
        csi, img = batch
        recon, mu, logvar, z = self(csi)

        recon_loss  = self.ms_loss(recon, img)
        kl_loss     = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        smooth_loss = temporal_smoothness_loss(z)
        freq_loss   = self.freq_loss(recon, img)
        grad_loss   = self.grad_loss(recon, img)

        beta = self._current_beta()
        loss = (recon_loss
                + beta                       * kl_loss
                + self.hparams.lambda_smooth * smooth_loss
                + self.hparams.lambda_freq   * freq_loss
                + self.hparams.lambda_grad   * grad_loss)

        self.log_dict({
            'train_loss': loss, 'train_recon': recon_loss, 'train_kl': kl_loss,
            'train_smooth': smooth_loss, 'train_freq': freq_loss, 'train_grad': grad_loss
        }, prog_bar=True)

        if self.ema is not None:
            self.ema.update()
        return loss

    def validation_step(self, batch, batch_idx):
        csi, img = batch
        if self.ema is not None:
            with self.ema.average_parameters():
                recon, mu, logvar, z = self(csi)
        else:
            recon, mu, logvar, z = self(csi)

        recon_loss  = self.ms_loss(recon, img)
        kl_loss     = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        smooth_loss = temporal_smoothness_loss(z)
        freq_loss   = self.freq_loss(recon, img)
        grad_loss   = self.grad_loss(recon, img)

        beta = self._current_beta()
        loss = (recon_loss
                + beta                       * kl_loss
                + self.hparams.lambda_smooth * smooth_loss
                + self.hparams.lambda_freq   * freq_loss
                + self.hparams.lambda_grad   * grad_loss)

        self.log_dict({
            'val_loss': loss, 'val_recon': recon_loss, 'val_kl': kl_loss,
            'val_smooth': smooth_loss, 'val_freq': freq_loss, 'val_grad': grad_loss
        }, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return [opt], [sch]
