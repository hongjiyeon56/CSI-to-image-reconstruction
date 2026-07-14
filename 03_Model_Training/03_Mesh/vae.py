from __future__ import annotations

import math

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class VAE(L.LightningModule):
    def __init__(
        self,
        window_size: int,
        num_subcarriers: int,
        z_dim: int = 256,
        lr: float = 3e-4,
        embed_dim: int = 128,
        max_len: int = 256,
        kl_weight: float = 0.01,
        l1_weight: float = 0.5,
        grad_weight: float = 0.1,
        freq_weight: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(lr)
        self.kl_weight = float(kl_weight)
        self.l1_weight = float(l1_weight)
        self.grad_weight = float(grad_weight)
        self.freq_weight = float(freq_weight)

        self.encoder = Encoder(
            num_subcarriers=num_subcarriers,
            embed_dim=embed_dim,
            max_len=max_len,
            z_dim=z_dim,
        )
        self.decoder = Decoder(z_dim=z_dim)

    def encode(
        self,
        csi: torch.Tensor,
        window_mean: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor]:
        mu, logvar, skip = self.encoder.encode(csi, window_mean)
        scale = torch.exp(0.5 * logvar)
        qz_x = Normal(loc=mu, scale=scale)
        return qz_x, skip

    def decode(self, qz_x: Normal, skip: torch.Tensor) -> torch.Tensor:
        z = qz_x.rsample() if self.training else qz_x.loc
        return self.decoder.decode(z, skip)

    def forward(
        self,
        csi: torch.Tensor,
        window_mean: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor]:
        qz_x, skip = self.encode(csi, window_mean)
        prediction = self.decode(qz_x, skip)
        return qz_x, prediction

    def loss_function(
        self,
        target: torch.Tensor,
        qz_x: Normal,
        prediction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        kl = self.calc_kl(qz_x)
        l1 = self.calc_l1(target, prediction)
        grad = self.calc_gradient(prediction, target)
        freq = self.calc_frequency(prediction, target) / 10.0

        total = (
            self.kl_weight * kl
            + self.l1_weight * l1
            + self.grad_weight * grad
            + self.freq_weight * freq
        )
        return {"loss": total, "kl": kl, "l1": l1, "grad": grad, "freq": freq}

    @staticmethod
    def calc_kl(qz_x: Normal) -> torch.Tensor:
        variance = qz_x.scale**2
        kl = 0.5 * (
            variance
            + qz_x.loc**2
            - 1.0
            - torch.log(variance + 1e-8)
        )
        return kl.mean(dim=0).sum()

    @staticmethod
    def calc_l1(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return torch.abs(target - prediction).mean()

    @staticmethod
    def calc_gradient(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return torch.abs(pred_dx - target_dx).mean() + torch.abs(pred_dy - target_dy).mean()

    @staticmethod
    def calc_frequency(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # 일부 GPU에서 fp16 FFT가 제한되므로 float32로 계산한다.
        prediction_gray = prediction.mean(dim=1, keepdim=True).float()
        target_gray = target.mean(dim=1, keepdim=True).float()
        pred_fft = torch.fft.fft2(prediction_gray)
        target_fft = torch.fft.fft2(target_gray)
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

    def _shared_step(
        self,
        batch: dict[str, object],
        stage: str,
    ) -> torch.Tensor:
        csi = batch["csi"]
        window_mean = batch["window_mean"]
        target = batch["image"]

        assert isinstance(csi, torch.Tensor)
        assert isinstance(window_mean, torch.Tensor)
        assert isinstance(target, torch.Tensor)

        qz_x, prediction = self.forward(csi, window_mean)
        losses = self.loss_function(target, qz_x, prediction)
        batch_size = int(csi.shape[0])

        for name, value in losses.items():
            self.log(
                f"{stage}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "loss",
                logger=True,
                batch_size=batch_size,
            )

        return losses["loss"]

    def training_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            amsgrad=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(self.trainer.max_epochs), 1),
            eta_min=self.lr * 0.05,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class Embedding(nn.Module):
    def __init__(self, num_subcarriers: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(num_subcarriers, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim은 num_heads로 나누어져야 합니다.")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(num_heads, max_len, max_len)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channels = x.shape
        if token_count > self.relative_position_bias.shape[1]:
            raise ValueError(f"token_count={token_count}가 max_len보다 큽니다.")

        qkv = self.qkv(x).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) / self.scale
        attention = attention + self.relative_position_bias[:, :token_count, :token_count]
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        output = attention @ value
        output = output.transpose(1, 2).reshape(batch_size, token_count, channels)
        output = self.proj(output)
        x = self.norm1(x + output)
        x = self.norm2(x + self.mlp(x))
        return x


class Tokenizer(nn.Module):
    def __init__(
        self,
        num_subcarriers: int,
        embed_dim: int,
        max_len: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        depth: int = 4,
    ):
        super().__init__()
        self.embed = Embedding(num_subcarriers, embed_dim)
        self.blocks = nn.Sequential(
            *[
                Attention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    max_len=max_len,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.embed(x))


class TcnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return self.act(x + identity)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        weight = torch.softmax(self.attn(x), dim=1)
        return (x * weight).sum(dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        num_subcarriers: int,
        embed_dim: int,
        max_len: int,
        z_dim: int,
    ):
        super().__init__()
        self.tokenizer = Tokenizer(
            num_subcarriers=num_subcarriers,
            embed_dim=embed_dim,
            max_len=max_len,
            num_heads=4,
            dropout=0.1,
            depth=4,
        )
        self.tcn = nn.Sequential(
            TcnBlock(embed_dim, 128, dilation=1),
            TcnBlock(128, 256, dilation=2),
            TcnBlock(256, 256, dilation=4),
        )
        self.pool = AttentionPool(256)

        # window_mean scalar를 작은 feature vector로 변환한다.
        self.mean_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

    def encode(
        self,
        csi: torch.Tensor,
        window_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.tokenizer(csi)
        sequence = sequence.permute(0, 2, 1)
        sequence = self.tcn(sequence)
        skip = sequence

        sequence_feature = self.pool(sequence)
        mean_feature = self.mean_encoder(window_mean)
        fused = torch.cat([sequence_feature, mean_feature], dim=1)
        fused = self.fc(fused)

        mu = self.mu(fused)
        logvar = torch.clamp(self.logvar(fused), min=-5.0, max=5.0)
        return mu, logvar, skip


class Decoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.decoder_input = nn.Linear(z_dim, 512 * 6 * 8)
        self.skip_conv = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.skip_pool = AttentionPool(128)
        self.skip_proj = nn.Linear(128, 128 * 6 * 8)

        self.layers = nn.ModuleList(
            [
                self._make_layer(512 + 128, 256),
                self._make_layer(256, 128),
                self._make_layer(128, 64),
                self._make_layer(64, 32),
                self._make_layer(32, 16),
                self._make_layer(16, 8),
            ]
        )
        self.final = nn.Sequential(
            nn.Upsample(size=(480, 640), mode="bilinear", align_corners=False),
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            ResidualBlock(out_channels),
        )

    def decode(self, z: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z).view(-1, 512, 6, 8)

        skip = self.skip_conv(skip)
        skip = self.skip_pool(skip)
        skip = self.skip_proj(skip).view(-1, 128, 6, 8)

        x = torch.cat([x, skip], dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.final(x)
