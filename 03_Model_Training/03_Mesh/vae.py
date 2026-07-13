import torch
import torch.nn as nn
import pytorch_lightning as L
import torch.nn.functional as F
from torch.distributions import Normal as Normal
import math


z_dim = 128
# 0516 - bce, mse, mse2
# 05162 - bce0516_2, mse0516_2
# 1. mse / 2. bce / 3. kl 0.5 bce / 4. kl 0.5 mse
    
class VAE(L.LightningModule):
    def __init__(self, window_size, num_subcarriers, embed_dim=128, max_len=256):
        super(VAE, self).__init__()

        self.encoder = Encoder(num_subcarriers=num_subcarriers, embed_dim=embed_dim, max_len=max_len)
        self.decoder = Decoder()

    def encode(self, x):
        mu, logvar, skip = self.encoder.encode(x)
        scale = torch.exp(0.5 * logvar)
        qz_x = Normal(loc=mu, scale=scale)
        return qz_x, skip
    
    def decode(self, qz_x, skip):
        if self.training:
            z = qz_x.rsample()
        else:
            z = qz_x.loc

        x_hat = self.decoder.decode(z, skip)
        return x_hat

    def forward(self, x):
        qz_x, skip = self.encode(x)
        x_hat = self.decode(qz_x, skip)
        return qz_x, x_hat

    def loss_function(self, x, qz_x, x_hat):
        """
        VAE reconstruction loss for black-background human mesh images.

        Main idea:
        - L1 keeps pixel-level reconstruction.
        - Dice Loss helps foreground/body region overlap.
        - Soft IoU Loss directly encourages area overlap.
        - Gradient/Frequency losses preserve boundary and global structure.
        - KL keeps the latent distribution regularized.
        """
        kl = self.calc_kl(qz_x)
        l1 = self.calc_l1(x, x_hat)
        dice = self.calc_dice_loss(x_hat, x)
        iou = self.calc_soft_iou_loss(x_hat, x)
        grad = self.calc_gradient(x_hat, x)
        freq = self.calc_frequency(x_hat, x) / 10.0

        # Recommended starting weights for mesh/silhouette-like image generation.
        # If outputs become blurry, lower dice/iou slightly.
        # If foreground disappears, raise dice first.
        total = (
            0.01 * kl
            + 0.50 * l1
            + 0.30 * dice
            + 0.30 * iou
            + 0.10 * grad
            + 0.01 * freq
        )

        losses = {
            "loss": total,
            "kl": kl,
            "l1": l1,
            "dice": dice,
            "iou": iou,
            "grad": grad,
            "freq": freq,
        }
        return losses
    
    def calc_kl(self, qz_x):
        variance = qz_x.scale**2
        kl = 0.5 * (variance + qz_x.loc**2 - 1 - torch.log(variance))
        return kl.mean(0).sum()

    def calc_l1(self, x, x_hat):
        return torch.abs(x - x_hat).mean()

    def _to_foreground_mask(self, img, threshold=0.05):
        """
        Convert RGB/grayscale mesh image to foreground mask.

        x_hat already passes through Sigmoid(), so predicted mask can stay soft.
        GT is binarized because the background is mostly black and the target mesh
        region is the main area we want to optimize with Dice/IoU.
        """
        # If RGB, collapse channel dimension to one foreground map.
        if img.size(1) > 1:
            img = img.mean(dim=1, keepdim=True)

        # Safety: if images are accidentally loaded as 0~255, map them to 0~1.
        if img.detach().max() > 1.0:
            img = img / 255.0

        img = img.clamp(0.0, 1.0)
        return img, (img > threshold).float()

    def calc_dice_loss(self, pred, target, eps=1e-6):
        pred_soft, _ = self._to_foreground_mask(pred)
        _, target_mask = self._to_foreground_mask(target)

        pred_soft = pred_soft.flatten(start_dim=1)
        target_mask = target_mask.flatten(start_dim=1)

        intersection = (pred_soft * target_mask).sum(dim=1)
        dice = (2.0 * intersection + eps) / (pred_soft.sum(dim=1) + target_mask.sum(dim=1) + eps)
        return 1.0 - dice.mean()

    def calc_soft_iou_loss(self, pred, target, eps=1e-6):
        pred_soft, _ = self._to_foreground_mask(pred)
        _, target_mask = self._to_foreground_mask(target)

        pred_soft = pred_soft.flatten(start_dim=1)
        target_mask = target_mask.flatten(start_dim=1)

        intersection = (pred_soft * target_mask).sum(dim=1)
        union = pred_soft.sum(dim=1) + target_mask.sum(dim=1) - intersection
        iou = (intersection + eps) / (union + eps)
        return 1.0 - iou.mean()

    def calc_gradient(self, pred, target):
        diff_x = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (target[:, :, :, 1:] - target[:, :, :, :-1])
        diff_y = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) - (target[:, :, 1:, :] - target[:, :, :-1, :])
        return diff_x.abs().mean() + diff_y.abs().mean()
    
    def calc_frequency(self, pred, target):
        pred = pred.mean(dim=1, keepdim=True)
        target = target.mean(dim=1, keepdim=True)

        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
    
    def __step(self, batch, batch_idx, stage):
        qz_x, x_hat = self.forward(batch[0])
        loss = self.loss_function(batch[1], qz_x, x_hat)
        for loss_n, loss_val in loss.items():
            self.log(f"{stage}_{loss_n}", loss_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss["loss"]

    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.__step(batch, batch_idx, stage="val")
  
    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=3e-4, amsgrad=True)


class Embedding(nn.Module):
    def __init__(self, num_subcarriers, embed_dim):
        super().__init__()

        self.proj = nn.Linear(num_subcarriers, embed_dim)

    def forward(self, x):
        # x: (B, T, C)
        return self.proj(x)  # (B, T, D)
    

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, max_len=256, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        self.proj = nn.Linear(embed_dim, embed_dim)

        # Relative Position Bias
        self.relative_position_bias = nn.Parameter(
            torch.zeros(num_heads, max_len, max_len)
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, N, C = x.shape

        # QKV
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention score
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, N, N)

        # Relative position bias 추가
        attn = attn + self.relative_position_bias[:, :N, :N]

        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)

        # Residual 1
        x = x + out
        x = self.norm1(x)

        # MLP
        mlp_out = self.mlp(x)

        # Residual 2
        x = x + mlp_out
        x = self.norm2(x)

        return x
    

class Tokenizer(nn.Module):
    def __init__(self, num_subcarriers, embed_dim, max_len, num_heads=8, dropout=0.1, depth=4):
        super().__init__()

        self.embed = Embedding(num_subcarriers, embed_dim)
        self.blocks = nn.Sequential(
            *[Attention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                max_len=max_len,
                dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return x  # (B, N_tokens, D)
    

class TcnBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation):
        super().__init__()

        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=dilation, dilation=dilation)

        self.act = nn.GELU()

        if in_c != out_c:
            self.skip = nn.Conv1d(in_c, out_c, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        x = self.act(self.conv1(x))
        x = self.conv2(x)

        x = self.act(x + identity)
        return x
    

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Linear(dim,1)

    def forward(self,x):
        # x (B, C, T)
        x = x.permute(0,2,1)   # (B, T, C)

        w = torch.softmax(self.attn(x),dim=1)
        x = (x * w).sum(dim=1)

        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)
    
    
class Encoder(nn.Module):
    def __init__(self, num_subcarriers, embed_dim, max_len):
        super().__init__()

        self.tokenizer = Tokenizer(
            num_subcarriers=num_subcarriers,
            embed_dim=embed_dim,
            max_len=max_len,
            num_heads=4,
            dropout=0.1,
            depth=4
        )

        self.tcn = nn.Sequential(
            TcnBlock(embed_dim, 128, 1),
            TcnBlock(128, 256, 2),
            TcnBlock(256, 256, 4),
        )

        self.pool = AttentionPool(256)
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        self.mu = nn.Linear(256, z_dim) 
        self.logvar = nn.Sequential(
            nn.Linear(256, z_dim),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.tokenizer(x)
        # x: (B, T, D) → (B, D, T)
        x = x.permute(0, 2, 1)
        x = self.tcn(x) # (B, 256, T)
        skip = x
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        mu = self.mu(x)
        logvar = self.logvar(x) * 10.0
        return mu, logvar, skip
    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_input = nn.Linear(z_dim, 512 * 6 * 8)

        self.skip_conv = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.skip_pool = AttentionPool(128)
        self.skip_proj = nn.Linear(128, 128 * 6 * 8)

        self.layers = nn.ModuleList([
            self._make_layer(512 + 128, 256),   # 6x8   -> 12x16
            self._make_layer(256, 128),         # 12x16  -> 24x32
            self._make_layer(128, 64),          # 24x32  -> 48x64
            self._make_layer(64, 32),           # 48x64  -> 96x128
            self._make_layer(32, 16),           # 96x128 -> 192x256
            self._make_layer(16, 8),            # 192x256 -> 384x512
        ])
        self.final = nn.Sequential(
            nn.Upsample(size=(480, 640), mode='bilinear', align_corners=True),  # 384x512 -> 480x640
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            ResidualBlock(out_c)
        )

    def decode(self, z, skip):
        x = self.decoder_input(z).view(-1, 512, 6, 8)

        skip = self.skip_conv(skip)         # (B, 128, N)
        skip = self.skip_pool(skip)         # (B, 128)
        skip = self.skip_proj(skip)         # (B, 128 * 6 * 8)
        skip = skip.view(-1, 128, 6, 8)

        x = torch.cat([x, skip], dim=1)     # (B, 512+128, 6, 8)

        x = self.layers[0](x)

        for layer in self.layers[1:]:
            x = layer(x)

        return self.final(x)
