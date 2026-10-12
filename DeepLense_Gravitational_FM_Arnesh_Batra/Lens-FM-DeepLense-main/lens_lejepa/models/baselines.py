"""Supervised reference models trained from scratch.

The physics-informed baselines in the paper (Lensiformer, LensPINN) and the
supervised ViT variants come from the ML4SCI DeepLense repository and are not
re-implemented here; see the README for links.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


def build_resnet18(num_outputs: int) -> nn.Module:
    """ResNet18 with a single-channel 3x3 stem and no max-pool (small images)."""
    model = resnet18(weights=None, num_classes=num_outputs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class ResNetRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = build_resnet18(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).squeeze(-1)


class _ChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = max(4, channels // 16)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        return x + residual * self.channel_attention(residual)


class RCAN(nn.Module):
    """Compact RCAN-style 2x upsampler (8 channel-attention blocks).

    The output layer is linear, as in the original RCAN. An output sigmoid
    (used in an earlier version of this baseline) saturates within a few steps
    on these mostly-black images, after which its gradient is exactly zero and
    the network is stuck predicting a black image. Predictions are clamped to
    the target range ``[0, 1]`` at evaluation time only.
    """

    def __init__(self, channels: int = 64, blocks: int = 8) -> None:
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[_ChannelAttentionBlock(channels) for _ in range(blocks)])
        self.tail = nn.Sequential(nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2), nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        features = self.head(low_res)
        output = self.tail(self.body(features) + features)
        return output if self.training else output.clamp(0.0, 1.0)


class Bicubic(nn.Module):
    """Parameter-free bicubic upsampling baseline."""

    def __init__(self, output_size: int = 64) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        size = (self.output_size, self.output_size)
        return F.interpolate(low_res, size=size, mode="bicubic", align_corners=False, antialias=True)


# --------------------------------------------------------------------------------------
# Further 2x super-resolution baselines. All use a linear output and are clamped to
# [0, 1] at evaluation only, like RCAN above.
# --------------------------------------------------------------------------------------
class _ClampAtEval(nn.Module):
    def output(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.training else x.clamp(0.0, 1.0)


class FSRCNN(_ClampAtEval):
    """FSRCNN (Dong et al., 2016): d=56, s=12, m=4, learned 9x9 deconvolution."""

    def __init__(self, d: int = 56, s: int = 12, m: int = 4) -> None:
        super().__init__()
        layers = [nn.Conv2d(1, d, 5, padding=2), nn.PReLU(d), nn.Conv2d(d, s, 1), nn.PReLU(s)]
        for _ in range(m):
            layers += [nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)]
        layers += [nn.Conv2d(s, d, 1), nn.PReLU(d)]
        self.body = nn.Sequential(*layers)
        self.deconv = nn.ConvTranspose2d(d, 1, 9, stride=2, padding=4, output_padding=1)

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        return self.output(self.deconv(self.body(low_res)))


class _ResBlock(nn.Module):
    def __init__(self, channels: int, batch_norm: bool, activation: nn.Module) -> None:
        super().__init__()
        norm = (lambda: nn.BatchNorm2d(channels)) if batch_norm else nn.Identity
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), norm(), activation,
                                  nn.Conv2d(channels, channels, 3, padding=1), norm())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SRResNet(_ClampAtEval):
    """SRResNet (Ledig et al., 2017): 16 residual blocks with batch norm, PReLU, sub-pixel x2."""

    def __init__(self, channels: int = 64, blocks: int = 16) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(1, channels, 9, padding=4), nn.PReLU(channels))
        self.body = nn.Sequential(*[_ResBlock(channels, True, nn.PReLU(channels)) for _ in range(blocks)],
                                  nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels))
        self.tail = nn.Sequential(nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(channels),
                                  nn.Conv2d(channels, 1, 9, padding=4))

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        features = self.head(low_res)
        return self.output(self.tail(self.body(features) + features))


class EDSR(_ClampAtEval):
    """EDSR-baseline (Lim et al., 2017): 16 residual blocks, no batch norm, sub-pixel x2."""

    def __init__(self, channels: int = 64, blocks: int = 16) -> None:
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[_ResBlock(channels, False, nn.ReLU(inplace=True)) for _ in range(blocks)],
                                  nn.Conv2d(channels, channels, 3, padding=1))
        self.tail = nn.Sequential(nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2), nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        features = self.head(low_res)
        return self.output(self.tail(self.body(features) + features))


class _DenseBlock(nn.Module):
    """Residual dense block: ``layers`` densely connected convs, local fusion, local residual."""

    def __init__(self, channels: int, growth: int, layers: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels + i * growth, growth, 3, padding=1), nn.ReLU(inplace=True)) for i in range(layers)
        )
        self.fuse = nn.Conv2d(channels + layers * growth, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in self.convs:
            features.append(conv(torch.cat(features, dim=1)))
        return x + self.fuse(torch.cat(features, dim=1))


class RDN(_ClampAtEval):
    """RDN (Zhang et al., 2018): D=16 residual dense blocks, C=8 layers, G=64, global fusion."""

    def __init__(self, channels: int = 64, blocks: int = 16, layers: int = 8, growth: int = 64) -> None:
        super().__init__()
        self.shallow1 = nn.Conv2d(1, channels, 3, padding=1)
        self.shallow2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList(_DenseBlock(channels, growth, layers) for _ in range(blocks))
        self.global_fusion = nn.Sequential(nn.Conv2d(channels * blocks, channels, 1), nn.Conv2d(channels, channels, 3, padding=1))
        self.tail = nn.Sequential(nn.Conv2d(channels, channels * 4, 3, padding=1), nn.PixelShuffle(2), nn.Conv2d(channels, 1, 3, padding=1))

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        first = self.shallow1(low_res)
        x = self.shallow2(first)
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return self.output(self.tail(self.global_fusion(torch.cat(outputs, dim=1)) + first))
