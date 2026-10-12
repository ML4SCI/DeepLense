"""Small heads that sit on top of the ViT patch tokens."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LeJEPAProjector(nn.Module):
    """Mean-pool the tokens, then a two-layer BatchNorm MLP (pretraining only)."""

    def __init__(self, input_dim: int, output_dim: int = 256) -> None:
        super().__init__()
        hidden_dim = max(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens.mean(dim=1) if tokens.ndim == 3 else tokens)


class EncoderClassifier(nn.Module):
    """Mean-pooled tokens followed by one linear layer."""

    def __init__(self, encoder: nn.Module, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images).mean(dim=1))


class EncoderRegressor(nn.Module):
    """Mean-pooled tokens, LayerNorm, one linear output (normalised target)."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(encoder.embed_dim), nn.Linear(encoder.embed_dim, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images).mean(dim=1)).squeeze(-1)


class PatchSuperResolution(nn.Module):
    """Token-to-patch decoder: each token predicts its own ``P x P`` pixel block.

    The ``G x G`` grid of predicted patches is stitched into an image at the
    encoder resolution, resized bicubically to the target size and squashed
    to ``[0, 1]`` with a sigmoid.
    """

    def __init__(self, encoder: nn.Module, image_size: int, patch_size: int, output_size: int = 64) -> None:
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.output_size = output_size
        self.patch_head = nn.Sequential(nn.LayerNorm(encoder.embed_dim), nn.Linear(encoder.embed_dim, patch_size * patch_size))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_head(self.encoder(images))
        batch, grid, size = patches.shape[0], self.grid_size, self.patch_size
        image = patches.reshape(batch, grid, grid, size, size).permute(0, 1, 3, 2, 4).reshape(batch, 1, grid * size, grid * size)
        image = F.interpolate(image, size=(self.output_size, self.output_size), mode="bicubic", align_corners=False)
        return torch.sigmoid(image)
