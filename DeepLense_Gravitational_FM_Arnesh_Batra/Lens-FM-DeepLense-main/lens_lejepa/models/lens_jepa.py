"""Encoder of the Lens-JEPA family (Rishi et al., 2025), used as a reference.

A Lensiformer-style backbone: shifted-patch tokens of the observed image, a
learned SIS lens inversion that reconstructs the source, and no-self-attention
blocks that combine observed-image and source tokens. It returns one token per
patch, like the ViT, and accepts I-JEPA context masks.

Parameter names match the original experiment code so its checkpoints load
with ``strict=True``.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from torch import nn

from .masking import apply_masks


class _FeedForward(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation, hidden: int, hidden_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        layers = [nn.Linear(in_features, hidden), activation(), nn.Dropout(dropout)]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden, hidden), activation(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, out_features), nn.Identity()]
        self.feedforward = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.feedforward:
            x = layer(x)
        return x


class _NoSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

    def forward(self, key, query, value):
        mask = None
        if query.shape[1] > 1 and key.shape[1] > 1:
            mask = torch.eye(query.shape[1], key.shape[1], dtype=torch.bool, device=query.device)
        return self.mha(query, key, value, attn_mask=mask)[0]


class _LSABlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, activation, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlsa = _NoSelfAttention(dim, heads, dropout)
        self.first_norm = nn.LayerNorm(dim)
        self.feedforward = _FeedForward(dim, dim, activation, hidden, hidden_layers=1, dropout=dropout)
        self.second_norm = nn.LayerNorm(dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, key, query=None, value=None):
        query = key / self.temperature if query is None else query
        value = key if value is None else value
        value = self.first_norm(value + self.mlsa(key, query, value))
        value = self.second_norm(value + self.feedforward(value))
        return self.dropout_layer(value)


class _ShiftedPatchTokenization(nn.Module):
    def __init__(self, image_size: int, patch: int, dim: int, channels: int = 1) -> None:
        super().__init__()
        self.shift = patch // 2
        self.num_patches = (image_size // patch) ** 2
        self.projection = nn.Conv2d(channels * 5, dim, kernel_size=patch, stride=patch)
        self.layer_norm = nn.LayerNorm(dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        s = self.shift
        shifted = [TF.affine(image, angle=0, translate=(dx, dy), fill=0, scale=1, shear=0) for dx, dy in ((-s, -s), (s, -s), (-s, s), (s, s))]
        tokens = self.projection(torch.cat([image, *shifted], dim=1)).flatten(2).transpose(1, 2)
        return self.layer_norm(tokens) + self.positional_encoding


class _PatchDecoder(nn.Module):
    def __init__(self, image_size: int, patch: int, dim: int) -> None:
        super().__init__()
        self.image_size, self.patch, self.grid = image_size, patch, image_size // patch
        self.patch_decoder = nn.Linear(dim, patch * patch)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, g, p = tokens.shape[0], self.grid, self.patch
        return self.patch_decoder(tokens).view(b, g, g, p, p).permute(0, 1, 3, 2, 4).reshape(b, self.image_size, self.image_size)


class _SISInverter(nn.Module):
    """Predicts a per-pixel Einstein-radius factor ``k`` in ``[k_min, k_max]``
    and moves every pixel to ``theta - k * theta / |theta|`` (SIS lens model)."""

    def __init__(self, image_size, patch, dim, heads, hidden, activation, blocks, dropout=0.1,
                 pixel_scale=0.101, k_min=0.8, k_max=1.2, eps=1e-8) -> None:
        super().__init__()
        self.size, self.pixel_scale, self.k_min, self.k_max, self.eps = image_size, pixel_scale, k_min, k_max, eps
        layers: list[nn.Module] = [_LSABlock(dim, heads, hidden, activation, dropout) for _ in range(blocks)]
        layers += [_PatchDecoder(image_size, patch, dim), nn.Sigmoid()]
        self.transformer = nn.ModuleList(layers)
        half = image_size // 2
        coords = torch.linspace(-half, half - 1, image_size) * pixel_scale
        gx, gy = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("flat_grid_x", gx.flatten(), persistent=False)
        self.register_buffer("flat_grid_y", gy.flatten(), persistent=False)

    def forward(self, images: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        b, n, half = images.shape[0], self.size, self.size // 2
        k = patches
        for layer in self.transformer:
            k = layer(k)
        k = self.k_min + (self.k_max - self.k_min) * k.reshape(b, n * n)
        nonzero = (self.flat_grid_x != 0) | (self.flat_grid_y != 0)
        x, y = self.flat_grid_x[nonzero], self.flat_grid_y[nonzero]
        radius = torch.sqrt(x**2 + y**2)
        k = k[:, nonzero]
        sx = x - k * x / radius
        sy = y - k * y / radius
        ix = torch.round(sx / self.pixel_scale + half).long().clamp(0, n - 1)
        iy = torch.round(sy / self.pixel_scale + half).long().clamp(0, n - 1)
        index = ix * n + iy
        flat_in = images.reshape(b, -1)
        out = torch.zeros(b, n * n, device=images.device, dtype=flat_in.dtype)
        current = out.gather(1, index)
        values = flat_in[:, nonzero]
        out.scatter_(1, index, torch.where(current == 0, values, (current + values) / 2))
        out = out.view(b, n, n)
        return out / (out.amax(dim=(1, 2), keepdim=True) + self.eps)


class _Lensiformer(nn.Module):
    def __init__(self, image_size, patch, dim, heads, hidden, blocks, dropout=0.1) -> None:
        super().__init__()
        self.image_size = image_size
        act = nn.ELU
        self.initial_tokenizer = _ShiftedPatchTokenization(image_size, patch, dim)
        self.secondary_tokenizer = _ShiftedPatchTokenization(image_size, patch, dim)
        self.encoder = _SISInverter(image_size, patch, dim, heads, hidden, act, blocks, dropout)
        self.transformer_blocks = nn.ModuleList(_LSABlock(dim, heads, hidden, act, dropout) for _ in range(blocks))

    def forward(self, images: torch.Tensor, masks=None) -> torch.Tensor:
        b, n = images.shape[0], self.image_size
        observed = self.initial_tokenizer(images.reshape(b, 1, n, n))
        source = self.encoder(images, observed)
        source_tokens = self.secondary_tokenizer(source.reshape(b, 1, n, n))
        if masks is not None:
            masks = masks if isinstance(masks, list) else [masks]
            source_tokens = apply_masks(source_tokens, masks)
            observed = apply_masks(observed, masks)
        for block in self.transformer_blocks:
            observed = block(key=observed, value=source_tokens)
        return observed


class LensJEPAEncoder(nn.Module):
    """Lens-JEPA backbone: 384-d tokens, 16 heads, one block (2.45M parameters)."""

    def __init__(self, image_size: int = 160, patch_size: int = 16, embed_dim: int = 384, num_heads: int = 16, depth: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.backbone = _Lensiformer(image_size, patch_size, embed_dim, num_heads, hidden=64, blocks=depth)
        for module in self.modules():  # the initialisation used by the original runs
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02, a=-2.0, b=2.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor, masks=None) -> torch.Tensor:
        return self.backbone(images, masks=masks)
