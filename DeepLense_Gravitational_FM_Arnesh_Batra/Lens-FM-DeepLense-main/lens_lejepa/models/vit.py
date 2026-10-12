"""A plain Vision Transformer encoder that returns patch tokens.

Adapted from the I-JEPA reference implementation (Meta, CC BY-NC 4.0).
Parameter names are kept identical so checkpoints from the original
experiments load with ``strict=True``. There is no class token: the encoder
returns one token per patch, ``[B, (H/P) * (W/P), D]``, and downstream heads
mean-pool when they need a single vector.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .masking import apply_masks, repeat_interleave_batch


def sincos_2d_position_embedding(dim: int, grid_size: int) -> np.ndarray:
    """Fixed 2-D sine-cosine position embedding, ``[grid_size**2, dim]``."""
    if dim % 4:
        raise ValueError("Embedding dimension must be divisible by 4.")

    def one_axis(positions: np.ndarray, axis_dim: int) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(axis_dim // 2, dtype=np.float64) / (axis_dim / 2.0))
        angles = np.einsum("m,d->md", positions.reshape(-1), omega)
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    coords = np.arange(grid_size, dtype=np.float64)
    grid_w, grid_h = np.meshgrid(coords, coords)  # width first, as in I-JEPA
    return np.concatenate([one_axis(grid_w, dim // 2), one_axis(grid_h, dim // 2)], axis=1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        if img_size % patch_size:
            raise ValueError("Image size must be a multiple of the patch size.")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, norm_layer) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """ViT encoder: patchify, add fixed position embedding, transformer blocks."""

    def __init__(
        self,
        img_size: int = 160,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        position = sincos_2d_position_embedding(embed_dim, self.grid_size)
        self.pos_embed = nn.Parameter(torch.from_numpy(position).float().unsqueeze(0), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_residual_branches()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.init_std, a=-2.0, b=2.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _rescale_residual_branches(self) -> None:
        # I-JEPA / BEiT trick: shrink each residual branch by sqrt(2 * depth).
        for layer_id, block in enumerate(self.blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def forward(self, x: torch.Tensor, masks: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Patch tokens ``[B, N, D]``; with I-JEPA ``masks``, only the kept tokens."""
        x = self.patch_embed(x) + self.pos_embed
        if masks is not None:
            x = apply_masks(x, masks if isinstance(masks, list) else [masks])
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


VIT_CONFIGS: dict[str, dict[str, int]] = {
    "vit_tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    # The wide 3-block encoder used by the 300-epoch I-JEPA reference run (21.5M parameters).
    "vit_base_3blocks": {"embed_dim": 768, "depth": 3, "num_heads": 12},
}


def build_vit(name: str, img_size: int, patch_size: int) -> VisionTransformer:
    if name not in VIT_CONFIGS:
        raise ValueError(f"Unknown ViT {name!r}; choose from {sorted(VIT_CONFIGS)}.")
    return VisionTransformer(img_size=img_size, patch_size=patch_size, **VIT_CONFIGS[name])


class JEPAPredictor(nn.Module):
    """I-JEPA predictor: a narrow ViT that fills in target tokens from context tokens."""

    def __init__(self, num_patches: int, embed_dim: int, predictor_dim: int = 384, depth: int = 6, num_heads: int = 12, init_std: float = 0.02) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.predictor_embed = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        position = sincos_2d_position_embedding(predictor_dim, int(num_patches**0.5))
        self.predictor_pos_embed = nn.Parameter(torch.from_numpy(position).float().unsqueeze(0), requires_grad=False)
        self.predictor_blocks = nn.ModuleList([Block(predictor_dim, num_heads, 4.0, norm_layer) for _ in range(depth)])
        self.predictor_norm = norm_layer(predictor_dim)
        self.predictor_proj = nn.Linear(predictor_dim, embed_dim)
        self.init_std = init_std
        nn.init.trunc_normal_(self.mask_token, std=init_std, a=-2.0, b=2.0)
        self.apply(self._init_weights)
        for layer_id, block in enumerate(self.predictor_blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std, a=-2.0, b=2.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor, masks_enc: list[torch.Tensor], masks_pred: list[torch.Tensor]) -> torch.Tensor:
        batch = len(context) // len(masks_enc)
        x = self.predictor_embed(context)
        x = x + apply_masks(self.predictor_pos_embed.repeat(batch, 1, 1), masks_enc)
        n_context = x.shape[1]
        target_pos = apply_masks(self.predictor_pos_embed.repeat(batch, 1, 1), masks_pred)
        target_pos = repeat_interleave_batch(target_pos, batch, repeat=len(masks_enc))
        targets = self.mask_token.repeat(target_pos.shape[0], target_pos.shape[1], 1) + target_pos
        x = torch.cat([x.repeat(len(masks_pred), 1, 1), targets], dim=1)
        for block in self.predictor_blocks:
            x = block(x)
        return self.predictor_proj(self.predictor_norm(x)[:, n_context:])
