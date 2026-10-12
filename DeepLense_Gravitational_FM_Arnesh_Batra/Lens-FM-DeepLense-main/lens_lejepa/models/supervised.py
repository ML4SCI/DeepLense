"""Supervised baselines from the paper: ViT, ViT-SD, Lensiformer and LensPINN.

These follow the DeepLense implementations used for the reported numbers,
with parameter names kept identical so the original checkpoints load with
``strict=True``. ResNet18 lives in :mod:`lens_lejepa.models.baselines`.

Lensiformer and LensPINN take a second input, the lensing *distortion map*
computed by :func:`lens_lejepa.data.io.distortion_map`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn


def _patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """``b c (h p1) (w p2) -> b (h w) (p1 p2 c)``."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // patch, patch, w // patch, patch)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // patch) * (w // patch), patch * patch * c)


# --------------------------------------------------------------------------------------
# ViT and ViT-SD (Lee et al., 2021: shifted patch tokenization + locality self-attention)
# --------------------------------------------------------------------------------------
class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Attention(nn.Module):
    """Multi-head attention; ``locality=True`` gives ViT-SD's LSA (learned
    temperature, no attention to self)."""

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float, locality: bool) -> None:
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.locality = locality
        if locality:
            self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head**-0.5)))
        else:
            self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in self.to_qkv(self.norm(x)).chunk(3, dim=-1))
        dots = q @ k.transpose(-1, -2) * (self.temperature.exp() if self.locality else self.scale)
        if self.locality:
            eye = torch.eye(n, device=x.device, dtype=torch.bool)
            dots = dots.masked_fill(eye, -torch.finfo(dots.dtype).max)
        out = self.dropout(dots.softmax(dim=-1)) @ v
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, locality: bool) -> None:
        super().__init__()
        self.locality = locality
        if not locality:
            self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            nn.ModuleList([_Attention(dim, heads, dim_head, dropout, locality), _FeedForward(dim, mlp_dim, dropout)]) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x if self.locality else self.norm(x)


class _Patchify(nn.Module):
    def __init__(self, patch: int) -> None:
        super().__init__()
        self.patch = patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _patchify(x, self.patch)


class _ShiftedPatchTokenization(nn.Module):
    """ViT-SD tokenizer: the image plus four half-pixel-free one-pixel shifts."""

    def __init__(self, dim: int, patch: int, channels: int) -> None:
        super().__init__()
        patch_dim = patch * patch * 5 * channels
        self.patch = patch
        self.to_patch_tokens = nn.Sequential(_Patchify(patch), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        return self.to_patch_tokens(torch.cat([x, *(F.pad(x, s) for s in shifts)], dim=1))


class SupervisedViT(nn.Module):
    """Class-token ViT trained from scratch. ``shifted_patches=True`` is ViT-SD."""

    def __init__(
        self,
        image_size: int = 160,
        patch_size: int = 16,
        num_classes: int = 3,
        dim: int = 1024,
        depth: int = 2,
        heads: int = 16,
        mlp_dim: int = 2048,
        channels: int = 1,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        shifted_patches: bool = False,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size
        if shifted_patches:
            self.to_patch_embedding = _ShiftedPatchTokenization(dim, patch_size, channels)
        else:
            self.to_patch_embedding = nn.Sequential(_Patchify(patch_size), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, locality=shifted_patches)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes)) if shifted_patches else nn.Linear(dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(images)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + self.pos_embedding[:, : x.shape[1] + 1]
        return self.mlp_head(self.transformer(self.dropout(x))[:, 0])


# --------------------------------------------------------------------------------------
# LensPINN (Ojha et al., 2024)
# --------------------------------------------------------------------------------------
class LensPINN(nn.Module):
    """Physics-informed CNN on the image and its lensing distortion map."""

    def __init__(self, num_classes: int = 3, channels: int = 2, width: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        def conv(cin, cout, k):
            return [nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU()]

        self.features = nn.Sequential(
            *conv(channels, width, 5), nn.MaxPool2d(2),
            *conv(width, width * 2, 3), nn.MaxPool2d(2),
            *conv(width * 2, width * 4, 3), nn.MaxPool2d(2),
            *conv(width * 4, width * 4, 3), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 4, num_classes))

    def forward(self, images: torch.Tensor, distortions: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(torch.cat([images, distortions], dim=1)))


class LensPINNRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = LensPINN(num_classes=1)

    def forward(self, images: torch.Tensor, distortions: torch.Tensor) -> torch.Tensor:
        return self.backbone(images, distortions).squeeze(-1)


# --------------------------------------------------------------------------------------
# Lensiformer (Velôso et al., 2023)
# --------------------------------------------------------------------------------------
class _MLP(nn.Module):
    """Linear -> act -> dropout, ``hidden_layers`` more of those, then a linear output."""

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, activation, dropout=0.1, softmax_output=False) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation(), nn.Dropout(dropout)]
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, out_dim), nn.Softmax(dim=1) if softmax_output else nn.Identity()]
        self.feed_list = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.feed_list:
            x = layer(x)
        return x


class _NoSelfAttention(nn.Module):
    """Multi-head attention in which no token attends to itself."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, key, query, value):
        mask = None
        if query.shape[1] > 1 and key.shape[1] > 1:
            mask = torch.eye(query.shape[1], key.shape[1], dtype=torch.bool, device=query.device)
        return self.mha(query, key, value, attn_mask=mask)[0]


class _LSABlock(nn.Module):
    def __init__(self, dim, heads, hidden, hidden_layers, activation, dropout=0.1) -> None:
        super().__init__()
        self.mlsa = _NoSelfAttention(dim, heads, dropout)
        self.first_norm = nn.LayerNorm(dim)
        self.feedforward = _MLP(dim, dim, hidden, hidden_layers, activation, dropout)
        self.second_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1))
        self.query_T = nn.Linear(dim, dim)
        self.value_T = nn.Linear(dim, dim)

    def forward(self, key, query=None, value=None):
        if query is None:
            query = self.query_T(key) / self.temperature
        if value is None:
            value = self.value_T(key)
        value = self.first_norm(value + self.mlsa(key, query, value))
        value = self.second_norm(value + self.feedforward(value))
        return self.dropout(value)


class _ShiftTokenizer(nn.Module):
    """Image + four diagonal half-patch shifts -> patch tokens, with class token."""

    def __init__(self, image_size: int, dim: int, patch: int, channels: int = 1) -> None:
        super().__init__()
        self.shift = patch // 2
        self.num_patches = (image_size // patch) ** 2
        self.tokenizer = nn.Conv2d(channels * 5, dim, kernel_size=patch, stride=patch)
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.positional = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

    def _translate(self, image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        return TF.affine(image, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        s = self.shift
        shifted = [self._translate(image, dx, dy) for dx, dy in ((-s, -s), (s, -s), (-s, s), (s, s))]
        patches = self.tokenizer(torch.cat([image, *shifted], dim=1)).flatten(2).transpose(1, 2)
        return torch.cat((self.class_embedding.expand(image.shape[0], -1, -1), patches), dim=1) + self.positional


class _PatchDecoder(nn.Module):
    def __init__(self, image_size: int, patch: int, dim: int) -> None:
        super().__init__()
        self.image_size, self.patch, self.grid = image_size, patch, image_size // patch
        self.patch_decoder = nn.Linear(dim, patch * patch)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == self.grid * self.grid + 1:
            tokens = tokens[:, 1:]
        b, g, p = tokens.shape[0], self.grid, self.patch
        return self.patch_decoder(tokens).view(b, g, g, p, p).permute(0, 1, 3, 2, 4).reshape(b, self.image_size, self.image_size)


def invert_lens(images: torch.Tensor, einstein_angle: torch.Tensor, min_angle: float = -3.323, max_angle: float = 3.232) -> torch.Tensor:
    """Map lensed images back to the source plane with a per-pixel SIS deflection.

    ``images`` and ``einstein_angle`` are ``[B, N, N]``. Each pixel is moved to
    ``beta = theta - theta_E * theta / |theta|`` and pixels landing on the same
    source pixel are averaged. Not differentiable (integer indexing).
    """
    b, n, _ = images.shape
    pixel = (max_angle - min_angle) / n
    centre = n // 2
    axis = torch.arange(-(centre - 1), n - (centre - 1), device=images.device, dtype=images.dtype)
    x, y = torch.meshgrid(axis * pixel, axis * pixel, indexing="ij")
    r = torch.sqrt(x**2 + y**2)
    r = torch.where(r == 0, torch.ones_like(r), r)
    bx = ((x - einstein_angle * x / r) / pixel + centre).clamp(0, n - 1).long()
    by = ((y - einstein_angle * y / r) / pixel + centre).clamp(0, n - 1).long()
    index = (bx * n + by).view(b, -1)
    total = torch.zeros(b, n * n, device=images.device, dtype=images.dtype).scatter_add_(1, index, images.reshape(b, -1))
    count = torch.zeros_like(total).scatter_add_(1, index, torch.ones_like(total))
    return torch.where(count > 0, total / count.clamp_min(1), total).view(b, n, n)


class _PhysicsEncoder(nn.Module):
    """Predicts a per-pixel Einstein-angle correction and inverts the lens."""

    def __init__(self, image_size, patch, dim, heads, hidden, activation, blocks, dropout=0.1) -> None:
        super().__init__()
        self.image_size = image_size
        layers: list[nn.Module] = [_LSABlock(dim, heads, hidden, 1, activation, dropout) for _ in range(blocks)]
        layers.append(_PatchDecoder(image_size, patch, dim))
        self.transformer = nn.ModuleList(layers)

    def forward(self, images, patches, distortion):
        k = patches
        for layer in self.transformer:
            k = layer(k)
        b = images.shape[0]
        angle = k.view(b, self.image_size, self.image_size) * distortion.view(b, self.image_size, self.image_size)
        with torch.no_grad():
            source = invert_lens(images.view(b, self.image_size, self.image_size).detach().float(), angle.detach().float())
        return k, source.to(images.dtype)


class Lensiformer(nn.Module):
    """Physics-informed ViT: tokens of the distortion map, the observed image
    and the reconstructed source are combined by a no-self-attention block.

    As in the reference implementation, the classifier ends in a softmax, so
    ``forward`` returns class probabilities (the training loss is applied to
    them directly). ``forward_with_source`` also returns the source image.
    """

    def __init__(
        self,
        image_size: int = 160,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_classes: int = 3,
        num_heads: int = 16,
        hidden: int = 64,
        hidden_layers: int = 3,
        blocks: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        act = nn.ELU
        self.initial_tokenizer = _ShiftTokenizer(image_size, embed_dim, patch_size)
        self.secondary_tokenizer = _ShiftTokenizer(image_size, embed_dim, patch_size)
        self.tertiary_tokenizer = _ShiftTokenizer(image_size, embed_dim, patch_size)
        self.encoder = _PhysicsEncoder(image_size, patch_size, embed_dim, num_heads, hidden, act, blocks, dropout)
        self.transformer_blocks = nn.ModuleList(_LSABlock(embed_dim, num_heads, hidden, 1, act, dropout) for _ in range(blocks))
        flat = (self.initial_tokenizer.num_patches + 1) * embed_dim
        self.feedforward_layer = _MLP(flat, num_classes, hidden, hidden_layers, act, dropout, softmax_output=True)

    def forward_with_source(self, images: torch.Tensor, distortions: torch.Tensor):
        b, n = images.shape[0], self.image_size
        distortion_tokens = self.tertiary_tokenizer(distortions.reshape(b, 1, n, n))
        image_tokens = self.initial_tokenizer(images.reshape(b, 1, n, n))
        _, source = self.encoder(images, image_tokens, distortions)
        source = source.view(b, 1, n, n)
        source_tokens = self.secondary_tokenizer(source)
        x = image_tokens
        for block in self.transformer_blocks:
            # Every block reads the same three token sets, as in the reference code.
            x = block(key=distortion_tokens, value=image_tokens, query=source_tokens)
        return self.feedforward_layer(x.reshape(b, -1)), source

    def forward(self, images: torch.Tensor, distortions: torch.Tensor) -> torch.Tensor:
        return self.forward_with_source(images, distortions)[0]
