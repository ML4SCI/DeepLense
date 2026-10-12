"""The two lensing priors of Lens-LeJEPA.

Prior 1 - exact D4 patch correspondence
    A rotation by a multiple of 90 degrees or a reflection (the dihedral group
    D4) of a centred lens is an equally valid observation of the same system.
    When the image side is a multiple of the patch size, such a transform
    moves whole patches onto whole patches, so it induces an *exact*
    permutation of patch indices. We compare the tokens of the transformed
    view with the canonical tokens re-indexed by that permutation.

Prior 2 - arc-weighted ring consistency
    Tokens are pooled into ``K`` lens-centred rings. Each patch is weighted by a
    label-free arc detector (image gradient + Laplacian magnitude), so patches
    on bright, curved arcs dominate. D4 preserves distance from the centre, so
    every ring maps onto itself and the descriptor is D4-invariant.

Nothing here uses labels, a lens model or the Einstein radius.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

D4_SIZE = 8  # four rotations, each with or without a reflection


def d4_transform(x: torch.Tensor, index: int) -> torch.Tensor:
    """Apply D4 element ``index`` (0-7) to the last two dimensions of ``x``.

    ``index % 4`` counter-clockwise quarter turns, then a left-right flip when
    ``index >= 4``.
    """
    if not 0 <= index < D4_SIZE:
        raise ValueError("D4 index must be in [0, 8).")
    out = torch.rot90(x, k=index % 4, dims=(-2, -1))
    if index >= 4:
        out = torch.flip(out, dims=(-1,))
    return out.contiguous()


def d4_patch_permutation(grid_size: int, index: int, device: torch.device | None = None) -> torch.Tensor:
    """Patch permutation induced by D4 element ``index`` on a ``G x G`` grid.

    ``perm[p] = q`` means patch ``p`` of the *transformed* image contains the
    pixels of patch ``q`` of the original image. Applying the image transform to
    the grid of patch indices gives exactly this map.
    """
    grid = torch.arange(grid_size * grid_size, device=device).reshape(grid_size, grid_size)
    return d4_transform(grid, index).reshape(-1)


def arc_weights(images: torch.Tensor, patch_size: int, strength: float) -> torch.Tensor:
    """Per-patch arc-detector weights with mean one, ``[B, num_patches]``.

    Saliency is ``|grad I| + |laplacian I|`` averaged over each patch and
    normalised to unit mean per image. ``strength`` interpolates between uniform
    weights (0) and the raw normalised saliency (1).
    """
    if images.shape[-1] % patch_size or images.shape[-2] % patch_size:
        raise ValueError("Image size must be a multiple of the patch size.")
    source = images.float()
    sobel_x = source.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    laplacian = source.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(source, sobel_x, padding=1)
    grad_y = F.conv2d(source, sobel_x.transpose(-1, -2), padding=1)
    saliency = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8) + F.conv2d(source, laplacian, padding=1).abs()
    per_patch = F.avg_pool2d(saliency, patch_size, patch_size).flatten(1)
    per_patch = torch.nan_to_num(per_patch, nan=0.0, posinf=0.0, neginf=0.0)
    normalised = per_patch / per_patch.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return 1.0 + strength * (normalised - 1.0)


def ring_masks(grid_size: int, rings: int, device: torch.device | None = None) -> torch.Tensor:
    """One-hot ring membership ``[rings, num_patches]``.

    Ring ``k`` holds patches whose centre distance, divided by the half
    diagonal of the grid, lies in ``[k / K, (k + 1) / K)``.
    """
    if grid_size <= 0 or rings <= 0:
        raise ValueError("grid_size and rings must be positive.")
    coords = torch.arange(grid_size, device=device, dtype=torch.float32)
    rows, cols = torch.meshgrid(coords, coords, indexing="ij")
    centre = (grid_size - 1) / 2.0
    radius = torch.sqrt((rows - centre).square() + (cols - centre).square()) / max(centre * 2.0**0.5, 1e-6)
    index = torch.clamp((radius * rings).long(), max=rings - 1).reshape(-1)
    return F.one_hot(index, num_classes=rings).T.float()


def ring_descriptor(tokens: torch.Tensor, images: torch.Tensor, patch_size: int, rings: int, strength: float) -> torch.Tensor:
    """Arc-weighted mean token in each ring, ``[B, rings, D]``."""
    grid_size = images.shape[-1] // patch_size
    if grid_size * grid_size != tokens.shape[1]:
        raise ValueError("Token count does not match the image patch grid.")
    weights = arc_weights(images, patch_size, strength).to(tokens.dtype)            # [B, N]
    masks = ring_masks(grid_size, rings, tokens.device).to(tokens.dtype)            # [K, N]
    weighted = weights.unsqueeze(1) * masks.unsqueeze(0)                            # [B, K, N]
    return torch.einsum("bkn,bnd->bkd", weighted, tokens) / weighted.sum(-1, keepdim=True).clamp_min(1e-6)


def d4_token_loss(canonical_tokens: torch.Tensor, transformed_tokens: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Prior 1: smooth-L1 between layer-normed corresponding patch tokens."""
    if canonical_tokens.shape != transformed_tokens.shape:
        raise ValueError("Token maps must have the same shape.")
    if permutation.numel() != canonical_tokens.shape[1]:
        raise ValueError("Permutation size does not match the number of tokens.")
    dim = canonical_tokens.shape[-1]
    target = F.layer_norm(canonical_tokens[:, permutation].float(), (dim,))
    return F.smooth_l1_loss(F.layer_norm(transformed_tokens.float(), (dim,)), target)


def ring_consistency_loss(
    canonical_tokens: torch.Tensor,
    transformed_tokens: torch.Tensor,
    canonical_images: torch.Tensor,
    transformed_images: torch.Tensor,
    patch_size: int,
    rings: int,
    strength: float,
) -> torch.Tensor:
    """Prior 2: smooth-L1 between layer-normed ring descriptors of both views."""
    a = ring_descriptor(canonical_tokens, canonical_images, patch_size, rings, strength).float()
    b = ring_descriptor(transformed_tokens, transformed_images, patch_size, rings, strength).float()
    dim = a.shape[-1]
    return F.smooth_l1_loss(F.layer_norm(b, (dim,)), F.layer_norm(a, (dim,)))
