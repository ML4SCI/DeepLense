"""The full pretraining objective.

    L = (1 - lambda) * L_inv + lambda * L_SIGReg              (LeJEPA base)
        + w_d4 * L_D4 + w_ring * L_ring                       (lensing priors)

With ``use_lens_priors=False`` this is exactly the LeJEPA control used in the
paper: same encoder, same views (minus the D4 transform), same schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .lens_priors import D4_SIZE, d4_patch_permutation, d4_token_loss, d4_transform, ring_consistency_loss
from .sigreg import SIGReg
from .views import lens_safe_view


def lejepa_loss(projected_views: torch.Tensor, sigreg: SIGReg, sigreg_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LeJEPA's teacher-free loss on ``[V, B, D]`` projected views.

    The invariance term pulls every view to the per-image view mean; gradients
    flow through all views (no teacher, no stop-gradient).
    """
    if projected_views.ndim != 3 or projected_views.shape[0] < 2:
        raise ValueError("Expected at least two projected views shaped [V, B, D].")
    invariance = (projected_views - projected_views.mean(dim=0, keepdim=True)).square().mean()
    regulariser = sigreg(projected_views)
    return (1 - sigreg_weight) * invariance + sigreg_weight * regulariser, invariance, regulariser


@dataclass
class ObjectiveConfig:
    use_lens_priors: bool = True
    sigreg_weight: float = 0.02
    sigreg_slices: int = 256
    d4_weight: float = 0.10
    ring_weight: float = 0.10
    rings: int = 3
    arc_strength: float = 0.75
    blur_probability: float = 0.5
    blur_sigma_max: float = 1.0
    noise_std_max: float = 0.03


class LensLeJEPAObjective(nn.Module):
    """Builds two views, encodes them and returns the loss and its parts."""

    def __init__(self, config: ObjectiveConfig, patch_size: int) -> None:
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.sigreg = SIGReg(num_slices=config.sigreg_slices)

    def _view(self, images: torch.Tensor) -> torch.Tensor:
        c = self.config
        return lens_safe_view(images, c.blur_probability, c.blur_sigma_max, c.noise_std_max)

    def forward(self, encoder: nn.Module, projector: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        c = self.config
        canonical_images = self._view(images)
        if c.use_lens_priors:
            # One D4 element for the whole minibatch, applied before the second view.
            index = int(torch.randint(0, D4_SIZE, ()).item())
            peer_images = self._view(d4_transform(images, index))
        else:
            peer_images = self._view(images)

        canonical_tokens = encoder(canonical_images)
        peer_tokens = encoder(peer_images)
        # Project both views in one batch so they share BatchNorm statistics.
        projected = projector(torch.cat((canonical_tokens, peer_tokens), dim=0)).reshape(2, images.shape[0], -1)
        loss, invariance, sigreg = lejepa_loss(projected, self.sigreg, c.sigreg_weight)

        zero = loss.new_zeros(())
        parts = {"invariance": invariance, "sigreg": sigreg, "d4": zero, "ring": zero}
        if c.use_lens_priors:
            permutation = d4_patch_permutation(images.shape[-1] // self.patch_size, index, images.device)
            parts["d4"] = d4_token_loss(canonical_tokens, peer_tokens, permutation)
            parts["ring"] = ring_consistency_loss(
                canonical_tokens, peer_tokens, canonical_images, peer_images, self.patch_size, c.rings, c.arc_strength
            )
            loss = loss + c.d4_weight * parts["d4"] + c.ring_weight * parts["ring"]
        parts["canonical_tokens"] = canonical_tokens.detach()
        return loss, parts
