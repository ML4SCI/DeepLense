"""Masked-prediction JEPA objectives used as references in the paper.

* ``ijepa``          I-JEPA (Assran et al., 2023): predict EMA-teacher features of
                     masked target blocks from a context block.
* ``lens_jepa``      the same objective on the Lens-JEPA encoder.
* ``lens_jepa_sym``  + a *global* D4 loss: the pooled feature of a rotated or
                     mirrored image must match the pooled teacher feature.
* ``lens_jepa_focus`` prediction loss re-weighted towards arc-like patches.

All four add a variance floor on pooled context features, as in the reference
runs, to prevent a position-only solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..models.masking import apply_masks, repeat_interleave_batch
from .lens_priors import D4_SIZE, arc_weights, d4_transform

MASKED_METHODS = ("ijepa", "lens_jepa", "lens_jepa_sym", "lens_jepa_focus")


@dataclass
class MaskedJEPAConfig:
    predictor_dim: int = 384
    predictor_depth: int = 6
    ema_start: float = 0.996
    ema_end: float = 1.0
    npred: int = 4
    pred_scale: tuple[float, float] = (0.15, 0.2)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)
    enc_scale: tuple[float, float] = (0.85, 1.0)
    min_keep: int = 1
    variance_weight: float = 10.0
    variance_target_std: float = 0.1
    symmetry_weight: float = 0.1        # lens_jepa_sym only
    focus_strength: float = 1.0         # lens_jepa_focus only

    def __post_init__(self) -> None:
        self.pred_scale, self.aspect_ratio, self.enc_scale = map(tuple, (self.pred_scale, self.aspect_ratio, self.enc_scale))


def variance_floor(tokens: torch.Tensor, target_std: float) -> torch.Tensor:
    """Hinge on the per-dimension std of image-pooled features."""
    pooled = tokens.float().mean(dim=1)
    return F.relu(target_std - torch.sqrt(pooled.var(dim=0, unbiased=False) + 1e-4)).mean()


def focus_weighted_loss(pred, target, images, masks_pred, batch, nenc, patch_size, strength) -> torch.Tensor:
    weights = arc_weights(images, patch_size, strength).unsqueeze(-1)
    weights = repeat_interleave_batch(apply_masks(weights, masks_pred), batch, repeat=nenc)
    per_token = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(dim=-1, keepdim=True)
    return (per_token * weights).sum() / weights.sum().clamp_min(1e-6)


def global_symmetry_loss(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
    dim = student_tokens.shape[-1]
    return F.smooth_l1_loss(F.layer_norm(student_tokens.float().mean(1), (dim,)), F.layer_norm(teacher_tokens.float().mean(1), (dim,)))


class MaskedJEPAObjective(nn.Module):
    def __init__(self, method: str, config: MaskedJEPAConfig, patch_size: int) -> None:
        super().__init__()
        if method not in MASKED_METHODS:
            raise ValueError(f"Unknown masked JEPA method {method!r}.")
        self.method, self.config, self.patch_size = method, config, patch_size

    def forward(self, encoder, teacher, predictor, images, masks_enc, masks_pred):
        """``images`` must already be standardised. Returns ``(loss, parts)``."""
        c, batch = self.config, images.shape[0]
        with torch.no_grad():
            teacher_full = F.layer_norm(teacher(images), (teacher.embed_dim,))
            target = repeat_interleave_batch(apply_masks(teacher_full, masks_pred), batch, repeat=len(masks_enc))
        context = encoder(images, masks_enc)
        predicted = predictor(context, masks_enc, masks_pred)
        if self.method == "lens_jepa_focus":
            prediction = focus_weighted_loss(predicted, target, images, masks_pred, batch, len(masks_enc), self.patch_size, c.focus_strength)
        else:
            prediction = F.smooth_l1_loss(predicted, target)
        loss = prediction
        parts = {"prediction": prediction, "symmetry": prediction.new_zeros(()), "variance": prediction.new_zeros(())}
        if self.method == "lens_jepa_sym":
            rotated = encoder(d4_transform(images, int(torch.randint(0, D4_SIZE, ()).item())))
            parts["symmetry"] = global_symmetry_loss(rotated, teacher_full)
            loss = loss + c.symmetry_weight * parts["symmetry"]
        if c.variance_weight > 0:
            parts["variance"] = variance_floor(context, c.variance_target_std)
            loss = loss + c.variance_weight * parts["variance"]
        parts["canonical_tokens"] = context.detach()
        return loss, parts


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(momentum).add_(s.detach(), alpha=1.0 - momentum)
