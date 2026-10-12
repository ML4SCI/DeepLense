"""Self-supervised pretraining: LeJEPA base loss plus the two lensing priors."""

from .diagnostics import pretraining_diagnostics, representation_stats
from .lens_priors import (
    D4_SIZE,
    arc_weights,
    d4_patch_permutation,
    d4_token_loss,
    d4_transform,
    ring_consistency_loss,
    ring_descriptor,
    ring_masks,
)
from .masked_jepa import MASKED_METHODS, MaskedJEPAConfig, MaskedJEPAObjective, ema_update
from .objective import LensLeJEPAObjective, ObjectiveConfig, lejepa_loss
from .sigreg import SIGReg
from .views import gaussian_blur, lens_safe_view

__all__ = [
    "D4_SIZE",
    "LensLeJEPAObjective",
    "MASKED_METHODS",
    "MaskedJEPAConfig",
    "MaskedJEPAObjective",
    "ema_update",
    "ObjectiveConfig",
    "SIGReg",
    "arc_weights",
    "d4_patch_permutation",
    "d4_token_loss",
    "d4_transform",
    "gaussian_blur",
    "lejepa_loss",
    "lens_safe_view",
    "pretraining_diagnostics",
    "representation_stats",
    "ring_consistency_loss",
    "ring_descriptor",
    "ring_masks",
]
