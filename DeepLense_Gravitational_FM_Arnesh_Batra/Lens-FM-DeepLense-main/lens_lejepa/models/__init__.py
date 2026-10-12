"""Encoders, heads, adapters and baselines."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .adapters import ADAPTERS, adapter_param_groups, canonical_adapter_name, inject_adapters
from .baselines import EDSR, FSRCNN, RCAN, RDN, Bicubic, ResNetRegressor, SRResNet, build_resnet18
from .heads import EncoderClassifier, EncoderRegressor, LeJEPAProjector, PatchSuperResolution
from .lens_jepa import LensJEPAEncoder
from .masking import BlockMaskCollator, MaskConfig, apply_masks, repeat_interleave_batch
from .supervised import Lensiformer, LensPINN, LensPINNRegressor, SupervisedViT
from .vit import VIT_CONFIGS, JEPAPredictor, VisionTransformer, build_vit

# Self-supervised encoders that return patch tokens [B, N, D].
ENCODERS: tuple[str, ...] = (*VIT_CONFIGS, "lens_jepa")
# Baselines that need the lensing distortion map as a second input.
NEEDS_DISTORTION: tuple[str, ...] = ("lensiformer", "lenspinn")


def is_encoder(name: str) -> bool:
    return name in ENCODERS


def build_encoder(name: str, image_size: int, patch_size: int) -> nn.Module:
    if name == "lens_jepa":
        return LensJEPAEncoder(image_size, patch_size)
    return build_vit(name, image_size, patch_size)


def load_pretrained_encoder(name: str, image_size: int, patch_size: int, checkpoint: str | Path | None) -> nn.Module:
    """Build an encoder and, if a checkpoint is given, load its ``encoder`` weights."""
    encoder = build_encoder(name, image_size, patch_size)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["encoder"] if "encoder" in state else state, strict=True)
    return encoder


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


__all__ = [
    "ADAPTERS",
    "ENCODERS",
    "NEEDS_DISTORTION",
    "EDSR",
    "FSRCNN",
    "RCAN",
    "RDN",
    "SRResNet",
    "VIT_CONFIGS",
    "Bicubic",
    "BlockMaskCollator",
    "EncoderClassifier",
    "EncoderRegressor",
    "JEPAPredictor",
    "LeJEPAProjector",
    "LensJEPAEncoder",
    "LensPINN",
    "LensPINNRegressor",
    "Lensiformer",
    "MaskConfig",
    "PatchSuperResolution",
    "ResNetRegressor",
    "SupervisedViT",
    "VisionTransformer",
    "adapter_param_groups",
    "apply_masks",
    "build_encoder",
    "build_resnet18",
    "build_vit",
    "canonical_adapter_name",
    "count_parameters",
    "inject_adapters",
    "is_encoder",
    "load_pretrained_encoder",
    "repeat_interleave_batch",
]
