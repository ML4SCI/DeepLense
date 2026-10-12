"""Synthetic 2x super-resolution: reconstruct the native 64x64 image from a
bicubic 32x32 copy.

The ViT sees the low-resolution image resized to the encoder resolution; CNN
baselines (RCAN) and bicubic interpolation see the 32x32 image directly.
Loss: Charbonnier + 0.1 * (1 - SSIM).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

from ..data import SyntheticSuperResolutionDataset, few_shot_split, stratified_split
from ..metrics import ssim, super_resolution_metrics
from ..models import EDSR, FSRCNN, RCAN, RDN, Bicubic, PatchSuperResolution, SRResNet, is_encoder
from .base import Task


def charbonnier_ssim_loss(prediction: torch.Tensor, target: torch.Tensor, ssim_weight: float = 0.1) -> torch.Tensor:
    charbonnier = torch.sqrt((prediction - target).square() + 1e-6).mean()
    return charbonnier + ssim_weight * (1.0 - ssim(prediction, target))


class SuperResolutionTask(Task):
    """Synthetic 2x super-resolution (32 -> 64 pixels)."""

    name = "super_resolution"
    monitor = "psnr"
    higher_is_better = True
    hr_size = 64

    def build_datasets(self):
        c = self.config
        dataset = SyntheticSuperResolutionDataset(f"{c.data_root}/{c.train_dataset}", c.image_size, self.hr_size, cache=c.cache)
        train_idx, val_idx = stratified_split(dataset.labels, c.train_fraction, c.seed)
        if c.shots is not None:
            # Few-shot: keep ``shots`` training images per class from the train split.
            local, _ = few_shot_split(dataset.labels[train_idx], c.shots, c.val_fraction, c.seed)
            train_idx = train_idx[local]
        return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())

    def build_test_dataset(self, root: str):
        return SyntheticSuperResolutionDataset(root, self.config.image_size, self.hr_size, cache=self.config.cache)

    def build_head(self, encoder: nn.Module) -> nn.Module:
        return PatchSuperResolution(encoder, self.config.image_size, self.config.patch_size, self.hr_size)

    def build_baseline(self, name: str) -> nn.Module:
        baselines = {"rcan": RCAN, "edsr": EDSR, "rdn": RDN, "srresnet": SRResNet, "fsrcnn": FSRCNN}
        if name in baselines:
            return baselines[name]()
        if name == "bicubic":
            return Bicubic(self.hr_size)
        return super().build_baseline(name)

    def model_input(self, batch):
        return (batch["encoder_input"] if is_encoder(self.config.backbone) else batch["low_res"],)

    def loss(self, output, batch):
        return charbonnier_ssim_loss(output.float(), batch["high_res"].float())

    def collect(self, output, batch):
        prediction, target = output.float(), batch["high_res"].float()
        return {
            "predictions": prediction.cpu().numpy(),
            "targets": target.cpu().numpy(),
            "ssim": ssim(prediction, target, reduce=False).cpu().numpy(),
        }

    def metrics(self, outputs: dict[str, np.ndarray]) -> dict:
        return super_resolution_metrics(outputs["predictions"], outputs["targets"], outputs["ssim"])
