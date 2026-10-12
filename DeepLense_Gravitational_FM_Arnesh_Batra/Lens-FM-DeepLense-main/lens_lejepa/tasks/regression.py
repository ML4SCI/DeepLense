"""Axion mass regression: predict ``log10(m_axion / eV)`` from axion images.

Targets are standardised with the mean/std of the *training* split only; the
normaliser is saved with the checkpoint and undone before computing metrics,
so all reported errors are in dex.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset

from ..data import AxionMassDataset, quantile_split, quantile_subsample
from ..metrics import regression_metrics
from ..models import EncoderRegressor, LensPINNRegressor, ResNetRegressor
from .base import Task


class RegressionTask(Task):
    """Axion mass regression, target log10(m / eV)."""

    name = "regression"
    monitor = "rmse_dex"
    higher_is_better = False

    def __init__(self, config) -> None:
        super().__init__(config)
        self.mean, self.std = 0.0, 1.0

    def build_datasets(self):
        c = self.config
        dataset = AxionMassDataset(f"{c.data_root}/{c.train_dataset}", c.image_size, cache=c.cache, distortion=self.needs_distortion)
        train_idx, val_idx = quantile_split(dataset.targets, c.train_fraction, c.seed)
        if c.shots is not None:
            train_idx = train_idx[quantile_subsample(dataset.targets[train_idx], c.shots, c.seed)]
        train_targets = dataset.targets[train_idx]
        self.mean = float(train_targets.mean())
        self.std = max(float(train_targets.std()), 1e-6)
        return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())

    def build_test_dataset(self, root: str):
        return AxionMassDataset(root, self.config.image_size, cache=self.config.cache, distortion=self.needs_distortion)

    def build_head(self, encoder: nn.Module) -> nn.Module:
        return EncoderRegressor(encoder)

    def build_baseline(self, name: str) -> nn.Module:
        if name == "resnet18":
            return ResNetRegressor()
        if name == "lenspinn":
            return LensPINNRegressor()
        return super().build_baseline(name)

    def loss(self, output, batch):
        target = (batch["target"] - self.mean) / self.std
        return F.huber_loss(output.float(), target, delta=1.0)

    def collect(self, output, batch):
        return {
            "predictions": (output.float() * self.std + self.mean).cpu().numpy(),
            "targets": batch["target"].cpu().numpy(),
        }

    def metrics(self, outputs: dict[str, np.ndarray]) -> dict:
        return regression_metrics(outputs["targets"], outputs["predictions"])

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state):
        self.mean, self.std = float(state["mean"]), float(state["std"])
