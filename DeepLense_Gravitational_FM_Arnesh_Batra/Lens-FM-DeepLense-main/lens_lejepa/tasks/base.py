"""The interface every downstream task implements.

A task owns everything that differs between classification, regression and
super-resolution: which dataset to read, how to split it, which head to put
on the backbone, the loss, and the metrics. The training loop in
:mod:`lens_lejepa.engine.finetune` only talks to this interface.

To add a task, subclass :class:`Task` and register it with
:func:`lens_lejepa.tasks.register`. See ``docs/ADDING_A_TASK.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ..config import FinetuneConfig
from ..models import NEEDS_DISTORTION, canonical_adapter_name, inject_adapters, is_encoder, load_pretrained_encoder


class Task(ABC):
    name: str = ""
    monitor: str = ""          # validation metric used for checkpoint selection
    higher_is_better: bool = True

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config

    # ---- data ---------------------------------------------------------------
    @abstractmethod
    def build_datasets(self) -> tuple[Dataset, Dataset]:
        """Return ``(train, validation)`` datasets from ``config.train_dataset``."""

    @abstractmethod
    def build_test_dataset(self, root: str) -> Dataset:
        """Return the held-out dataset rooted at ``root``."""

    # ---- model --------------------------------------------------------------
    @abstractmethod
    def build_head(self, encoder: nn.Module) -> nn.Module:
        """Wrap a ViT encoder with this task's head."""

    def build_baseline(self, name: str) -> nn.Module:
        raise ValueError(f"Task {self.name!r} has no baseline called {name!r}.")

    def build_model(self) -> nn.Module:
        """Backbone + head, with the requested adaptation applied."""
        c = self.config
        if not is_encoder(c.backbone):
            return self.build_baseline(c.backbone)
        encoder = load_pretrained_encoder(c.backbone, c.image_size, c.patch_size, c.pretrained)
        if c.adaptation == "full":
            pass
        elif c.adaptation == "linear_probe":
            encoder.requires_grad_(False)
        else:
            inject_adapters(encoder, canonical_adapter_name(c.adaptation), c.rank, c.alpha, c.dropout, c.rank_schedule)
        return self.build_head(encoder)

    @property
    def needs_distortion(self) -> bool:
        """Lensiformer and LensPINN also read the lensing distortion map."""
        return self.config.backbone in NEEDS_DISTORTION

    def model_input(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Positional arguments for ``model(...)``."""
        if self.needs_distortion:
            return batch["image"], batch["distortion"]
        return (batch["image"],)

    # ---- optimisation and evaluation ---------------------------------------
    @abstractmethod
    def loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Training loss from model output and the batch."""

    @abstractmethod
    def collect(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Per-batch arrays needed to compute metrics (moved to CPU)."""

    @abstractmethod
    def metrics(self, outputs: dict[str, np.ndarray]) -> dict[str, Any]:
        """Scalar metrics (plus optional arrays) from the concatenated outputs."""

    # ---- state that must be saved with the checkpoint (e.g. a normaliser) ---
    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass
