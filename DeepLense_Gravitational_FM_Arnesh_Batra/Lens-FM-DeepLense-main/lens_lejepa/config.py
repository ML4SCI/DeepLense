"""Typed run configurations, loaded from YAML with command-line overrides.

Every field has a default equal to the setting used in the paper, so a YAML
file only needs to list what differs. Overrides use ``key=value`` pairs, with
values parsed as YAML (``shots=100``, ``shots=null``, ``rank_schedule=[8,16]``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .ssl.masked_jepa import MASKED_METHODS, MaskedJEPAConfig
from .ssl.objective import ObjectiveConfig


@dataclass
class PretrainConfig:
    """Self-supervised pretraining on unlabelled images (Model I in the paper)."""

    method: str = "lens_lejepa"            # lens_lejepa | lejepa | ijepa | lens_jepa | lens_jepa_sym | lens_jepa_focus
    data_root: str = "../datasets"
    dataset: str = "Model_I"
    max_samples: int | None = None
    image_size: int = 160
    patch_size: int = 16
    encoder: str = "vit_small"             # vit_small | vit_base_3blocks | lens_jepa | ...
    projector_dim: int = 256               # LeJEPA methods only
    epochs: int = 100
    batch_size: int = 128
    lr: float = 5e-4
    start_lr: float = 5e-5
    final_lr: float = 5e-7
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    final_weight_decay: float = 0.05
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)      # lens_lejepa / lejepa
    masked: MaskedJEPAConfig = field(default_factory=MaskedJEPAConfig)      # masked-prediction references
    # A run is rejected if pooled features stay collapsed this many epochs.
    collapse_std_threshold: float = 0.02
    collapse_rank_threshold: float = 0.04
    collapse_patience: int = 3
    bf16: bool = True
    seed: int = 42
    workers: int = 8
    device: str = "cuda"
    output_dir: str = "runs"
    run_name: str | None = None
    max_steps_per_epoch: int | None = None  # for quick smoke tests only

    def __post_init__(self) -> None:
        if self.method not in ("lens_lejepa", "lejepa", *MASKED_METHODS):
            raise ValueError(f"Unknown method {self.method!r}.")
        if isinstance(self.objective, dict):
            self.objective = ObjectiveConfig(**self.objective)
        if isinstance(self.masked, dict):
            self.masked = MaskedJEPAConfig(**self.masked)
        self.objective.use_lens_priors = self.method == "lens_lejepa"

    @property
    def is_masked(self) -> bool:
        return self.method in MASKED_METHODS


@dataclass
class FinetuneConfig:
    """Downstream training of one task on top of a (pretrained) backbone."""

    task: str = "classification"           # classification | regression | super_resolution
    data_root: str = "../datasets"
    train_dataset: str = "Model_II"
    test_dataset: str | None = "Model_II_test"
    shots: int | None = None               # labels per class (classification/SR) or axion images (regression); null = all
    train_fraction: float = 0.8            # full-data train/validation split
    val_fraction: float = 0.25             # few-shot: validation images per class = shots * val_fraction
    cache: bool = False                    # keep raw arrays in RAM
    image_size: int = 160
    patch_size: int = 16
    # Backbone: a ViT (optionally from a pretraining checkpoint) or a baseline.
    backbone: str = "vit_small"            # encoder: vit_small | vit_tiny | vit_base | vit_base_3blocks | lens_jepa
                                           # or a baseline: resnet18 | vit | vitsd | lensiformer | lenspinn | rcan | bicubic
    pretrained: str | None = None          # path to last_pretrain.pt
    adaptation: str = "rslora"             # an adapter name, "full" or "linear_probe" (encoder backbones only)
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.0
    rank_schedule: list[int] | None = None # layerwise adapter only
    loraplus_ratio: float = 2.0
    epochs: int = 50
    batch_size: int = 128
    grad_accum: int = 1                    # effective batch = batch_size * grad_accum
    lr: float = 3e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1           # classification only
    train_jitter: int = 0                  # classification only: random +-N native-pixel shifts of training images
    val_interval: int = 10
    bf16: bool = True
    seed: int = 42
    workers: int = 8
    device: str = "cuda"
    output_dir: str = "runs"
    run_name: str | None = None
    max_steps_per_epoch: int | None = None


C = TypeVar("C", PretrainConfig, FinetuneConfig)


def _parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ValueError(f"Override {item!r} must look like key=value.")
    key, raw = item.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw)


def _set_nested(data: dict, keys: list[str], value: Any) -> None:
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def load_config(cls: type[C], path: str | Path | None = None, overrides: list[str] | None = None) -> C:
    """Build ``cls`` from an optional YAML file plus ``key=value`` overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        keys, value = _parse_override(item)
        _set_nested(data, keys, value)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def config_to_dict(config: Any) -> dict[str, Any]:
    return dataclasses.asdict(config)
