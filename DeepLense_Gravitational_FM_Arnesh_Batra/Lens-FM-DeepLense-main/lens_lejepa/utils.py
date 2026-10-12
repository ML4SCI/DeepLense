"""Small helpers shared by the training engines."""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("lens_lejepa")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def make_run_dir(output_dir: str, group: str, run_name: str | None) -> Path:
    run_dir = Path(output_dir) / group / (run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append one row, rewriting the header if new columns appear."""
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows.append({key: value for key, value in row.items() if np.isscalar(value) or value is None})
    fieldnames = list(dict.fromkeys(key for r in rows for key in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values in {name}.")


class WarmupCosine:
    """Linear warm-up from ``start`` to ``peak``, then cosine decay to ``final``.

    Call :meth:`step` once per optimiser step; it sets ``lr`` (or
    ``weight_decay``) on every group except those with ``no_decay=True`` when
    scheduling weight decay.
    """

    def __init__(self, optimizer, key: str, start: float, peak: float, final: float, warmup_steps: int, total_steps: int) -> None:
        self.optimizer, self.key = optimizer, key
        self.start, self.peak, self.final = start, peak, final
        self.warmup_steps, self.total_steps = warmup_steps, max(total_steps, 1)
        self.step_count = 0

    def value(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.start + (self.peak - self.start) * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.final + (self.peak - self.final) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def step(self) -> float:
        self.step_count += 1
        value = self.value(self.step_count)
        for group in self.optimizer.param_groups:
            if self.key == "weight_decay" and group.get("no_decay", False):
                continue
            group[self.key] = value
        return value
