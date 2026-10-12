"""Deterministic, stratified train/validation splits.

All functions return integer index arrays into a dataset and depend only on the
labels/targets and the seed, so the same seed always gives the same split.
"""

from __future__ import annotations

import numpy as np


def stratified_split(labels: np.ndarray, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split every class into ``train_fraction`` train and the rest validation."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train, val = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cut = int(len(indices) * train_fraction)
        train.extend(indices[:cut])
        val.extend(indices[cut:])
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def few_shot_split(
    labels: np.ndarray,
    shots_per_class: int,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Take ``shots_per_class`` training images per class.

    The validation set is drawn from the remaining images of each class and has
    ``int(shots_per_class * val_fraction)`` images per class (at least one), so
    model selection never sees more labels than the budget implies.
    """
    if shots_per_class <= 0:
        raise ValueError("shots_per_class must be positive.")
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    val_per_class = max(1, int(shots_per_class * val_fraction))
    train, val = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        train.extend(indices[:shots_per_class])
        val.extend(indices[shots_per_class : shots_per_class + val_per_class])
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def quantile_groups(values: np.ndarray, bins: int = 10) -> np.ndarray:
    """Assign each continuous value to one of ``bins`` quantile groups."""
    values = np.asarray(values)
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.int64)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, min(bins, len(values)) + 1)))
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=True)


def quantile_split(values: np.ndarray, train_fraction: float, seed: int, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Train/validation split of a continuous target, stratified by quantile.

    Every quantile group contributes to both sides, so the full target range
    appears in training and in validation.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if len(values) < 2:
        raise ValueError("Need at least two targets to split.")
    rng = np.random.RandomState(seed)
    groups = quantile_groups(values, bins)
    train, val = [], []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        rng.shuffle(indices)
        if len(indices) == 1:
            train.extend(indices)
            continue
        cut = min(len(indices) - 1, max(1, int(round(len(indices) * train_fraction))))
        train.extend(indices[:cut])
        val.extend(indices[cut:])
    if not val:
        val.append(train.pop())
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def quantile_subsample(values: np.ndarray, count: int, seed: int, bins: int = 10) -> np.ndarray:
    """Pick ``count`` items whose targets cover every quantile proportionally."""
    values = np.asarray(values)
    if count >= len(values):
        return np.arange(len(values), dtype=np.int64)
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = np.random.RandomState(seed)
    groups = quantile_groups(values, bins)
    selected: list[int] = []
    for group in np.unique(groups):
        candidates = np.flatnonzero(groups == group)
        rng.shuffle(candidates)
        quota = max(1, round(count * len(candidates) / len(values)))
        selected.extend(candidates[:quota].tolist())
    if len(selected) > count:
        rng.shuffle(selected)
        selected = selected[:count]
    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(len(values)), selected)
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)].tolist())
    return np.asarray(selected, dtype=np.int64)
