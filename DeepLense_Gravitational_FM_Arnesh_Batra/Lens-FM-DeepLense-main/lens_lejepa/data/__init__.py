"""Datasets, splits and loaders for the DeepLense simulations."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .datasets import (
    CLASSES,
    AxionMassDataset,
    LensClassificationDataset,
    SyntheticSuperResolutionDataset,
    UnlabelledLensDataset,
    list_class_files,
)
from .splits import few_shot_split, quantile_split, quantile_subsample, stratified_split


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, drop_last: bool = False) -> DataLoader:
    """A ``DataLoader`` with sensible defaults for GPU training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=drop_last,
    )


__all__ = [
    "CLASSES",
    "AxionMassDataset",
    "LensClassificationDataset",
    "SyntheticSuperResolutionDataset",
    "UnlabelledLensDataset",
    "few_shot_split",
    "list_class_files",
    "make_loader",
    "quantile_split",
    "quantile_subsample",
    "stratified_split",
]
