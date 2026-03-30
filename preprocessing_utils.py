"""
Data Preprocessing Utilities for DeepLense.

This module provides reusable data preprocessing functions for
gravitational lensing image datasets. It includes image normalization,
augmentation helpers, and dataset splitting utilities that can be used
across different DeepLense sub-projects.

Author: Kamala Hasini Burra
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def normalize_images(
    images: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Normalize a batch of images using the specified method.

    Parameters
    ----------
    images : np.ndarray
        Batch of images with shape (N, H, W) or (N, H, W, C).
    method : str
        Normalization method. Options:
        - 'minmax': Scale to [0, 1] range
        - 'zscore': Zero mean, unit variance
        - 'robust': Median-centered with IQR scaling

    Returns
    -------
    np.ndarray
        Normalized images with the same shape.

    Raises
    ------
    ValueError
        If an unknown normalization method is specified.

    Examples
    --------
    >>> imgs = np.random.randint(0, 255, (10, 64, 64)).astype(float)
    >>> normed = normalize_images(imgs, method='minmax')
    >>> assert normed.min() >= 0.0 and normed.max() <= 1.0
    """
    images = images.astype(np.float64)

    if method == "minmax":
        img_min = images.min()
        img_max = images.max()
        if img_max - img_min > 0:
            return (images - img_min) / (img_max - img_min)
        return np.zeros_like(images)

    elif method == "zscore":
        mean = images.mean()
        std = images.std()
        if std > 0:
            return (images - mean) / std
        return np.zeros_like(images)

    elif method == "robust":
        median = np.median(images)
        q75 = np.percentile(images, 75)
        q25 = np.percentile(images, 25)
        iqr = q75 - q25
        if iqr > 0:
            return (images - median) / iqr
        return np.zeros_like(images)

    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Choose from: 'minmax', 'zscore', 'robust'."
        )


def augment_image(
    image: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotate_90: int = 0,
    add_noise: bool = False,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Apply augmentation transforms to a single image.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, C).
    flip_horizontal : bool
        Whether to flip horizontally.
    flip_vertical : bool
        Whether to flip vertically.
    rotate_90 : int
        Number of 90-degree rotations (0, 1, 2, or 3).
    add_noise : bool
        Whether to add Gaussian noise.
    noise_std : float
        Standard deviation of Gaussian noise (only used if add_noise=True).

    Returns
    -------
    np.ndarray
        Augmented image with the same dtype.
    """
    result = image.copy()

    if flip_horizontal:
        result = np.flip(result, axis=1)

    if flip_vertical:
        result = np.flip(result, axis=0)

    if rotate_90 > 0:
        axes = (0, 1)
        result = np.rot90(result, k=rotate_90, axes=axes)

    if add_noise:
        noise = np.random.normal(0, noise_std, result.shape)
        result = result + noise

    return np.ascontiguousarray(result)


def stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset indices into train/val/test with stratification.

    Ensures each class is proportionally represented in all splits.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer class labels.
    train_ratio : float
        Proportion for training set.
    val_ratio : float
        Proportion for validation set.
    test_ratio : float
        Proportion for test set.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of np.ndarray
        (train_indices, val_indices, test_indices)

    Raises
    ------
    ValueError
        If ratios don't sum to approximately 1.0.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        raise ValueError(
            f"Ratios must sum to 1.0, got {ratio_sum:.2f}"
        )

    rng = np.random.RandomState(random_seed)
    classes = np.unique(labels)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:])

    return (
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx),
    )


def compute_class_weights(
    labels: np.ndarray,
    method: str = "balanced",
) -> Dict[int, float]:
    """Compute class weights for handling imbalanced datasets.

    Parameters
    ----------
    labels : np.ndarray
        Array of integer class labels.
    method : str
        Weighting method:
        - 'balanced': Inversely proportional to class frequency
        - 'sqrt': Square root of balanced weights (softer rebalancing)

    Returns
    -------
    dict
        Mapping from class label to weight.
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(classes)

    weights: Dict[int, float] = {}

    for cls, count in zip(classes, counts):
        if method == "balanced":
            weights[int(cls)] = n_samples / (n_classes * count)
        elif method == "sqrt":
            weights[int(cls)] = np.sqrt(n_samples / (n_classes * count))
        else:
            raise ValueError(f"Unknown method '{method}'")

    return weights


def create_image_patches(
    image: np.ndarray,
    patch_size: int = 32,
    stride: int = 16,
) -> np.ndarray:
    """Extract overlapping patches from an image.

    Useful for patch-based training of gravitational lens models.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W) or (H, W, C).
    patch_size : int
        Size of each square patch.
    stride : int
        Step size between consecutive patches.

    Returns
    -------
    np.ndarray
        Array of patches with shape (N, patch_size, patch_size, ...).
    """
    h, w = image.shape[:2]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)

    return np.array(patches)
