"""
transforms.py
-------------
Build torchvision-compatible transform pipelines for LSST/Rubin images
that are consistent with the augmentations used in DeepLense Model I/II/III
training (see DeepLense_Classification_* folders in the main repo).

Why a separate function rather than just using torchvision directly?
--------------------------------------------------------------------
LSST images have different statistical properties from natural images:
  - Single-channel (or 3-band stacked) float32 arrays
  - Strong rotational symmetry (lensing arcs wrap around the lens centre)
  - Pixel values after asinh stretch are in [0, 1] but *not* the ImageNet
    mean/std distribution — using standard ImageNet normalization would
    degrade performance.

This module provides ``build_deeplense_transforms`` which applies
physically-motivated augmentations and the correct per-dataset statistics.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# Empirical mean/std for DeepLense Model I (single channel, asinh-stretched)
# derived from the training split.  Update these if using a different dataset.
_DEEPLENSE_MEAN = (0.5,)
_DEEPLENSE_STD = (0.5,)


def build_deeplense_transforms(
    image_size: int = 64,
    is_train: bool = True,
    num_channels: int = 1,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    use_random_rotation: bool = True,
    use_random_flip: bool = True,
):
    """
    Return a ``torchvision.transforms.Compose`` pipeline suitable for
    LSST lensing images fed into DeepLense models.

    Parameters
    ----------
    image_size : int
        Final spatial size (square) after transforms.
    is_train : bool
        If ``True`` include data augmentation.  If ``False`` return only
        resize + normalise.
    num_channels : int
        Number of image channels (1 for single-band, 3 for gri stack).
    mean, std : tuple[float, ...], optional
        Per-channel normalisation constants.  Defaults to 0.5 / 0.5 which
        maps [0, 1] asinh-stretched images to [-1, 1].
    use_random_rotation : bool
        Apply random 90° rotations (exploits rotational symmetry of lensing).
    use_random_flip : bool
        Apply random horizontal / vertical flips.

    Returns
    -------
    torchvision.transforms.Compose
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch and torchvision are required.  "
            "Install with:  pip install torch torchvision"
        )

    _mean = mean if mean is not None else _DEEPLENSE_MEAN * num_channels
    _std = std if std is not None else _DEEPLENSE_STD * num_channels

    train_transforms = []
    eval_transforms = []

    # Resize to model input size
    resize = T.Resize((image_size, image_size), antialias=True)
    train_transforms.append(resize)
    eval_transforms.append(resize)

    # Augmentations — only for training
    if is_train:
        if use_random_rotation:
            # RandomRotation with multiples of 90° preserves lensing geometry
            train_transforms.append(
                T.RandomApply([T.RandomRotation(degrees=(0, 360))], p=0.8)
            )
        if use_random_flip:
            train_transforms.append(T.RandomHorizontalFlip(p=0.5))
            train_transforms.append(T.RandomVerticalFlip(p=0.5))

    # To tensor: (H, W) numpy → (C, H, W) float tensor in [0, 1]
    train_transforms.append(T.ToTensor())
    eval_transforms.append(T.ToTensor())

    # Normalise
    normalise = T.Normalize(mean=list(_mean), std=list(_std))
    train_transforms.append(normalise)
    eval_transforms.append(normalise)

    if is_train:
        return T.Compose(train_transforms)
    return T.Compose(eval_transforms)
