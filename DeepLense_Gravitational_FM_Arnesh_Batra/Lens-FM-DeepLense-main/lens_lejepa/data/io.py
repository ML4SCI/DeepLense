"""Reading DeepLense ``.npy`` files and turning them into model-ready tensors.

The DeepLense releases store images in a few slightly different ways:

* a plain ``float`` array of shape ``[H, W]`` (or ``[1, H, W]``),
* an object array that wraps the image (``array(image, dtype=object)``),
* for axion images in Model II/III, an object array ``[image, mass]`` where
  ``mass`` is the simulated axion particle mass in eV.

Everything in this module is pure NumPy/PyTorch and has no dataset state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize


def _unwrap_object_array(array: np.ndarray, path: Path) -> tuple[np.ndarray, list[float]]:
    """Split an object array into its single image payload and any scalars."""
    values = array.reshape(-1)
    images = [np.asarray(value) for value in values if np.asarray(value).ndim >= 2]
    scalars = [float(value) for value in values if np.asarray(value).ndim == 0]
    if len(images) != 1:
        raise ValueError(f"Expected exactly one image inside {path}, found {len(images)}.")
    return images[0], scalars


def load_lens_image(path: str | Path) -> np.ndarray:
    """Return the image stored in a DeepLense file as ``float32 [C, H, W]``."""
    path = Path(path)
    array = np.load(path, allow_pickle=True)
    while array.dtype == object:
        array, _ = _unwrap_object_array(array, path)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return array[None]
    if array.ndim == 3 and array.shape[0] in (1, 3):
        return array
    if array.ndim == 3 and array.shape[-1] in (1, 3):
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"Unsupported image shape {array.shape} in {path}.")


def load_axion_image_and_log_mass(path: str | Path) -> tuple[np.ndarray, float]:
    """Return ``(image [C, H, W], log10(mass / eV))`` for an axion file."""
    path = Path(path)
    array = np.load(path, allow_pickle=True)
    if array.dtype != object:
        raise ValueError(f"{path} has no mass metadata (expected an object array).")
    image, scalars = _unwrap_object_array(array, path)
    if len(scalars) != 1:
        raise ValueError(f"Expected one scalar mass in {path}, found {len(scalars)}.")
    mass = scalars[0]
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError(f"Axion mass must be finite and positive in {path}, got {mass!r}.")
    image = np.asarray(image, dtype=np.float32)
    return (image[None] if image.ndim == 2 else image), float(np.log10(mass))


def shift_image(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate ``[C, H, W]`` by whole pixels; uncovered pixels get the image minimum."""
    out = np.full_like(image, image.min())
    h, w = image.shape[-2:]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[..., yd, xd] = image[..., ys, xs]
    return out


def minmax_normalise(image: np.ndarray) -> np.ndarray:
    """Average channels to one and rescale each image to ``[0, 1]``.

    Non-finite pixels are replaced by the finite minimum/maximum so a single
    bad pixel cannot poison an entire batch.
    """
    if image.shape[0] > 1:
        image = image.mean(axis=0, keepdims=True)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    low, high = float(image[finite].min()), float(image[finite].max())
    image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
    return ((image - low) / max(high - low, 1e-8)).astype(np.float32)


def standardise(images: torch.Tensor) -> torch.Tensor:
    """Per-image zero mean / unit variance over the spatial dimensions."""
    mean = images.mean(dim=(-2, -1), keepdim=True)
    std = images.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
    return (images - mean) / std


def to_model_tensor(image: np.ndarray, size: int, standardize: bool = True) -> torch.Tensor:
    """Min-max normalise, resize to ``size x size`` and optionally standardise.

    This is the preprocessing used for pretraining (with ``standardize=False``,
    because the SSL views are perturbed first and standardised afterwards) and
    for classification and regression (``standardize=True``).
    """
    tensor = torch.from_numpy(minmax_normalise(image))
    if tensor.shape[-2:] != (size, size):
        tensor = resize(tensor, [size, size], antialias=True)
    return standardise(tensor) if standardize else tensor


def bicubic(images: torch.Tensor, size: int, antialias: bool = False) -> torch.Tensor:
    """Bicubic resize of a ``[C, H, W]`` or ``[B, C, H, W]`` tensor."""
    batched = images if images.ndim == 4 else images.unsqueeze(0)
    out = F.interpolate(batched, size=(size, size), mode="bicubic", align_corners=False, antialias=antialias)
    return out if images.ndim == 4 else out.squeeze(0)


def distortion_map(image: np.ndarray, size: int) -> torch.Tensor:
    """Label-free lensing distortion map used by Lensiformer and LensPINN.

    ``|tanh(d²/dxdy [log(I_max / I)]²)|`` of the raw image (shifted to be
    non-negative), resized to ``size x size``. Values lie in ``[0, 1]``.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image.mean(axis=0)
    finite = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    shifted = finite - float(finite.min())
    eps = 1e-8
    log_ratio = np.square(np.log((float(shifted.max()) + eps) / (shifted + eps)))
    curvature = np.gradient(np.gradient(log_ratio, axis=0), axis=1)
    result = np.nan_to_num(np.abs(np.tanh(curvature)), nan=0.0, posinf=1.0, neginf=0.0)
    tensor = torch.from_numpy(result.astype(np.float32)).unsqueeze(0)
    if tensor.shape[-2:] != (size, size):
        tensor = resize(tensor, [size, size], antialias=True)
    return tensor
