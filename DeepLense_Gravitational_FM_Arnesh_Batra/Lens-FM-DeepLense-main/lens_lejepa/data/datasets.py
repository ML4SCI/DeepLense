"""PyTorch datasets for the three DeepLense tasks.

Expected directory layout (one folder per class, one ``.npy`` file per image)::

    datasets/Model_II/{axion,cdm,no_sub}/*.npy
    datasets/Model_II_test/{axion,cdm,no_sub}/*.npy

File lists are always sorted, so splits are identical on every machine.
Every dataset returns a dictionary, which keeps the training loop task-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import bicubic, distortion_map, shift_image, load_axion_image_and_log_mass, load_lens_image, minmax_normalise, to_model_tensor

CLASSES: tuple[str, ...] = ("axion", "cdm", "no_sub")


def list_class_files(root: str | Path, classes: tuple[str, ...] = CLASSES) -> tuple[list[Path], np.ndarray]:
    """Return sorted file paths and integer labels for a class-folder dataset."""
    root = Path(root)
    paths: list[Path] = []
    labels: list[int] = []
    for label, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        files = sorted(folder.glob("*.npy"))
        paths.extend(files)
        labels.extend([label] * len(files))
    if not paths:
        raise RuntimeError(f"No .npy files found under {root}")
    return paths, np.asarray(labels, dtype=np.int64)


class UnlabelledLensDataset(Dataset):
    """Images only, for self-supervised pretraining (labels are never read).

    Images are min-max normalised and resized but *not* standardised: the SSL
    objective perturbs each view first and standardises afterwards.
    """

    def __init__(self, root: str | Path, image_size: int, max_samples: int | None = None, seed: int = 0) -> None:
        self.paths, _ = list_class_files(root)
        if max_samples is not None and max_samples < len(self.paths):
            keep = np.random.RandomState(seed).choice(len(self.paths), max_samples, replace=False)
            self.paths = [self.paths[i] for i in sorted(keep)]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = load_lens_image(self.paths[index])
        return {"image": to_model_tensor(image, self.image_size, standardize=False)}


class LensClassificationDataset(Dataset):
    """Three-class substructure classification (axion / CDM / no substructure)."""

    def __init__(self, root: str | Path, image_size: int, cache: bool = False, distortion: bool = False, jitter: int = 0) -> None:
        self.paths, self.labels = list_class_files(root)
        self.image_size = image_size
        self.distortion = distortion
        self.jitter = jitter  # random +-jitter native-pixel shift per item (training-time robustness only)
        self._cache = [load_lens_image(path) for path in self.paths] if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self._cache[index] if self._cache is not None else load_lens_image(self.paths[index])
        if self.jitter > 0:
            dy, dx = torch.randint(-self.jitter, self.jitter + 1, (2,)).tolist()
            image = shift_image(image, dy, dx)
        item = {
            "image": to_model_tensor(image, self.image_size, standardize=True),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }
        if self.distortion:
            item["distortion"] = distortion_map(image, self.image_size)
        return item


class AxionMassDataset(Dataset):
    """Axion images only, with target ``log10(m_axion / eV)``."""

    def __init__(self, root: str | Path, image_size: int, cache: bool = False, distortion: bool = False) -> None:
        folder = Path(root) / "axion"
        self.paths = sorted(folder.glob("*.npy"))
        if not self.paths:
            raise RuntimeError(f"No axion .npy files found under {folder}")
        self.image_size = image_size
        self.distortion = distortion
        records = [load_axion_image_and_log_mass(path) for path in self.paths]
        self.targets = np.asarray([target for _, target in records], dtype=np.float32)
        self._cache = [image for image, _ in records] if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._cache is not None:
            image = self._cache[index]
        else:
            image, _ = load_axion_image_and_log_mass(self.paths[index])
        item = {
            "image": to_model_tensor(image, self.image_size, standardize=True),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }
        if self.distortion:
            item["distortion"] = distortion_map(image, self.image_size)
        return item


class SyntheticSuperResolutionDataset(Dataset):
    """Synthetic 2x super-resolution pairs built from native 64x64 images.

    * ``high_res``: the native image, min-max scaled to ``[0, 1]`` (the target).
    * ``low_res``: antialiased bicubic downsample to 32x32 (input for CNN baselines).
    * ``encoder_input``: ``low_res`` bicubic-upsampled to the encoder resolution.

    This is image reconstruction, not physical resolution recovery: the
    DeepLense release has no paired low/high-resolution simulation.
    """

    def __init__(self, root: str | Path, image_size: int, hr_size: int = 64, cache: bool = False) -> None:
        self.paths, self.labels = list_class_files(root)
        self.image_size = image_size
        self.hr_size = hr_size
        self._cache = [load_lens_image(path) for path in self.paths] if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self._cache[index] if self._cache is not None else load_lens_image(self.paths[index])
        high_res = torch.from_numpy(minmax_normalise(image))
        if high_res.shape[-1] != self.hr_size:
            high_res = bicubic(high_res, self.hr_size, antialias=True)
        low_res = bicubic(high_res, self.hr_size // 2, antialias=True)
        return {
            "encoder_input": bicubic(low_res, self.image_size),
            "low_res": low_res,
            "high_res": high_res,
        }
