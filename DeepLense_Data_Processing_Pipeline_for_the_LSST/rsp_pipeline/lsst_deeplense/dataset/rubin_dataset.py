"""
rubin_dataset.py
----------------
PyTorch Dataset that retrieves LSST/Rubin images on-the-fly (or from a
pre-fetched cache) and presents them in the same interface as DeepLense's
``LensDataset``, making it a drop-in replacement for the existing
classification / super-resolution / lens-finding training loops.

Two modes of operation
----------------------
1.  **Online mode** – a ``pandas.DataFrame`` of sky positions (ra, dec) is
    supplied together with a ``RubinSIAClient``.  Images are retrieved
    lazily on ``__getitem__`` and optionally cached to disk.

2.  **Offline mode** – a directory of pre-downloaded ``.npy`` files
    (one per sky position) is provided.  Useful for reproducibility and
    when network access to the RSP is not available during training.

Example (online)
----------------
>>> from lsst_deeplense import RubinSIAClient
>>> from lsst_deeplense.dataset import RubinLensDataset
>>> from lsst_deeplense.preprocessing import build_deeplense_transforms
>>>
>>> sia = RubinSIAClient()
>>> candidates_df = pd.read_csv("candidates.csv")  # must have 'ra', 'dec'
>>> dataset = RubinLensDataset(
...     candidates=candidates_df,
...     sia_client=sia,
...     transform=build_deeplense_transforms(is_train=True),
...     cache_dir="./cache/rubin_cutouts",
... )
>>> loader = torch.utils.data.DataLoader(dataset, batch_size=32)

Example (offline)
-----------------
>>> dataset = RubinLensDataset.from_npy_dir(
...     npy_dir="./cache/rubin_cutouts",
...     transform=build_deeplense_transforms(is_train=False),
... )
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    # Allow module import without torch for inspection / testing
    Dataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False


class RubinLensDataset(Dataset):
    """
    PyTorch Dataset for LSST/Rubin gravitational lensing images.

    Compatible with the DeepLense ``LensDataset`` interface so that it can
    be passed directly to existing classification and super-resolution
    training scripts.

    Parameters
    ----------
    candidates : pd.DataFrame
        Must contain columns ``'ra'`` and ``'dec'`` (degrees, ICRS).
        An optional ``'label'`` column is used when available (int class index).
    sia_client : RubinSIAClient, optional
        Image retrieval client.  Required in online mode.
    transform : callable, optional
        torchvision-compatible transform applied to each image array before
        it is returned.  Build one with ``build_deeplense_transforms()``.
    normaliser : Normaliser, optional
        Pre-transform pixel normalisation (default: asinh stretch).
    cache_dir : str or Path, optional
        Directory to cache downloaded ``.npy`` cutouts.  If a cached file
        exists it will be loaded instead of re-fetching from the RSP.
    bands : list[str]
        Photometric bands to retrieve (default ``['g', 'r', 'i']``).
    size_arcsec : float
        Cutout half-width in arcseconds (default 10 arcsec).
    output_size : int
        Target spatial dimension after resize (default 64, matching
        DeepLense Model I/II/III).
    """

    def __init__(
        self,
        candidates: pd.DataFrame,
        sia_client=None,
        transform: Optional[Callable] = None,
        normaliser=None,
        cache_dir: Optional[Union[str, Path]] = None,
        bands: List[str] = None,
        size_arcsec: float = 10.0,
        output_size: int = 64,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for RubinLensDataset.")

        if "ra" not in candidates.columns or "dec" not in candidates.columns:
            raise ValueError("candidates DataFrame must contain 'ra' and 'dec' columns.")

        self.candidates = candidates.reset_index(drop=True)
        self.sia_client = sia_client
        self.transform = transform
        self.normaliser = normaliser
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.bands = bands or ["g", "r", "i"]
        self.size_arcsec = size_arcsec
        self.output_size = output_size

        # Lazy import to keep module importable without these deps
        from ..preprocessing.cutout import CutoutExtractor
        from ..preprocessing.normalise import Normaliser

        self._extractor = CutoutExtractor(output_size=output_size)
        if self.normaliser is None:
            self.normaliser = Normaliser(strategy="asinh")

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cutout cache directory: %s", self.cache_dir)

        self._has_labels = "label" in self.candidates.columns

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int):
        row = self.candidates.iloc[idx]
        ra, dec = float(row["ra"]), float(row["dec"])

        # 1. Load or retrieve the image array
        array = self._load_or_fetch(ra, dec)  # (C, H, W) float32

        # 2. Apply normalisation per band
        normalised = np.stack(
            [self.normaliser(array[c]) for c in range(array.shape[0])],
            axis=0,
        )  # (C, H, W)

        # 3. Convert to (H, W) or (H, W, C) for torchvision ToTensor
        if normalised.shape[0] == 1:
            img_hwc = normalised[0]  # (H, W)
        else:
            img_hwc = normalised.transpose(1, 2, 0)  # (H, W, C)

        # 4. Apply transform (ToTensor + augmentations)
        if self.transform is not None:
            tensor = self.transform(img_hwc)
        else:
            tensor = torch.from_numpy(normalised)

        if self._has_labels:
            label = int(row["label"])
            return tensor, label
        return tensor

    # ------------------------------------------------------------------
    # Class methods for alternate construction
    # ------------------------------------------------------------------

    @classmethod
    def from_npy_dir(
        cls,
        npy_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        normaliser=None,
        label_csv: Optional[Union[str, Path]] = None,
    ) -> "RubinLensDataset":
        """
        Build a dataset from a directory of pre-downloaded ``.npy`` files.

        Each file should be named ``{ra:.6f}_{dec:.6f}.npy`` (matching the
        naming convention used by ``_cache_path``).

        Parameters
        ----------
        npy_dir : str or Path
            Directory containing ``.npy`` cutout files.
        transform : callable, optional
        normaliser : Normaliser, optional
        label_csv : str or Path, optional
            CSV with columns ``ra``, ``dec``, ``label``.

        Returns
        -------
        RubinLensDataset
        """
        npy_dir = Path(npy_dir)
        npy_files = sorted(npy_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {npy_dir}")

        records = []
        for fp in npy_files:
            stem = fp.stem
            parts = stem.split("_")
            try:
                ra = float(parts[0])
                dec = float(parts[1])
            except (IndexError, ValueError):
                logger.warning("Skipping unrecognised file name: %s", fp.name)
                continue
            records.append({"ra": ra, "dec": dec, "_npy_path": str(fp)})

        df = pd.DataFrame(records)

        if label_csv is not None:
            label_df = pd.read_csv(label_csv)
            df = df.merge(label_df[["ra", "dec", "label"]], on=["ra", "dec"], how="left")

        inst = cls.__new__(cls)
        inst.candidates = df.reset_index(drop=True)
        inst.sia_client = None
        inst.transform = transform
        inst.bands = ["g", "r", "i"]
        inst.size_arcsec = 10.0
        inst.output_size = 64
        inst.cache_dir = npy_dir

        from ..preprocessing.cutout import CutoutExtractor
        from ..preprocessing.normalise import Normaliser

        inst._extractor = CutoutExtractor(output_size=inst.output_size)
        inst.normaliser = normaliser or Normaliser(strategy="asinh")
        inst._has_labels = "label" in inst.candidates.columns
        return inst

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_fetch(self, ra: float, dec: float) -> np.ndarray:
        """Return a (C, H, W) float32 array, using cache if available."""
        cache_path = self._cache_path(ra, dec) if self.cache_dir else None

        # Try cache first
        if cache_path and cache_path.exists():
            return np.load(str(cache_path))

        # Check if preloaded npy path stored in row
        row_mask = (
            (np.isclose(self.candidates["ra"], ra, atol=1e-9))
            & (np.isclose(self.candidates["dec"], dec, atol=1e-9))
        )
        row = self.candidates[row_mask]
        if not row.empty and "_npy_path" in row.columns:
            npy_path = row.iloc[0]["_npy_path"]
            if npy_path and os.path.exists(npy_path):
                return np.load(npy_path)

        # Fetch from RSP
        if self.sia_client is None:
            raise RuntimeError(
                f"No cached file found for (ra={ra}, dec={dec}) and no "
                "sia_client configured.  Either provide a cache_dir with "
                "pre-fetched data or supply a RubinSIAClient."
            )

        band_arrays = self.sia_client.fetch_cutouts(
            ra=ra, dec=dec, size_arcsec=self.size_arcsec, bands=self.bands
        )
        if not band_arrays:
            raise RuntimeError(
                f"SIA returned no images for (ra={ra}, dec={dec})."
            )

        # Stack bands that are available
        available = [b for b in self.bands if b in band_arrays]
        stacked = self._extractor.stack_bands(
            {b: band_arrays[b] for b in available},
            band_order=tuple(available),
        )  # (C, H, W)

        # Cache to disk
        if cache_path is not None:
            np.save(str(cache_path), stacked)
            logger.debug("Cached cutout → %s", cache_path)

        return stacked

    def _cache_path(self, ra: float, dec: float) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        key = f"{ra:.6f}_{dec:.6f}"
        return self.cache_dir / f"{key}.npy"

    # ------------------------------------------------------------------
    # Convenience: statistics
    # ------------------------------------------------------------------

    def compute_dataset_stats(
        self, max_samples: int = 500
    ) -> Dict[str, float]:
        """
        Estimate per-channel mean and std over a random subset of the dataset.

        Useful for calibrating the ``Normaliser`` parameters.

        Returns
        -------
        dict with keys 'mean' and 'std' (averaged over channels)
        """
        n = min(max_samples, len(self))
        indices = np.random.choice(len(self), n, replace=False)

        sums, sums_sq, count = 0.0, 0.0, 0
        for idx in indices:
            sample = self[int(idx)]
            arr = sample[0].numpy() if isinstance(sample, tuple) else sample.numpy()
            sums += arr.mean()
            sums_sq += (arr**2).mean()
            count += 1

        mean = sums / count
        std = np.sqrt(sums_sq / count - mean**2)
        return {"mean": float(mean), "std": float(std)}
