"""
DeepLense Dataset Utilities
============================
Robust loader for DeepLense .npy files which exist in
multiple formats depending on the simulation pipeline.

Supported formats:
    1. Plain float array    : shape (64, 64)
    2. Object array of rows : shape (64,) of 64-element arrays
    3. Flat array + label   : shape (4096+1,) object array

Usage:
    from dataset_utils import load_npy_image, LensingDataset
    img = load_npy_image('path/to/image.npy', target_size=64)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image as PILImage


def load_npy_image(path: str, target_size: int = 64) -> torch.Tensor:
    """
    Robustly load a DeepLense .npy image file regardless of format.

    Args:
        path        : Path to the .npy file
        target_size : Output image size (default 64x64)

    Returns:
        torch.Tensor of shape (1, target_size, target_size)
        normalized to [0, 1]
    """
    raw = np.load(path, allow_pickle=True)

    # Format 1: Plain float array (64, 64)
    if raw.dtype != object:
        img = raw.astype(np.float32).squeeze()

    # Format 2 & 3: Object array of rows
    else:
        rows = [np.array(r, dtype=np.float32).flatten() for r in raw]
        # Take longest row = the image data (4096 = 64x64)
        img_row = max(rows, key=len)
        side = int(np.sqrt(len(img_row)))
        img = img_row.reshape(side, side)

    # Handle extra dimensions
    if img.ndim > 2:
        img = img[0] if img.shape[0] <= 4 else img[:, :, 0]

    # Resize if needed
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = np.array(
            PILImage.fromarray(img).resize(
                (target_size, target_size), PILImage.BILINEAR
            ),
            dtype=np.float32
        )

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return torch.from_numpy(img).unsqueeze(0).float()


class LensingDataset(Dataset):
    """
    PyTorch Dataset for DeepLense gravitational lensing images.

    Folder structure expected:
        root_dir/
            no_sub/   (no substructure)
            cdm/      (cold dark matter substructure)
            axion/    (axion-like particle substructure)

    Args:
        root_dir   : Path to dataset root folder
        only_class : Load only one class (e.g. 'no_sub')
        augment    : Apply random flips and rotations
        val_split  : Fraction of data for validation (default 0.2)
        is_train   : If True load train split, else val split
        target_size: Image size (default 64)
    """

    CLASS_NAMES = ['No Substructure', 'CDM', 'Axion']
    FOLDERS     = ['no_sub', 'cdm', 'axion']

    def __init__(
        self,
        root_dir   : str,
        only_class : str  = None,
        augment    : bool = False,
        val_split  : float = 0.2,
        is_train   : bool = True,
        target_size: int  = 64
    ):
        self.augment     = augment
        self.target_size = target_size
        all_samples      = []
        targets = [only_class] if only_class else self.FOLDERS

        for label, folder in enumerate(self.FOLDERS):
            if folder not in targets:
                continue
            files = sorted([
                f for f in glob.glob(
                    os.path.join(root_dir, folder, '*.npy'))
                if not os.path.basename(f).startswith('._')
            ])
            all_samples += [(f, label) for f in files]

        # Reproducible train/val split
        import random
        random.seed(42)
        random.shuffle(all_samples)
        n_val = int(len(all_samples) * val_split)
        self.samples = (
            all_samples[n_val:] if is_train else all_samples[:n_val]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = load_npy_image(path, target_size=self.target_size)
        if self.augment:
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, dims=[1, 2])
        return img, label
```

Commit message:
```
Add dataset_utils.py with robust npy loader (fixes #[issue_number])
```

---

**Step 3 — Open PR**

Go to your fork → Click **Contribute** → **Open Pull Request**

Title:
```
Add dataset_utils.py with robust loader for inconsistent .npy formats
```

Description:
```
Fixes #[issue_number]

## Problem
DeepLense .npy files exist in 3 different formats causing 
ValueError for new contributors loading the dataset.

## Solution
Added dataset_utils.py with:
- load_npy_image() — handles all 3 .npy formats automatically
- LensingDataset — clean PyTorch Dataset class with train/val split

## Formats handled
1. Plain (64,64) float arrays
2. Object arrays of row vectors
3. Flat (4096,) arrays with label row appended

## Testing
Tested on all 3 classes (no_sub, cdm, axion) with 
~90,000 files on Google Colab T4 GPU ✅

## Example usage
from dataset_utils import load_npy_image, LensingDataset
img = load_npy_image('sample.npy', target_size=64)
