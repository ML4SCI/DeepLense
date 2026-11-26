"""
get_data_diff.py  (patched)
---------------------------
DataLoader for the Super Resolution diffusion pipeline.

Fixes applied (Issue #135):
  • Guard against IndexError when dataset size < requested index / batch size.
  • Clear FileNotFoundError message when HR/LR directories are missing.
  • Added __len__ and __repr__ for easier debugging.
  • Graceful handling of mismatched HR ↔ LR counts.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SuperResDataset(Dataset):
    """
    Loads matched HR / LR pairs from two directories.

    Expected structure:
        root/
            HR/  ← high-resolution .npy files
            LR/  ← low-resolution  .npy files

    File names in HR/ and LR/ must match exactly.
    """

    def __init__(self, root: str, transform=None):
        self.root      = Path(root)
        self.transform = transform

        hr_dir = self.root / "HR"
        lr_dir = self.root / "LR"

        # ── Validate directories ─────────────────────────────────────────────
        for d, label in [(hr_dir, "HR"), (lr_dir, "LR")]:
            if not d.exists():
                raise FileNotFoundError(
                    f"'{label}' directory not found: {d}\n"
                    "Run generate_dataset.py first to create HR/LR pairs.\n"
                    "Example:\n"
                    "  python generate_dataset.py --input_dir raw/ "
                    "--output_dir dataset/ --scale_factor 4"
                )

        hr_files = sorted(hr_dir.glob("*.npy"))
        lr_files = sorted(lr_dir.glob("*.npy"))

        # ── Guard: empty directories ─────────────────────────────────────────
        if len(hr_files) == 0:
            raise RuntimeError(
                f"No .npy files found in '{hr_dir}'. "
                "Please run generate_dataset.py to populate the dataset."
            )

        # ── Guard: mismatched counts ─────────────────────────────────────────
        hr_stems = {f.stem for f in hr_files}
        lr_stems = {f.stem for f in lr_files}
        common   = hr_stems & lr_stems

        if len(common) == 0:
            raise RuntimeError(
                "No matching file names found between HR and LR directories. "
                "Ensure generate_dataset.py produced paired files."
            )

        if len(common) < len(hr_files) or len(common) < len(lr_files):
            print(
                f"[WARNING] HR has {len(hr_files)} files, "
                f"LR has {len(lr_files)} files. "
                f"Using {len(common)} matched pairs."
            )

        self.hr_files = sorted([hr_dir / f"{s}.npy" for s in common])
        self.lr_files = sorted([lr_dir / f"{s}.npy" for s in common])

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self):
        return len(self.hr_files)

    def __repr__(self):
        return (
            f"SuperResDataset(root='{self.root}', "
            f"n_pairs={len(self)})"
        )

    def __getitem__(self, idx: int):
        # Fix for original IndexError: clamp idx to valid range
        # (This can happen with certain samplers or off-by-one bugs.)
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}. "
                "Check that your DataLoader batch_size / num_workers settings "
                "are compatible with the dataset size."
            )

        hr = np.load(self.hr_files[idx]).astype(np.float32)
        lr = np.load(self.lr_files[idx]).astype(np.float32)

        # Ensure channel dimension: (H, W) → (1, H, W)
        if hr.ndim == 2:
            hr = hr[np.newaxis]
        if lr.ndim == 2:
            lr = lr[np.newaxis]

        hr_tensor = torch.from_numpy(hr)
        lr_tensor = torch.from_numpy(lr)

        if self.transform:
            hr_tensor = self.transform(hr_tensor)
            lr_tensor = self.transform(lr_tensor)

        return lr_tensor, hr_tensor   # (LR, HR) — input first, target second


# ── Factory functions ────────────────────────────────────────────────────────

def get_dataloader(
    root: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    transform=None,
    drop_last: bool = False,
) -> DataLoader:
    """
    Returns a DataLoader for the SuperResDataset at `root`.

    Args:
        root        : Path to split directory (e.g. 'dataset/train').
        batch_size  : Mini-batch size.
        shuffle     : Shuffle samples each epoch.
        num_workers : Worker processes for data loading.
        transform   : Optional torchvision transform applied to both HR and LR.
        drop_last   : Drop the last incomplete batch.

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = SuperResDataset(root=root, transform=transform)

    # Guard: batch_size larger than dataset → clamp and warn
    if batch_size > len(dataset):
        print(
            f"[WARNING] batch_size ({batch_size}) > dataset size ({len(dataset)}). "
            f"Clamping batch_size to {len(dataset)}."
        )
        batch_size = len(dataset)
        drop_last  = False   # cannot drop_last when batch == full dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    print(f"DataLoader ready: {dataset}")
    return loader


# ── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "dataset/train"
    loader    = get_dataloader(data_root, batch_size=4)

    lr_batch, hr_batch = next(iter(loader))
    print(f"LR batch shape : {lr_batch.shape}")
    print(f"HR batch shape : {hr_batch.shape}")
    print("Smoke-test passed ✅")