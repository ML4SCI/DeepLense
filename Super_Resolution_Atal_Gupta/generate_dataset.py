"""
generate_dataset.py
-------------------
Generates High-Resolution (HR) and Low-Resolution (LR) image pairs
for the Super Resolution pipeline in the DeepLense project.

Usage:
    python generate_dataset.py --input_dir <path_to_raw_npy_files> \
                               --output_dir <path_to_save_dataset> \
                               --scale_factor 4 \
                               --split 0.8

Directory structure created:
    output_dir/
        train/
            HR/   <- high-resolution .npy files
            LR/   <- low-resolution  .npy files
        val/
            HR/
            LR/
"""

import os
import argparse
import numpy as np
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def downsample(image: np.ndarray, scale: int) -> np.ndarray:
    """
    Simple average-pooling downsample (no external deps required).
    Works on (H, W) or (C, H, W) arrays.
    """
    if image.ndim == 2:
        h, w = image.shape
        h_new, w_new = h // scale, w // scale
        lr = image[:h_new * scale, :w_new * scale]
        lr = lr.reshape(h_new, scale, w_new, scale).mean(axis=(1, 3))
    elif image.ndim == 3:
        c, h, w = image.shape
        h_new, w_new = h // scale, w // scale
        lr = image[:, :h_new * scale, :w_new * scale]
        lr = lr.reshape(c, h_new, scale, w_new, scale).mean(axis=(2, 4))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return lr.astype(image.dtype)


def collect_files(input_dir: Path) -> list:
    """Collect all .npy files recursively from input_dir."""
    files = sorted(input_dir.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No .npy files found in '{input_dir}'. "
            "Please download the DeepLense dataset first.\n"
            "Dataset link: https://drive.google.com/file/d/1gTBsn4N9MkbVwHlTZtJQI2-YsNnMHOHC"
        )
    return files


def make_dirs(output_dir: Path):
    for split in ("train", "val"):
        for res in ("HR", "LR"):
            (output_dir / split / res).mkdir(parents=True, exist_ok=True)


# ── main ─────────────────────────────────────────────────────────────────────

def generate(input_dir: str, output_dir: str, scale_factor: int, split: float):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    files = collect_files(input_path)
    print(f"Found {len(files)} .npy files in '{input_path}'.")

    # Shuffle deterministically
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(files))
    files   = [files[i] for i in indices]

    n_train = max(1, int(len(files) * split))
    splits  = {"train": files[:n_train], "val": files[n_train:]}

    # Edge-case: if val is empty, borrow one sample
    if not splits["val"]:
        splits["val"]   = [splits["train"][-1]]
        splits["train"] = splits["train"][:-1]

    make_dirs(output_path)

    for split_name, split_files in splits.items():
        print(f"\nGenerating {split_name} set ({len(split_files)} samples)…")
        for i, fpath in enumerate(split_files):
            hr = np.load(fpath)

            # Normalise to [0, 1] float32
            hr = hr.astype(np.float32)
            vmin, vmax = hr.min(), hr.max()
            if vmax - vmin > 0:
                hr = (hr - vmin) / (vmax - vmin)

            lr = downsample(hr, scale_factor)

            stem = fpath.stem
            np.save(output_path / split_name / "HR" / f"{stem}.npy", hr)
            np.save(output_path / split_name / "LR" / f"{stem}.npy", lr)

            if (i + 1) % 100 == 0 or (i + 1) == len(split_files):
                print(f"  [{i+1}/{len(split_files)}] processed {fpath.name}")

    # Summary
    for split_name in ("train", "val"):
        n = len(list((output_path / split_name / "HR").glob("*.npy")))
        print(f"\n{split_name}: {n} HR/LR pairs saved → {output_path / split_name}")

    print("\n✅ Dataset generation complete.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HR–LR pairs for the DeepLense Super Resolution pipeline."
    )
    parser.add_argument(
        "--input_dir",  required=True,
        help="Directory containing raw .npy lensing images."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where the HR/LR dataset will be saved."
    )
    parser.add_argument(
        "--scale_factor", type=int, default=4,
        help="Downscaling factor for LR images (default: 4)."
    )
    parser.add_argument(
        "--split", type=float, default=0.8,
        help="Train/val split ratio (default: 0.8)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.input_dir, args.output_dir, args.scale_factor, args.split)