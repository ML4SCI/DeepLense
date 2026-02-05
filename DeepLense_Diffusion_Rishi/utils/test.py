"""
Usage:
    python test.py --index 100 --output_dir plots_real --filename lens_100.jpg
"""
import argparse
import os

import numpy as np
import torch
import torchvision.transforms as Transforms
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from dataset.transforms import get_transforms

# Default paths to check for data
# Determine the absolute path to the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default paths to check for data (relative to script location)
DEFAULT_DATA_PATHS = [
    os.path.join(SCRIPT_DIR, "../Data/cdm_regress_multi_param_model_ii/cdm_regress_multi_param/"),
    os.path.join(SCRIPT_DIR, "../Data/npy_lenses-20240731T044737Z-001/npy_lenses/"),
    os.path.join(SCRIPT_DIR, "../Data/real_lenses_dataset/lenses"),
]

# def get_transforms() -> Transforms.Compose:  <-- Removing local definition
# Using imported get_transforms instead



def find_valid_data_dir(paths: List[str]) -> Optional[str]:
    """Iterates through possible paths and returns the first valid directory."""
    for path in paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    return None


def load_file_list(data_dir: str) -> List[str]:
    """Returns a sorted list of .npy files in the directory."""
    try:
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
        return files
    except OSError as e:
        print(f"Error accessing directory {data_dir}: {e}")
        return []


def load_data(file_path: str) -> Optional[np.ndarray]:
    """Loads numpy data from a file."""
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalizes data to [0, 1] range."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val > 0:
        return (data - min_val) / (max_val - min_val)
    else:
        print("Warning: Data is constant. Skipping normalization.")
        return data


def process_data(data: np.ndarray, transforms: Transforms.Compose) -> torch.Tensor:
    """Converts to tensor and applies transforms."""
    data_torch = torch.from_numpy(data)
    # Ensure float type for potential transforms
    if data_torch.dtype not in [torch.float32, torch.float64]:
        data_torch = data_torch.float()
    return transforms(data_torch)


def save_plot(data_torch: torch.Tensor, output_dir: str, filename: str) -> bool:
    """Saves a visualization of the data."""
    try:
        # Permute (C, H, W) -> (H, W, C) for plotting if 3D
        if data_torch.ndim == 3:
            data_to_plot = data_torch.permute(1, 2, 0).to("cpu").numpy()
        else:
            data_to_plot = data_torch.to("cpu").numpy()

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)

        plt.figure()  # Create new figure to avoid state leak
        plt.imshow(data_to_plot)
        plt.axis("off")  # Optional: remove axes for clean image
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {save_path}")
        return True
    except Exception as e:
        print(f"Error saving plot: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test script for DeepLense Diffusion")
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=DEFAULT_DATA_PATHS,
        help="List of dataset directories",
    )
    parser.add_argument(
        "--index", type=int, default=50, help="Index of file to process"
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Output directory"
    )
    parser.add_argument(
        "--filename", type=str, default="ddpm_ssl_actual.jpg", help="Output filename"
    )

    args = parser.parse_args()

    data_dir = find_valid_data_dir(args.data_dirs)
    if not data_dir:
        print(f"Error: No valid data directory found in {args.data_dirs}")
        return

    print(f"Using data directory: {data_dir}")
    files = load_file_list(data_dir)
    if not files:
        print("No .npy files found.")
        return

    if args.index < 0 or args.index >= len(files):
        print(f"Error: Index {args.index} out of bounds ({len(files)} files).")
        return

    full_path = os.path.join(data_dir, files[args.index])
    print(f"Processing: {full_path}")

    data = load_data(full_path)
    if data is None:
        return

    print(f"Original Shape: {data.shape}")
    print(f"Range: [{np.min(data)}, {np.max(data)}]")

    data = normalize_data(data)

    try:
        data_torch = process_data(data, get_transforms())
    print(f"After transforms: {data_torch.shape}, "
          f"range: [{data_torch.min().item():.4f}, {data_torch.max().item():.4f}]")
    except Exception as e:
        print(f"Transformation failed: {e}")
        return

    save_plot(data_torch, args.output_dir, args.filename)


if __name__ == "__main__":
    main()
