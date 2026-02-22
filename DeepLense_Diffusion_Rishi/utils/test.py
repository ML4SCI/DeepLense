"""
utils/test.py — Refactored for DeepLense Diffusion Dataset Preprocessing

Fixes over original:
- Multiple dataset paths via CLI (--data_dirs) instead of hardcoded root_dir
- Graceful handling of missing directories and corrupted .npy files
- Modular transform pipeline (crop size, normalize toggle via CLI)
- Automatic creation of output directory
- Configurable sample index and random seed
- Proper main() structure
"""

import os
import argparse
import logging

import torch
import numpy as np
import torchvision.transforms as Transforms
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Transform builder ────────────────────────────────────────────────────────
def build_transforms(crop_size: int = 64, normalize: bool = False) -> Transforms.Compose:
    """
    Build a modular transform pipeline.

    Args:
        crop_size: Size for CenterCrop. Set to 0 to skip.
        normalize: Whether to apply Normalize with dataset statistics.

    Returns:
        A torchvision Compose transform.
    """
    pipeline = []

    if crop_size > 0:
        pipeline.append(Transforms.CenterCrop(crop_size))

    if normalize:
        pipeline.append(
            Transforms.Normalize(
                mean=[0.06814773380756378, 0.21582692861557007, 0.4182431399822235],
                std=[0.16798585653305054, 0.5532506108283997, 1.1966736316680908],
            )
        )

    return Transforms.Compose(pipeline)


# ── File collection ──────────────────────────────────────────────────────────
def collect_npy_files(data_dirs: list) -> list:
    """
    Collect .npy file paths from multiple directories.

    Skips directories that don't exist with a warning instead of crashing.

    Args:
        data_dirs: List of dataset root directories.

    Returns:
        Sorted combined list of .npy file paths.
    """
    all_files = []

    for root_dir in data_dirs:
        if not os.path.isdir(root_dir):
            logger.warning("Directory not found, skipping: %s", root_dir)
            continue

        files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ])

        if not files:
            logger.warning("No .npy files found in: %s", root_dir)
        else:
            logger.info("Found %d .npy files in: %s", len(files), root_dir)

        all_files.extend(files)

    return all_files


# ── Safe loader ──────────────────────────────────────────────────────────────
def safe_load_npy(path: str):
    """
    Load a .npy file with error handling.

    Args:
        path: Path to the .npy file.

    Returns:
        Loaded numpy array, or None on failure.
    """
    try:
        return np.load(path)
    except FileNotFoundError:
        logger.warning("File not found: %s", path)
    except ValueError as e:
        logger.warning("Corrupted .npy file %s: %s", path, e)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
    return None


# ── Main processing ──────────────────────────────────────────────────────────
def process(args) -> None:
    # Collect files from all directories
    all_files = collect_npy_files(args.data_dirs)

    if not all_files:
        logger.error("No valid .npy files found. Exiting.")
        return

    logger.info("Total files collected: %d", len(all_files))

    # Clamp index safely — no magic IndexError
    index = args.index
    if index >= len(all_files):
        logger.warning(
            "Index %d out of range (total: %d). Using last file.",
            index, len(all_files),
        )
        index = len(all_files) - 1

    data_file_path = all_files[index]
    logger.info("Loading sample [%d]: %s", index, data_file_path)

    # Load
    data = safe_load_npy(data_file_path)
    if data is None:
        logger.error("Could not load sample. Exiting.")
        return

    print("data.shape:", data.shape)

    # Normalize to [0, 1]  — same as original
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    print("min:", np.min(data))
    print("max:", np.max(data))

    # Convert to torch tensor — same as original (npy loader returns ndarray)
    data_torch = torch.from_numpy(data)

    # Apply modular transforms
    transforms = build_transforms(crop_size=args.crop_size, normalize=args.normalize)
    data_torch = transforms(data_torch)

    # Back to numpy for visualization — same as original
    data_viz = data_torch.permute(1, 2, 0).to("cpu").numpy()

    # Save — with automatic directory creation
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_tag = os.path.basename(os.path.dirname(data_file_path))
    out_path = os.path.join(args.output_dir, f"ddpm_ssl_actual_{dataset_tag}_{index}.jpg")

    plt.imshow(data_viz)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved -> %s", out_path)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepLense Diffusion — Dataset Preprocessing Utility"
    )
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=[
            "../Data/cdm_regress_multi_param_model_ii/cdm_regress_multi_param",
            "../Data/npy_lenses-20240731T044737Z-001/npy_lenses",
            "../Data/real_lenses_dataset/lenses",
        ],
        help="One or more dataset directories containing .npy files.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=50,
        help="Index of the sample to visualize from the combined file list.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=64,
        help="CenterCrop size. Set to 0 to disable cropping.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply dataset-specific Normalize transform.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save output visualizations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("data_dirs : %s", args.data_dirs)
    logger.info("index=%d | crop_size=%d | normalize=%s | seed=%d",
                args.index, args.crop_size, args.normalize, args.seed)

    process(args)


if __name__ == "__main__":
    main()