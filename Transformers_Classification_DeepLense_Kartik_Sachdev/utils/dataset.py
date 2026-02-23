"""
dataset.py — Refactored for modularity, dynamic augmentation support,
and clean separation of concerns.

Changes from original:
- Removed duplicate imports
- Extracted hardcoded category logic into a configurable loader
- Added WrapperDataset for dynamic transform injection
- Unified download/extract logic via a single helper
- Removed hardcoded sample counts in dataeff variant
- Added LensDataset as the clean base class
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

import gdown
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

from config.data_config import DATASET
from utils.augmentation import TransformationsSLL
from utils.util import make_directories


# ─────────────────────────────────────────────────────────────────────────────
# Download & Extraction Helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(
    filename: str,
    url: str = "https://drive.google.com/uc?id=1m7QzSzXyE8u_QoYplN9dIe-X2pf1KXxt",
) -> None:
    """Downloads dataset from Google Drive.

    Args:
        filename: Output file path.
        url: Google Drive URL to dataset.
    """
    if not os.path.isfile(filename):
        try:
            gdown.download(url, filename, quiet=False)
        except Exception as e:
            print(f"Download failed: {e}")
    else:
        print("File already exists, skipping download.")


def extract_split_dataset(
    filename: str,
    destination_dir: str = "data",
    dataset_name: str = "Model_I",
    split: bool = False,
) -> None:
    """Extracts a .tar file and optionally splits into train/val (90:10).

    Args:
        filename: Path to tar file.
        destination_dir: Output directory.
        dataset_name: Name of the dataset (Model_I, Model_II, Model_III).
        split: Whether to split into train/val subsets.
    """
    if not split:
        print("Extracting folder...")
        os.system(f"tar xf {filename} --directory {destination_dir}")
        print("Extraction complete.")
    else:
        os.system(
            f"tar xf {filename} --directory {destination_dir} ; "
            f"mv {destination_dir}/{dataset_name} {destination_dir}/{dataset_name}_raw"
        )
        splitfolders.ratio(
            f"{destination_dir}/{dataset_name}_raw",
            output=f"{destination_dir}/{dataset_name}",
            seed=1337,
            ratio=(0.9, 0.1),
        )
        os.system(f"rm -r {destination_dir}/{dataset_name}_raw")


def _resolve_dataset_dir(
    destination_dir: str,
    dataset_name: str,
    mode: str,
    download: bool,
) -> str:
    """Resolves dataset directory, downloading if needed.

    This replaces the repeated download/check logic that was copy-pasted
    across DeepLenseDataset, DeepLenseDatasetSSL, etc.

    Args:
        destination_dir: Root data directory.
        dataset_name: Dataset name key (e.g. 'Model_I').
        mode: One of 'train', 'val', 'test'.
        download: Whether to download if missing.

    Returns:
        Path to the resolved dataset folder.
    """
    if mode == "test":
        filename = f"{destination_dir}/{dataset_name}_test.tgz"
        foldername = f"{destination_dir}/{dataset_name}_test"
    else:
        filename = f"{destination_dir}/{dataset_name}.tgz"
        foldername = f"{destination_dir}/{dataset_name}"

    url = DATASET[dataset_name][f"{mode}_url"]

    if download and not os.path.isdir(foldername):
        if not os.path.isfile(filename):
            download_dataset(filename, url=url)
        extract_split_dataset(filename, destination_dir)
    else:
        assert os.path.isdir(foldername), (
            f"Dataset not found at '{foldername}'. Set download=True to download it."
        )
        print(f"{dataset_name} dataset already exists.")

    return foldername


# ─────────────────────────────────────────────────────────────────────────────
# Normalize .npy Image
# ─────────────────────────────────────────────────────────────────────────────

def _load_npy_image(
    path: str,
    label: int,
    npy_index_map: Optional[Dict[int, int]] = None,
) -> np.ndarray:
    """Loads and normalizes a .npy lens image.

    Previously, label == 0 always triggered image[0] — this was hardcoded
    category-specific logic. Now it's driven by an optional npy_index_map
    dict that maps label → array index. If not provided, no slicing occurs.

    Args:
        path: Path to .npy file.
        label: Integer class label.
        npy_index_map: Optional dict mapping label to array index for slicing.
                       Example: {0: 0} means label-0 images are stored as image[0].

    Returns:
        Normalized 2D numpy array (H, W).
    """
    image = np.load(path, allow_pickle=True)

    if npy_index_map is not None and label in npy_index_map:
        image = image[npy_index_map[label]]

    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    return image


# ─────────────────────────────────────────────────────────────────────────────
# LensDataset — clean base dataset, no hardcoded logic
# ─────────────────────────────────────────────────────────────────────────────

class LensDataset(Dataset):
    """Base dataset for gravitational lens .npy images.

    Replaces DeepLenseDataset with:
    - No hardcoded category/label logic
    - Configurable npy_index_map for array slicing
    - Clean separation of loading and transforming
    """

    def __init__(
        self,
        destination_dir: str,
        dataset_name: str,
        mode: str,
        transform: Optional[Callable] = None,
        download: bool = False,
        channels: int = 1,
        npy_index_map: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            destination_dir: Root directory where dataset is stored.
            dataset_name: Name of dataset (e.g. 'Model_I').
            mode: 'train', 'val', or 'test'.
            transform: Albumentations or torchvision transform.
            download: Download dataset if not present.
            channels: Number of image channels (1 or 3).
            npy_index_map: Maps label index → array slice index for .npy files.
                           Pass {0: 0} to replicate old `if label == 0: image = image[0]`.
        """
        assert mode in ["train", "val", "test"]

        self.root_dir = _resolve_dataset_dir(
            destination_dir, dataset_name, mode, download
        )
        self.transform = transform
        self.channels = channels
        self.npy_index_map = npy_index_map  # replaces hardcoded label==0 logic

        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        self.imagefilename: List[str] = []
        self.labels: List[int] = []

        for cls in classes:
            for fname in os.listdir(os.path.join(self.root_dir, cls)):
                self.imagefilename.append(os.path.join(self.root_dir, cls, fname))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        path, label = self.imagefilename[index], self.labels[index]

        image = _load_npy_image(path, label, self.npy_index_map)
        image = np.expand_dims(image, axis=2)  # (H, W, 1)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"].float().clone().detach()

        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# WrapperDataset — dynamic transform injection
# ─────────────────────────────────────────────────────────────────────────────

class WrapperDataset(Dataset):
    """Wraps any Dataset and applies a dynamically swappable transform.

    This solves the core issue: previously transforms were baked into
    each dataset class. Now you can wrap a base dataset and change
    the transform at runtime — e.g. different augmentations for
    train vs val without reloading data.

    Example:
        >>> base = LensDataset(...)
        >>> train_ds = WrapperDataset(base, transform=train_transform)
        >>> train_ds.set_transform(stronger_augment)  # swap at runtime
    """

    def __init__(
        self,
        base_dataset: Dataset,
        transform: Optional[Callable] = None,
    ):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"].float().clone().detach()
        return image, label

    def set_transform(self, transform: Callable) -> None:
        """Dynamically update the transform without reloading data."""
        self.transform = transform


# ─────────────────────────────────────────────────────────────────────────────
# SSL Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LensDatasetSSL(LensDataset):
    """SSL variant of LensDataset — returns multiple augmented views.

    Replaces DeepLenseDatasetSSL. Uses LensDataset as base so download/load
    logic is not duplicated.
    """

    def __init__(
        self,
        destination_dir: str,
        dataset_name: str,
        mode: str,
        transforms: Optional[List[Callable]] = None,
        download: bool = False,
        channels: int = 1,
        npy_index_map: Optional[Dict[int, int]] = None,
    ):
        super().__init__(
            destination_dir=destination_dir,
            dataset_name=dataset_name,
            mode=mode,
            transform=None,  # transforms handled per-view below
            download=download,
            channels=channels,
            npy_index_map=npy_index_map,
        )
        self.transforms = transforms  # list of transforms, one per view

    def __getitem__(self, index: int):
        path, label = self.imagefilename[index], self.labels[index]

        image = _load_npy_image(path, label, self.npy_index_map)
        image = np.expand_dims(image, axis=2)

        ret = []
        if self.transforms is not None:
            for t in self.transforms:
                transformed = t(image=image)
                img_t = transformed["image"].float().clone().detach()
                ret.append(img_t)
        ret.append(label)
        return ret


class LensDatasetSSLRegression(LensDatasetSSL):
    """SSL regression variant — returns mass instead of class label."""

    def __getitem__(self, index: int):
        path, label = self.imagefilename[index], self.labels[index]

        data = np.load(path, allow_pickle=True)
        image = data[0]
        mass = np.float32(data[1])

        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        image = np.expand_dims(image, axis=2)

        ret = []
        if self.transforms is not None:
            for t in self.transforms:
                transformed = t(image=image)
                img_t = transformed["image"].float().clone().detach()
                ret.append(img_t)
        ret.append(mass)
        return ret


# ─────────────────────────────────────────────────────────────────────────────
# Custom Datasets (Image files, not .npy)
# ─────────────────────────────────────────────────────────────────────────────

class CustomDataset(Dataset):
    """Dataset for standard image files (PNG/JPG) organized in class folders."""

    def __init__(self, root_dir: str, mode: str, transform: Optional[Callable] = None):
        assert mode in ["train", "val", "test"]
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform

        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.imagefilename: List[str] = []
        self.labels: List[int] = []

        for cls in classes:
            for fname in os.listdir(os.path.join(self.root_dir, cls)):
                self.imagefilename.append(os.path.join(self.root_dir, cls, fname))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.open(self.imagefilename[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


class CustomDatasetSSL(Dataset):
    """SSL variant of CustomDataset — returns multiple augmented views."""

    def __init__(
        self,
        root_dir: str,
        mode: str,
        transforms: Optional[List[Callable]] = None,
    ):
        assert mode in ["train", "val", "test"]
        self.root_dir = os.path.join(root_dir, mode)
        self.transforms = transforms

        classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.imagefilename: List[str] = []
        self.labels: List[int] = []

        for cls in classes:
            for fname in os.listdir(os.path.join(self.root_dir, cls)):
                self.imagefilename.append(os.path.join(self.root_dir, cls, fname))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.open(self.imagefilename[index])
        ret = []
        if self.transforms is not None:
            for t in self.transforms:
                ret.append(t(image))
        ret.append(self.labels[index])
        return ret


# ─────────────────────────────────────────────────────────────────────────────
# Default Dataset Setup (SSL)
# ─────────────────────────────────────────────────────────────────────────────

class DefaultDatasetSetupSSL:
    """Convenience class for setting up SSL training with default config."""

    def __init__(
        self,
        dataset_name: str = "Model_II",
        image_size: int = 224,
        dir: Optional[str] = None,
    ) -> None:
        current_file = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file)

        self.data_dir = dir if dir is not None else os.path.join(parent_directory, "../data")
        make_directories([self.data_dir])

        self.setup(dataset_name=dataset_name)
        self.setup_transforms(image_size=image_size)

    def setup(self, dataset_name: str = "Model_II") -> None:
        self.cfg = {
            "dataset_name": dataset_name,
            "dataset": DATASET[dataset_name],
            "classes": DATASET[dataset_name]["classes"],
            "train_url": DATASET[dataset_name]["train_url"],
        }

    def setup_transforms(self, image_size: int) -> None:
        self.train_transforms = TransformationsSLL().get_transforms_multiple(
            final_size=image_size
        )

    def get_dataset(self, mode: str = "train") -> LensDatasetSSL:
        assert mode in ["train", "val", "test"]
        dataset = LensDatasetSSL(
            destination_dir=self.data_dir,
            dataset_name=self.cfg["dataset_name"],
            mode=mode,
            transforms=self.train_transforms,
            download=True,
            channels=1,
        )
        print(f"{mode} data: {len(dataset)} samples")
        return dataset

    def visualize_dataset(self, dataset: Dataset) -> None:
        visualize_samples_ssl(dataset, labels_map=self.cfg["classes"])


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────────────

def visualize_samples(
    dataset: Dataset,
    labels_map: Dict[int, str],
    fig_height: int = 15,
    fig_width: int = 15,
    num_cols: int = 5,
    cols_rows: int = 5,
) -> None:
    """Visualize random samples from a dataset."""
    figure = plt.figure(figsize=(fig_height, fig_width))
    cols, rows = num_cols, cols_rows
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"{labels_map[label]}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def visualize_samples_ssl(
    dataset: Dataset,
    labels_map: Dict[int, str],
    fig_height: int = 15,
    fig_width: int = 15,
    num_cols: int = 5,
    cols_rows: int = 5,
    num_rows_inner: int = 1,
    num_cols_inner: int = 2,
    regression: bool = False,
) -> None:
    """Visualize SSL samples (multiple views per sample)."""
    fig = plt.figure(figsize=(fig_height, fig_width))
    outer_subplot_index = 1

    for row in range(num_cols):
        for col in range(cols_rows):
            outer_subplot = fig.add_subplot(num_cols, cols_rows, outer_subplot_index)
            outer_subplot.set_xticklabels([])
            outer_subplot_index += 1

            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            batch = dataset[sample_idx]

            title = f"{batch[-1]:.4f}" if regression else f"{labels_map[batch[-1]]}"
            outer_subplot.set_title(title)
            img = batch[:-1]

            for inner_col in range(num_cols_inner):
                inner_subplot = outer_subplot.inset_axes([
                    inner_col / num_cols_inner, 0,
                    1 / num_cols_inner, 1,
                ])
                inner_subplot.imshow(img[inner_col].squeeze())
                plt.axis("off")

    fig.suptitle("Dataset")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Distributed Sampler Helpers
# ─────────────────────────────────────────────────────────────────────────────

class SubsetRandomSampler(torch.utils.data.Sampler):
    """Samples elements randomly from a given list of indices, without replacement."""

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def get_samplers(config, trainset: Dataset, testset: Dataset):
    """Build distributed samplers for train and test sets."""
    config.defrost()
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    indices = np.arange(global_rank, len(testset), num_tasks)
    sampler_val = SubsetRandomSampler(indices)

    return sampler_train, sampler_val