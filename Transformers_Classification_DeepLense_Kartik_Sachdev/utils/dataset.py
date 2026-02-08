import os
import torch
import numpy as np
import gdown
import splitfolders
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from typing import List, Optional, Tuple

# --- CORE REFACTOR (Issue #123) ---

class LensDataset(Dataset):
    """
    Refactored LensDataset: Handles only raw data fetching.
    Removes hardcoded category logic and direct augmentation.
    """
    def __init__(self, file_paths: List[str], labels: List[int], channels: int = 1):
        assert len(file_paths) == len(labels), "Mismatched paths and labels"
        self.file_paths = file_paths
        self.labels = labels
        self.channels = channels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        label = self.labels[index]
        
        # Load and normalize
        if path.endswith('.npy'):
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = arr[0]
            # Squeeze channel dim if it's (1, H, W)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            
            # Min-Max Normalization
            arr = arr.astype(np.float32)
            den = np.max(arr) - np.min(arr)
            norm = (arr - np.min(arr)) / (den + 1e-8)
            img_uint8 = (norm * 255).astype("uint8")
            img = Image.fromarray(img_uint8).convert("RGB" if self.channels == 3 else "L")
        else:
            img = Image.open(path).convert("RGB" if self.channels == 3 else "L")

        return img, torch.tensor(label).long()

class WrapperDataset(Dataset):
    """
    Wrapper for dynamic transform handling as requested in #123.
    """
    def __init__(self, base_dataset: Dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        img, label = self.base[index]
        if self.transform is not None:
            # Handle Albumentations vs Torchvision
            img_np = np.array(img)
            try:
                out = self.transform(image=img_np)
                img = out["image"] if isinstance(out, dict) else out
            except:
                img = self.transform(img)
        return img, label

# Alias for backward compatibility
DeepLenseDataset = LensDataset

# --- UTILITY & DATA DISCOVERY ---

def discover_lens_data(root_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Dynamically categorizes data to remove hardcoded logic.
    """
    categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    paths, labels = [], []
    for idx, cat in enumerate(categories):
        cat_path = os.path.join(root_dir, cat)
        for f in os.listdir(cat_path):
            if f.endswith(('.npy', '.png', '.jpg')):
                paths.append(os.path.join(cat_path, f))
                labels.append(idx)
    return paths, labels, categories

# --- VISUALIZATION ---

def visualize_samples(dataset, labels_map, num_samples=10):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        idx = torch.randint(len(dataset), (1,)).item()
        img, label = dataset[idx]
        plt.subplot(2, 5, i+1)
        plt.title(labels_map[label])
        plt.imshow(np.array(img), cmap='gray')
        plt.axis('off')
    plt.show()

# --- LEGACY SUPPORT (Optional, kept if needed for your specific env) ---

def download_dataset(filename, url):
    if not os.path.isfile(filename):
        gdown.download(url, filename, quiet=False)