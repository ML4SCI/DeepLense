from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDatasetViT(Dataset):
    """
    Vision Transformer dataset for .npy gravitational lensing data.

    Expected directory structure:
        root/
        ├── axion/
        ├── cdm/
        └── no_sub/

    Each directory contains .npy files.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        self.samples = []
        self.class_to_idx = {}

        self._build_index()

    def _build_index(self):
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])

        if not classes:
            raise RuntimeError(f"No class folders found in {self.data_dir}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = self.data_dir / cls
            for file_path in cls_dir.glob("*.npy"):
                self.samples.append((file_path, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"No .npy files found in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError("Dataset index out of range")

        file_path, label = self.samples[idx]

        data = np.load(file_path, allow_pickle=True)

        if file_path.parent.name == "axion":
            image = data[0]
        else:
            image = data

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")

        if image.ndim > 2:
            image = image[0]

        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        return image, label
