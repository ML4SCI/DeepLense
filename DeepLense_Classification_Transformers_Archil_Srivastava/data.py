import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os

from constants import *


class LensDataset(Dataset):
    """
    Dataset class to convert the raw lens data into pytorch dataset
    """

    def __init__(self, root_dir, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # --- NEW: Defensive programming check ---
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"The directory '{root_dir}' was not found. Please check your data path.")
        
        # Check if directory is empty (ignores hidden files like .DS_Store)
        entries = [e for e in os.scandir(root_dir) if not e.name.startswith('.')]
        if len(entries) == 0:
            raise ValueError(f"The directory '{root_dir}' is empty! Please ensure you have category subfolders (e.g., Axion, CDM) inside.")
        # ----------------------------------------

        self.image_paths, self.categories = [], []
        for category_dir in entries:
            # We only want to look inside subdirectories
            if category_dir.is_dir():
                for raw_image_file in os.scandir(category_dir.path):
                    # Only add files that look like numpy arrays (or whatever your format is)
                    if raw_image_file.name.endswith(('.npy', '.npz')): 
                        self.image_paths.append(raw_image_file.path)
                        self.categories.append(category_dir.name)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, category = self.image_paths[idx], self.categories[idx]

        # Use allow_pickle=True carefully
        image = np.load(image_path, allow_pickle=True)
        
        # Normalize category names to match LABEL_MAP keys
        category_key = category.lower() 
        
        if category_key == "axion":
            # Some datasets have a tuple/list format, ensuring we get the array
            if isinstance(image, (list, tuple, np.ndarray)) and len(image) > 0:
                image = image[0]

        if self.transform:
            image = self.transform(image)

        # Ensure category exists in your LABEL_MAP
        return image, LABEL_MAP.get(category, 0)


class WrapperDataset(Dataset):
    def __init__(self, subset, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(config, initial_size, final_size, mode="test"):
    transform_pipeline = [transforms.ToTensor()]

    # Added check to prevent IndexErrors if sizes are None
    if initial_size is not None and initial_size == 150:
        transform_pipeline.append(transforms.CenterCrop(100))
    elif initial_size is not None:
        transform_pipeline.append(transforms.CenterCrop(50))

    if initial_size != final_size and final_size is not None:
        transform_pipeline.append(transforms.Resize(final_size))

    return transforms.Compose(transform_pipeline)