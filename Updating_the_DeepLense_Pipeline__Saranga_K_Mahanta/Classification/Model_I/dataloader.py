import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import utils
import matplotlib.pyplot as plt

from config import BATCH_SIZE, TRAIN_DATA_PATH, TEST_DATA_PATH
from utils import transforms


class CustomDataset(Dataset):
    """
    Custom dataset for loading .npy lensing image files.
    Expected structure:

    root_dir/
        class_name_1/
            sample1.npy
            sample2.npy
        class_name_2/
            sample1.npy
            ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Collect all .npy files inside class folders
        root_list = glob.glob(os.path.join(root_dir, "*", "*.npy"))

        #  Guard against empty dataset
        if len(root_list) == 0:
            raise ValueError(
                f"No .npy files found in {root_dir}. "
                "Expected structure: root/class_name/*.npy"
            )

        # Build class distribution
        self.class_distribution = {}
        for img_path in root_list:
            class_name = os.path.basename(os.path.dirname(img_path))
            self.class_distribution[class_name] = \
                self.class_distribution.get(class_name, 0) + 1

        #  Deterministic class mapping (important for reproducibility)
        self.class_map = {
            cls_name: idx
            for idx, cls_name in enumerate(sorted(self.class_distribution.keys()))
        }

        # Store dataset entries
        self.data = [
            (img_path, os.path.basename(os.path.dirname(img_path)))
            for img_path in root_list
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]

        # Load numpy array
        img = np.load(img_path, allow_pickle=True)

        # Special handling for axion class
        if class_name == "axion":
            img = img[0]

        # Apply transforms if provided
        if self.transform:
            aug = self.transform(image=img)
            img = aug["image"]

        img = img.to(torch.float32)
        class_id = torch.tensor(self.class_map[class_name], dtype=torch.long)

        return img, class_id


def create_data_loaders(
    train_data_path,
    test_data_path,
    val_split=0.1,
    batch_size=128,
    transforms=None,
    class_map=False,
):
    """
    Creates train, validation, and test dataloaders.
    """

    dataset = CustomDataset(train_data_path, transform=transforms)
    total_size = len(dataset)

    print(f"\nTotal training samples: {total_size}")

    # Split train/validation
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    test_set = CustomDataset(test_data_path, transform=transforms)

    print(
        f"\nNumber of training samples: {len(train_set)}"
        f"\nNumber of validation samples: {len(val_set)}"
        f"\nNumber of test samples: {len(test_set)}"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if class_map:
        return train_loader, val_loader, test_loader, dataset.class_map

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader, test_loader = create_data_loaders(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        val_split=0.2,
        batch_size=BATCH_SIZE,
        transforms=transforms,
    )

    single_batch = next(iter(train_loader))
    print(f"\nShape of one batch: {single_batch[0].shape}")

    grid = utils.make_grid(single_batch[0], nrow=8)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Sample Training Batch")
    plt.show()
