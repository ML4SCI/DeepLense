import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import utils
from torch.utils.data import DataLoader, Dataset, random_split

from config import BATCH_SIZE, TRAIN_DATA_PATH, TEST_DATA_PATH
from utils import transforms


class CustomDataset(Dataset):
    """
    Dataset loader for Model_II classification.

    Expected directory structure:

    root_dir/
        axion/
            *.npy
        cdm/
            *.npy
        no_sub/
            *.npy
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Collect all .npy files under class folders
        root_list = glob.glob(os.path.join(root_dir, "*", "*.npy"))

        if len(root_list) == 0:
            raise ValueError(
                f"No .npy files found in {root_dir}. "
                "Expected structure: root/class_name/*.npy"
            )

        # Build class distribution
        self.class_distribution = {}
        for img_path in root_list:
            class_name = os.path.basename(os.path.dirname(img_path))
            self.class_distribution[class_name] = (
                self.class_distribution.get(class_name, 0) + 1
            )

        # Deterministic class mapping
        self.class_map = {
            class_name: idx
            for idx, class_name in enumerate(sorted(self.class_distribution.keys()))
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

        img = np.load(img_path, allow_pickle=True)

        if class_name == "axion":
            img = img[0]

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

    dataset = CustomDataset(train_data_path, transform=transforms)
    total_samples = len(dataset)

    print("\nTotal training data:", total_samples)

    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    test_set = CustomDataset(test_data_path, transform=transforms)

    print(
        f"\nNumber of training examples: {len(train_set)}\n"
        f"Number of validation examples: {len(val_set)}\n"
        f"Number of test examples: {len(test_set)}"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if class_map:
        return train_loader, val_loader, test_loader, dataset.class_map

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_data_loaders(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
        val_split=0.2,
        batch_size=BATCH_SIZE,
        transforms=transforms,
    )

    single_batch = next(iter(train_loader))
    print("\nShape of one batch:", single_batch[0].shape)

    grid = utils.make_grid(single_batch[0], nrow=8)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Sample Batch")
    plt.show()
