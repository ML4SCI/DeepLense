import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os

from constants import *


class LensDataset(Dataset):
    """
    Dataset class to convert the raw lens data into pytorch dataset

    Parameters
    ----------
    root_dir : str
        Root directory of the data
    transform : Function
        The transformation pipeline. If None, no transformations are applied
    """

    def __init__(self, root_dir, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_paths, self.categories = [], []
        for category_dir in os.scandir(root_dir):
            for raw_image_file in os.scandir(category_dir.path):
                self.image_paths.append(raw_image_file.path)
                self.categories.append(category_dir.name)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, category = self.image_paths[idx], self.categories[idx]

        image = np.load(image_path, allow_pickle=True)
        if category == "axion":
            image = image[0]

        if self.transform:
            image = self.transform(image)

        return image, LABEL_MAP[category]


class WrapperDataset(Dataset):
    """
    Dataset to wrap train/test dataset and their transformations

    Parameters
    ----------
    subset : torch.utils.data.Dataset
        The dataset to use
    transform : Function
        The transformation pipeline. If None, no transformations are applied
    """

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
    """
    Get the preprocessing and augmentation pipeline based on the dataset used

    Parameters
    ----------
    config : dict
        Config params given to the script
    initial_size : list
        Dimensions of the original image
    final_size : list
        Dimensions of the augmented image
    mode : str, optional
        Whether the transformation is being done on train or test dataset, by default 'test'
        We don't perform augmentations on the test dataset

    Returns
    -------
    Function
        The transformation pipeline
    """
    transform_pipeline = [transforms.ToTensor()]
    # if mode == 'train':
    #     transform_pipeline.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    #     if config.random_rotation > 0:
    #         transform_pipeline.append(transforms.RandomRotation(config.random_rotation, interpolation=transforms.InterpolationMode.BILINEAR))
    #     if config.random_zoom < 1: # 1 is when the random crop is the whole image
    #         transform_pipeline.append(transforms.RandomResizedCrop(final_size, scale=(config.random_zoom**2, 1.), ratio=(1., 1.)))

    if initial_size == 150:  # Model I
        transform_pipeline.append(transforms.CenterCrop(100))
    else:  # Model II and Model III
        transform_pipeline.append(transforms.CenterCrop(50))

    if initial_size != final_size:
        transform_pipeline.append(transforms.Resize(final_size))

    return transforms.Compose(transform_pipeline)
