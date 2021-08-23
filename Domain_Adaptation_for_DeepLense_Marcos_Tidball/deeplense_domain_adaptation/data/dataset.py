import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NumpyDataset(Dataset):
    def __init__(self, data_path, labels_path, transform, three_channels=False):
        """
        Loads image NumPy arrays and labels NumPy arrays and applies transforms to them in a memory-efficient manner.

        Arguments:
        ----------
        data_path: str
            Path to image data of shape [number_of_images, 1, height, width].

        labels_path: str
            Path to label data of shape [number_of_labels, 1].

        transform: Torchvision transforms
            Augmentations to be applied on the data.

        three_channels: bool
            If True the one-channel image is copied over the other channels, creating a three-channel image.
        """

        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')

        self.transform = transform
        self.three_channels = three_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.three_channels:
            data = np.tile(data, (3, 1, 1)) # copies 2d array to the other 3 channels
        
        data = data.astype(np.float32)
        data = self.transform(torch.from_numpy(data))

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return data, label

def get_dataloader(data_path, labels_path, augmentations, bs=100, three_channels=False, num_workers=2, shuffle=False):
    """
    Creates PyTorch DataLoaders that load image NumPy arrays and labels NumPy arrays and applies transforms to them in a memory-efficient manner.

    Arguments:
    ----------
    data_path: str
        Path to image data of shape [number_of_images, 1, height, width].

    labels_path: str
        Path to label data of shape [number_of_labels, 1].

    transform: Torchvision transforms
        Augmentations to be applied on the data.

    bs: int
        Batch size for loading the data.

    three_channels: bool
        If True the one-channel image is copied over the other channels, creating a three-channel image.

    num_workers: int
        Number of workers to be used (a number too high may slow down loading).

    shuffle: bool
        If True shuffles the data in the DataLoader.
        Some algorithms (Self-Ensemble & AdaMatch) require different DataLoaders that are not shuffled between one another!
    """

    # create dataset
    dataset = NumpyDataset(data_path, labels_path, augmentations, three_channels)
    
    # create dataloader
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=bs, num_workers=num_workers)
    
    return dataloader