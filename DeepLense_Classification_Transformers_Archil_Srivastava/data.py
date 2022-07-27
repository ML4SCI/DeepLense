import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os

from constants import *

class LensDataset(Dataset):
    def __init__(self, image_size, memmap_path, *args,
                 transform=None, mean=None, std=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Hack where shape of memmap is inferred by creating memmap object twice. TODO: Find cleaner way
        self.x = np.memmap(os.path.join(memmap_path, 'images.npy'), dtype='int32', mode='r')
        self.length = self.x.shape[0] // (image_size * image_size)
        self.x = np.memmap(os.path.join(memmap_path, 'images.npy'), dtype='int32', mode='r',
                           shape=(self.length, image_size, image_size))
        self.y = np.load(os.path.join(memmap_path, 'labels.npy'))

        self.mean = mean
        if self.mean is None:
            self.mean = np.mean(self.x)
        
        self.std = std
        if self.std is None:
            self.std = np.std(self.x)

        self.transform = transform
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img = (self.x[idx] - self.mean) / self.std
        img = np.expand_dims(img, axis=0) # Add channel axis
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        
        label = self.y[idx]

        return img, label

class WrapperDataset(Dataset):
    def __init__(self, subset, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label 


def get_transforms(config, final_size, mode='test'):
    transform_pipeline = []
    if mode == 'train':
        transform_pipeline.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        if config.random_rotation > 0:
            transform_pipeline.append(transforms.RandomRotation(config.random_rotation, interpolation=transforms.InterpolationMode.BILINEAR))
        if config.random_zoom < 1: # 1 is when the random crop is the whole image
            transform_pipeline.append(transforms.RandomResizedCrop(final_size, scale=(config.random_zoom**2, 1.), ratio=(1., 1.)))
    
    transform_pipeline.append(transforms.Resize(final_size))

    return transforms.Compose(transform_pipeline)