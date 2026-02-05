
import torch
import numpy as np
from torchvision import transforms

class MinMaxNormalize:
    """Normalizes the image to [0, 1] range using min-max scaling."""
    def __call__(self, x):
        min_val = x.min()
        max_val = x.max()
        if max_val - min_val > 0:
            return (x - min_val) / (max_val - min_val)
        return x

class SelectChannel:
    """Selects a specific channel/index from the input array."""
    def __init__(self, index=0):
        self.index = index

    def __call__(self, x):
        # If x is numpy array or tensor with more than 1 dimension
        if hasattr(x, "ndim") and x.ndim > 2:
             return x[self.index]
        return x

class ToTensor:
    """Converts a numpy array to a torch tensor."""
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float()

class AddChannel:
    """Adds a channel dimension at the specified index."""
    def __init__(self, dim=0):
        self.dim = dim
    
    def __call__(self, x):
        return x.unsqueeze(self.dim)

def get_transforms(config=None):
    """
    Returns a composition of transforms based on configuration.
    For now, returns a default set of transforms if config is not provided.
    """
    # Default transforms replicating the logic in CustomDataset
    # 1. Select logical channel (if necessary, though logic was specific to 'axion' files)
    # 2. MinMax Normalize
    # 3. ToTensor
    # 4. Add channel dimension
    
    transform_list = [
        MinMaxNormalize(),
        ToTensor(),
        AddChannel(dim=0)
    ]
    
    return transforms.Compose(transform_list)

def get_conditional_transforms():
    """Returns transforms specifically for conditional generation."""
    return transforms.Compose([
         MinMaxNormalize(),
         ToTensor(),
         AddChannel(dim=0)
    ])
