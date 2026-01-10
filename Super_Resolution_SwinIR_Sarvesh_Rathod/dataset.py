import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

class LensingDataset(Dataset):
    def __init__(self, hr_path, lr_path, transform=None):
        """
        Args:
            hr_path (string): Path to the high-resolution .npy file.
            lr_path (string): Path to the low-resolution .npy file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the numpy arrays
        # Note: The original get_data_diff.py script saves them as (N, 1, H, W) or similar
        try:
            self.hr_images = np.load(hr_path)
            self.lr_images = np.load(lr_path)
        except Exception as e:
            print(f"Error loading files: {e}")
            self.hr_images = []
            self.lr_images = []

        self.transform = transform

        # Ensure correct shape if necessary (N, C, H, W)
        # Check if loaded data needs reshaping. 
        # Usually PyTorch expects (C, H, W). 
        # If the data is (N, 128, 128), we might need to add channel dim.
        if len(self.hr_images) > 0:
             if self.hr_images.ndim == 3: # (N, H, W)
                self.hr_images = self.hr_images[:, np.newaxis, :, :]
                self.lr_images = self.lr_images[:, np.newaxis, :, :]
        
        print(f"Loaded {len(self.hr_images)} samples.")

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr = self.hr_images[idx]
        lr = self.lr_images[idx]

        # Convert to torch tensor
        # Assuming data is already float32 and normalized 0-1 or -1 to 1 based on get_data.py
        # If get_data.py outputs numpy arrays, just convert them.
        
        hr_tensor = torch.from_numpy(hr).float()
        lr_tensor = torch.from_numpy(lr).float()

        if self.transform:
            # Apply any transforms if needed
            pass

        return lr_tensor, hr_tensor
