
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Callable, List, Tuple

class LensDataset(Dataset):
    """
    Base dataset class for loading lens data.
    
    Args:
        root_dir (str): Path to the directory containing .npy files.
        transform (callable, optional): Optional transform to be applied on a sample.
        max_samples (int, optional): Maximum number of samples to load.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None, max_samples: Optional[int] = None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load and sort file list for reproducibility
        self.file_list = sorted([f for f in os.listdir(root_dir) if f.endswith('.npy')])
        if max_samples:
            self.file_list = self.file_list[:max_samples]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        
        try:
            # Load numpy array
            data = np.load(file_path, allow_pickle=True)
            
            # Apply transforms
            # Note: Normalization and type conversion should be handled by the transform
            if self.transform:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # Identify a strategy for failed loads; for now return None or raise
            raise e

class WrapperDataset(Dataset):
    """
    Wrapper dataset for conditional generation or multi-class scenarios.
    Automatically iterates through subdirectories as classes.
    
    Args:
        root_dir (str): Root directory containing class subdirectories.
        transform (callable, optional): Transform to apply to data.
        max_samples_per_class (int, optional): Limit samples per class.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None, max_samples_per_class: int = 5000):
        self.root_dir = root_dir
        self.transform = transform
        
        # Identify class folders
        self.class_folders = sorted([
            f for f in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        
        if not self.class_folders:
            raise FileNotFoundError(f"No class subdirectories found in {root_dir}")

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.class_folders)
        
        self.data_index = [] # List of (file_path, class_name)
        
        for class_name in self.class_folders:
            class_path = os.path.join(root_dir, class_name)
            files = sorted([f for f in os.listdir(class_path) if f.endswith('.npy')])
            
            if max_samples_per_class:
                files = files[:max_samples_per_class]
                
            for f in files:
                self.data_index.append((os.path.join(class_path, f), class_name))
        
        print(f"Found {len(self.class_folders)} classes: {self.class_folders}")
        print(f"Total samples: {len(self.data_index)}")

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, class_name = self.data_index[idx]
        
        # Load Data
        data = np.load(file_path, allow_pickle=True)
        
        # Specific handling for 'axion' like cases could go into a specific transform 
        # or pre-processing function if strictly required, but generally we rely on shape.
        # Minimal legacy support: if data is wrapper in an extra array [data, label, ...] like structure
        # Checking if data has a shape like (1, 64, 64) vs (64, 64) done via transforms usually.
        
        if self.transform:
            data = self.transform(data)
            
        # Get Label
        label = self.label_encoder.transform([class_name])[0]
        
        return data, label
