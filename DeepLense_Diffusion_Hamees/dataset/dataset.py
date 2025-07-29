import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the class subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        """Finds the class folders in a dataset."""
        classes = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        """Creates a list of samples (file_path, class_index)."""
        samples = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = self.root_dir / class_name
            for file_path in class_dir.glob("*.npy"):
                samples.append((file_path, class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, class_idx = self.samples[idx]
        # Load data, add a channel dimension, making it (H, W, C)
        sample = np.load(file_path).astype(np.float32) # Ensure float32
        sample = np.expand_dims(sample, axis=-1) # Shape: (64, 64, 1)

        if self.transform:
            sample = self.transform(sample)

        return sample, class_idx

# Pipeline A: For the NanoDiT Diffusion Model
# Input: (64, 64, 1) NumPy array
# Output: (1, 64, 64) Tensor, normalized to [-1, 1]
diffusion_transforms = transforms.Compose([
    transforms.ToTensor(), # Converts (H, W, C) NumPy to (C, H, W) Tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizes a 1-channel image to [-1, 1]
])

# Pipeline B: For the ResNet34 Classifier
# Input: (64, 64, 1) NumPy array
# Output: (3, 224, 224) Tensor, normalized with ImageNet stats
classifier_transforms = transforms.Compose([
    transforms.ToTensor(), # Converts to (1, 64, 64) Tensor and scales to [0, 1]
    transforms.Resize((224, 224), antialias=True), # Upsample to 224x224
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert grayscale to 3-channel
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])
