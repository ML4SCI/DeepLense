import os
import torch
from torch.utils.data import Dataset
import numpy as np

class NPYDataset(Dataset):
    """
    Custom PyTorch Dataset for loading .npy-based DeepLense data.
    """

    def __init__(self, root_dir):
        self.samples = []

        # Only include directories (ignore files like dataset.py)
        self.class_names = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            class_dir = os.path.join(root_dir, cls)

            for file in os.listdir(class_dir):
                if file.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(class_dir, file), self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        data = np.load(file_path)
        data = torch.tensor(data, dtype=torch.float32)

        # Ensure shape is compatible with CNN (3 channels, 224x224)
        if len(data.shape) == 2:
            data = data.unsqueeze(0)

        data = torch.nn.functional.interpolate(
            data.unsqueeze(0), size=(224, 224), mode="bilinear"
        ).squeeze(0)

        data = data.repeat(3, 1, 1)

        return data, label
