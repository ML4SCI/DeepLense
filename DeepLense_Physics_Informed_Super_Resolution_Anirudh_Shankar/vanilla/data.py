import torch
import numpy as np
from pathlib import Path
    
class LensingDataset(torch.utils.data.Dataset):
    def __init__(self, directory, classes, num_samples, aux='_sim_'):
        """
        The dataset class

        :param directory: Path to the dataset directory
        :param classes: List of lensing image classes
        :param num_samples: Number of images in the dataset
        :param aux: Used to indicate whether the dataset contains image sumulations ('_sim_') or deflection angles ('_alpha_')
        """
        super(LensingDataset, self).__init__()
        self.directory = Path(directory)
        self.classes = classes
        self.num_samples = num_samples
        self.aux = aux

    def __len__(self):
        """
        :return: Returns the length of the dataset
        """
        return self.num_samples*len(self.classes)
    
    def __getitem__(self, index):
        """
        Supplies LR images

        :param index: Index in the dataset to look for
        :return: LR image, min-max normalized
        """
        selected_class = self.classes[index//self.num_samples]
        sample_index = index%self.num_samples
        image_path = self.directory / selected_class / f"{selected_class}{self.aux}{sample_index}.npy"

        if not image_path.exists():
            raise FileNotFoundError(f"Expected sample not found: {image_path}")

        # Keep the returned tensor in the original single-channel format: [1, H, W].
        image = torch.tensor(np.array([np.load(image_path)]))

        if self.aux == '_sim_':
            image_min = torch.min(image)
            image_max = torch.max(image)

            # Avoid dividing by zero if a sample is constant-valued.
            if torch.isclose(image_max, image_min):
                return torch.zeros_like(image)

            image = (image - image_min)/(image_max - image_min)

        return image
