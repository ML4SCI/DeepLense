import os
import torch
import numpy as np

class LensingDataset(torch.utils.data.Dataset):
    def __init__(self, directory, classes, num_samples):
        """
        The dataset class

        :param directory: Path to the dataset directory
        :param classes: List of lensing image classes
        :param num_samples: Number of images per class
        """
        super(LensingDataset, self).__init__()
        self.directory = directory
        self.classes = classes
        self.num_samples = num_samples

    def __len__(self):
        """
        :return: Returns the length of the dataset
        """
        return self.num_samples * len(self.classes)

    def __getitem__(self, index):
        """
        Supplies LR images

        :param index: Index in the dataset to look for
        :return: LR image, min-max normalized
        """
        selected_class = self.classes[index // self.num_samples]
        class_index = index % self.num_samples
        
        file_path = os.path.join(self.directory, selected_class, f'sim_{class_index}.npy')
        
        # Load array, convert to tensor, cast to float32, and add channel dimension
        np_img = np.load(file_path)
        image = torch.from_numpy(np_img).float().unsqueeze(0)
        
        # Min-max normalization with epsilon to prevent division by zero
        img_min = torch.min(image)
        img_max = torch.max(image)
        image = (image - img_min) / (img_max - img_min + 1e-8)
        
        return image
