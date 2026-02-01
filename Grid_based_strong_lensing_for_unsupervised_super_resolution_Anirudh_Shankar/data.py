import torch
import numpy as np
import os    
class LensingDataset(torch.utils.data.Dataset):
    def __init__(self, directory, classes, num_samples):
        """
        The dataset class

        :param directory: Path to the dataset directory
        :param classes: List of lensing image classes
        :param num_samples: Number of images in the dataset
        """
        super(LensingDataset, self).__init__()
        self.directory = directory
        self.classes = classes
        self.num_samples = num_samples
    def __len__(self):
        """
        :return: Returns the length of the dataset
        """
        return self.num_samples*len(self.classes)
    
    



    def __getitem__(self, index):
        """
        Supplies LR images
        """
        selected_class = self.classes[index // self.num_samples]
        class_index = index % self.num_samples
        
        # USE os.path.join for portability
        file_path = os.path.join(self.directory, selected_class, 'sim_%d.npy' % class_index)
        
        image = torch.tensor(np.array([np.load(file_path)]))
        
        # Small safety check: handle division by zero if image is blank
        img_min = torch.min(image)
        img_max = torch.max(image)
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
            
        return image