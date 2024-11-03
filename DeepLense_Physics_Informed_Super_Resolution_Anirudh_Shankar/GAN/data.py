import torch
import numpy as np
    
class LensingDataset(torch.utils.data.Dataset):
    def __init__(self, directory, classes, num_samples, aux='_sim_'):
        """
        The dataset class

        :param directory: Path to the dataset directory
        :param classes: List of lensing image classes (usually 1)
        :param num_samples: Number of images in the dataset
        """
        super(LensingDataset, self).__init__()
        self.directory = directory
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
        class_index = index%self.num_samples
        image = torch.tensor(np.array([np.load(self.directory+selected_class+'/%s'%selected_class+self.aux+'%d.npy'%(class_index))]))
        if self.aux == '_sim_':
            image = (image - torch.min(image))/(torch.max(image)-torch.min(image))
        return image
    
class LensingDataset4(torch.utils.data.Dataset):
    def __init__(self, directory, classes, num_samples):
        """
        The dataset class

        :param directory: Path to the dataset directory
        :param classes: List of lensing image classes (usually 1)
        :param num_samples: Number of images in the dataset
        """
        super(LensingDataset4, self).__init__()
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

        :param index: Index in the dataset to look for
        :return: LR image, min-max normalized
        """
        selected_class = self.classes[index//self.num_samples]
        class_index = index%self.num_samples
        image = torch.tensor(np.array(np.load(self.directory+selected_class+'/%s_sim_%d.npy'%(selected_class,class_index))))
        image = (image - torch.min(image))/(torch.max(image)-torch.min(image))
        return image