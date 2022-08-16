import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import utils
from torch.utils.data import DataLoader, Dataset, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        root_list = glob.glob(root_dir)
        self.class_map = {}
        self.class_distribution = {}
        self.transform = transform

        for img_path in root_list:
            class_name = img_path.split(os.sep)[-2]
            if class_name not in self.class_distribution:
                self.class_distribution[class_name] = 1
            else:
                self.class_distribution[class_name] +=1

        for index, entity in enumerate(self.class_distribution):
            self.class_map[entity] = index

        # print("\nDataset Distribution:")
        # print(self.class_distribution)
        # print("\nClass indices:")
        # print(self.class_map)

        self.data = []
        for img_path in root_list:
            class_name = img_path.split(os.sep)[-2]
            self.data.append([img_path, class_name])
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = np.load(img_path, allow_pickle = True)
        if class_name == 'axion':
            img = img[0]
        
        
        if self.transform:
            aug = self.transform(image = img)
            img = aug['image']
        
        img = img.to(torch.float)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)

        return img, class_id


def create_data_loaders(train_data_path, test_data_path, val_split = 0.1, batch_size = 128, transforms = None, class_map = False):

    dataset = CustomDataset(train_data_path, transform = transforms)
    m = len(dataset)
    print("\nTotal training data: " + str(m))
    try:
        train_set,val_set=random_split(dataset,[int(m-m*val_split),int(m*val_split)])
    except:
        train_set,val_set=random_split(dataset,[int(m-m*val_split),int(m*val_split+1)])
        
    test_set = CustomDataset(test_data_path, transform = transforms)

    print(f"\n    Number of training set examples: {len(train_set)} \n\
    Number of validation set examples: {len(val_set)} \n\
    Number of test set examples: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    if class_map:
        return train_loader, val_loader, test_loader, dataset.class_map

    return train_loader, val_loader, test_loader



if __name__ == "__main__":

    batch_size = 128
    transforms = A.Compose(
            [
                A.CenterCrop(height = 50, width = 50, p=1.0),
                ToTensorV2()
            ]
        )

    train_data_path = r'C:\Users\Saranga\Desktop\ML4SCI\Work\Model_I_subset\*\*'
    test_data_path = r'C:\Users\Saranga\Desktop\ML4SCI\Work\Model_I_test_subset\*\*'

    train_loader, val_loader, test_loader = create_data_loaders(train_data_path, test_data_path, 
                                                                val_split = 0.2, batch_size = batch_size,
                                                                transforms = transforms)

    single_batch = next(iter(train_loader))
    print(f"\nShape of one batch of training data: {single_batch[0].shape}")
    single_batch_grid = utils.make_grid(single_batch[0], nrow=8)
    plt.figure(figsize = (20,700))
    plt.imshow(single_batch_grid.permute(1, 2, 0))
    plt.show()

    

    




   