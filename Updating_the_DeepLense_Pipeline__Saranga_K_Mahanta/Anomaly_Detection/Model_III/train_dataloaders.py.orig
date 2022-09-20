import glob
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import utils
from torch.utils.data import Dataset, DataLoader
<<<<<<< HEAD
import albumentations as A
from albumentations.pytorch import ToTensorV2
=======

from config import *
from utils import *
>>>>>>> 3e766e076bb0754e9e764e03244fcb57e4a077d0


def img_paths_list(root_dir):
    root_list = glob.glob(root_dir)

    data = []
    for img_path in tqdm(root_list):
        data.append(img_path)
        
    return data


class CustomDataset(Dataset):
    def __init__(self, paths_list, transform = None):

        self.transform = transform
        self.data = paths_list
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        img = np.load(path, allow_pickle = True)
        img = (img - np.min(img))/(np.max(img) - np.min(img))

       
        if self.transform:
            aug = self.transform(image = img)
            img = aug['image']
        
        img = img.to(torch.float)        

        return img


def create_dataloaders(train_data_path, test_data_path, train_transforms, test_transforms, batch_size):
    dataset_img_paths_list = img_paths_list(train_data_path)
    test_paths_list = img_paths_list(test_data_path)

    val_split = int(0.1 * len(dataset_img_paths_list))
    random.shuffle(dataset_img_paths_list)
    val_paths_list = dataset_img_paths_list[:val_split]
    train_paths_list = dataset_img_paths_list[val_split:]

    assert len(dataset_img_paths_list) == (len(train_paths_list) + len(val_paths_list))

    train_dataset = CustomDataset(train_paths_list, transform = train_transforms)
    print(f"Training dataset size: {len(train_dataset)}")
    val_dataset = CustomDataset(val_paths_list, transform = test_transforms)
    print(f"Validation dataset size: {len(val_dataset)}")
    test_dataset = CustomDataset(test_paths_list,transform = test_transforms)
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
<<<<<<< HEAD
    BATCH_SIZE = 256
    TRAIN_DATA_PATH = r'C:\Users\Saranga\Desktop\ML4SCI\Work\Model_III_subset\no_sub\*'
    TEST_DATA_PATH = r'C:\Users\Saranga\Desktop\ML4SCI\Work\Model_III_test\no_sub\*'

    train_transforms = A.Compose(
            [
                A.HorizontalFlip(p = 0.5),
                A.VerticalFlip(p = 0.5),
                A.Rotate(limit = 360, p = 0.4),
                ToTensorV2()
            ]
        )
    test_transforms = A.Compose(
                [
                    ToTensorV2()
                ]
            )

=======
>>>>>>> 3e766e076bb0754e9e764e03244fcb57e4a077d0
    train_loader, val_loader, test_loader = create_dataloaders(
        TRAIN_DATA_PATH, TEST_DATA_PATH, 
        train_transforms, test_transforms, 
        BATCH_SIZE
        )
    single_batch = next(iter(test_loader))
    print(f"Shape of a single batch of data: {single_batch[0].shape}")

    single_batch_grid = utils.make_grid(single_batch[:16], nrow=8)
    plt.figure(figsize = (20,70))
    plt.imshow(single_batch_grid.permute(1, 2, 0))
    plt.show()
