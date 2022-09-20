import glob
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import utils
from torch.utils.data import Dataset, DataLoader

from config import *
from utils import *


def img_paths_list(root_dir):
    root_list = glob.glob(root_dir)
    data = []
    for img_path in tqdm(root_list):
        data.append(img_path)
        
    return data


class CustomDataset(Dataset):
    def __init__(self, img_paths_and_labels_list,transform = None):
        self.data = img_paths_and_labels_list
        self.transform = transform
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx]
        array = np.load(path, allow_pickle = True)
        img = array[0]
        img = img.astype('float32')
        mass = torch.tensor(-(math.log10(array[1])))
       
        if self.transform:
            aug = self.transform(image = img)
            img = aug['image']      

        return img, mass


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
    set_seed(7)
    train_loader, val_loader, test_loader = create_dataloaders(
        TRAIN_DATA_PATH, TEST_DATA_PATH, 
        train_transforms, test_transforms, 
        BATCH_SIZE
        )
    single_batch = next(iter(test_loader))
    print(f"Shape of a single batch of data: {single_batch[0].shape}")

    single_batch_grid = utils.make_grid(single_batch[0][:16], nrow=8)
    plt.figure(figsize = (20,70))
    plt.imshow(single_batch_grid.permute(1, 2, 0))
    plt.show()