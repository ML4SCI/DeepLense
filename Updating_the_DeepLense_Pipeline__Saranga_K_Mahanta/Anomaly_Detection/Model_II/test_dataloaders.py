import os
import glob
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def img_paths_list(root_dir):
    root_list = glob.glob(root_dir)
    class_map = {}
    class_distribution = {}
    
    for img_path in root_list:
        class_name = img_path.split(os.sep)[-2]
        if class_name not in class_distribution:
            class_distribution[class_name] = 1
        else:
            class_distribution[class_name] +=1
                
    for index, entity in enumerate(class_distribution):
        class_map[entity] = index
    print("Dataset Distribution:\n")
    print(class_distribution)
    print("\n\nClass indices:\n")
    print(class_map)

    data = []
    for img_path in tqdm(root_list):
        class_name = img_path.split(os.sep)[-2]
        data.append([img_path, class_name])
        
    return data, class_map

class CustomDataset(Dataset):
    def __init__(self, img_paths_and_labels_list, class_map,transform = None):
        self.data = img_paths_and_labels_list
        self.class_map = class_map
        self.transform = transform
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = np.load(img_path, allow_pickle = True)
        if class_name == 'axion':
            img = img[0]
        img = (img - np.min(img))/(np.max(img) - np.min(img))
        
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']
        
        img = img.to(torch.float)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)

        return img, class_id

def create_full_test_dataloader(full_test_data_path, test_transforms, batch_size):
    full_test_dataset_img_paths_list, class_map = img_paths_list(full_test_data_path)
    full_test_dataset = CustomDataset(full_test_dataset_img_paths_list, class_map, transform = test_transforms)
    full_test_loader = DataLoader(full_test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    return full_test_loader, class_map


if __name__ == '__main__':

    FULL_TEST_DATA_PATH = r'C:\Users\Saranga\Desktop\ML4SCI\Work\Model_II_test\*\*'
    BATCH_SIZE = 256
    
    test_transforms = A.Compose(
                [
                    ToTensorV2()
                ]
            )

    full_test_loader, class_map = create_full_test_dataloader(FULL_TEST_DATA_PATH, test_transforms, BATCH_SIZE)
    