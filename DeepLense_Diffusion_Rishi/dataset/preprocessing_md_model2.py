import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict

class NumbersDataset(Dataset):
    def __init__(self, main_folder_path, max_examples=10000):
        self.numbers = self.read_numbers_from_folders(main_folder_path, max_examples)
    
    def read_numbers_from_folders(self, main_folder_path, max_examples):
        numbers = []
        count = 0
        for subdir, _, files in os.walk(main_folder_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(subdir, file)
                    number = np.load(file_path, allow_pickle=True)  # Load single number
                    numbers.append(number[1])
                    count += 1
                    if count>= max_examples:
                        return numbers
        return numbers

    def __len__(self):
        return len(self.numbers)
    
    def __getitem__(self, idx):
        return torch.tensor(np.log(self.numbers[idx]), dtype=torch.float).unsqueeze(0)


class CustomDataset_AE_Conditional(Dataset):

    def __init__(self, root_dir, max_samples, config, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.file_list = self.file_list[:max_samples]
        self.mass_max = config.data.max_value
        self.mass_min = config.data.min_value

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data_load = np.load(file_path, allow_pickle=True)
        data = (data_load[0] - np.min(data_load[0]))/(np.max(data_load[0])-np.min(data_load[0]))
        mass = data_load[1]
        mass = np.log(mass)#.repeat(128)
        #mass = (mass - self.mass_min)/(self.mass_max - self.mass_min)
        mass = torch.tensor(mass)
        data = torch.from_numpy(data).float()
        data = data.unsqueeze(0)
        if self.transform:
            data = self.transform(data)

        return data, mass.unsqueeze(0)


if __name__ == '__main__':
    folder_path = '../Data/Model_II/axion'
    #dataset = CustomDataset_AE_Conditional(folder_path, max_samples=10000)
    mass = -51
    mass_min = -55.26202392578125
    mass_max = -50.65694808959961
    print((mass-mass_min)/(mass_max-mass_min))
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # train_size = 0.9
    # indices = list(range(len(dataset)))
    # train_dataset, test_dataset = train_test_split(indices, train_size=train_size, random_state=42)
    # train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    #print(dataset[1][1])
    #Check the DataLoader
    #print(dataloader[0].shape)
    # Initialize variables to store the max and min values
    # max_value = float('-inf')
    # min_value = float('inf')
    # for batch in dataloader:
    #     batch_max = torch.max(batch)
    #     batch_min = torch.min(batch)
        
    #     if batch_max > max_value:
    #         max_value = batch_max.item()
        
    #     if batch_min < min_value:
    #         min_value = batch_min.item()
    # print(f'Max value: {max_value}')
    # print(f'Min value: {min_value}') 