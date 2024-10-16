import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, root_dir, config, max_samples=10000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.file_list = self.file_list[:max_samples]
        self.max_v4 = config.data.max
        self.min_v4 = config.data.min

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = np.load(file_path, allow_pickle=True)
        data_new = (data[0] - np.min(data[0]))/(np.max(data[0])-np.min(data[0]))
        v1 = data[1]
        v2 = data[2]
        v3 = data[3]
        v4 = data[4]
        data_new = torch.from_numpy(data_new).float()
        data_new = data_new.unsqueeze(0)
        v1 = torch.tensor(v1).float()
        v1 = v1.unsqueeze(0)
        v2 = torch.tensor(v2).float()
        v2 = v2.unsqueeze(0)
        v3 = torch.tensor(v3).float()
        v3 = v3.unsqueeze(0)
        v4 = torch.tensor(v4).float()
        ## Normalised
        #v4 = (v4 - self.min_v4)/(self.max_v4-self.min_v4)
        v4 = v4.unsqueeze(0)
        #v4 = v4.unsqueeze(0)

        if self.transform:
            data = self.transform(data)

        return data_new, v1, v2, v3, v4


class CustomDataset_1(Dataset):
    def __init__(self, root_dir, config, max_samples=10000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.file_list = self.file_list[:max_samples]
        self.max_v4 = config.data.max
        self.min_v4 = config.data.min

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = np.load(file_path, allow_pickle=True)
        data_new = (data[0] - np.min(data[0]))/(np.max(data[0])-np.min(data[0]))
        v1 = data[1]
        v2 = data[2]
        v3 = data[3]
        v4 = data[4]
        data_new = torch.from_numpy(data_new).float()
        data_new = data_new.unsqueeze(0)
        v1 = torch.tensor(v1).float()
        v1 = v1.unsqueeze(0)
        v2 = torch.tensor(v2).float()
        v2 = v2.unsqueeze(0)
        v3 = torch.tensor(v3).float()
        v3 = v3.unsqueeze(0)
        v4 = torch.tensor(v4).float()
        ## Normalised
        #v4 = (v4 - self.min_v4)/(self.max_v4-self.min_v4)
        v4 = v4.unsqueeze(0)
        #v4 = v4.unsqueeze(0)

        if self.transform:
            data = self.transform(data)

        return data_new, v1, v2, v3, v4

class CustomDataset_v1(Dataset):
    def __init__(self, root_dir, config, max_samples=10000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.file_list = self.file_list[:max_samples]
        self.max_v4 = config.data.max
        self.min_v4 = config.data.min

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = np.load(file_path, allow_pickle=True)
        data_new = (data[0] - np.min(data[0]))/(np.max(data[0])-np.min(data[0]))
        v1 = data[1]
        v2 = data[2]
        v3 = data[3]
        v4 = data[4]
        data_new = torch.from_numpy(data_new).float()
        data_new = data_new.unsqueeze(0)
        v1 = torch.tensor(v1).float()
        v1 = v1.unsqueeze(0)
        v2 = torch.tensor(v2).float()
        v2 = v2.unsqueeze(0)
        v3 = torch.tensor(v3).float()
        v3 = v3.unsqueeze(0)
        v4 = torch.tensor(v4).float()
        
        

        ## Normalised
        #v4 = (v4 - self.min_v4)/(self.max_v4-self.min_v4)
        v4 = v4.unsqueeze(0)
        

        if self.transform:
            data = self.transform(data)

        return data_new, v4
    


if __name__ == '__main__':
    folder_path = '../Data/cdm_regress_multi_param_model_ii/cdm_regress_multi_param/'
    dataset = CustomDataset_v1(root_dir=folder_path, max_samples=10000)
    # mass = -51
    # mass_min = -55.26202392578125
    # mass_max = -50.65694808959961
    # print((mass-mass_min)/(mass_max-mass_min))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    # train_size = 0.9
    # indices = list(range(len(dataset)))
    # train_dataset, test_dataset = train_test_split(indices, train_size=train_size, random_state=42)
    # train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    #print(dataset[1][1])
    #Check the DataLoader
    #print(dataloader[0].shape)
    # Initialize variables to store the max and min values
    max_value = float('-inf')
    min_value = float('inf')
    i = 0
    for image,batch in dataloader:
        batch_max = torch.max(batch)
        batch_min = torch.min(batch)
        
        if batch_max > max_value:
            max_value = batch_max.item()
        
        if batch_min < min_value:
            min_value = batch_min.item()

        print(i)
        i = i+1
    print(f'Max value: {max_value}')
    print(f'Min value: {min_value}') 