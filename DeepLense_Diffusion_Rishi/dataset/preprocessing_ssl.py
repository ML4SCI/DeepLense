import os
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, root_dir, config, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]
        self.mean = config.data.mean
        self.std = config.data.std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = np.load(file_path, allow_pickle=True)
        #data = (data - np.min(data))/(np.max(data)-np.min(data))
        # Normalize each channel separately
        # for i in range(3):
        #     data[i, :, :] = (data[i, :, :] - self.mean[i]) / self.std[i]
        data = torch.from_numpy(data).float()
        #data = data.unsqueeze(0)
        #print(data.shape)
        if self.transform:
            data = self.transform(data)

        return data

class CustomDataset_Conditional(Dataset):
    def __init__(self, folder_path, max_samples=5000, transforms=None):
        self.folder_path = folder_path
        self.class_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        #print(self.class_folders)
        # Initialize LabelEncoder
        self.label_encoder = LabelEncoder()
        
        # Fit and transform class labels to numerical labels
        self.labels = self.label_encoder.fit_transform(self.class_folders)
        self.transform = transforms
        self.data = []
        
        for class_folder in self.class_folders:
            class_path = os.path.join(folder_path, class_folder)
            #print(class_folder)
            file_list = [f for f in os.listdir(class_path) if f.endswith('.npy')]
            file_list = file_list[:max_samples]

            for file_name in file_list:
                file_path = os.path.join(class_path, file_name)
                data_point = np.load(file_path, allow_pickle=True)
                if file_name.startswith('axion'):
                    data_point = data_point[0]
                self.data.append((data_point, class_folder))
                #print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point, class_name = self.data[idx]
        #print(class_name)
        label = self.label_encoder.transform([class_name])[0]
        #print(label)
        
        #data_point = (data_point - np.mean(data_point, axis=(1,2)))/(np.std(data_point, axis=(1,2)))
        
        # normalise numpy array and convert to PyTorch tensor
        data_point = (data_point-np.min(data_point))/(np.max(data_point)-np.min(data_point))
        data_point = torch.from_numpy(data_point).float()
        data_point = data_point.unsqueeze(0)

        if self.transform:
            data_point = self.transform(data_point)

        # # Convert label to one-hot vector
        # one_hot_label = F.one_hot(torch.tensor(label), num_classes=len(self.labels))

        
        return data_point, label

if __name__ == '__main__':
    print('hi')
    dataset =  CustomDataset_Conditional('../../Data/Model_II/')
    print(dataset[80000])