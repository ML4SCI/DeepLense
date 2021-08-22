
import sys
import os
import time
import logging
import argparse
import shutil
import torch
import torchvision
import numpy as np
from os import listdir
import pandas as pd
from e2cnn import gspaces
from e2cnn import nn
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os.path import join
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import RandomRotation, Pad, Resize, ToTensor, Compose
import torchvision.transforms as transforms



class CustomDataset(Dataset):
    
    def __init__(self, root_dir, mode, transform=None):
        assert mode in ['train', 'test']

        self.root_dir = root_dir
            
        if mode == "train":
            self.root_dir = self.root_dir+"/train"
        else:
            self.root_dir = self.root_dir+"/val"

        
        self.transform = transform
        classes = [join(self.root_dir, x).split('/')[3] for x in listdir(self.root_dir)]


        classes = [join(self.root_dir, x).split('/')[3] for x in listdir(self.root_dir)]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        self.imagefilename = []
        self.labels = []

        for i in classes:
            for x in listdir(join(self.root_dir, i)):
                self.imagefilename.append(join(self.root_dir, i,x))
                self.labels.append(self.class_to_idx[i])

    
    def __getitem__(self, index):
        image, label = self.imagefilename[index], self.labels[index]
        
        image = np.load(image)
        image = image / image.max() #normalizes data in range 0 - 255
        image = 255 * image
        image  = Image.fromarray(image.astype('uint8')).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)