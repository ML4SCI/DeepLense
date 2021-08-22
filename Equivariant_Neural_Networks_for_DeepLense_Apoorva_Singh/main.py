#Import Dependencies

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
from trainer import ECNN

from dataset import CustomDataset


#Selecting Device type to run the code
def get_device(use_cuda=True, cuda_idx=0):
    if use_cuda:
        if torch.cuda.is_available():
            assert cuda_idx in range(0, torch.cuda.device_count()),\
                "GPU index out of range. index lies in [{}, {})".format(0, torch.cuda.device_count())
            device = torch.device("cuda:"+str(cuda_idx))
        else:
            print("cuda not found, will switch to cpu")
    else:
        device = torch.device("cpu")
    print(f'Using device = {str(device)}')
    return device



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_cuda', type=bool, default=True, help="True, if use cuda")
    parser.add_argument('--cuda_idx', type=int, default=0, help="cuda device index")
    parser.add_argument('--data_dir', type=str, default='images_f', help='Data directory')
    parser.add_argument('--sym_group', type=str, default='Circular', help='Symmetry Group')
    parser.add_argument('--use_CNN', type = bool, default = False, help = 'True, if use simple convolution (ResNet18)')
    parser.add_argument('--epochs', type=int, default=30, help='Count of epochs') 
    parser.add_argument('--N', type=int, default=4, help='N')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size') 
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate') 
    parser.add_argument('--mode', type = str, default = 'Train', help = 'Training or Testing mode')
    parser.add_argument('--test_time', type = str, default = '', help = 'Name of the folder containing the model')

    return parser



def init_logging_handler(log_dir, current_time, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(os.path.join(log_dir, current_time)):
        os.makedirs(os.path.join(log_dir, current_time))

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/{}/log_{}.txt'.format(
        log_dir, current_time, current_time + extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    shutil.copyfile('general_code.py', os.path.join(log_dir, current_time) + "/code.py")
    
    return logger


parser = get_parser()
argv = sys.argv[1:]
args, _ = parser.parse_known_args(argv)

# Logging
if args.mode == 'Train':
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
else:
    current_time = args.test_time
if args.use_CNN == False:
    log_dir = 'Model_' + args.data_dir[-1] + '_' +  args.sym_group[0]+str(args.N)
else:
    log_dir = 'Model_' + args.data_dir[-1] + '_CNN'
logger = init_logging_handler(log_dir, current_time)
logger.setLevel(logging.DEBUG)

logging.debug(str(args))

device = get_device(args.use_cuda, args.cuda_idx)


#Initializing the Model

model = ECNN(logger,log_dir = log_dir, current_time = current_time, n_classes=3, sym_group = args.sym_group, N = args.N,device = device,lr=args.lr, use_CNN = args.use_CNN, mode = args.mode)


# images are padded to have shape 129x129.
# this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
pad = Pad((0, 0, 1, 1), fill=0)
# to reduce interpolation artifacts (e.g. when testing the model on rotated images),
# we upsample an image by a factor of 3, rotate it and finally downsample it again
resize1 = Resize(387)
resize2 = Resize(129)
# topil = transforms.ToPILImage()
totensor = ToTensor()
togray = transforms.Grayscale(num_output_channels=1)

if args.use_CNN:
    transform_train = transforms.Compose([
    transforms.RandomCrop(128),
    pad,
#     resize1,
#     RandomRotation(180, resample=Image.BILINEAR, expand=False),
#     resize2,
    totensor,
    togray,
])
    transform_test = transforms.Compose([
        transforms.RandomCrop(128),
        pad,
        totensor,
        togray,

])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(128),
        pad,
        resize1,
        RandomRotation(180, resample=Image.BILINEAR, expand=False),
        resize2,
        totensor,
        togray,
    ])
    transform_test = transforms.Compose([
        transforms.RandomCrop(128),
        pad,
        totensor,
        togray,

    ])

trainset = CustomDataset(args.data_dir+'/','train',transform_train)
testset = CustomDataset(args.data_dir+'/', 'test',transform=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)

total_epochs = args.epochs

if args.mode == 'Train':
    all_train_loss,all_test_loss,all_train_accuracy,all_test_accuracy = model.train(train_loader,test_loader,total_epochs)
    model.save_results()

model.plot_ROC(testset, test_loader)
model.plot_CM(testset, test_loader)

