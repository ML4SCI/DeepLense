import os
import torch 
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_model_2 import CustomDataset
import torch.nn.functional as F
from tqdm import tqdm 
from torchmetrics.image.fid import FrechetInceptionDistance

from torchmetrics import StructuralSimilarityIndexMeasure

latent_size = 64
device = 'cuda'

# Set seed for PyTorch
torch.manual_seed(42)

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./ddpm_config.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
config = pipe.read_conf()
#pipe.log()
#print(config.unet.input_channels)

# Load the Dataset
dataset = CustomDataset(root_dir=config.data.folder)
train_dl = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def calculate_psnr(real_images, generated_images, max_pixel_value=1.0):
    mse = F.mse_loss(generated_images, real_images)
    if mse == 0:
        return float('inf')

    psnr = 10*torch.log10(max_pixel_value **2/mse)
    return psnr

def calc_fid(model, checkpoint = None):
    fid = FrechetInceptionDistance(feature=2048, reset_real_features = True, normalize = True).to(device)
    if checkpoint != None:
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
    n = 1000
    z_dim = latent_size
    with torch.no_grad():
      z = torch.randn((1000, z_dim, 1, 1), device = device)
      fake_imgs = model(z).detach()

    
    image_list = []
    num_images_to_sample = 1000  # The number of images you want to sample
    total_sampled = 0
    
    # Sample a specific number of images (1000 in this example)
    for i, data in enumerate(train_dl):
        images = data
        image_list.append(images)
        total_sampled += images.size(0)  # Increment the total number of sampled images
        if total_sampled >= num_images_to_sample:
            break
    
    # Concatenate the list of images into a single tensor
    image_tensor = torch.cat(image_list[:num_images_to_sample], dim=0)
    real_imgs = image_tensor[0:1000, :, :, :]
    
    real_imgs = real_imgs.to(device)
    real_imgs_rgb = convert_to_rgb(real_imgs, device)  # Convert to RGB
    fake_imgs_rgb = convert_to_rgb(fake_imgs, device)  # Convert to RGB
    fid.update(real_imgs_rgb, real=True)
    fid.update(fake_imgs_rgb, real=False)
    #fid.update(real_imgs, real=True)
    #fid.update(fake_imgs, real=False)
    score = fid.compute()
    # return score/len(real_data_loader)
    return score

def calc_psnr(model, checkpoint = None):
    #fid = FrechetInceptionDistance(feature=2048, reset_real_features = True, normalize = True).to(device)
    if checkpoint != None:
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
    n = 1000
    z_dim = latent_size
    with torch.no_grad():
      z = torch.randn((1000, z_dim, 1, 1), device = device)
      fake_imgs = model(z).detach()

    
    image_list = []
    num_images_to_sample = 1000  # The number of images you want to sample
    total_sampled = 0
    
    # Sample a specific number of images (1000 in this example)
    for i, data in enumerate(train_dl):
        images = data
        image_list.append(images)
        total_sampled += images.size(0)  # Increment the total number of sampled images
        if total_sampled >= num_images_to_sample:
            break
    
    # Concatenate the list of images into a single tensor
    image_tensor = torch.cat(image_list[:num_images_to_sample], dim=0)
    real_imgs = image_tensor[0:1000, :, :, :]
    
    real_imgs = real_imgs.to(device)
    score = calculate_psnr(real_imgs, fake_imgs)
    print(real_imgs.shape)
    print(fake_imgs.shape)
    return score

def calc_ssim(model, checkpoint = None):
    #fid = FrechetInceptionDistance(feature=2048, reset_real_features = True, normalize = True).to(device)
    if checkpoint != None:
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
    n = 100
    z_dim = latent_size
    with torch.no_grad():
      z = torch.randn((100, z_dim, 1, 1), device = device)
      fake_imgs = model(z).detach()

    
    image_list = []
    num_images_to_sample = 100  # The number of images you want to sample
    total_sampled = 0
    
    # Sample a specific number of images (1000 in this example)
    for i, data in enumerate(train_dl):
        images = data
        image_list.append(images)
        total_sampled += images.size(0)  # Increment the total number of sampled images
        if total_sampled >= num_images_to_sample:
            break
    
    # Concatenate the list of images into a single tensor
    image_tensor = torch.cat(image_list[:num_images_to_sample], dim=0)
    real_imgs = image_tensor[0:100, :, :, :]
    
    real_imgs = real_imgs.to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    score = ssim_metric(real_imgs, fake_imgs)
    # print(real_imgs.shape)
    # print(fake_imgs.shape)
    return score

def calc_mse(model, checkpoint = None):
    #fid = FrechetInceptionDistance(feature=2048, reset_real_features = True, normalize = True).to(device)
    if checkpoint != None:
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
    n = 100
    z_dim = latent_size
    with torch.no_grad():
      z = torch.randn((100, z_dim, 1, 1), device = device)
      fake_imgs = model(z).detach()

    
    image_list = []
    num_images_to_sample = 1000  # The number of images you want to sample
    total_sampled = 0
    
    # Sample a specific number of images (1000 in this example)
    for i, data in enumerate(train_dl):
        images = data
        image_list.append(images)
        total_sampled += images.size(0)  # Increment the total number of sampled images
        if total_sampled >= num_images_to_sample:
            break
    
    # Concatenate the list of images into a single tensor
    image_tensor = torch.cat(image_list[:num_images_to_sample], dim=0)
    real_imgs = image_tensor[0:100, :, :, :]
    
    real_imgs = real_imgs.to(device)
    #ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    score = F.mse_loss(real_imgs, fake_imgs)
    # print(real_imgs.shape)
    # print(fake_imgs.shape)
    return score
def convert_to_rgb(images, device):
    
        colormap = cm.viridis
        input_rgb_list = []
        for image in images:
            # Apply colormap
            image = image.cpu()
            input_rgb = colormap(image.numpy())  
            # Keep only RGB channels
            input_rgb = input_rgb[0, :, :, :3]
            # Convert numpy array back to tensor and permute dimensions to (channels, height, width)
            input_rgb_tensor = torch.from_numpy(input_rgb.astype(np.float32)).permute(2, 0, 1)
            input_rgb_list.append(input_rgb_tensor)
        
        # Stack the list of tensors along the batch dimension
        input_rgb_batch = torch.stack(input_rgb_list, dim=0).to(device)
    
        return input_rgb_batch



# Model
generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 256, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=False),
    # nn.BatchNorm2d(16),
    # nn.ReLU(True),
    # nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Sigmoid()
)
generator = to_device(generator, device)

checkpoint = 'scripts/checkpointsG/1/g_checkpoint_1_epoch_91.pth'

score = calc_mse(generator, checkpoint=checkpoint)
print("psnr Score:", score)