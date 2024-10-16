import sys
import os
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader 
from torchvision import models 

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_all_model2 import CustomDataset, CustomDataset_1
from models.ddpm_all import Diffusion
from models.autoencoder import Autoencoder
from models.unet_all import UNet_all_conditional
from train.train_ae_ddpm import Trainer

import matplotlib.pyplot as plt
# Set seed for PyTorch
torch.manual_seed(42)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./conditional_ddpm_config.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
config = pipe.read_conf()
#pipe.log()
#print(config.unet.input_channels)


## Load model
# Auto encoder
# model_latent = Autoencoder(latent_dim = config.ae.latent_dim, hidden_dim = config.ae.hidden_dim, input_dim =config.ae.input_dim)
# model_latent.load_state_dict(torch.load('saved_models/ae_log_md_bestmodel.pt'))
# model_latent = model_latent.to(config.device)
# model_latent.eval()
# Conditional DDPM
model = UNet_all_conditional(config)
model.load_state_dict(torch.load('saved_models/all_conditional_ckpt_model2.pt'))
model = model.to(device=config.device)
# Load resnet
# Model 
# model_res = models.resnet18(pretrained=False)
# model_res.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# num_ftrs = model_res.fc.in_features
# model_res.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
# model_res.load_state_dict(torch.load('saved_models/resnet_new_v4_bestmodel.pt'))
# model_res.to(config.device)
# model_res.eval()

model_res = models.wide_resnet50_2(pretrained=False)
#model = models.resnet18(pretrained=True)
model_res.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model_res.fc.in_features
model_res.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
model_res.to(config.device)

# class Encoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.input_channels = config.unet.input_channels
#         self.latent_dimension = config.unet.latent_dimension

#         channels = [config.unet.input_channels, 8, 16, 32, 64, 128, 256,  2*config.unet.latent_dimension]  # Shape B,2*z_dim,1,1

#         layers = []

#         default_activation = nn.LeakyReLU(0.2, inplace=True)

#         for i in range(len(channels)-1):
#             layers.append(nn.Conv2d(channels[i], channels[i+1], 3, 2, 1))
#             layers.append(nn.BatchNorm2d(channels[i+1]))
#             activation = default_activation if i < len(channels) -2 else nn.Identity()
#             layers.append(activation)

#         layers.append(View((-1, 2*self.latent_dimension)))

#         self.encoder = nn.Sequential(*layers)
#         self.linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(2*config.unet.latent_dimension, 1))

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.linear(x)
#         return x#[:,:self.latent_dimension], x[:,self.latent_dimension:]


# model_res = Encoder(config)
model_res.load_state_dict(torch.load('saved_models/resnet_new_v4_bestmodel.pt'))
model_res.to(config.device)

# Load diffusion 
diffusion = Diffusion(noise_steps=config.diff.noise_steps, beta_start=config.diff.beta_start, beta_end=config.diff.beta_end, img_size=config.data.image_size, device=config.device)

# Load Data
# Load the Dataset
dataset = CustomDataset(root_dir=config.data.folder, config=config)#, max_samples=config.data.max_samples, config=config)
data_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=config.data.shuffle)
#print(data_loader)
dataset_1 = CustomDataset_1(root_dir=config.data.folder, config=config)
data_loader_1 = DataLoader(dataset=dataset_1, batch_size=100, shuffle=config.data.shuffle)

for batch_1 in data_loader_1:
    data, v1, v2, v3, v4 = batch_1
    v4_1 = v4.to(config.device).float()
    break

# For batch data
for batch in data_loader:
    data, v1, v2, v3, v4 = batch
    v1 = v1.to(config.device).float()
    v2 = v2.to(config.device).float()
    v3 = v3.to(config.device).float()
    v4 = v4.to(config.device).float()
    data = data.to(config.device).float()
    #print(v4)
    #print(mass)
    # observed_values = model_latent.encode(mass)
    # print(observed_values)
    sample_images = diffusion.sample_conditional(model, 100, v1, v2, v3, v4, cfg_scale=0)
    observed_values = model_res(sample_images)

    ## Plot for true and predicted
    # fig, axs = plt.subplots(1, 2)
    # true = axs[0].imshow(data.squeeze(0).squeeze(0).to('cpu').numpy())
    # axs[0].set_title('True')
    # predic = axs[1].imshow(sample_images.squeeze(0).squeeze(0).to('cpu').numpy())
    # axs[1].set_title('Predicted')
    # # grid = torchvision.utils.make_grid(sample_images)
    # # ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    # # plt.imshow(ndarr)
    # plt.savefig(os.path.join("plots", f"all_variables.jpg"))
    # plt.figure(figsize=(6,6), dpi=100)
    # Renormalise 
    #mass = mass.to('cpu').numpy()
    #observed_values = observed_values.to('cpu').detach().numpy()
    #mass = mass*(config.data.max_value-config.data.min_value) + config.data.min_value
    #observed_values = observed_values*(config.data.max_value-config.data.min_value) + config.data.min_value
    plt.scatter(v4_1.to('cpu').numpy(), observed_values.to('cpu').detach().numpy(), color='black')
    #plt.scatter(, observed_values, color='black')
    plt.xlabel('Observed variable 4')
    plt.ylabel('Predicted variable 4')
    plt.savefig(os.path.join("plots", f"all_new_variables_4.jpg"))
    break
# Load mass and axion 
# dir = '../Data/Model_II/axion/axion_sim_356113875435765399182650162198846.npy'
# #dir = '../Data/Model_II/axion/axion_sim_76051909974444833968539282987196377.npy'
# data = np.load(dir, allow_pickle=True)
# data_input = (data[0] - np.min(data[0]))/((np.max(data[0])-np.min(data[0])))
# labels = torch.tensor(np.log(data[1])).to(config.device)
# print(labels.unsqueeze(0).unsqueeze(0))
# sample_images = diffusion.sample_conditional(model, 1, labels)
# print(sample_images)
#print(model_latent(labels))


# Load resnet
# Model 
# model_res = models.resnet18(pretrained=False)
# model_res.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# num_ftrs = model_res.fc.in_features
# model_res.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
# model_res.load_state_dict(torch.load('saved_models/resnet_log_md_bestmodel.pt'))
# model_res.eval()
# #print(data_input)
# data_input = torch.from_numpy(data_input).float()
# print(model_res(data_input.unsqueeze(0).unsqueeze(0)))


## Plot for true and predicted
# fig, axs = plt.subplots(1, 2)
# true = axs[0].imshow(data_input)
# axs[0].set_title('True')
# predic = axs[1].imshow(sample_images.squeeze(0).squeeze(0).to('cpu').numpy())
# axs[1].set_title('Predicted')
# # grid = torchvision.utils.make_grid(sample_images)
# # ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
# # plt.imshow(ndarr)
# plt.savefig(os.path.join("plots", f"mass_distribution_axion_5.jpg"))