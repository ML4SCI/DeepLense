import sys
import os
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader 
import torchvision
import torchvision.transforms as Transforms

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_ssl import CustomDataset
from models.ddpm import Diffusion
from models.autoencoder import Autoencoder
from models.unet_sa import UNet_linear_conditional, UNet_mass_em_conditional, UNet
from train.train_ddpm import Trainer

import matplotlib.pyplot as plt
# Set seed for PyTorch
torch.manual_seed(42)


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./ddpm_ssl_config.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
config = pipe.read_conf()
#pipe.log()
#print(config.unet.input_channels)


## Load model
model = UNet(config)
model.load_state_dict(torch.load('saved_models/ssl_ddpm_lenses.pt'))
model = model.to(device=config.device)

# Load diffusion 
diffusion = Diffusion(noise_steps=config.diff.noise_steps, beta_start=config.diff.beta_start, beta_end=config.diff.beta_end, img_size=config.data.image_size, device=config.device)

# Load the Dataset
transforms = Transforms.Compose([Transforms.CenterCrop(config.data.image_size)])
dataset = CustomDataset(root_dir=config.data.folder, transform=transforms)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=config.data.shuffle)

# # For batch data
for batch in data_loader:
    data = batch
    #mass = mass.to(config.device).float()
    data = data.to(config.device).float()
    #print(mass)
    # observed_values = model_latent.encode(mass)
    # print(observed_values)
    sample_images = diffusion.sample(model, 5)#, cfg_scale=0)
    #observed_values = model_res(sample_images)
    # plt.figure(figsize=(6,6), dpi=100)
    # fig, axs = plt.subplots(1, 2)
    # true = axs[0].imshow(data)
    # axs[0].set_title('True')
    # predic = axs[1].imshow(sample_images.squeeze(0).to('cpu').numpy())
    # axs[1].set_title('Predicted')
    grid = torchvision.utils.make_grid(sample_images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    plt.imshow(ndarr)
    plt.savefig(os.path.join("plots", f"ddpm_ssl_2.jpg"))
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
