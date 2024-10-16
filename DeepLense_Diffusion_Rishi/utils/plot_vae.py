import sys
import os
import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader 
import torchvision
import torchvision.transforms as Transforms

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_model_2 import CustomDataset
from models.vae import vae
from train.train_vae import Trainer
from models.vae_sample import plot, sample, reconstruction

import matplotlib.pyplot as plt
# Set seed for PyTorch
torch.manual_seed(42)


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./vae_config.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
config = pipe.read_conf()
#pipe.log()
#print(config.unet.input_channels)

# Load the Dataset
# dataset = CustomDataset(root_dir=config.data.folder)
# data_loader = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

## Load model
model = vae(config)
model.load_state_dict(torch.load('saved_models/new_vae_cdm.pt'))
model = model.to(device=config.device)

dataset = CustomDataset(root_dir=config.data.folder)#, transform=transforms)
data_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=config.data.shuffle)


#reconstruction(model,trainset=dataset)
#sampled_images = sample(model=model)
# # For batch data
for batch in data_loader:
    data = batch
    #mass = mass.to(config.device).float()
    data = data.to(config.device).float()
    output, _, _ = model(data)
    sample(model=model)
    plot(output.cpu(), save_path="plots/vae_recon3")
    plot(data.cpu(), save_path="plots/vae_original3")
    #print(output)
    # plt.figure(figsize=(6,6), dpi=100)
    # fig, axs = plt.subplots(1, 2)
    # true = axs[0].imshow(data.squeeze(0).squeeze(0).to('cpu').numpy())
    # axs[0].set_title('True')
    # predic = axs[1].imshow(output.squeeze(0).squeeze(0).to('cpu').detach().numpy())
    # axs[1].set_title('Reconstruction')
    # plt.savefig(os.path.join("plots", f"vae_recon8.jpg"))
    break
