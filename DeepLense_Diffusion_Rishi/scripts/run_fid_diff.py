import sys
import os
import torch

from torch import nn
from torch.utils.data import DataLoader 
import torchvision.transforms as Transforms

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_model_2 import CustomDataset
from models.unet_sa import UNet_conditional, UNet
from models.ddpm import Diffusion

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
data_loader = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

# Load model
model = UNet(config)
model.load_state_dict(torch.load('saved_models/ddpm_cdm_100.pt'))#, map_location=torch.device('cuda'))
model = model.to(device=config.device)


# Load diffusion 
diffusion = Diffusion(noise_steps=config.diff.noise_steps, beta_start=config.diff.beta_start, beta_end=config.diff.beta_end, img_size=config.data.image_size, device=config.device)
#labels = torch.ones([8,], dtype=torch.long).to(config.device)


# Sampling Images
# labels = torch.zeros([8,], dtype=torch.long).to(config.device)
# sampled_images = diffusion.sample_conditional(model, n=8, labels=labels)
# diffusion.save_images(sampled_images, os.path.join("plots", f"no_sub_420.jpg"))

# Calculating Fid scores
FID_Score = diffusion.cal_ssim(model=model, train_dl=data_loader, device=config.device)
print("FID score: ", FID_Score)
