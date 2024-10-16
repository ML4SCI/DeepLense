import sys
import os
import torch

from torch import nn
from torch.utils.data import DataLoader 
import torchvision.transforms as Transforms

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_ssl import CustomDataset
from models.unet_sa import UNet_conditional, UNet
from models.ddpm import Diffusion

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

# Load the Dataset
# transforms = Transforms.Compose([Transforms.CenterCrop(config.data.image_size)])
# dataset = CustomDataset(root_dir=config.data.folder, config=config, transform=transforms)
# data_loader = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

eval_transforms = Transforms.Compose([
                # Transforms.ToTensor(), # npy loader returns torch.Tensor
                Transforms.CenterCrop(64),
                Transforms.Normalize(mean = [0.06814773380756378, 0.21582692861557007, 0.4182431399822235],\
                                        std = [0.16798585653305054, 0.5532506108283997, 1.1966736316680908]),
            ])
dataset = CustomDataset(root_dir=config.data.folder, config=config, transform=eval_transforms)
data_loader = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

# Load model
model = UNet(config)
model.load_state_dict(torch.load('saved_models/sup_ddpm_lenses_mean.pt'))#, map_location=torch.device('cpu'))
model = model.to(device=config.device)


# Load diffusion 
diffusion = Diffusion(noise_steps=config.diff.noise_steps, beta_start=config.diff.beta_start, beta_end=config.diff.beta_end, img_size=config.data.image_size, device=config.device)
#labels = torch.ones([8,], dtype=torch.long).to(config.device)


# Sampling Images
# labels = torch.zeros([8,], dtype=torch.long).to(config.device)
# sampled_images = diffusion.sample_conditional(model, n=8, labels=labels)
# diffusion.save_images(sampled_images, os.path.join("plots", f"no_sub_420.jpg"))

# Calculating Fid scores
FID_Score = diffusion.cal_fid(model=model, train_dl=data_loader, device=config.device)
print("FID score: ", FID_Score)
