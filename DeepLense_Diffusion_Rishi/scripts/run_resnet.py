import torch
import pickle 
import torch.nn as nn
from torchvision import models

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from models.unet_sa import UNet_conditional, UNet_linear_conditional
from models.ddpm import Diffusion

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

model_ft = models.resnet18(pretrained=False)
model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 3))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model_2_path = "saved_models/ResNet18_Model2.pth"
model_ft = torch.load(resnet_model_2_path)#, map_location=device)
model = model_ft.to(device)
model.eval()

# Load model
model_diffusion = UNet_linear_conditional(config)
model_diffusion.load_state_dict(torch.load('saved_models/new_label_conditional_ckpt_model2.pt'))#, map_location=torch.device('cpu'))
model_diffusion = model_diffusion.to(device=config.device)
# Load diffusion 
diffusion = Diffusion(noise_steps=config.diff.noise_steps, beta_start=config.diff.beta_start, beta_end=config.diff.beta_end, img_size=config.data.image_size, device=config.device)
#labels = torch.ones([8,], dtype=torch.long).to(config.device)
labels_axion = torch.ones([5,],dtype=torch.long).to(device)
samples = diffusion.sample_conditional(model_diffusion, 5, labels_axion)
print(samples.shape)
output = model(samples)
print(output)