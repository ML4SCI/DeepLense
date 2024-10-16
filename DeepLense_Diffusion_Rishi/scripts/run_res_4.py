import torch
import torch.nn as nn

from torchvision import models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_all_model2 import CustomDataset, CustomDataset_v1
from train.train_resnet import Trainer

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

# Load the Dataset
dataset = CustomDataset_v1(root_dir=config.data.folder, config= config)#, max_samples=config.data.max_samples, config=config)
train_size = config.data.train_test_split
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, train_size=train_size, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

# Model 
model = models.wide_resnet50_2(pretrained=False)
#model = models.resnet18(pretrained=True)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
model.to(config.device)

#model = models.mobilenet_v2(pretrained=False)
#model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#num_ftrs = model.fc.in_features
#model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
#model.to(config.device)

# class Encoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.input_channels = config.unet.input_channels
#         self.latent_dimension = config.unet.latent_dimension

#         channels = [config.unet.input_channels, 8, 16, 32, 64, 128,  2*config.unet.latent_dimension]  # Shape B,2*z_dim,1,1

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


# model = Encoder(config)
# model.to(config.device)

# Create Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.opt.lr,
    eps=config.opt.eps,
)

if config.opt.scheduler == "OneCycleLR":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = config.opt.lr,
        steps_per_epoch = len(train_data_loader),
        epochs = config.opt.epochs,
    ) 
elif config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Create the loss
mse = nn.MSELoss()

trainer = Trainer(model=model, config=config)

if config.verbose:
    print("\n### MODEL ###\n", print(model))
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULAR ###\n", scheduler)
    print("\n### LOSSES ###\n")
    print("\n### Beginning Training ...\n")


# Train
trainer.train(train_data_loader=train_data_loader, test_data_loader=test_data_loader, mse=mse, optimizer=optimizer, scheduler=scheduler)