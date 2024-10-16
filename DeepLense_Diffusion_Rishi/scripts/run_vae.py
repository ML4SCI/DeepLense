import sys
import torch

from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_model_2 import CustomDataset
from models.vae import vae
from models.vae_sample import loss_beta
from train.train_vae import Trainer

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
dataset = CustomDataset(root_dir=config.data.folder)
#data_loader = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

train_size = config.data.train_test_split
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, train_size=train_size, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)
# Load model
model = vae(config)
model = model.to(device=config.device)

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
vae_loss = loss_beta()

trainer = Trainer(model=model, config=config, dataset=dataset)

if config.verbose:
    print("\n### MODEL ###\n", print(model))
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULAR ###\n", scheduler)
    print("\n### LOSSES ###\n")
    print("\n### Beginning Training ...\n")


# Train
trainer.train(train_data_loader=train_data_loader, test_data_loader=test_data_loader, vae_loss=vae_loss, optimizer=optimizer, scheduler=scheduler)