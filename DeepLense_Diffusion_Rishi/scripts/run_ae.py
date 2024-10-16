import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_md_model2 import NumbersDataset
from models.autoencoder import Autoencoder
from train.train_ae import Trainer


# Set seed for PyTorch
torch.manual_seed(42)

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./ae_md_config.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
config = pipe.read_conf()

# Load the Dataset
dataset = NumbersDataset(main_folder_path=config.data.folder)
train_size = config.data.train_test_split
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, train_size=train_size, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

# Load Model
model = Autoencoder(latent_dim = config.ae.latent_dim, hidden_dim = config.ae.hidden_dim, input_dim =config.ae.input_dim)
model = model.to(config.device)

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