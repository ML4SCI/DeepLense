import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import DefaultDatasetSetup
from models.cnn_zoo import CustomResNet
from utils.losses.contrastive_loss import ContrastiveLossEuclidean
from utils.train import train_simplistic
from utils.util import load_model_add_head
from torchsummary import summary

# Set device
device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_method = "contrastive_embedding"
saved_model_path = "/home/kartik/git/deepLense_transformer_ssl/output/pretrained_contrastive_embedding.pth"

# Set hyperparameters
batch_size = 128
epochs = 10
learning_rate = 0.001
margin = 1.0
num_channels = 1
temperature = 0.5
num_classes = 2

# Load dataset
default_dataset_setup = DefaultDatasetSetup()
train_dataset = default_dataset_setup.get_default_trainset_ssl()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load pretrained model and add head
pretrain_head = nn.Identity()
pretrain_model = CustomResNet(num_channels, head=pretrain_head)
pretrain_model.summarize()

in_features = pretrain_model.get_last_layer_features()
finetune_head = nn.Sequential(
    nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, num_classes)
)

model = load_model_add_head(
    pretrain_model=pretrain_model,
    saved_model_path=saved_model_path,
    head=finetune_head,
    freeze_pretrain_layers=True,
)

model.to(device)
summary(model=model, input_size=(1, 1, 224, 224), device=device)


# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
train_simplistic(
    epochs, model, device, train_loader, criterion, optimizer, saved_model_path
)
