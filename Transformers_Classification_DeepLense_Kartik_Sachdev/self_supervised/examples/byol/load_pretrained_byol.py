import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import DeepLenseDatasetSSL, DefaultDatasetSetupSSL
from torch.utils.data import DataLoader, random_split

from models.cnn_zoo import CustomResNet
from utils.losses.contrastive_loss import ContrastiveLossEuclidean
from utils.train import train_simplistic
from utils.util import (
    load_model_add_head,
    get_second_last_layer,
    get_last_layer_features,
)
from torchsummary import summary
from models.byol import BYOLSingleChannel, FinetuneModelByol
import torchvision
from models.utils.finetune_model import FinetuneModel

# Set device
device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_method = "contrastive_embedding"
pretrained_model_path = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/2023-07-23-13-30-24/checkpoint/Resnet_finetune_Model_II_2023-07-23-13-30-24.pt"
finetuned_model_path = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/2023-07-23-13-30-24/checkpoint/Resnet_finetune_Model_II.pt"

# Set hyperparameters
batch_size = 512
epochs = 10
learning_rate = 0.0001
margin = 1.0
num_channels = 1
temperature = 0.5
num_classes = 3

# Load dataset
default_dataset_setup = DefaultDatasetSetupSSL(dataset_name="Model_II")
train_dataset = default_dataset_setup.get_dataset()

# split in train and valid set
split_ratio = 0.25  # 0.25
valid_len = int(split_ratio * len(train_dataset))
train_len = len(train_dataset) - valid_len

train_dataset, val_set = random_split(train_dataset, [train_len, valid_len])

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
)

val_loader = DataLoader(
    dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=8
)


# Load pretrained model and add head
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = BYOLSingleChannel(backbone, num_ftrs=512)
model.to(device)
input_feature = 256
finetune_head = nn.Sequential(
    nn.Linear(input_feature, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(512, input_feature),
    nn.BatchNorm1d(input_feature),
    nn.ReLU(),
    nn.Linear(input_feature, num_classes),
)
finetune_model = FinetuneModelByol(backbone=model, head=finetune_head)
finetune_model.to(device=device)
finetune_model.load_state_dict(torch.load(pretrained_model_path))
print(">>>> Keys matched")
summary(finetune_model, input_size=(10, 1, 224, 224), device=device)


# Define optimizer and loss function
optimizer = optim.Adam(finetune_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
train_simplistic(
    epochs,
    finetune_model,
    device,
    train_loader,
    criterion,
    optimizer,
    finetuned_model_path,
    valid_loader=val_loader,
)
