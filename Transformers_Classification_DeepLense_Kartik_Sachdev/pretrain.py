import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import DefaultDatasetSetup
from models.cnn_zoo import CustomResNet
from utils.losses.contrastive_loss import ContrastiveLossEuclidean, ContrastiveLossEmbedding
from utils.train import train_contrastive_with_labels, train_contrastive

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_method = "contrastive_embedding"
saved_model_path = f"output/pretrained_{learning_method}.pth"

# Set hyperparameters
batch_size = 128
epochs = 10
learning_rate = 0.001
margin = 1.0
num_channels = 1 
temperature = 0.5

# Load dataset
default_dataset_setup = DefaultDatasetSetup()
train_dataset = default_dataset_setup.get_default_trainset_ssl()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Create model
head = nn.Identity()
model = CustomResNet(num_channels, head=head)
model.to(device)
model.inspect_layers()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = ContrastiveLossEmbedding(margin, device=device)

# Training loop
# train_contrastive_with_labels(epochs, model, device, train_loader, criterion, optimizer, saved_model_path)
train_contrastive(epochs, model, device, train_loader, criterion, optimizer, saved_model_path)
