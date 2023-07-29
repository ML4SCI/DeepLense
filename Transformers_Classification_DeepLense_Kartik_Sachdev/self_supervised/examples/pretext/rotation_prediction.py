import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the rotation prediction task model
class RotationModel(nn.Module):
    def __init__(self, num_classes):
        super(RotationModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer
        self.rotation_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        predictions = self.rotation_classifier(features)
        return predictions

# Set hyperparameters
batch_size = 32
epochs_pretrain = 10
epochs_finetune = 10
learning_rate_pretrain = 0.001
learning_rate_finetune = 0.001
num_classes = 10  # Number of classes for classification

# Load dataset for pretext task (rotation prediction)
pretrain_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])
pretrain_dataset = datasets.CIFAR10('path_to_pretrain_dataset', train=True, download=True, transform=pretrain_transform)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

# Load dataset for fine-tuning (classification)
finetune_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])
finetune_dataset = datasets.CIFAR10('path_to_finetune_dataset', train=True, download=True, transform=finetune_transform)
finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)

# Pretraining phase: rotation prediction
pretrain_model = RotationModel(num_classes)
pretrain_model.to(device)

criterion_pretrain = nn.CrossEntropyLoss()
optimizer_pretrain = optim.Adam(pretrain_model.parameters(), lr=learning_rate_pretrain)

# Pretraining loop
for epoch in range(epochs_pretrain):
    for batch_idx, (images, _) in enumerate(pretrain_loader):
        images = images.to(device)
        
        optimizer_pretrain.zero_grad()
        predictions = pretrain_model(images)
        loss = criterion_pretrain(predictions, torch.arange(4).repeat(images.size(0))[:images.size(0)].to(device))
        loss.backward()
        optimizer_pretrain.step()
        
        if batch_idx % 100 == 0:
            print(f"Pretraining Epoch [{epoch}/{epochs_pretrain}], Batch [{batch_idx}/{len(pretrain_loader)}], Loss: {loss.item()}")

# Fine-tuning phase: classification
finetune_model = RotationModel(num_classes)
finetune_model.to(device)

criterion_finetune = nn.CrossEntropyLoss()
optimizer_finetune = optim.Adam(finetune_model.parameters(), lr=learning_rate_finetune)

# Fine-tuning loop
for epoch in range(epochs_finetune):
    for batch_idx, (images, labels) in enumerate(finetune_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer_finetune.zero_grad()
        predictions = finetune_model(images)
        loss = criterion_finetune(predictions, labels)
        loss.backward()
        optimizer_finetune.step()
        
        if batch_idx % 100 == 0:
            print(f"Fine-tuning Epoch [{epoch}/{epochs_finetune}], Batch [{batch_idx}/{len(finetune_loader)}], Loss: {loss.item()}")

# Save the fine-tuned model
torch.save(finetune_model.state_dict(), 'selfsupervised_classification_model.pth')