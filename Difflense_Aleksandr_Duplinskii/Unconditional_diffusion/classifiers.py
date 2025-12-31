import torch.nn as nn
import torchvision.models as models
import torch


class SmallCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32×32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16×16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1×1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    
class MediumCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),  # 32×32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),  # 16×16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1×1
        )
        
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
def get_resnet18_grayscale(num_classes, dropout_p=0.5):
    model = models.resnet18(weights='IMAGENET1K_V1')  

    original_conv = model.conv1

    new_conv = nn.Conv2d(
    in_channels=1,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias is not None
    )

    with torch.no_grad():
        new_conv.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

    model.conv1 = new_conv
    
    
        # Add dropout before final linear layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),         # 👈 Dropout goes here
        nn.Linear(in_features, num_classes)
    )

    return model

import torch.nn as nn
from torchvision import models

class ResNet18GrayscaleFD(nn.Module):
    def __init__(self, num_classes=3, bottleneck_dim=32, dropout_p=0.5, use_classifier=True):
        super().__init__()
        base_model = models.resnet18(weights='IMAGENET1K_V1')

        # Convert input to grayscale (1 channel)
        original_conv = base_model.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
        base_model.conv1 = new_conv

        # Remove original classifier
        in_features = base_model.fc.in_features
        base_model.fc = nn.Identity()  # We take control

        self.backbone = base_model
        self.feature_bottleneck = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, bottleneck_dim),
            # nn.GELU()
        )

        self.classifier_head = nn.Linear(bottleneck_dim, num_classes) if use_classifier else nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = self.feature_bottleneck(x)
        x = self.classifier_head(x)
        return x
