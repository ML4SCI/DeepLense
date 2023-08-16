import timm
import torch.nn as nn
import torch.nn as nn
from torchvision.models import resnet18, resnet
from utils.util import check_trainable_layers, get_second_last_layer
from torchsummary import summary
from typing import Optional
import torch


class Model(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=1
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lin = nn.Linear(1408, 3)
        self.do = nn.Dropout(p=0.5)

    def forward(self, image):
        image = self.backbone(image)
        image = self.pool(image)

        image = image.view(image.shape[0], -1)
        image = self.do(image)
        image = self.lin(image)
        return image


class ViT(nn.Module):
    def __init__(self, model="convit_base", num_classes=3, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model, num_classes=0, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class ConViT(nn.Module):
    def __init__(self, model="convit_base", num_classes=3, pretrained=False):
        super().__init__()
        assert model in [
            "convit_base",
            "convit_small",
            "convit_tiny",
        ], "Wrong model name check timm libray for more models"
        self.model = timm.create_model(model, pretrained=pretrained, in_chans=1)
        num_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.PReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class CustomResNet(nn.Module):
    def __init__(
        self,
        num_channels,
        device: Optional[str] = "cuda",
    ):
        super(CustomResNet, self).__init__()
        self.model = resnet18(pretrained=False)
        self.device = device
        self.num_channels = num_channels

        # TODO: add conv layer depending on channel else use pretrained model
        self.model.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.to(device)

    def add_head(self, head: Optional[nn.Module] = None, freeze_backbone=False):
        requires_grad = not freeze_backbone
        for param in self.model.parameters():
            param.requires_grad = requires_grad

        self.model.fc = nn.Identity() if head is None else head
        self.model.fc.to(self.device)
        return self.model
        # self.model.to(self.device)

    def append_layer(self, layer, freeze_backbone=True):
        requires_grad = not freeze_backbone

        self.model = nn.Sequential(self.model, layer)
        # Freezing network Sequential at index 0
        self.model[0].requires_grad_(requires_grad)
        return self.model

    def forward(self, x):
        return self.model(x)

    def get_random_input(self):
        return torch.randn(
            self.get_input_size()[0],
            self.get_input_size()[1],
            self.get_input_size()[2],
            self.get_input_size()[3],
        ).to(self.device)

    def get_input_size(self):
        return (10, self.num_channels, 224, 224)

    def summarize(self):
        # summarize whole network
        summary(
            self.model, input_size=self.get_input_size(), device=self.device
        )  # torch.randn(1, 3, 224, 224)

    def inspect_layers(self):
        # check trainable layers
        check_trainable_layers(self.model)

    def freeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_last_layer_features(self):
        output = self.model(self.get_random_input())
        num_last_features = output.size(1)

        return num_last_features

    def get_fc_in_features(self):
        return self.model.fc.in_features
    
    def get_second_last_in_features(self):
        layer = get_second_last_layer(self.model)
        return layer

    

if __name__ == "__main__":
    resnet = CustomResNet(num_channels=1)
    resnet.to(device="cuda")
    resnet.inspect_layers()
