import timm
import torch.nn as nn


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
        self.model.head =  nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.PReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x
