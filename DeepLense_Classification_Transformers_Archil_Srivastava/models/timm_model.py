import torch
from torch.nn import (
    Sequential,
    Linear,
    PReLU,
    BatchNorm1d,
    Dropout,
    Flatten,
)
import timm

class TimmModelSimple(torch.nn.Module):
    def __init__(self, name, in_chans=1, num_classes=3, pretrained=True, tune=True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes)

        # Freeze or tune entire model
        for param in self.backbone.parameters():
            param.requires_grad = tune  

        # Ensure classifier layers are always trainable
        for param in self.backbone.get_classifier().parameters():
            param.requires_grad = True

        self.num_classes = num_classes

    def get_representation_features(self):
        return self.backbone.get_classifier().in_features

    def forward(self, img):
        return self.backbone(img)

    def print_model_summary(self, input_size=(1, 1, 224, 224)):
        from torchinfo import summary
        summary(self, input_size)

class TimmModelComplex(TimmModelSimple):
    def __init__(self, *args, dropout_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate

        # Remove old classifier
        self.backbone.reset_classifier(0)

        # Multi-layer classifier
        self.classifier = Sequential(
            Flatten(),
            Linear(self.get_representation_features(), 1024),
            PReLU(),
            BatchNorm1d(1024),
            Dropout(self.dropout_rate),
            Linear(1024, 512),
            PReLU(),
            BatchNorm1d(512),
            Dropout(self.dropout_rate),
            Linear(512, 128),
            PReLU(),
            BatchNorm1d(128),
            Dropout(self.dropout_rate * 0.6),
            Linear(128, self.num_classes),
        )

    def forward(self, img):
        representations = self.backbone(img)
        return self.classifier(representations)
