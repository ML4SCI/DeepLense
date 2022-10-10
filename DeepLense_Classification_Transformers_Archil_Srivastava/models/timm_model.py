import torch
from torch.nn import Sequential, ModuleList, Linear, PReLU, BatchNorm1d, Dropout, Identity, LazyLinear, Flatten
import timm

class TimmModelSimple(torch.nn.Module):
    def __init__(self, name, *args, in_chans=1, num_classes=3, pretrained=True, tune=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = timm.create_model(name, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes)
        for param in self.backbone.parameters():
            param.requires_grad = tune
        if type(self.backbone.get_classifier()) == tuple or type(self.backbone.get_classifier()) == list:
            for component in self.backbone.get_classifier():
                for param in component.parameters():
                    param.requires_grad = True
        else:
            for param in self.backbone.get_classifier().parameters():
                param.requires_grad = True
        
        self.num_classes = num_classes
    
    def get_representation_features(self):
        return self.backbone.get_classifier().in_features
    
    def forward(self, img):
        return self.backbone(img)


class TimmModelComplex(TimmModelSimple):
    def __init__(self, *args, dropout_rate=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        # in_features = self.backbone.get_classifier().in_features
        # self.backbone.reset_classifier(0)
        self.classifier = Sequential(
                            Flatten(),
                            LazyLinear(1024),
                            PReLU(),
                            BatchNorm1d(1024),
                            Dropout(0.5),
                            
                            Linear(1024, 512),
                            BatchNorm1d(512),
                            PReLU(),
                            Dropout(0.5),
    
                            Linear(512, 128),
                            PReLU(),
                            BatchNorm1d(128),
                            Dropout(0.3),
                            
                            Linear(128, self.num_classes)
                            )
    
    def forward(self, img):
        representations = self.backbone(img)
        return self.classifier(representations)
