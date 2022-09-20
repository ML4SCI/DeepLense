import timm
import torch.nn as nn


class EffNetB1_backbone_model(nn.Module):
    
    def __init__(self, pretrained = True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b1', pretrained = pretrained, in_chans = 1)

        self.fc = nn.Sequential(
                                nn.Linear(1280 * 2 * 2, 1024),
                                nn.PReLU(),
                                nn.BatchNorm1d(1024),
                                nn.Dropout(p = 0.5),
                                
                                nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.PReLU(),
                                nn.Dropout(p = 0.5),
        
                                nn.Linear(512, 128),
                                nn.PReLU(),
                                nn.BatchNorm1d(128),
                                nn.Dropout(p = 0.3),
                                
                                nn.Linear(128, 3)
                                )
        
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.view(-1, 1280 * 2 * 2)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = EffNetB1_backbone_model(pretrained = True)
    print(model)
