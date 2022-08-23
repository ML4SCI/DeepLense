import torch.nn as nn
from activation_funcs import Mish_layer

class Regressor(nn.Module):
    
    def __init__(self, in_channels = 1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = 7, stride = 2),
            nn.BatchNorm2d(16),
            Mish_layer(),

            nn.Conv2d(16, 32, 7, stride = 2),
            nn.BatchNorm2d(32),
            Mish_layer(),
            
            nn.Conv2d(32, 64, 7),
            nn.BatchNorm2d(64),
            Mish_layer(),
        )
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
                                nn.Linear(2304, 1024),
                                Mish_layer(),
                                nn.BatchNorm1d(1024),
                                nn.Dropout(p = 0.3),
                                
                                nn.Linear(1024, 512),
                                Mish_layer(),
                                nn.BatchNorm1d(512),
                                nn.Dropout(p = 0.2),
        
                                nn.Linear(512, 256),
                                Mish_layer(),
                                nn.BatchNorm1d(256),
                                
                                nn.Linear(256, 1)
                                )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = Regressor()
    print(model)