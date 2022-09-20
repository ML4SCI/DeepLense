import torch.nn as nn
from CBAM import CBAMBlock
from activation_funcs import Mish_layer

class Regressor(nn.Module):
    
    def __init__(self, in_channels = 1):
        super().__init__()
        self.ConvBlock1 = self._create_conv_block(in_channels = in_channels, out_channels = 16, kernel_size = 7, stride = 2)
        self.CBAMBlock1 = CBAMBlock(channel = 16,reduction = 2,kernel_size = 7)
        self.ConvBlock2 = self._create_conv_block(16, 32, kernel_size = 7, stride = 2)
        self.CBAMBlock2 = CBAMBlock(channel = 32,reduction = 4,kernel_size = 7)
        self.ConvBlock3 = self._create_conv_block(32, 64, 7, 2)
        self.CBAMBlock3 = CBAMBlock(channel = 64,reduction = 8,kernel_size = 7)
               
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
                                nn.Linear(12544, 2048),
                                Mish_layer(),
                                nn.BatchNorm1d(2048),
                                nn.Dropout(p = 0.2),
                                
                                nn.Linear(2048, 1024),
                                Mish_layer(),
#                                 nn.BatchNorm1d(1024),
                                nn.Dropout(p = 0.1),
        
                                nn.Linear(1024, 512),
                                Mish_layer(),
#                                 nn.BatchNorm1d(512),
                                
                                nn.Linear(512, 1)
                                )

        
    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.CBAMBlock1(x)
        x = self.ConvBlock2(x)
        x = self.CBAMBlock2(x)
        x = self.ConvBlock3(x)
        x = self.CBAMBlock3(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

    def _create_conv_block(self, in_channels, out_channels, kernel_size, stride):
        ConvBlock = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            Mish_layer()
        )
        
        return ConvBlock

if __name__ == '__main__':
    model = Regressor()
    print(model)