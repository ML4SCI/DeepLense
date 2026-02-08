import torch
import torch.nn as nn

class SISR(nn.Module):
    def __init__(self, magnification, n_mag, residual_depth, in_channels=1, out_channels=1, latent_channel_count=64):
        """
        Single image super-resolution module.
        """
        super().__init__()
        self.magnification = magnification
        self.residual_depth = residual_depth
        self.in_channels = in_channels
        self.latent_channel_count = latent_channel_count
        self.out_channels = out_channels
        
        self.residual_layer_list = nn.ModuleList()
        self.subpixel_layer_list = nn.ModuleList()
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.latent_channel_count, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.latent_channel_count)
        self.relu1 = nn.ReLU()
        
        for _ in range(residual_depth):
            self.residual_layer_list.append(self.make_residual_layer(latent_channel_count))
            
        self.conv2 = nn.Conv2d(in_channels=latent_channel_count, out_channels=latent_channel_count, kernel_size=9, padding=4)
        
        for _ in range(n_mag):
            self.subpixel_layer_list.append(self.make_subpixel_layer(latent_channel_count))
        
        self.conv3 = nn.Conv2d(in_channels=self.latent_channel_count, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initial Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Save input for Global Residual Connection
        x_res_0 = x 
        
        # Residual Blocks (Local Residual Learning)
        for module in self.residual_layer_list:
            x_res = x
            x = module(x)
            x = x + x_res # Element-wise Sum
        
        # Post-Residual Block
        x = self.conv2(x)
        
        # Global Residual Connection
        x = x + x_res_0
        
        # Upsampling
        for module in self.subpixel_layer_list:
            x = module(x)
            
        # Output Block
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
    
    def make_residual_layer(self, channels):
        """
        Generates a residual block WITHOUT the final ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=channels)
        )

    def make_subpixel_layer(self, channels):
        """
        Generates and returns a single subpixel layer
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * (self.magnification**2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.magnification),
            nn.ReLU()
        )
