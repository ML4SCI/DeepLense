import torch

class SISR(torch.nn.Module):
    def __init__(self, magnification, n_mag, residual_depth, in_channels=1, out_channels = 1, latent_channel_count=64):
        """
        Single image super-resolution module, to upscale an image to a decided magnification

        :param magnification: Magnification value
        :param n_mag: Number of times the above magnification is applied
        :param residual_depth: Number of residual modules used
        :param in_channels: Number of channels in the image (here 1)
        :param latent_channel_count: Dimensions of the residual module layers
        """
        super(SISR, self).__init__()
        self.magnification = magnification
        self.residual_depth = residual_depth
        self.in_channels = in_channels
        self.latent_channel_count=latent_channel_count
        self.residual_layer_list = torch.nn.ModuleList()
        self.subpixel_layer_list = torch.nn.ModuleList()
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,out_channels=self.latent_channel_count,kernel_size=3,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=self.latent_channel_count)
        self.out_channels = out_channels

        self.relu1 = torch.nn.ReLU()
        for _ in range(residual_depth):
            self.residual_layer_list.append(self.make_residual_layer(latent_channel_count))
        self.conv2 = torch.nn.Conv2d(in_channels=latent_channel_count,out_channels=latent_channel_count,kernel_size=9,padding=4)
        for _ in range(n_mag):
            self.subpixel_layer_list.append(self.make_subpixel_layer(latent_channel_count))
        
        self.conv3 = torch.nn.Conv2d(in_channels=self.latent_channel_count,out_channels=self.out_channels,kernel_size=3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Feed-forward 
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x_res_0 = x.clone()
        for module in self.residual_layer_list:
            x_res = x.clone()
            x = module(x)
            x = x + x_res
        x = self.conv2(x)
        x = x + x_res_0
        for module in self.subpixel_layer_list:
            x = module(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
    def make_residual_layer(self, channels):
        """
        Generates and returns a single residual layer
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(num_features=channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(num_features=channels),
            torch.nn.ReLU()
        )
    def make_subpixel_layer(self, channels):
        """
        Generates and returns a single subpixel layer
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels,out_channels=channels*self.magnification*self.magnification,kernel_size=3,padding=1),
            torch.nn.PixelShuffle(self.magnification),
            torch.nn.ReLU()
        )