import torch
from numpy import ceil

class Generator(torch.nn.Module):
    def __init__(self, residual_depth, out_channels, in_channels=1, maginfication=2, num_magnifications=1, latent_channel_count=64):
        """
        The Resnet_SISR module

        :param residual_depth: Number of residual blocks
        :param in_channels: Number of image channels (here 1, as image is b/w)
        :param magnification: Magnification factor at each iteration
        :param num_magnifications: Number of magnifications to be done
        :param latent_channel_count: Number of image channels the model is to be trained with 
        """
        super(Generator, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=latent_channel_count,kernel_size=9,padding=4)
        self.relu = torch.nn.ReLU()
        self.bn_res = torch.nn.BatchNorm2d(latent_channel_count)
        self.residual_depth = residual_depth
        self.residual_layer_list = torch.nn.ModuleList()
        self.magnification_list = torch.nn.ModuleList()
        self.magnification = maginfication
        self.num_magnifications = num_magnifications
        for _ in range(residual_depth):
            self.residual_layer_list.append(self.make_residual_block(latent_channel_count))
        self.conv2 = torch.nn.Conv2d(in_channels=latent_channel_count,out_channels=latent_channel_count,kernel_size=9,padding=4)
        for _ in range(num_magnifications):
            self.magnification_list.append(self.make_subpixel_block(latent_channel_count))
        self.conv3 = torch.nn.Conv2d(in_channels=latent_channel_count,out_channels=out_channels,kernel_size=3,padding=1)


    def forward(self, x):
        """
        Forward propagation

        :param x: Low resolution image to be upscaled
        :return: SR image
        """
        x = self.conv1(x)
        x = self.relu(x)
        res_0 = x.clone()
        for i in range(self.residual_depth):
            res = x.clone()
            x = self.residual_layer_list[i](x)
            x = x + res
        x = self.conv2(x)
        x = self.bn_res(x)
        x = x + res_0
        for i in range(self.num_magnifications):
            x = self.magnification_list[i](x)
        x = self.conv3(x)
        x = self.relu(x)
        return x

    def make_residual_block(self, channels):
        """
        Makes a residual block

        :param channels: Number of channels the image will come in
        :return: The residual block
        """
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1),
                                   torch.nn.BatchNorm2d(channels),
                                   self.relu,
                                   torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,padding=1),
                                   torch.nn.BatchNorm2d(channels),
                                   self.relu)
    def make_subpixel_block(self, channels):
        """
        Makes a subpixel block which increases image dimensions

        :param channels: Number of channels the image will come in
        :return: The subpixel block
        """
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels=channels,out_channels=channels*self.magnification*self.magnification,kernel_size=3,padding=1),
                                   torch.nn.PixelShuffle(self.magnification),
                                   self.relu)
    
class Discriminator(torch.nn.Module):
    def __init__(self, residual_depth, in_shape, in_channels=1, latent_channel_count=64) -> None:
        """
        Discriminator module to be used in the GAN

        :param residual_depth: Number of residual blocks
        :param in_channels: Number of image channels (here 1, as image is b/w)
        :param latent_channel_count: Number of image channels the model is to be trained with 
        """
        super(Discriminator, self).__init__()
        img_shape = in_shape
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=latent_channel_count,kernel_size=9,padding=4)
        img_shape = int(img_shape - 9 + 8 + 1)
        self.relu = torch.nn.ReLU()
        convolution_layer_list = torch.nn.ModuleList()
        channel_tracker = latent_channel_count
        for i in range(residual_depth):
            if i%2: 
                convolution_layer_list.append(self.make_convolution_layer(channel_tracker,stride=2))
            else:
                convolution_layer_list.append(self.make_convolution_layer(channel_tracker,maginfication=2,stride=2))
                channel_tracker *= 2
            img_shape = int((img_shape - 3 + 2)/2+1)
        self.conv_layers = torch.nn.Sequential(*convolution_layer_list)
        self.fc1 = torch.nn.Linear(in_features=img_shape**2*channel_tracker,out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64,out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Forward propagation

        :param x: Low resolution image to judged by the discriminator
        :return: Shape 1 judgement
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv_layers(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def make_convolution_layer(self, channels, maginfication=1, stride=1):
        """
        Makes a convolution block

        :param channels: Number of channels the image will come in
        :return: The convolution block
        """
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels=channels,out_channels=channels*maginfication,kernel_size=3,padding=1,stride=stride),
                                   torch.nn.BatchNorm2d(channels*maginfication),
                                   self.relu)

class Resnet_simple(torch.nn.Module):
    def __init__(self, fc_in = 25088) -> None:
        """
        Resnet_simple model class

        :param fc_in: Optional number of in_channels of the fc layer that creates the latent space
        """
        super(Resnet_simple, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        """
        Forward propagation

        :param x: Image to be classified
        :return: Class predicted by the model
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp1(x)

        return x