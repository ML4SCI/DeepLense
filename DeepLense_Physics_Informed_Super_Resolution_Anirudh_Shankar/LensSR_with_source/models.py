import torch
import numpy as np
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib.pyplot as plt

class SersicProfiler(torch.nn.Module):
    def __init__(self, in_shape, resolution, device, alpha, BATCH_SIZE, sersic_args = [20, 1, 0.25]) -> None:
        """
        Constructs a sersic profile

        :param in_shape: Image dimensions
        :param device: Device the model is being trained on
        :param alpha: Deflection angle of the lens being studied
        :param resolution: Pixel to arcsec conversion
        :param BATCH_SIZE: 
        :param sersic_args: Definition of the sersic profile
        """
        super(SersicProfiler, self).__init__()
        self.sersic_args = sersic_args
        pos_y = torch.tensor([[i for _ in range(in_shape)]for i in range(in_shape-1,-1,-1)]).to(device)
        pos_x = torch.tensor([[i for i in range(in_shape)]for _ in range(in_shape)]).to(device)
        pos_x = self.origin_shift(pos_x, in_shape//2)*resolution
        pos_y = self.origin_shift(pos_y, in_shape//2)*resolution
        self.pos_x, self.pos_y = pos_x.flatten(), pos_y.flatten()
        r = torch.sqrt(pos_x**2+pos_y**2) #r(i,j) in pixel indices
        theta = torch.arctan2(input=pos_y,other=pos_x) #theta(x,y)
        self.device = device
        self.resolution = resolution

        dest_r = r-alpha
        dest_x, dest_y = torch.round(dest_r/resolution*torch.cos(theta)).int(), torch.round(dest_r/resolution*torch.sin(theta)).int()
        dest_y = torch.flip(dest_y, dims=[0])
        dest_x, dest_y = self.origin_shift(dest_x, -in_shape//2), self.origin_shift(dest_y, -in_shape//2)
        dest_x, dest_y = dest_x.view(1,-1).repeat(BATCH_SIZE, 1), dest_y.view(1,-1).repeat(BATCH_SIZE, 1)
        self.dest_indices = dest_y*in_shape+dest_x
        self.dest_indices = self.dest_indices.type(torch.int64)

        dest_x, dest_y = dest_r*torch.cos(theta), dest_r*torch.sin(theta)
        # dest_y = torch.flip(dest_y, dims=[0])
        self.dest_x, self.dest_y = dest_x, dest_y

        sersic_profile = self.sersic_law(dest_x.view(-1),dest_y.view(-1),0.1,0.1)
        sersic_profile = (sersic_profile - torch.min(sersic_profile))/(torch.max(sersic_profile)-torch.min(sersic_profile))


    def sersic_law(self, x, y, x_center, y_center):
        """
        Constructs a (displaced) Sérsic profile for a set of positions
        
        :param x: x coordinate list (in arcsec)
        :param y: y coordinate list (in arcsec)
        :param x_center, y_center: Defines the displacement of the Sérsic center
        :return: Intensity values for the given coordinates
        """
        amp, n_sersic, R_sersic = self.sersic_args 
        b_n = 1.999*n_sersic-0.327
        R = torch.sqrt(torch.pow(x-x_center,2)+torch.pow(y-y_center,2))
        I = amp*torch.pow(torch.e,-b_n*(torch.pow(R/R_sersic,1/n_sersic)-1))
        return I
    
    def forward(self, image, LR, norm=True):
        """
        Constructs a Sérsic profile as the source and then performs lensing to train the image network
        """
        B, _, x, y = image.shape
        image = image.view(image.shape[0],-1).to(self.device)
        source_profile = torch.zeros_like(image)
        source_profile = source_profile.scatter_(1, self.dest_indices.view(B,-1), LR.view(B,-1))
        _, max_index = torch.max(source_profile, dim=1)
        y_center, x_center = max_index//x, max_index%x
        y_center = y - y_center
        x_center, y_center = self.origin_shift(x_center, x/2)*self.resolution, self.origin_shift(y_center, y/2)*self.resolution

        sersic_profile = self.sersic_law(self.dest_x.view(1,-1).repeat(B,1), self.dest_y.view(1,-1).repeat(B,1), x_center.view(B,1), y_center.view(B,1))
        if norm==True:
            sersic_profile = (sersic_profile - torch.min(sersic_profile))/(torch.max(sersic_profile)-torch.min(sersic_profile))
        sersic_profile = sersic_profile.view(B,1,x,y)
        image = image.view(B,1,x,y)
        source_profile = source_profile.view(B,1,x,y)
        
        return torch.nn.functional.mse_loss(sersic_profile, image)
    def get_sample(self, image, plot):
        """
        Used in generating examples
        """
        B, _, x, y = image.shape
        image = image.view(image.shape[0],-1).to(self.device)
        source_profile = torch.zeros_like(image)
        source_profile = source_profile.scatter_(1, self.dest_indices.split([1 for _ in range(5)],0)[0], image.view(image.shape[0],-1))
        _, max_index = torch.max(source_profile, dim=1)
        y_center, x_center = max_index//x, max_index%x
        y_center = y - y_center
        x_center, y_center = self.origin_shift(x_center, x/2)*self.resolution, self.origin_shift(y_center, y/2)*self.resolution

        sersic_profile = self.sersic_law(self.dest_x.view(-1), self.dest_y.view(-1), x_center, y_center)
        sersic_profile = (sersic_profile - torch.min(sersic_profile))/(torch.max(sersic_profile)-torch.min(sersic_profile))
        sersic_profile = sersic_profile.view(x,y)

        if not plot: return sersic_profile, image
        plot, axes = plt.subplots(1,2)
        plot.set_size_inches(10,5)
        y1, x1 = np.histogram(image.view(-1).detach().cpu(), bins=20)
        y2, x2 = np.histogram(sersic_profile.view(-1).detach().cpu(), bins=20)
        axes[0].stairs(np.sqrt(y1), x1, label="Image: %.3f"%torch.sum(image).detach().cpu().float())
        axes[0].stairs(np.sqrt(y2), x2, label="Re-lensing profile: %.3f"%torch.sum(sersic_profile).cpu().float())
        axes[1].plot(image.view(-1).detach().cpu(), label="Image")
        axes[1].plot(sersic_profile.view(-1).detach().cpu(), label="Re-lensing profile")
        axes[1].legend()
        axes[0].legend()
        return sersic_profile, image
    
    def intensity_conservation(self, tensor_1, tensor_2):
        """
        Imposes intensity conservation on the two tensors (presently unused in this model)
        """
        return torch.mean((torch.sum(tensor_1, dim=[1,2,3], keepdim=True)/(tensor_1.shape[2]**2)-torch.sum(tensor_2, dim=[1,2,3], keepdim=True)/(tensor_2.shape[2]**2))**2)
    
    def origin_shift(self, source, shift):
        return source - shift


class SISR(torch.nn.Module):
    def __init__(self, magnification, n_mag, residual_depth, in_channels=1, latent_channel_count=64):
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

        self.relu1 = torch.nn.ReLU()
        for _ in range(residual_depth):
            self.residual_layer_list.append(self.make_residual_layer(latent_channel_count))
        self.conv2 = torch.nn.Conv2d(in_channels=latent_channel_count,out_channels=latent_channel_count,kernel_size=9,padding=4)
        for _ in range(n_mag):
            self.subpixel_layer_list.append(self.make_subpixel_layer(latent_channel_count))
        
        self.conv3 = torch.nn.Conv2d(in_channels=self.latent_channel_count,out_channels=self.in_channels,kernel_size=3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=self.in_channels)

        self.sigmoid = torch.nn.Sigmoid()

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
        x = self.sigmoid(x)
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