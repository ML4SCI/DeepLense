import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SersicProfiler(torch.nn.Module):
    def __init__(self, resolution, device, sersic_args = [1, 1, 0.25], vdl_weight=1, multi_scale_loss_scales = [1, 0.5, 0.25]) -> None:
        """
        The physics-based loss module that performs everything required in training 

        :param resolution: Pixel to arcsec conversion
        :param device: Device the model is being trained on
        :param sersic_args: Definition of the sersic profile
        :param vdl_weight: Weight of the VDL module
        :param multi_scale_loss_scales: List of scales the multi_scale_loss module will operate in
        """
        super(SersicProfiler, self).__init__()
        self.sersic_args = sersic_args
        self.device = device
        self.resolution = resolution
        self.vdl_weight = vdl_weight
        self.scale_list = multi_scale_loss_scales

    def variation_density_loss(self, alpha):
        """
        Computes the vadiation density loss for a given tensor, which is the sum of the differences between adjescent pixels in both axes, normalized by image shape 

        :param alpha: Input tensor (here, deflection angle)
        :return: The VDL of the deflection angle
        """
        diff_x = torch.abs(alpha[:, :, 1:, :] - alpha[:, :, :-1, :])
        diff_y = torch.abs(alpha[:, :, :, 1:] - alpha[:, :, :, :-1])
        return torch.sum(diff_x)/(diff_x.shape[2]*diff_x.shape[3]) + torch.sum(diff_y)/(diff_y.shape[2]*diff_y.shape[3])
    
    def create_source(self, alpha, resolution, LR):
        """
        Uses the provided deflection angle to reconstruct the source image from the lensing image
        
        :param alpha: Input deflection angle
        :param resolution: Arcsec per pixel of the image
        :param LR: Image with which source is to be reconstructed
        :return: Source image
        """
        B, _, x, y = LR.shape
        alpha = torch.split(alpha, [1,1], dim=1)
        alpha_r, alpha_t = alpha[0], alpha[1]
        arcsec_bound = resolution*x/2 #
        pos_x = torch.linspace(-arcsec_bound, arcsec_bound, x).to(self.device)
        pos_y = torch.linspace(-arcsec_bound, arcsec_bound, y).to(self.device)
        theta_y, theta_x = torch.meshgrid(pos_x, pos_y)

        # reverse lensing to get the source

        theta_r = torch.sqrt(theta_x**2 + theta_y**2)
        theta_t = torch.arctan2(input=theta_y,other=theta_x)
        beta_r = theta_r - alpha_r

        beta_x, beta_y = beta_r*torch.cos(theta_t), beta_r*torch.sin(theta_t) #gradient flow
        beta_x, beta_y = beta_x - alpha_t*torch.cos(theta_t), beta_y - alpha_t*torch.sin(theta_t) #gradient flow
        beta_y = torch.flip(beta_y, dims=[2]) #gradient flow

        theta_r_source = torch.sqrt(theta_x**2 + theta_y**2) #gradient flow
        theta_t_source = torch.arctan2(input=theta_y,other=theta_x) #gradient flow
        beta_r_source = theta_r_source + alpha_r #gradient flow
        beta_x_source, beta_y_source = beta_r_source*torch.cos(theta_t_source), beta_r_source*torch.sin(theta_t_source) #gradient flow
        beta_x_source, beta_y_source = beta_x_source + alpha_t*torch.cos(theta_t), beta_y_source + alpha_t*torch.sin(theta_t) #gradient flow
        beta_x_source, beta_y_source = beta_x_source/resolution, beta_y_source/resolution #gradient flow
        beta_x_source, beta_y_source = beta_x_source/(x//2), beta_y_source/(y//2) #gradient flow

        grid = torch.stack((beta_x_source, beta_y_source), dim=-1) #gradient flow
        grid = grid.view(B,x,y,2) #gradient flow
        source_profile_regenerated = torch.nn.functional.grid_sample(LR, grid, mode='bilinear', padding_mode='zeros', align_corners=True) #gradient flow
        return source_profile_regenerated
    
    def create_sersic(self, source_profile_regenerated, resolution, R_sersic):
        """
        Creates and fits a Sérsic profile to a source image

        :param source_profile_regenerated: Input source image
        :param resolution: Arcsec per pixel of the source image
        :param R_sersic: Half light radius of the Sérsic profile
        :return: Sérsic profile fit to the source image, at it's resolution 
        """
        B, _, x, y = source_profile_regenerated.shape
        mean_indices, _ = self.approximate_center(source_profile_regenerated.view(B,-1))

        # re-lensing
        arcsec_bound = resolution*x/2
        pos_x_source = torch.linspace(-arcsec_bound, arcsec_bound, x).to(self.device)
        pos_y_source = torch.linspace(-arcsec_bound, arcsec_bound, y).to(self.device)
        theta_y_source, theta_x_source = torch.meshgrid(pos_x_source, pos_y_source)

        y_center_LR, x_center_LR = mean_indices//y, mean_indices%x
        y_center_LR = x - y_center_LR
        x_center_LR, y_center_LR = self.origin_shift(x_center_LR, x/2)*resolution, self.origin_shift(y_center_LR, y/2)*resolution

        S_lens = self.sersic_law(theta_x_source.reshape(1,-1).repeat(B,1), theta_y_source.reshape(1,-1).repeat(B,1), x_center_LR.view(B,-1), y_center_LR.view(B,-1), R_sersic) #no gradient
        S_lens = (S_lens - torch.min(S_lens, dim=-1)[0].view(B,1))/(torch.max(S_lens, dim=-1)[0].view(B,1)-torch.min(S_lens, dim=-1)[0].view(B,1))
        return S_lens

    def create_lensing(self, LR, alpha, alpha_interpolated, resolution, resolution_, magnification, R_sersic = None):
        """
        Performs lensing twice, to create the source image using the deflection angle, and then to recreate the lensing image at the required resolution

        :param LR: Low resolution lensing image
        :param alpha: Deflection angle extracted from the LR image, at the higher resolution
        :param alpha_interpolated: Deflection angle interpolated to the resolution of the LR image
        :param resolution: Arcsec per pixel of the LR image
        :param resolution: Target arcsec per pixel
        :param magnification: Target magnification
        :param R_sersic: Half light radius of the Sérsic profile
        :return: Source image, Sérsic profile, Re-lensed image
        """
        B, _, x, y = LR.shape
        alpha = torch.split(alpha, [1,1], dim=1)
        alpha_r, alpha_t = alpha[0], alpha[1]
        arcsec_bound = resolution*x/2 #
        pos_x = torch.linspace(-arcsec_bound, arcsec_bound, x).to(self.device)
        pos_y = torch.linspace(-arcsec_bound, arcsec_bound, y).to(self.device)
        theta_y, theta_x = torch.meshgrid(pos_x, pos_y)

        # reverse lensing to get the source

        theta_r = torch.sqrt(theta_x**2 + theta_y**2)
        theta_t = torch.arctan2(input=theta_y,other=theta_x)
        beta_r = theta_r - alpha_r

        beta_x, beta_y = beta_r*torch.cos(theta_t), beta_r*torch.sin(theta_t) #gradient flow
        beta_x, beta_y = beta_x - alpha_t*torch.cos(theta_t), beta_y - alpha_t*torch.sin(theta_t) #gradient flow
        beta_y = torch.flip(beta_y, dims=[2]) #gradient flow

        theta_r_source = torch.sqrt(theta_x**2 + theta_y**2) #gradient flow
        theta_t_source = torch.arctan2(input=theta_y,other=theta_x) #gradient flow
        beta_r_source = theta_r_source + alpha_r #gradient flow
        beta_x_source, beta_y_source = beta_r_source*torch.cos(theta_t_source), beta_r_source*torch.sin(theta_t_source) #gradient flow
        beta_x_source, beta_y_source = beta_x_source + alpha_t*torch.cos(theta_t), beta_y_source + alpha_t*torch.sin(theta_t) #gradient flow
        beta_x_source, beta_y_source = beta_x_source/resolution, beta_y_source/resolution #gradient flow
        beta_x_source, beta_y_source = beta_x_source/(x//2), beta_y_source/(y//2) #gradient flow

        grid = torch.stack((beta_x_source, beta_y_source), dim=-1) #gradient flow
        grid = grid.view(B,x,y,2) #gradient flow
        source_profile_regenerated = torch.nn.functional.grid_sample(LR, grid, mode='bilinear', padding_mode='zeros', align_corners=True) #gradient flow

        # fitting the source onto a Sérsic

        LR = LR.view(B,-1)
        mean_indices, _ = self.approximate_center(source_profile_regenerated.view(B,-1))
        beta_x_LR, beta_y_LR = beta_x, beta_y

        # re-lensing

        in_shape = int(x*magnification)
        alpha_interpolated = torch.split(alpha_interpolated, [1,1], dim=1)
        alpha_r_interpolated, alpha_t_interpolated = alpha_interpolated[0], alpha_interpolated[1]
        arcsec_bound = resolution_*in_shape/2
        pos_x_source = torch.linspace(-arcsec_bound, arcsec_bound, in_shape).to(self.device)
        pos_y_source = torch.linspace(-arcsec_bound, arcsec_bound, in_shape).to(self.device)
        theta_y_source, theta_x_source = torch.meshgrid(pos_x_source, pos_y_source)
        theta_r_source = torch.sqrt(theta_x_source**2 + theta_y_source**2)
        theta_t_source = torch.arctan2(input=theta_y_source,other=theta_x_source)
        theta_y_source = torch.flip(theta_y_source, dims=[0]) #gradient flow

        beta_r_source = theta_r_source - alpha_r_interpolated #gradient flow
        beta_x_source, beta_y_source = beta_r_source*torch.cos(theta_t_source), beta_r_source*torch.sin(theta_t_source) #gradient flow
        beta_x_source, beta_y_source = beta_x_source - alpha_t_interpolated*torch.cos(theta_t_source), beta_y_source - alpha_t_interpolated*torch.sin(theta_t_source) #gradient flow
        beta_y_source = torch.flip(beta_y_source, dims=[2]) #gradient flow

        y_center_LR, x_center_LR = mean_indices//x, mean_indices%x
        y_center_LR = x - y_center_LR
        x_center_LR, y_center_LR = self.origin_shift(x_center_LR, x/2)*resolution, self.origin_shift(y_center_LR, y/2)*resolution

        I_lens_LR = self.sersic_law(beta_x_LR.view(B,-1), beta_y_LR.view(B,-1), x_center_LR.view(B,1), y_center_LR.view(B,1), R_sersic) #gradient flow
        I_lens = self.sersic_law(beta_x_source.view(B,-1), beta_y_source.view(B,-1), x_center_LR.view(B,1), y_center_LR.view(B,1), R_sersic) #gradient flow
        S_lens = self.sersic_law(theta_x_source.reshape(1,-1).repeat(B,1), theta_y_source.reshape(1,-1).repeat(B,1), x_center_LR.view(B,-1), y_center_LR.view(B,-1), R_sersic) #no gradient

        I_lens_LR = (I_lens_LR - torch.min(I_lens_LR, dim=-1)[0].view(B,1))/(torch.max(I_lens_LR, dim=-1)[0].view(B,1)-torch.min(I_lens_LR, dim=-1)[0].view(B,1))
        S_lens = (S_lens - torch.min(S_lens, dim=-1)[0].view(B,1))/(torch.max(S_lens, dim=-1)[0].view(B,1)-torch.min(S_lens, dim=-1)[0].view(B,1))
        I_lens = (I_lens - torch.min(I_lens, dim=-1)[0].view(B,1))/(torch.max(I_lens, dim=-1)[0].view(B,1)-torch.min(I_lens, dim=-1)[0].view(B,1))
        source_profile_regenerated = source_profile_regenerated.view(B,-1)
        source_profile_regenerated = (source_profile_regenerated - torch.min(source_profile_regenerated, dim=-1)[0].view(B,1))/(torch.max(source_profile_regenerated, dim=-1)[0].view(B,1)-torch.min(source_profile_regenerated, dim=-1)[0].view(B,1))


        I_lens_LR = I_lens_LR.view(B,1,x,y)
        I_lens = I_lens.view(B,1,in_shape,in_shape)
        S_lens = S_lens.view(B,1,in_shape,in_shape)
        source_profile_regenerated = source_profile_regenerated.view(B,1,x,y)
        if False:
            LR = LR.view(B,1,x,y)
            plot, axes = plt.subplots(1,5)
            plot.set_size_inches(25,5)
            axes[0].imshow(Image.fromarray(source_profile_regenerated[0][0].detach().cpu().numpy()*255))
            axes[1].imshow(Image.fromarray(S_lens.detach()[0][0].cpu().numpy()*255))
            axes[2].imshow(Image.fromarray(I_lens.detach()[0][0].cpu().numpy()*255))
            axes[3].imshow(Image.fromarray(I_lens_LR.detach()[0][0].cpu().numpy()*255))
            axes[4].imshow(Image.fromarray(LR.detach()[0][0].cpu().numpy()*255))

        return source_profile_regenerated, S_lens, I_lens_LR, I_lens# gradients on 1st, 3rd and 4th

    def multi_scale_loss(self, alpha, LR, resolution, R_sersic = None):
        """
        Computes several loss values and constraints at different length scales

        :param alpha: Deflection angle at high resolution
        :param LR: Low resolution images (input)
        :param resolution: Arcsec per pixel of the LR images
        :param R_sersic: Half light radius of the Sérsic profile
        :return: The compunded loss value
        """
        losses = 0
        individual_losses = {'source_0':[], 'source_1':[], 'source_2':[], 'image_0':[], 'alpha_0':[]}
        for scale in self.scale_list:
            if scale == 1:
                alpha_LR = torch.nn.functional.interpolate(alpha, scale_factor=0.5, mode='bicubic')
                source_profile_regenerated, S_lens, _, I_lens = self.create_lensing(LR, alpha_LR, alpha, resolution, resolution/2, 2, R_sersic=R_sersic)
                source_interpolated = torch.nn.functional.interpolate(source_profile_regenerated, scale_factor = 2, mode='bicubic') #gradient flow
                LR_interpolated = torch.nn.functional.interpolate(LR, scale_factor = 2, mode='bicubic') #gradient flow
                losses += torch.nn.functional.mse_loss(source_interpolated, S_lens)
                losses += self.vdl_weight*self.variation_density_loss(alpha)
                losses += torch.nn.functional.mse_loss(LR_interpolated, I_lens)
                individual_losses['source_0'] = torch.nn.functional.mse_loss(source_interpolated, S_lens)
                individual_losses['image_0'] = torch.nn.functional.mse_loss(LR_interpolated, I_lens)
                individual_losses['alpha_0'] = self.vdl_weight*self.variation_density_loss(alpha)
            
            elif scale == 0.5:
                alpha_LR = torch.nn.functional.interpolate(alpha, scale_factor=0.5, mode='bicubic')
                source_profile_regenerated, S_lens, I_lens, _ = self.create_lensing(LR, alpha_LR, alpha_LR, resolution, resolution, 1, R_sersic=R_sersic)
                # losses += torch.nn.functional.mse_loss(LR, I_lens)
                losses += torch.nn.functional.mse_loss(source_profile_regenerated, S_lens)
                individual_losses['source_1'] = torch.nn.functional.mse_loss(source_profile_regenerated, S_lens)
            
            else:
                LR_scale = scale*2
                alpha_interpolated = torch.nn.functional.interpolate(alpha, scale_factor=scale, mode='bicubic')
                alpha_LR = torch.nn.functional.interpolate(alpha, scale_factor=0.5, mode='bicubic')
                source_profile_regenerated, S_lens, _, __ = self.create_lensing(LR, alpha_LR, alpha_interpolated, resolution, resolution/LR_scale, LR_scale, R_sersic=R_sersic)
                source_interpolated = torch.nn.functional.interpolate(source_profile_regenerated, scale_factor = LR_scale, mode='bicubic')
                losses += torch.nn.functional.mse_loss(source_interpolated, S_lens)
                individual_losses['source_2'] = torch.nn.functional.mse_loss(source_interpolated, S_lens)
        return losses, individual_losses

    def sersic_law(self, x, y, x_center, y_center, R_sersic):
        """
        Constructs a (displaced) Sérsic profile for a set of positions
        
        :param x: x coordinate list (in arcsec)
        :param y: y coordinate list (in arcsec)
        :param x_center, y_center: Defines the displacement of the Sérsic center
        :return: Intensity values for the given coordinates
        """
        if R_sersic == None:
            amp, n_sersic, R_sersic = self.sersic_args
        else:
            amp, n_sersic = self.sersic_args[:2]
        b_n = 1.999*n_sersic-0.327
        R = torch.sqrt(torch.pow(x-x_center,2)+torch.pow(y-y_center,2))
        I = amp*torch.pow(torch.e,-b_n*(torch.pow(R/R_sersic,1/n_sersic)-1))
        return I
    
    def forward(self, alpha, LR, R_sersic = None):
        """
        Constructs a Sérsic profile as the source and then performs lensing to train the image network
        """
        return self.multi_scale_loss(alpha,LR,self.resolution, R_sersic=R_sersic)
    
    def get_sample(self, alpha, LR, plot, R_sersic = None):
        """
        Used in generating examples
        """
        alpha_LR = torch.nn.functional.interpolate(alpha, scale_factor=0.5, mode='bicubic')
        source_profile_regenerated, S_lens, _, I_lens = self.create_lensing(LR, alpha_LR, alpha, self.resolution, self.resolution/2, 2, R_sersic=R_sersic)
        if not plot: return I_lens[0], LR[0], source_profile_regenerated[0]
        plot, axes = plt.subplots(1,2)
        plot.set_size_inches(10,5)
        y1, x1 = np.histogram(LR.view(-1).detach().cpu(), bins=20)
        y2, x2 = np.histogram(I_lens.view(-1).detach().cpu(), bins=20)
        axes[0].stairs(np.sqrt(y2), x2, label="Sérsic profile: %.3f"%torch.sum(S_lens).cpu().float())
        axes[0].stairs(np.sqrt(y1), x1, label="Source: %.3f"%torch.sum(source_profile_regenerated).detach().cpu().float())
        axes[1].scatter(list(range(source_profile_regenerated.shape[2]*source_profile_regenerated.shape[3])), source_profile_regenerated.view(-1).detach().cpu(), label="Source", marker='.')
        axes[1].scatter(np.array(range(S_lens.shape[2]*S_lens.shape[3]))/(S_lens.shape[2]*S_lens.shape[3])*(source_profile_regenerated.shape[2]*source_profile_regenerated.shape[3]), S_lens.view(-1).detach().cpu(), label="Sérsic profile", marker='.')
        axes[1].legend()
        axes[0].legend()
        return I_lens[0], LR[0], source_profile_regenerated[0]
    
    def intensity_conservation(self, tensor_1, tensor_2):
        """
        Imposes intensity conservation on the two tensors (presently unused in this model)
        """
        return torch.mean((torch.sum(tensor_1, dim=[1,2,3], keepdim=True)/(tensor_1.shape[2]**2)-torch.sum(tensor_2, dim=[1,2,3], keepdim=True)/(tensor_2.shape[2]**2))**2)
    
    def origin_shift(self, source, shift):
        """
        A coordinate shift to recenter the origin
        """
        return source - shift
    
    def approximate_center(self, intensity_profile):
        _, max_index = torch.max(intensity_profile, dim=1)
        return max_index, None

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
        
        self.conv3 = torch.nn.Conv2d(in_channels=self.latent_channel_count,out_channels=self.in_channels*2,kernel_size=3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=self.in_channels*2)

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