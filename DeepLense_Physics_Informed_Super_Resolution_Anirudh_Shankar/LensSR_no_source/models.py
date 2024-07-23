import torch    

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

class LensingLoss(torch.nn.Module):
    def __init__(self, in_shape, device, alpha, resolution, BATCH_SIZE, source_scaling) -> None:
        """
        A custom loss that incorporates the strong lensing equation and some intensity constraints

        :param in_shape: Image dimensions
        :param device: Device the model is being trained on
        :param alpha: Deflection angle of the lens being studied
        :param resolution: Pixel to arcsec conversion
        :param BATCH_SIZE: 
        :param source_scaling: Scaling of the deflection angle to generate the intermediate representation
        """
        super(LensingLoss, self).__init__()
        pos_y = torch.tensor([[i for _ in range(in_shape)]for i in range(in_shape-1,-1,-1)]).to(device)
        pos_x = torch.tensor([[i for i in range(in_shape)]for _ in range(in_shape)]).to(device)
        pos_x = self.origin_shift(pos_x, in_shape//2)*resolution
        pos_y = self.origin_shift(pos_y, in_shape//2)*resolution
        self.pos_x, self.pos_y = pos_x.flatten(), pos_y.flatten()
        r = torch.sqrt(pos_x**2+pos_y**2) #r(i,j) in pixel indices
        theta = torch.arctan2(input=pos_y,other=pos_x) #theta(x,y)

        dest_r_source = r-alpha
        dest_x, dest_y = torch.round(dest_r_source/resolution*torch.cos(theta)).int(), torch.round(dest_r_source/resolution*torch.sin(theta)).int()
        dest_y = torch.flip(dest_y, dims=[0])
        dest_x, dest_y = self.origin_shift(dest_x, -in_shape//2), self.origin_shift(dest_y, -in_shape//2)
        dext_x, dest_y = dest_x.view(1,-1), dest_y.view(1,-1)
        dest_indices = dest_y*in_shape+dext_x
        self.dest_indices_source = dest_indices.type(torch.int64)

        dest_r = r+alpha/source_scaling
        dest_x, dest_y = dest_r/resolution*torch.cos(theta), dest_r/resolution*torch.sin(theta)
        dest_x, dest_y = dest_x/(in_shape//2), dest_y/(in_shape//2)
        dest_y = torch.flip(dest_y, dims=[0])
        grid = torch.stack((dest_x.view(1,dest_x.shape[1],dest_x.shape[1]), dest_y.view(1,dest_y.shape[1],dest_y.shape[1])), dim=-1)
        self.grid = grid.repeat(BATCH_SIZE, 1, 1, 1)
    
    def function_loss(self, source, image):
        """
        Computes the MSE between intermediate representations from the source network and the image network
        """
        source_f = torch.nn.functional.grid_sample(image, self.grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        function_loss = source_f-source
        return torch.mean((function_loss)**2)
    
    def get_samples(self, image):
        """
        Used in generating examples
        """
        B, _, x, y = image.shape
        source_f = torch.nn.functional.grid_sample(image, self.grid[0].view(1,x,y,2), mode='bilinear', padding_mode='zeros', align_corners=True)
        source_true = torch.zeros_like(image).view(1,-1)
        source_true = source_true.scatter_(1, self.dest_indices_source.view(1,-1), image.view(1,-1))
        source_true = source_true.view(1,1,x,y)
        return source_f, source_true
    
    def origin_shift(self, source, shift):
        return source - shift
    
    def intensity_conservation(self, tensor_1, tensor_2):
        """
        Imposes intensity conservation on the two tensors
        """
        return torch.mean((torch.sum(tensor_1, dim=[1,2,3], keepdim=True)/(tensor_1.shape[2]**2)-torch.sum(tensor_2, dim=[1,2,3], keepdim=True)/(tensor_2.shape[2]**2))**2)
    
    def forward(self, source, image, image_lr):
        """
        Feed-forward: Imposes the lensing equation and intensity constraints as a loss
        """
        return self.function_loss(source, image) + self.intensity_conservation(image_lr, image) + self.intensity_conservation(image_lr, source)