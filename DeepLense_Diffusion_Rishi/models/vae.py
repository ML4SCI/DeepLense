import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_channels = config.vae.input_channels
        self.latent_dimension = config.vae.latent_dimension

        channels = [config.vae.input_channels, 8, 16, 32, 64, 128, 256,  2*config.vae.latent_dimension]  # Shape B,2*z_dim,1,1

        layers = []

        default_activation = nn.LeakyReLU(0.2, inplace=True)

        for i in range(len(channels)-1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], 3, 2, 1))
            activation = default_activation if i < len(channels) -2 else nn.Identity()
            layers.append(activation)

        layers.append(View((-1, 2*self.latent_dimension)))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x[:,:self.latent_dimension], x[:,self.latent_dimension:]


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.nc = config.vae.input_channels
        self.latent_dimension = config.vae.latent_dimension

        default_activation = nn.LeakyReLU(0.2, inplace=True)

        self.decoder = nn.Sequential(
            # nn.Linear(config.vae.latent_dimension, 256),
            # View((-1, 256, 1, 1)),
            # default_activation,
            # nn.ConvTranspose2d(256, 64, 4),
            # default_activation,
            # nn.ConvTranspose2d(64, 64, 4, 2, 1),
            # default_activation,
            # nn.ConvTranspose2d(64, 32, 4, 2, 1),
            # default_activation,
            # nn.ConvTranspose2d(32, 32, 4, 2, 1),
            # default_activation,
            # nn.ConvTranspose2d(32, self.nc, 4, 2, 1),

            nn.Linear(config.vae.latent_dimension, 256),
            View((-1, 256, 1, 1)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, bias=False),
                                )
        
    def forward(self, z):
        x_rec = self.decoder(z)
        x_rec = F.sigmoid(x_rec)
        return x_rec

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class vae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dimension = config.vae.latent_dimension
        self.input_channels = config.vae.input_channels
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        #print(mu.shape)
        z = reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        #print(x_recon.shape)
        return x_recon, mu, logvar


