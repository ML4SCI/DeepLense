import torch.nn as nn
from activation_funcs import Mish_layer

class Encoder(nn.Module):
    
    def __init__(self, in_channels = 1, latent_dim = 2048):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size=2, stride = 2),
            nn.BatchNorm2d(64),
            Mish_layer(),

            nn.Conv2d(64, 32, 2, stride=1),
            nn.BatchNorm2d(32),
            Mish_layer(),
            
            nn.Conv2d(32, 16, 2, stride=1),
            nn.BatchNorm2d(16),
            Mish_layer(),
            
            nn.Conv2d(16, 8, 2, stride=1),
            nn.BatchNorm2d(8),
            Mish_layer()
        )


        self.bottleneck = nn.Sequential(
            nn.Linear(6728, latent_dim),
            Mish_layer()
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = x.view(x.shape[0],-1)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, out_channels = 1, latent_dim = 2048):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 6728),
            Mish_layer()
        )

        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride = 1),
            nn.BatchNorm2d(16),
            Mish_layer(),
            
            nn.ConvTranspose2d(16, 32, 2, stride = 1),
            nn.BatchNorm2d(32),
            Mish_layer(),

            nn.ConvTranspose2d(32, 64, 2, stride = 1),
            nn.BatchNorm2d(64),
            Mish_layer(),
            
            nn.ConvTranspose2d(64, out_channels, 2, stride = 2),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = x.view(x.shape[0], 8, 29, 29)
        x = self.decoder_conv(x)

        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channels = 1)
        self.decoder = Decoder(out_channels = 1)

    def forward(self,x):
        encoded = self.encoder(x)
        z = self.decoder(encoded)

        return z

if __name__ == '__main__':
    AE = Autoencoder()
    print(AE)
