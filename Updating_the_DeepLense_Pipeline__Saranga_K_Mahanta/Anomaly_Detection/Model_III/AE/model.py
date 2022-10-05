import torch.nn as nn
import torch

from config import FULL_TEST_DATA_PATH, BATCH_SIZE, MODEL_PATH
from utils import device, set_seed, test_transforms


class Encoder(nn.Module):
    
    def __init__(self, in_channels = 1, latent_dim = 2048):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size=2, stride = 2),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 32, 2, stride=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            
            nn.Conv2d(32, 16, 2, stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            
            nn.Conv2d(16, 8, 2, stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(6728, latent_dim),
            nn.PReLU()
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
            nn.PReLU()
        )

        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride = 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            
            nn.ConvTranspose2d(16, 32, 2, stride = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),

            nn.ConvTranspose2d(32, 64, 2, stride = 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
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
        self.enc = Encoder(in_channels = 1)
        self.dec = Decoder(out_channels = 1)

    def forward(self,x):
        encoded = self.enc(x)
        z = self.dec(encoded)

        return z

if __name__ == '__main__':
    model = Autoencoder()
    print(model)
