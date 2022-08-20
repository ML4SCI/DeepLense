import sys
import torch
import torch.nn as nn

sys.path.append('../')
from utils import device
from config import MODEL_PATH

class Encoder(nn.Module):
    
    def __init__(self, in_channels = 1, latent_dim = 2048):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = 7, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 32, 7, stride=2, padding = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            
            nn.Conv2d(32, 64, 7),
            nn.BatchNorm2d(64),
            nn.PReLU(),           
        )
       
        self.mu = nn.Linear(3136, latent_dim)
        self.var = nn.Linear(3136, latent_dim)
         

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = x.view(x.shape[0],-1)

        z_mu = self.mu(x)
        z_var = self.var(x)

        return z_mu, z_var



class Decoder(nn.Module):
    
    def __init__(self, out_channels = 1, latent_dim = 2048):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 3136),
            nn.PReLU()
        )
     
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            
            nn.ConvTranspose2d(32, 16, 7, stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.ConvTranspose2d(16, out_channels, 7, stride = 2, padding = 1, output_padding = 1),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = x.view(x.shape[0], 64, 7, 7)
        x = self.decoder_conv(x)
#         x = torch.tanh(x)
        return x


class VAE(nn.Module):

    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        
        z_mu, z_var = self.enc(x)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        predicted = self.dec(x_sample)
        
        return predicted, z_mu, z_var


if __name__ == '__main__':
    encoder = Encoder()
    decoder = Decoder()
    model = VAE(encoder, decoder)
    if device != 'cpu':
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location = torch.device('cpu')))
    print(model)