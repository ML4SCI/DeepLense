import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, in_channels=1, latent_dim = 2048):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = 7, stride = 2, padding = 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 32, 7, stride=2, padding = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            
            nn.Conv2d(32, 64, 7),
            nn.BatchNorm2d(64),
            nn.PReLU())

        self.flat = nn.Flatten()
        self.enc_lin = nn.Linear(3136, latent_dim)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flat(x)
        x = self.enc_lin(x)
        return x


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

class Discriminator(nn.Module):

    def __init__(self, dim_z = 2048 , dim_h = 256):
        super(Discriminator,self).__init__()
        self.dim_z = dim_z
        self.dim_h = dim_h
        self.network = []
        self.network.extend([
            nn.Linear(self.dim_z, self.dim_h),
            nn.PReLU(),
            nn.Linear(self.dim_h, self.dim_h),
            nn.PReLU(),
            nn.Linear(self.dim_h,1),
            nn.Sigmoid(),
        ])
        self.network = nn.Sequential(*self.network)

    def forward(self, z):
        disc = self.network(z)
        return disc

if __name__ == '__main__':
    encoder = Encoder()
    decoder = Decoder()
    disc = Discriminator()

    print(encoder, end = '\n\n')
    print(decoder, end = '\n\n')
    print(disc, end = '\n\n')