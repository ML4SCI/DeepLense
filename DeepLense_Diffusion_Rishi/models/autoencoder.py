import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, latent_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim*4, hidden_dim*8),
            # nn.ReLU(),
            # nn.Linear(hidden_dim*8, latent_dim),
            # nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim*2),
            nn.ReLU(),
            # nn.Linear(hidden_dim*8, hidden_dim*4),
            # nn.ReLU(),
            # nn.Linear(hidden_dim*4, hidden_dim*2),
            # nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        with torch.no_grad():
            decoded = self.decoder(x)
        return decoded

if __name__ =='__main__':
    data = torch.rand(32,1)
    model = Autoencoder(3,2,1)
    output = model(data)
    print(output.shape)