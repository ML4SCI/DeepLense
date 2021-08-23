from torch import nn

class Discriminator(nn.Module):
    def __init__(self, features_size=256, hidden_features=256):
        """
        Neural network that receives an array of size `features_size` and discriminates between encodings of the source domain and the target domain.

        Arguments:
        ----------
        features_size: int
            Size of encoded features array.

        hidden_features: int
            Size of the hidden layers.
        """

        super(Discriminator, self).__init__()

        self.discrim = nn.Sequential(
            nn.Linear(features_size, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 2)
        )

    def forward(self, x):
        x = self.discrim(x)
        return x