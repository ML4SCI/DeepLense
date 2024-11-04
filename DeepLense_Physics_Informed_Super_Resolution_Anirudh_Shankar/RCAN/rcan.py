from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, latent_dim, reduction) -> None:
        """
        A module that implements channel attention

        :param latent_dim: Latent dimension size
        :param reduction: Latent size reduction scale
        """
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(latent_dim, latent_dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // reduction, latent_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)
    
class RCAB(nn.Module):
    def __init__(self, latent_dim, reduction):
        """
        Implements the Residual Channel Attention Block module

        :param latent_dim: Latent dimension size
        :param reduction: Latent size reduction scale
        """
        super(RCAB, self).__init__()
        self.rcab = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            ChannelAttention(latent_dim, reduction)
        )

    def forward(self, x):
        return x + self.rcab(x)
    
class RG(nn.Module):
    def __init__(self, latent_dim, num_rcab, reduction):
        """
        Implements the Residual Group module

        :param latent_dim: Latent dimension size
        :param num_rcab: Number of RCAB blocks in the Residual Group
        :param reduction: Latent size reduction scale
        """
        super(RG, self).__init__()
        self.rg = [RCAB(latent_dim, reduction) for _ in range(num_rcab)]
        self.rg.append(nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1))
        self.rg = nn.Sequential(*self.rg)

    def forward(self, x):
        return x + self.rg(x)
    
class RCAN(nn.Module):
    def __init__(self, scale, latent_dim, num_rg, num_rcab, reduction, in_channels=1, out_channels=1):
        """
        Implements the Residual Channel Attention Network

        :param scale: Super-resolution scale
        :param latent_dim: Latent dimension size
        :param num_rg: Number of residual groups
        :param num_rcab: Number of RCAB modules
        :param reduction: Latent size reduction scale
        :param in_channels: Number of input image channels
        :param out_channels: Number of output image channels
        """
        super(RCAN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, latent_dim, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(latent_dim, num_rcab, reduction) for _ in range(num_rg)])
        self.conv2 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)
        self.upscale = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.conv3 = nn.Conv2d(latent_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Feed-forward
        """
        x = self.conv1(x)
        residual = x
        x = self.rgs(x)
        x = self.conv2(x)
        x += residual
        x = self.upscale(x)
        x = self.conv3(x)
        return x