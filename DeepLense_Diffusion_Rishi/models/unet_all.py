import torch
import torch.nn.functional as F

from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x+self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels,
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Down_mass(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels,
            ),
        )

        self.layer1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.layer2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.layer3 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.layer4 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )


    def forward(self, x, t, v1, v2, v3, v4):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        v1o = self.layer1(v1)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        v2o = self.layer2(v2)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        v3o = self.layer3(v3)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        v4o = self.layer4(v4)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb + v1o + v2o + v3o + v4o


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class Up_mass(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=128):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.layer1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.layer2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.layer3 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

        self.layer4 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        

    def forward(self, x, skip_x, t, v1, v2, v3, v4):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        v1o = self.layer1(v1)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        v2o = self.layer2(v2)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        v3o = self.layer3(v3)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        v4o = self.layer4(v4)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb + v1o + v2o + v3o + v4o

class UNet_all_conditional(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.time_dim = config.unet.time_emb
        self.inc = DoubleConv(config.unet.input_channels, 64)
        self.down1 = Down_mass(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down_mass(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down_mass(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up_mass(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up_mass(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up_mass(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, config.unet.output_channels, kernel_size=1)

        self.linear_1 = nn.Linear(1,config.unet.time_emb)
        self.linear_2 = nn.Linear(1,config.unet.time_emb)
        self.linear_3 = nn.Linear(1,config.unet.time_emb)
        self.linear_4 = nn.Linear(1,config.unet.time_emb)

        # if config.data.num_classes is not None:
        #     self.label_emb = nn.Embedding(config.data.num_classes, config.unet.time_emb)
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y1, y2, y3, y4):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        #print(t.shape)
        if y1 is not None:
            v1 = self.linear_1(y1)
            v2 = self.linear_2(y2)
            v3 = self.linear_3(y3)
            v4 = self.linear_4(y4)

        x1 = self.inc(x)
        x2 = self.down1(x1, t, v1, v2, v3, v4)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t, v1, v2, v3, v4)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t, v1, v2, v3, v4)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t, v1, v2, v3, v4)
        x = self.sa4(x)
        x = self.up2(x, x2, t, v1, v2, v3, v4)
        x = self.sa5(x)
        x = self.up3(x, x1, t, v1, v2, v3, v4)
        x = self.sa6(x)
        output = self.outc(x)
        return output

