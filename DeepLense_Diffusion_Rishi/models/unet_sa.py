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

        self.mass_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t, m):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        mass = self.mass_emb_layer(m)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb + mass


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

        self.mass_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
        

    def forward(self, x, skip_x, t, m):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        mass = self.mass_emb_layer(m)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb + mass

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=1, time_dim=128):
    #def __init__(self, config):
        super().__init__()
        #self.config = config
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)
        self.bot1 = DoubleConv(256, 256)
        self.bot3 = DoubleConv(256, 256)
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        # Sinusoidal embedding
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.config.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        #print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)



class UNet_conditional(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.time_dim = config.unet.time_emb
        self.inc = DoubleConv(config.unet.input_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, config.unet.output_channels, kernel_size=1)

        if config.data.num_classes is not None:
            self.label_emb = nn.Embedding(config.data.num_classes, config.unet.time_emb)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

class UNet_ae_conditional(nn.Module):
    def __init__(self, config, ae_model):
        super().__init__()
        self.device = config.device
        self.time_dim = config.unet.time_emb
        self.inc = DoubleConv(config.unet.input_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, config.unet.output_channels, kernel_size=1)

        self.ae_model = ae_model

        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            mass_embedding = self.ae_model.encode(y)
            t += mass_embedding

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_linear_conditional(nn.Module):
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

        self.linear = nn.Linear(1,config.unet.time_emb)

        if config.data.num_classes is not None:
            self.label_emb = nn.Embedding(config.data.num_classes, config.unet.time_emb)
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        #print(t.shape)
        if y is not None:
            #mass_embedding = self.ae_model.encode(y)
            #m = self.linear(y)
            #m = y.repeat(1, self.time_dim)
            m = self.label_emb(y)
            #print(m.shape)
            #t += mass_embedding

        x1 = self.inc(x)
        x2 = self.down1(x1, t, m)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t, m)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t, m)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t, m)
        x = self.sa4(x)
        x = self.up2(x, x2, t, m)
        x = self.sa5(x)
        x = self.up3(x, x1, t, m)
        x = self.sa6(x)
        output = self.outc(x)
        return output

class UNet_mass_em_conditional(nn.Module):
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

        self.linear = nn.Linear(1,config.unet.time_emb)

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

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        #print(t.shape)
        if y is not None:
            #mass_embedding = self.ae_model.encode(y)
            #m = self.linear(y)
            m = self.pos_encoding(y, self.time_dim)
            #m = y.repeat(1, self.time_dim)
            #m = self.label_emb(y)
            #print(m.shape)
            #t += mass_embedding

        x1 = self.inc(x)
        x2 = self.down1(x1, t, m)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t, m)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t, m)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t, m)
        x = self.sa4(x)
        x = self.up2(x, x2, t, m)
        x = self.sa5(x)
        x = self.up3(x, x1, t, m)
        x = self.sa6(x)
        output = self.outc(x)
        return output