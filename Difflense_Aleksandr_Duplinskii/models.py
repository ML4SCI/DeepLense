import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.utils.data import random_split
from einops import rearrange
import os
import random
from typing import Optional, Tuple, Union



def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = torch.clip(alphas_cumprod, 1e-8, 1.0)
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 1e-5, 0.999)
    return betas


def cosine_beta_schedule_custom(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Returns a cosine-shaped beta schedule scaled to [beta_start, beta_end].

    Args:
        timesteps (int): Number of diffusion steps.
        beta_start (float): Minimum beta value.
        beta_end (float): Maximum beta value.

    Returns:
        torch.Tensor: Tensor of shape (timesteps,) with the beta schedule.
    """
    x = torch.linspace(0, 1, timesteps)
    cosine = (1 + torch.cos(np.pi * x + np.pi)) / 2
    cosine = (cosine - cosine.min()) / (cosine.max() - cosine.min())  # Normalize to [0, 1]
    betas = beta_start + cosine * (beta_end - beta_start)  # Scale to desired range
    return betas



# --- Noise Schedule ---
# betas = torch.linspace(1e-4, 0.02, T).to(DEVICE)
betas = cosine_beta_schedule_custom(T).to(DEVICE)
# betas = cosine_beta_schedule(T).to(DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


# --- EMA Update ---
def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


# From Pranath code:
#-------------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g
        
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


#--------------------------------------------------------------------------------------------



# --- Timestep Embedding ---
def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half_dim).float() / half_dim).to(DEVICE)
    angles = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

def get_time_embedding_continuous(t, dim):
    """
    Continuous sinusoidal time embedding for flow matching.

    Args:
        t: Tensor of shape [batch] with values in [0, 1]
        dim: embedding dimension (must be even)

    Returns:
        Tensor of shape [batch, dim]
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
    )
    # multiply continuous times with frequencies
    angles = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

# --- Forward Diffusion ---
def q_sample(x0, t, noise):
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1).to(DEVICE)
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1).to(DEVICE)
    return sqrt_alphas_cumprod * x0 + sqrt_one_minus * noise

# --- Ensure shapes match before adding ---

def center_crop(tensor, target_height, target_width):
    _, _, h, w = tensor.shape
    start_h = (h - target_height) // 2
    start_w = (w - target_width) // 2
    return tensor[:, :, start_h:start_h + target_height, start_w:start_w + target_width]

# ---  U-Net ---
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.SiLU()
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.norm2 = nn.GroupNorm(1, out_ch)

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(self.conv1(x))
        h += self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        return h + self.res_conv(x)

    

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)  # Flatten spatial dims
        q, k, v = self.q(h), self.k(h), self.v(h)

        attn = torch.bmm(q.transpose(1, 2), k) / (C ** 0.5)  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)  # [B, C, HW]
        out = self.proj(out).view(B, C, H, W)

        return x + out
    
class AttnBlock(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.block1 = ResBlock(512, 512, time_emb_dim)
        self.attn   = SelfAttention(512)
        self.block2 = ResBlock(512, 512, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.attn(x)
        x = self.block2(x, t_emb)
        return x
    
    
class UNet_512_mix_attn(nn.Module):
    def __init__(self, time_emb_dim=EMBED_DIM):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Downsampling
        self.enc1 = ResBlock(1, 64, time_emb_dim)
        self.attn_enc1 = Residual(Rezero(LinearAttention(64)))
        self.enc2 = ResBlock(64, 128, time_emb_dim)
        self.attn_enc2 = Residual(Rezero(LinearAttention(128)))
        self.enc3 = ResBlock(128, 256, time_emb_dim)
        self.attn_enc3 = SelfAttention(256)
        self.enc4 = ResBlock(256, 512, time_emb_dim)

        # Bottleneck with attention
        self.middle = AttnBlock(time_emb_dim)

        # Upsampling — updated in_channels due to concatenation
        self.dec1 = ResBlock(512, 256, time_emb_dim)              # mid only
        self.attn_dec1 = Residual(Rezero(LinearAttention(256)))
        self.dec2 = ResBlock(256 + 256, 128, time_emb_dim)        # d1 + x3
        self.attn_dec2 = SelfAttention(128)
        self.dec3 = ResBlock(128 + 128, 64, time_emb_dim)         # d2 + x2
        self.attn_dec3 = Residual(Rezero(LinearAttention(64)))
        self.dec4 = ResBlock(64 + 64, 1, time_emb_dim)            # d3 + x1

    def forward(self, x, t_emb):
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.enc1(x, t_emb)
        x1 = self.attn_enc1(x1)
        x2 = self.enc2(F.avg_pool2d(x1, 2), t_emb)
        x2 = self.attn_enc2(x2)
        x3 = self.enc3(F.avg_pool2d(x2, 2), t_emb)
        x3 = self.attn_enc3(x3)
        x4 = self.enc4(F.avg_pool2d(x3, 2), t_emb)

        # Bottleneck
        mid = self.middle(x4, t_emb)

        # Decoder with concatenation
        d1 = F.interpolate(self.dec1(mid, t_emb), size=x3.shape[2:], mode='nearest')
        d1 = self.attn_dec1(d1)
        d2_input = torch.cat([d1, x3], dim=1)
        d2 = F.interpolate(self.dec2(d2_input, t_emb), size=x2.shape[2:], mode='nearest')
        d2 = self.attn_dec2(d2)
        d3_input = torch.cat([d2, x2], dim=1)
        d3 = F.interpolate(self.dec3(d3_input, t_emb), size=x1.shape[2:], mode='nearest')
        d3 = self.attn_dec3(d3)
        d4_input = torch.cat([d3, x1], dim=1)
        out = self.dec4(d4_input, t_emb)

        return out
    
class UNet_512_lin_attn(nn.Module):
    def __init__(self, time_emb_dim=EMBED_DIM):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Downsampling
        self.enc1 = ResBlock(1, 64, time_emb_dim)
        self.attn_enc1 = Residual(Rezero(LinearAttention(64)))
        self.enc2 = ResBlock(64, 128, time_emb_dim)
        self.attn_enc2 = Residual(Rezero(LinearAttention(128)))
        self.enc3 = ResBlock(128, 256, time_emb_dim)
        self.attn_enc3 = Residual(Rezero(LinearAttention(256)))
        self.enc4 = ResBlock(256, 512, time_emb_dim)

        # Bottleneck with attention
        self.middle = AttnBlock(time_emb_dim)

        # Upsampling — updated in_channels due to concatenation
        self.dec1 = ResBlock(512, 256, time_emb_dim)              # mid only
        self.attn_dec1 = Residual(Rezero(LinearAttention(256)))
        self.dec2 = ResBlock(256 + 256, 128, time_emb_dim)        # d1 + x3
        self.attn_dec2 = Residual(Rezero(LinearAttention(128)))
        self.dec3 = ResBlock(128 + 128, 64, time_emb_dim)         # d2 + x2
        self.attn_dec3 = Residual(Rezero(LinearAttention(64)))
        self.dec4 = ResBlock(64 + 64, 1, time_emb_dim)            # d3 + x1

    def forward(self, x, t_emb):
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.enc1(x, t_emb)
        x1 = self.attn_enc1(x1)
        x2 = self.enc2(F.avg_pool2d(x1, 2), t_emb)
        x2 = self.attn_enc2(x2)
        x3 = self.enc3(F.avg_pool2d(x2, 2), t_emb)
        x3 = self.attn_enc3(x3)
        x4 = self.enc4(F.avg_pool2d(x3, 2), t_emb)

        # Bottleneck
        mid = self.middle(x4, t_emb)

        # Decoder with concatenation
        d1 = F.interpolate(self.dec1(mid, t_emb), size=x3.shape[2:], mode='nearest')
        d1 = self.attn_dec1(d1)
        d2_input = torch.cat([d1, x3], dim=1)
        d2 = F.interpolate(self.dec2(d2_input, t_emb), size=x2.shape[2:], mode='nearest')
        d2 = self.attn_dec2(d2)
        d3_input = torch.cat([d2, x2], dim=1)
        d3 = F.interpolate(self.dec3(d3_input, t_emb), size=x1.shape[2:], mode='nearest')
        d3 = self.attn_dec3(d3)
        d4_input = torch.cat([d3, x1], dim=1)
        out = self.dec4(d4_input, t_emb)

        return out
    

class CondPyramid(nn.Module):
    """
    Lightweight LR feature pyramid producing c1 (H×W), c2 (H/2×W/2), c3 (H/4×W/4).
    Channel sizes match your encoder scales: 64, 128, 256.
    """
    def __init__(self, in_ch=1, chs=(64, 128, 256)):
        super().__init__()
        c1, c2, c3 = chs
        act = nn.SiLU()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1),
            act,
            nn.Conv2d(c1, c1, 3, padding=1),
            act,
        )
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)  # /2
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)  # /4
        self.block3 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(c3, c3, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x_cond):
        c1 = self.block1(x_cond)          # [B, 64,  H,   W]
        h  = self.down1(c1)               # [B, 128, H/2, W/2]
        c2 = self.block2(h)               # [B, 128, H/2, W/2]
        h  = self.down2(c2)               # [B, 256, H/4, W/4]
        c3 = self.block3(h)               # [B, 256, H/4, W/4]
        return c1, c2, c3


class UNet_512_mix_attn_conditional(nn.Module):
    def __init__(self, time_emb_dim=EMBED_DIM, use_cond_pyr: bool = True, use_global_cond: bool = True):
        super().__init__()
        self.use_cond_pyr = use_cond_pyr
        self.use_global_cond = use_global_cond
        
        if self.use_cond_pyr:
            self.gamma_c1 = nn.Parameter(torch.tensor(0.0))
            self.gamma_c2 = nn.Parameter(torch.tensor(0.0))
            self.gamma_c3 = nn.Parameter(torch.tensor(0.0))


        # Time embedding MLP (unchanged)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        
        
           # global conditioning path (GAP + tiny MLP), gated at 0
        if self.use_global_cond:
            self.cond_global_pool = nn.AdaptiveAvgPool2d(1)  # GAP over x_cond
            self.cond_mlp = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            self.gamma_cond = nn.Parameter(torch.tensor(0.5))
            self.mix_emb = nn.Linear(2 * time_emb_dim, time_emb_dim)
            # init so mix_emb([t, c]) ≈ t at step 0
            with torch.no_grad():
                self.mix_emb.weight.zero_()
                self.mix_emb.weight[:time_emb_dim, :time_emb_dim] = torch.eye(time_emb_dim)
                if self.mix_emb.bias is not None:
                    self.mix_emb.bias.zero_()

        # Encoder (unchanged, input has 2 channels: [noisy, LR])
        self.enc1 = ResBlock(2,   64, time_emb_dim)
        self.attn_enc1 = Residual(Rezero(LinearAttention(64)))
        self.enc2 = ResBlock(64,  128, time_emb_dim)
        self.attn_enc2 = Residual(Rezero(LinearAttention(128)))
        self.enc3 = ResBlock(128, 256, time_emb_dim)
        self.attn_enc3 = SelfAttention(256)
        self.enc4 = ResBlock(256, 512, time_emb_dim)

        self.middle = AttnBlock(time_emb_dim)

        # Decoder — channel sizes depend on whether we concat cond features
        if self.use_cond_pyr:
            # d2: [d1(256), x3(256), c3(256)] -> 128
            self.dec1 = ResBlock(512,              256, time_emb_dim)
            self.attn_dec1 = Residual(Rezero(LinearAttention(256)))
            self.dec2 = ResBlock(256 + 256 + 256,  128, time_emb_dim)
            self.attn_dec2 = SelfAttention(128)
            # d3: [d2(128), x2(128), c2(128)] -> 64
            self.dec3 = ResBlock(128 + 128 + 128,   64, time_emb_dim)
            self.attn_dec3 = Residual(Rezero(LinearAttention(64)))
            # d4: [d3(64), x1(64), c1(64)] -> 1
            self.dec4 = ResBlock(64 + 64 + 64,       1, time_emb_dim)

            self.cond_pyr = CondPyramid(in_ch=1, chs=(64, 128, 256))
        else:
            # Original decoder sizes (no cond features)
            self.dec1 = ResBlock(512,        256, time_emb_dim)
            self.attn_dec1 = Residual(Rezero(LinearAttention(256)))
            self.dec2 = ResBlock(256 + 256,  128, time_emb_dim)  # [d1, x3]
            self.attn_dec2 = SelfAttention(128)
            self.dec3 = ResBlock(128 + 128,   64, time_emb_dim)  # [d2, x2]
            self.attn_dec3 = Residual(Rezero(LinearAttention(64)))
            self.dec4 = ResBlock(64 + 64,      1, time_emb_dim)  # [d3, x1]

            self.cond_pyr = None  # not used

    def forward(self, x, t_emb, x_cond):
        """
        x:      [B, 1, H, W]   noisy image at step t
        x_cond: [B, 1, H, W]   LR conditional image
        t_emb:  [B, EMBED_DIM]
        """
        t_emb = self.time_mlp(t_emb)  # unchanged
        
        # ✅ inject global LR conditioning into time embedding (FiLM-style)
        if self.use_global_cond:
            c_global = self.cond_global_pool(x_cond).flatten(1)   # [B,1]
            c_emb = self.cond_mlp(c_global)                       # [B, EMBED_DIM]
            t_cat = torch.cat([t_emb, self.gamma_cond * c_emb], dim=1)  # [B, 2D]
            t_emb = self.mix_emb(t_cat)                                  # [B, D]

        # Concatenate condition at input (unchanged)
        x_in = torch.cat([x, x_cond], dim=1)  # [B,2,H,W]

        # Encoder
        x1 = self.enc1(x_in, t_emb)                 # [B,64,H,W]
        x1 = self.attn_enc1(x1)
        x2 = self.enc2(F.avg_pool2d(x1, 2), t_emb)  # [B,128,H/2,W/2]
        x2 = self.attn_enc2(x2)
        x3 = self.enc3(F.avg_pool2d(x2, 2), t_emb)  # [B,256,H/4,W/4]
        x3 = self.attn_enc3(x3)
        x4 = self.enc4(F.avg_pool2d(x3, 2), t_emb)  # [B,512,H/8,W/8]

        # Bottleneck
        mid = self.middle(x4, t_emb)

        # Multi-scale LR features (only if enabled)
        if self.use_cond_pyr:
            c1, c2, c3 = self.cond_pyr(x_cond)      # [B,64,H,W], [B,128,H/2,W/2], [B,256,H/4,W/4]
            c1s = self.gamma_c1 * c1
            c2s = self.gamma_c2 * c2
            c3s = self.gamma_c3 * c3

        # Decoder
        d1 = F.interpolate(self.dec1(mid, t_emb), size=x3.shape[2:], mode='nearest')
        d1 = self.attn_dec1(d1)

        if self.use_cond_pyr:
            d2_input = torch.cat([d1, x3, c3s], dim=1)
        else:
            d2_input = torch.cat([d1, x3], dim=1)
        d2 = F.interpolate(self.dec2(d2_input, t_emb), size=x2.shape[2:], mode='nearest')
        d2 = self.attn_dec2(d2)

        if self.use_cond_pyr:
            d3_input = torch.cat([d2, x2, c2s], dim=1)
        else:
            d3_input = torch.cat([d2, x2], dim=1)
        d3 = F.interpolate(self.dec3(d3_input, t_emb), size=x1.shape[2:], mode='nearest')
        d3 = self.attn_dec3(d3)

        if self.use_cond_pyr:
            d4_input = torch.cat([d3, x1, c1s], dim=1)
        else:
            d4_input = torch.cat([d3, x1], dim=1)
        out = self.dec4(d4_input, t_emb)

        return out
    
class UNet_512_mix_attn_conditional_cfg(nn.Module):
    def __init__(self, time_emb_dim=EMBED_DIM, use_cond_pyr: bool = True, use_global_cond: bool = True):
        super().__init__()
        self.use_cond_pyr = use_cond_pyr
        self.use_global_cond = use_global_cond

        if self.use_cond_pyr:
            self.gamma_c1 = nn.Parameter(torch.tensor(0.0))
            self.gamma_c2 = nn.Parameter(torch.tensor(0.0))
            self.gamma_c3 = nn.Parameter(torch.tensor(0.0))

        # Time embedding MLP (unchanged)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # 👇 NEW: tiny MLP for the cond flag (scalar -> emb)
        # cond_flag convention: 1 = condition present, 0 = dropped (uncond)
        self.flag_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.gamma_flag = nn.Parameter(torch.tensor(1.0))  # scale for flag injection

        # global conditioning path (GAP + tiny MLP), gated at 0
        if self.use_global_cond:
            self.cond_global_pool = nn.AdaptiveAvgPool2d(1)  # GAP over x_cond
            self.cond_mlp = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            self.gamma_cond = nn.Parameter(torch.tensor(0.5))
            self.mix_emb = nn.Linear(2 * time_emb_dim, time_emb_dim)
            with torch.no_grad():
                self.mix_emb.weight.zero_()
                self.mix_emb.weight[:time_emb_dim, :time_emb_dim] = torch.eye(time_emb_dim)
                if self.mix_emb.bias is not None:
                    self.mix_emb.bias.zero_()

            # 👇 NEW: learned null for the global-cond embedding path (optional but helpful)
            self.c_null_emb = nn.Parameter(torch.zeros(time_emb_dim))

        # Encoder (unchanged)
        self.enc1 = ResBlock(2,   64, time_emb_dim)
        self.attn_enc1 = Residual(Rezero(LinearAttention(64)))
        self.enc2 = ResBlock(64,  128, time_emb_dim)
        self.attn_enc2 = Residual(Rezero(LinearAttention(128)))
        self.enc3 = ResBlock(128, 256, time_emb_dim)
        self.attn_enc3 = SelfAttention(256)
        self.enc4 = ResBlock(256, 512, time_emb_dim)

        self.middle = AttnBlock(time_emb_dim)

        # Decoder — channel sizes depend on whether we concat cond features
        if self.use_cond_pyr:
            self.dec1 = ResBlock(512,              256, time_emb_dim)
            self.attn_dec1 = Residual(Rezero(LinearAttention(256)))
            self.dec2 = ResBlock(256 + 256 + 256,  128, time_emb_dim)
            self.attn_dec2 = SelfAttention(128)
            self.dec3 = ResBlock(128 + 128 + 128,   64, time_emb_dim)
            self.attn_dec3 = Residual(Rezero(LinearAttention(64)))
            self.dec4 = ResBlock(64 + 64 + 64,       1, time_emb_dim)

            self.cond_pyr = CondPyramid(in_ch=1, chs=(64, 128, 256))
        else:
            self.dec1 = ResBlock(512,        256, time_emb_dim)
            self.attn_dec1 = Residual(Rezero(LinearAttention(256)))
            self.dec2 = ResBlock(256 + 256,  128, time_emb_dim)
            self.attn_dec2 = SelfAttention(128)
            self.dec3 = ResBlock(128 + 128,   64, time_emb_dim)
            self.attn_dec3 = Residual(Rezero(LinearAttention(64)))
            self.dec4 = ResBlock(64 + 64,      1, time_emb_dim)
            self.cond_pyr = None

    def forward(self, x, t_emb, x_cond, cond_flag=None):
        """
        x:         [B, 1, H, W]   noisy image at step t
        x_cond:    [B, 1, H, W]   LR conditional image
        t_emb:     [B, EMBED_DIM]
        cond_flag: [B, 1]         1 = condition present, 0 = dropped (uncond). If None, assume 1.
        """
        B = x.shape[0]
        device = x.device

        if cond_flag is None:
            cond_flag = torch.ones(B, 1, device=device)

        # helpful broadcasters
        def b11(flag):  # [B,1,1,1] for feature gating
            return flag.view(B, 1, 1, 1)
        def b1(flag):   # [B,1] for embedding blending
            return flag.view(B, 1)

        # Time embedding base
        t_emb = self.time_mlp(t_emb)
        
        t_emb = t_emb + self.gamma_flag * self.flag_mlp(cond_flag)

        if self.use_global_cond:
            c_global = self.cond_global_pool(x_cond).flatten(1)      # [B,1]
            c_emb = self.cond_mlp(c_global)                          # [B,D]
            # Blend with learned null when dropped
            c_emb_eff = b1(cond_flag) * c_emb + (1.0 - b1(cond_flag)) * self.c_null_emb.unsqueeze(0)
            t_cat = torch.cat([t_emb, self.gamma_cond * c_emb_eff], dim=1)  # [B,2D]
            t_emb = self.mix_emb(t_cat)                                      # [B,D]

        # Concatenate condition at input, gated
        x_in = torch.cat([x, b11(cond_flag) * x_cond], dim=1)  # [B,2,H,W]

        # Encoder
        x1 = self.enc1(x_in, t_emb)
        x1 = self.attn_enc1(x1)
        x2 = self.enc2(F.avg_pool2d(x1, 2), t_emb)
        x2 = self.attn_enc2(x2)
        x3 = self.enc3(F.avg_pool2d(x2, 2), t_emb)
        x3 = self.attn_enc3(x3)
        x4 = self.enc4(F.avg_pool2d(x3, 2), t_emb)

        # Bottleneck
        mid = self.middle(x4, t_emb)

        # Multi-scale LR features (only if enabled) — gated
        if self.use_cond_pyr:
            c1, c2, c3 = self.cond_pyr(x_cond)
            c1s = b11(cond_flag) * (self.gamma_c1 * c1)
            c2s = b11(cond_flag) * (self.gamma_c2 * c2)
            c3s = b11(cond_flag) * (self.gamma_c3 * c3)

        # Decoder
        d1 = F.interpolate(self.dec1(mid, t_emb), size=x3.shape[2:], mode='nearest')
        d1 = self.attn_dec1(d1)

        d2_input = torch.cat([d1, x3, c3s], dim=1) if self.use_cond_pyr else torch.cat([d1, x3], dim=1)
        d2 = F.interpolate(self.dec2(d2_input, t_emb), size=x2.shape[2:], mode='nearest')
        d2 = self.attn_dec2(d2)

        d3_input = torch.cat([d2, x2, c2s], dim=1) if self.use_cond_pyr else torch.cat([d2, x2], dim=1)
        d3 = F.interpolate(self.dec3(d3_input, t_emb), size=x1.shape[2:], mode='nearest')
        d3 = self.attn_dec3(d3)

        d4_input = torch.cat([d3, x1, c1s], dim=1) if self.use_cond_pyr else torch.cat([d3, x1], dim=1)
        out = self.dec4(d4_input, t_emb)
        return out

    
def filter_by_shape(target_model, old_state):
    """Keep only tensors whose keys exist in target and shapes match."""
    new_sd = target_model.state_dict()
    filt = {k:v for k,v in old_state.items() if (k in new_sd and new_sd[k].shape == v.shape)}
    return filt


def get_state_dict(ckpt_path, key_candidates=("model", "state_dict", "ema_model")):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in key_candidates:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    # fallback: assume the file itself is a state_dict
    return ckpt

def strip_module_prefix(state):
    # remove leading "module." if present (from DataParallel)
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

@torch.no_grad()
def seed_widened_conv_from_old(new_conv, old_conv):
    """Copy old conv weights into the left slice of new_conv's input channels; zero the rest."""
    w_new = new_conv.weight
    w_new.zero_()
    in_old = old_conv.weight.shape[1]
    w_new[:, :in_old, :, :].copy_(old_conv.weight)
    if new_conv.bias is not None and old_conv.bias is not None:
        new_conv.bias.copy_(old_conv.bias)

def transplant_decoder_block_(new_block, old_block, extra_in_channels):
    """
    Seed widened decoder block (your ResBlock) from old one.
    Handles conv1 (in_channels changed) and res_conv (1x1 projection) if present.
    """
    # conv1 widened: [out, in_old + extra, k, k]
    seed_widened_conv_from_old(new_block.conv1, old_block.conv1)

    # conv2 usually same shape — copy directly if it matches
    if new_block.conv2.weight.shape == old_block.conv2.weight.shape:
        with torch.no_grad():
            new_block.conv2.weight.copy_(old_block.conv2.weight)
            if getattr(new_block.conv2, "bias", None) is not None and getattr(old_block.conv2, "bias", None) is not None:
                new_block.conv2.bias.copy_(old_block.conv2.bias)

    # residual 1x1 projection may also be widened
    if hasattr(new_block, "res_conv") and hasattr(old_block, "res_conv"):
        if new_block.res_conv.weight.shape[1] != old_block.res_conv.weight.shape[1]:
            seed_widened_conv_from_old(new_block.res_conv, old_block.res_conv)
        else:
            with torch.no_grad():
                new_block.res_conv.weight.copy_(old_block.res_conv.weight)
                if getattr(new_block.res_conv, "bias", None) is not None and getattr(old_block.res_conv, "bias", None) is not None:
                    new_block.res_conv.bias.copy_(old_block.res_conv.bias)


class UNet_512_mix_attn_conditional_flow(nn.Module):
    def __init__(self, time_emb_dim=EMBED_DIM, use_cond_pyr: bool = True, use_global_cond: bool = True):
        super().__init__()
        self.use_cond_pyr = use_cond_pyr
        self.use_global_cond = use_global_cond
        
        if self.use_cond_pyr:
            self.gamma_c1 = nn.Parameter(torch.tensor(0.0))
            self.gamma_c2 = nn.Parameter(torch.tensor(0.0))
            self.gamma_c3 = nn.Parameter(torch.tensor(0.0))


        # Time embedding MLP (unchanged)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
           # global conditioning path (GAP + tiny MLP), gated at 0
        if self.use_global_cond:
            self.cond_global_pool = nn.AdaptiveAvgPool2d(1)  # GAP over x_cond
            self.cond_mlp = nn.Sequential(
                nn.Linear(1, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            self.gamma_cond = nn.Parameter(torch.tensor(0.5))
            self.mix_emb = nn.Linear(2 * time_emb_dim, time_emb_dim)
            # init so mix_emb([t, c]) ≈ t at step 0
            with torch.no_grad():
                self.mix_emb.weight.zero_()
                self.mix_emb.weight[:time_emb_dim, :time_emb_dim] = torch.eye(time_emb_dim)
                if self.mix_emb.bias is not None:
                    self.mix_emb.bias.zero_()

        # Encoder (unchanged, input has 2 channels: [noisy, LR])
        self.enc1 = ResBlock(2,   64, time_emb_dim)
        self.attn_enc1 = Residual(Rezero(LinearAttention(64)))
        self.enc2 = ResBlock(64,  128, time_emb_dim)
        self.attn_enc2 = Residual(Rezero(LinearAttention(128)))
        self.enc3 = ResBlock(128, 256, time_emb_dim)
        self.attn_enc3 = SelfAttention(256)
        self.enc4 = ResBlock(256, 512, time_emb_dim)

        self.middle = AttnBlock(time_emb_dim)

        # Decoder — channel sizes depend on whether we concat cond features
        if self.use_cond_pyr:
            # d2: [d1(256), x3(256), c3(256)] -> 128
            self.dec1 = ResBlock(512,              256, time_emb_dim)
            self.attn_dec1 = Residual(Rezero(LinearAttention(256)))
            self.dec2 = ResBlock(256 + 256 + 256,  128, time_emb_dim)
            self.attn_dec2 = SelfAttention(128)
            # d3: [d2(128), x2(128), c2(128)] -> 64
            self.dec3 = ResBlock(128 + 128 + 128,   64, time_emb_dim)
            self.attn_dec3 = Residual(Rezero(LinearAttention(64)))
            # d4: [d3(64), x1(64), c1(64)] -> 1
            self.dec4 = ResBlock(64 + 64 + 64,       1, time_emb_dim)

            self.cond_pyr = CondPyramid(in_ch=1, chs=(64, 128, 256))
        else:
            # Original decoder sizes (no cond features)
            self.dec1 = ResBlock(512,        256, time_emb_dim)
            self.attn_dec1 = Residual(Rezero(LinearAttention(256)))
            self.dec2 = ResBlock(256 + 256,  128, time_emb_dim)  # [d1, x3]
            self.attn_dec2 = SelfAttention(128)
            self.dec3 = ResBlock(128 + 128,   64, time_emb_dim)  # [d2, x2]
            self.attn_dec3 = Residual(Rezero(LinearAttention(64)))
            self.dec4 = ResBlock(64 + 64,      1, time_emb_dim)  # [d3, x1]

            self.cond_pyr = None  # not used

    def forward(self, x, t_emb, x_cond):
        """
        x:      [B, 1, H, W]   noisy image at step t
        x_cond: [B, 1, H, W]   LR conditional image
        t_emb:  [B, EMBED_DIM]
        """
        t_emb = self.time_mlp(t_emb)  # unchanged
        
        #  inject global LR conditioning into time embedding (FiLM-style)
        if self.use_global_cond:
            c_global = self.cond_global_pool(x_cond).flatten(1)   # [B,1]
            c_emb = self.cond_mlp(c_global)                       # [B, EMBED_DIM]
            t_cat = torch.cat([t_emb, self.gamma_cond * c_emb], dim=1)  # [B, 2D]
            t_emb = self.mix_emb(t_cat)                                  # [B, D]

        # Concatenate condition at input (unchanged)
        x_in = torch.cat([x, x_cond], dim=1)  # [B,2,H,W]

        # Encoder
        x1 = self.enc1(x_in, t_emb)                 # [B,64,H,W]
        x1 = self.attn_enc1(x1)
        x2 = self.enc2(F.avg_pool2d(x1, 2), t_emb)  # [B,128,H/2,W/2]
        x2 = self.attn_enc2(x2)
        x3 = self.enc3(F.avg_pool2d(x2, 2), t_emb)  # [B,256,H/4,W/4]
        x3 = self.attn_enc3(x3)
        x4 = self.enc4(F.avg_pool2d(x3, 2), t_emb)  # [B,512,H/8,W/8]

        # Bottleneck
        mid = self.middle(x4, t_emb)

        # Multi-scale LR features (only if enabled)
        if self.use_cond_pyr:
            c1, c2, c3 = self.cond_pyr(x_cond)      # [B,64,H,W], [B,128,H/2,W/2], [B,256,H/4,W/4]
            c1s = self.gamma_c1 * c1
            c2s = self.gamma_c2 * c2
            c3s = self.gamma_c3 * c3

        # Decoder
        d1 = F.interpolate(self.dec1(mid, t_emb), size=x3.shape[2:], mode='bilinear')
        d1 = self.attn_dec1(d1)

        if self.use_cond_pyr:
            d2_input = torch.cat([d1, x3, c3s], dim=1)
        else:
            d2_input = torch.cat([d1, x3], dim=1)
        d2 = F.interpolate(self.dec2(d2_input, t_emb), size=x2.shape[2:], mode='bilinear')
        d2 = self.attn_dec2(d2)

        if self.use_cond_pyr:
            d3_input = torch.cat([d2, x2, c2s], dim=1)
        else:
            d3_input = torch.cat([d2, x2], dim=1)
        d3 = F.interpolate(self.dec3(d3_input, t_emb), size=x1.shape[2:], mode='bilinear')
        d3 = self.attn_dec3(d3)

        if self.use_cond_pyr:
            d4_input = torch.cat([d3, x1, c1s], dim=1)
        else:
            d4_input = torch.cat([d3, x1], dim=1)
        out = self.dec4(d4_input, t_emb)

        return out