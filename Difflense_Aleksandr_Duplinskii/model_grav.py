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

# --- Hyperparameters ---
T = 1000
BATCH_SIZE = 256
IMG_SIZE = 64
EMBED_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    

    
#---------------------------- Model I --------------------------------------------------------

# # --- Dataset ---
# class PTImageDataset(torch.utils.data.Dataset):
#     def __init__(self, pt_path, downsample_size=None):
#         self.data = torch.load(pt_path)
#         self.data = (self.data - 0.5) * 2  # normalize to [-1, 1]
#         self.downsample_size = downsample_size  # e.g., (75, 75)

#     def __len__(self):
#         return self.data.size(0)

#     def __getitem__(self, idx):
#         img = self.data[idx]  # shape: [1, H, W]
#         if self.downsample_size:
#             img = F.interpolate(
#                 img.unsqueeze(0),
#                 size=self.downsample_size,
#                 mode='bilinear',
#                 align_corners=False
#             ).squeeze(0)
#         return img, 0  # dummy label for uncond model

# # full_dataset = PTImageDataset("all_samples.pt")
# full_dataset = PTImageDataset("all_samples.pt")#, downsample_size=(75, 75))


# n_total = len(full_dataset)
# n_train = int(0.9 * n_total)
# n_test = n_total - n_train

# train_data, test_data = random_split(full_dataset, [n_train, n_test])

# train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)




#---------------------------- Model II --------------------------------------------------------


class NpyDiffusionDatasetLog(torch.utils.data.Dataset):
    def __init__(self, root_folder, downsample_size=None, apply_log=True, normalize=True, log_eps=1e-6):
        self.image_paths = []
        self.labels = []
        self.class_to_label = {'axion': 0, 'cdm': 1, 'no_sub': 2}
        self.downsample_size = downsample_size
        self.apply_log = apply_log
        self.normalize = normalize
        self.log_eps = log_eps

        for class_name, label in self.class_to_label.items():
            class_folder = os.path.join(root_folder, class_name)
            if not os.path.isdir(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if fname.endswith('.npy'):
                    self.image_paths.append(os.path.join(class_folder, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        data = np.load(path, allow_pickle=True)

        if label == 0:  # axion
            img = data[0]
        else:
            img = data

        img = img.astype(np.float32)

        # === Log scale transform ===
        if self.apply_log:
            img = np.log(img + self.log_eps)

        # === Normalize log image to [-1, 1] ===
        if self.normalize:
            log_min, log_max = img.min(), img.max()
            if log_max - log_min > 1e-8:
                img = (img - log_min) / (log_max - log_min)
                img = (img - 0.5) * 2
            else:
                img = np.zeros_like(img)

        if img.ndim == 2:
            img = img[None, :, :]

        img = torch.tensor(img, dtype=torch.float32)

        if self.downsample_size:
            img = F.interpolate(img.unsqueeze(0), size=self.downsample_size, mode='bilinear', align_corners=False).squeeze(0)

        return img, label



class NpyDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, downsample_size=None):
        self.image_paths = []
        self.labels = []
        self.class_to_label = {'axion': 0, 'cdm': 1, 'no_sub': 2}
        self.downsample_size = downsample_size

        # Walk through folders and store paths + labels
        for class_name, label in self.class_to_label.items():
            class_folder = os.path.join(root_folder, class_name)
            if not os.path.isdir(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if fname.endswith('.npy'):
                    self.image_paths.append(os.path.join(class_folder, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and parse based on label (e.g., axion = [image, intensity])
        data = np.load(path, allow_pickle=True)
        if label == 0:
            img = data[0]  # axion: [image, intensity]
        else:
            img = data     # others: plain image

        # Validate and normalize to [0, 1]
        if not isinstance(img, np.ndarray) or img.ndim < 2:
            raise ValueError(f"Invalid image at {path}, got shape {getattr(img, 'shape', None)}")

        img = img.astype(np.float32)
        img_min, img_max = np.min(img), np.max(img)
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # Add channel dim if needed
        if img.ndim == 2:
            img = img[None, :, :]

        img = torch.tensor(img, dtype=torch.float32)

        # Resize if needed
        if self.downsample_size:
            img = F.interpolate(img.unsqueeze(0), size=self.downsample_size, mode='bilinear', align_corners=False).squeeze(0)

        # Normalize to [-1, 1]
        img = (img - 0.5) * 2

        return img, label

# Training dataset and loader
train_ds = NpyDiffusionDataset('./Model_II')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=True)

# Test/validation dataset and loader
test_ds = NpyDiffusionDataset('./Model_II_test')
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


#--------------------------------------------------------------------------------------------



# --- Timestep Embedding ---
def get_timestep_embedding(timesteps, dim):
    half_dim = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half_dim).float() / half_dim).to(DEVICE)
    angles = timesteps[:, None].float() * freqs[None, :]
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

    
class UNet_512(nn.Module):
    def __init__(self, time_emb_dim=EMBED_DIM):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Downsampling path
        self.enc1 = ResBlock(1, 64, time_emb_dim)
        self.enc2 = ResBlock(64, 128, time_emb_dim)
        self.enc3 = ResBlock(128, 256, time_emb_dim)
        self.enc4 = ResBlock(256, 512, time_emb_dim)

        # Bottleneck
        self.middle = ResBlock(512, 512, time_emb_dim)

        # Upsampling path
        self.dec1 = ResBlock(512, 256, time_emb_dim)
        self.dec2 = ResBlock(256, 128, time_emb_dim)
        self.dec3 = ResBlock(128, 64, time_emb_dim)
        self.dec4 = ResBlock(64, 1, time_emb_dim)

    def forward(self, x, t_emb):
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.enc1(x, t_emb)                        # [B, 64, H, W]
        x2 = self.enc2(F.avg_pool2d(x1, 2), t_emb)      # [B, 128, H/2, W/2]
        x3 = self.enc3(F.avg_pool2d(x2, 2), t_emb)      # [B, 256, H/4, W/4]
        x4 = self.enc4(F.avg_pool2d(x3, 2), t_emb)      # [B, 512, H/8, W/8]

        mid = self.middle(x4, t_emb)

        # Decoder
        d1 = F.interpolate(self.dec1(mid, t_emb), size=x3.shape[2:], mode='nearest')
        d2 = F.interpolate(self.dec2(d1 + x3, t_emb), size=x2.shape[2:], mode='nearest')
        d3 = F.interpolate(self.dec3(d2 + x2, t_emb), size=x1.shape[2:], mode='nearest')
        out = self.dec4(d3 + x1, t_emb)

        return out



    
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
    

model = UNet_512_mix_attn().to(DEVICE)


def sample_epsilon(model, n_samples=1):
    model.eval()
    x = torch.randn(n_samples, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=DEVICE, dtype=torch.long)
        t_emb = get_timestep_embedding(t_batch, EMBED_DIM)

        with torch.no_grad():
            eps_theta = model(x, t_emb)
            # print(f"[t={t}] eps_theta stats — mean: {eps_theta.mean().item():.4f}, std: {eps_theta.std().item():.4f}")


        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        # Compute exact DDPM mean
        mu = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta)

        # Sample x_{t-1}
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = mu + torch.sqrt(beta_t) * noise

    return (x+1)/2

def sample_epsilon_v2(model, n_samples=1):
    model.eval()
    x = torch.randn(n_samples, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)  # Start with x_T

    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, device=DEVICE, dtype=torch.long)
            t_emb = get_timestep_embedding(t_batch, EMBED_DIM)

            eps_theta = model(x, t_emb)
            # print(f"[t={t}] eps_theta stats — mean: {eps_theta.mean().item():.4f}, std: {eps_theta.std().item():.4f}")

            alpha_t = alphas[t_batch].view(-1, 1, 1, 1).to(DEVICE)
            alpha_bar_t = alphas_cumprod[t_batch].view(-1, 1, 1, 1).to(DEVICE)
            beta_t = betas[t_batch].view(-1, 1, 1, 1).to(DEVICE)
            
            assert eps_theta.shape == x.shape, "Prediction shape mismatch"
            assert alpha_t.shape == x.shape[:1] + (1, 1, 1), "alpha_t shape mismatch"
            assert not torch.isnan(eps_theta).any(), "NaNs in prediction"
            assert not torch.isinf(eps_theta).any(), "Infs in prediction"

            mu = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta
            )

            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = mu + torch.sqrt(beta_t) * noise
    return (x + 1) / 2


def sample_v(model, n_samples=1):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)  # x_T

        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, device=DEVICE, dtype=torch.long)
            t_emb   = get_timestep_embedding(t_batch, EMBED_DIM)

            # Predict v
            v_pred = model(x, t_emb)
            # v_pred = model(x, t)

            # Get per-sample scalars
            alpha_bar_t = alphas_cumprod[t_batch].view(-1, 1, 1, 1)  # [B,1,1,1]
            sqrt_ab     = torch.sqrt(alpha_bar_t)
            sigma_t     = torch.sqrt(1. - alpha_bar_t)

            # Recover predicted noise ε̂
            eps_theta = sqrt_ab * v_pred + sigma_t * x

            alpha_t = alphas[t_batch].view(-1, 1, 1, 1)
            beta_t  = betas[t_batch].view(-1, 1, 1, 1)

            mu = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / sigma_t * eps_theta)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = mu + torch.sqrt(beta_t) * noise

        # Final output should be in [-1, 1]
        x = torch.clamp(x, -1., 1.)

        # Optional: convert to [0, 1] if needed for display
        # return F.interpolate((x + 1) / 2, size=(150, 150), mode='bilinear', align_corners=False)
        return (x + 1) / 2
