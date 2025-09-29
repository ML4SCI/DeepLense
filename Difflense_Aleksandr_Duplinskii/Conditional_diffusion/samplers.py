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
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, Union

from models import get_time_embedding_continuous
from torch.amp import GradScaler, autocast

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


# --- EMA Update ---
def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
            
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


def predict_x0_from_noise(x_t, t, pred_noise, alphas_cumprod):
    sqrt_alpha_bar_t = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
    return x0_hat.clamp(-1, 1)  # Optional clamp to valid range


def sample_epsilon(model, n_samples=1):
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

def sample_epsilon_conditional(model, x_cond, T=T):
    """
    Conditional sampling from diffusion model.

    Args:
        model: trained conditional diffusion model
        x_cond: conditioning image [B, 1, H, W] (e.g. blurred low-res image)
        T: number of diffusion steps

    Returns:
        Generated high-resolution image [B, 1, H, W] in [0, 1] range
    """
    model.eval()
    n_samples = x_cond.size(0)
    x = torch.randn_like(x_cond)  # Start from noise, same shape as condition

    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, device=DEVICE, dtype=torch.long)
            t_emb = get_timestep_embedding(t_batch, EMBED_DIM)

            eps_theta = model(x, t_emb, x_cond)  # 👈 Conditional input

            alpha_t = alphas[t_batch].view(-1, 1, 1, 1).to(DEVICE)
            alpha_bar_t = alphas_cumprod[t_batch].view(-1, 1, 1, 1).to(DEVICE)
            beta_t = betas[t_batch].view(-1, 1, 1, 1).to(DEVICE)

            mu = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta
            )

            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = mu + torch.sqrt(beta_t) * noise

    return (x + 1) / 2  # Return in [0, 1] range



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
    
@torch.no_grad()
def sample_v_conditional(model, x_cond, T=T):
    """
    Conditional sampling with v-parameterization.

    Args:
        model: trained conditional diffusion model that outputs v_pred
               signature: model(x_t, t_emb, x_cond) -> v_pred
        x_cond: conditioning image [B, 1, H, W] (e.g., LR/blurred)
        T: number of diffusion steps

    Returns:
        Generated high-resolution image [B, 1, H, W] in [0, 1].
    """
    model.eval()
    B = x_cond.size(0)
    x = torch.randn_like(x_cond, device=x_cond.device)  # x_T

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=x_cond.device, dtype=torch.long)
        t_emb   = get_timestep_embedding(t_batch, EMBED_DIM)

        # --- predict v (conditional) ---
        v_pred = model(x, t_emb, x_cond)  # same API as your epsilon-conditional model

        # --- convert v -> epsilon_hat ---
        alpha_bar_t = alphas_cumprod[t_batch].view(B, 1, 1, 1)          # \bar{alpha}_t
        sqrt_ab     = torch.sqrt(alpha_bar_t)                            # sqrt(\bar{alpha}_t)
        sigma_t     = torch.sqrt(1.0 - alpha_bar_t)                      # sqrt(1-\bar{alpha}_t)
        eps_hat     = sqrt_ab * v_pred + sigma_t * x                     # <- from your unconditional code

        # --- DDPM step (same as epsilon-param) ---
        alpha_t = alphas[t_batch].view(B, 1, 1, 1)
        beta_t  = betas[t_batch].view(B, 1, 1, 1)

        # mu_t = (1/sqrt(alpha_t)) * (x_t - ((1 - alpha_t)/sqrt(1 - \bar{alpha}_t)) * eps_hat)
        mu = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / sigma_t) * eps_hat)

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = mu + torch.sqrt(beta_t) * noise

    x = torch.clamp(x, -1.0, 1.0)
    return (x + 1.0) / 2.0

@torch.no_grad()
def sample_v_conditional_cfg(model, x_cond, T=T, guidance_scale=2.0, timesteps=None):
    """
    v-parameterized conditional sampling with classifier-free guidance.
    model: predicts v; signature model(x_t, t_emb, x_cond)
    x_cond: [B,1,H,W] in [-1,1]
    guidance_scale: s; 1.0 = no guidance; 2-3 gives stronger adherence to condition
    timesteps: optional list/1D tensor of descending ints; default uses all 0..T-1
    """
    device = x_cond.device
    B = x_cond.size(0)
    x = torch.randn_like(x_cond)

    if timesteps is None:
        t_schedule = torch.arange(T-1, -1, -1, device=device)
    else:
        t_schedule = torch.as_tensor(timesteps, device=device).long()

    zeros_cond = torch.zeros_like(x_cond)

    for t in t_schedule:
        t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
        t_emb   = get_timestep_embedding(t_batch, EMBED_DIM)

        # predict v for uncond and cond
        v_uncond = model(x, t_emb, zeros_cond)
        v_cond   = model(x, t_emb, x_cond)
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

        # convert v -> eps_hat
        alpha_bar_t = alphas_cumprod[t_batch].view(B,1,1,1)
        alpha_t = torch.sqrt(alpha_bar_t)
        sigma_t = torch.sqrt(1.0 - alpha_bar_t)
        eps_hat = alpha_t * v_guided + sigma_t * x

        # DDPM update
        alpha_t_scalar = alphas[t_batch].view(B,1,1,1)
        beta_t         = betas[t_batch].view(B,1,1,1)
        mu = (1.0 / torch.sqrt(alpha_t_scalar)) * (x - ((1.0 - alpha_t_scalar) / sigma_t) * eps_hat)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = mu + torch.sqrt(beta_t) * noise

    return (x.clamp(-1,1) + 1) / 2  # [0,1]

@torch.no_grad()
def sample_epsilon_conditional_cfg(
    model,
    x_cond,
    T=T,
    guidance_scale: float = 2.5,
    timesteps=None,
    to_image=True,
):
    """
    Classifier-free guidance sampler for ε-prediction models (DDPM update).

    model:  predicts epsilon; signature model(x_t, t_emb, x_cond, cond_flag)
    x_cond: [B,1,H,W] in [-1,1]
    T:      total diffusion steps (len of schedules)
    guidance_scale (s): 1.0 = no guidance; ~2–4 typical
    timesteps: optional 1D list/tensor of steps to run (DESCENDING). If None, uses T-1..0.
    to_image: if True, map from [-1,1] -> [0,1] at the end.

    Requires global (or outer-scope) 1D tensors on device:
      - betas            (len T)
      - alphas           (len T) where alphas = 1 - betas
      - alphas_cumprod   (len T) ᾱ_t = ∏_{i<=t} α_i
    """
    device = x_cond.device
    B = x_cond.size(0)
    x = torch.randn_like(x_cond)

    # default schedule: T-1, ..., 0
    if timesteps is None:
        t_schedule = torch.arange(T - 1, -1, -1, device=device)
    else:
        t_schedule = torch.as_tensor(timesteps, device=device).long()

    # handy indexer: take v[t] per-batch and shape to [B,1,1,1]
    def gather(v, t):  # v: [T], t: [B]
        return v[t].view(B, 1, 1, 1)

    for t in t_schedule:
        t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
        t_emb   = get_timestep_embedding(t_batch, EMBED_DIM)

        # ------- CFG: cond & uncond in ONE forward -------
        x_cat      = torch.cat([x, x], dim=0)
        t_emb_cat  = torch.cat([t_emb, t_emb], dim=0)
        xcond_cat  = torch.cat([x_cond, x_cond], dim=0)
        flags_cat  = torch.cat([torch.ones(B,1,device=device),  # cond
                                torch.zeros(B,1,device=device)], dim=0)  # uncond

        eps_cat = model(x_cat, t_emb_cat, xcond_cat, flags_cat)  # ε(x_t, t, cond_flag)
        eps_cond, eps_uncond = eps_cat[:B], eps_cat[B:]
        eps_hat = eps_uncond + guidance_scale * (eps_cond - eps_uncond)  # ε̂_guided

        # ------- DDPM update (use posterior variance \tilde{β}_t) -------
        alpha_t       = gather(alphas, t_batch)               # α_t
        beta_t        = gather(betas, t_batch)                # β_t
        alpha_bar_t   = gather(alphas_cumprod, t_batch)       # ᾱ_t
        prev_t        = torch.clamp(t_batch - 1, min=0)
        alpha_bar_tm1 = gather(alphas_cumprod, prev_t)        # ᾱ_{t-1}

        # mean μ_θ(x_t, t)
        # μ = (1/√α_t) * ( x_t - (β_t / √(1-ᾱ_t)) * ε̂ )
        sqrt_one_m_ab = torch.sqrt(1.0 - alpha_bar_t)
        mu = (x - (beta_t / sqrt_one_m_ab) * eps_hat) / torch.sqrt(alpha_t)

        # posterior variance \tilde{β}_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t
        tilde_beta = (1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t) * beta_t

        # sample next x
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = mu + torch.sqrt(tilde_beta) * noise

    if to_image:
        return (x.clamp(-1, 1) + 1) / 2.0
    return x

def cosine_timesteps(steps, device, eps=1e-3):
    s = torch.linspace(0, 1, steps + 1, device=device)
    return torch.cos((s * (1 - eps)) * (torch.pi / 2))  # 1 -> ~0



@torch.no_grad()
def sample_flow_conditional(
    model,
    x_cond,
    steps=100,
    solver="heun",       # "euler" or "heun"
    return_range="0_1",  # "0_1" or "minus1_1"
):
    """
    Conditional sampling for a flow-matching model (linear path).

    Args:
        model: trained conditional flow model (could be ema_model)
        x_cond: conditioning tensor [B, C, H, W]
        steps: number of ODE steps between t=1 -> t=0
        solver: "euler" (explicit Euler) or "heun" (RK2 predictor-corrector)
        return_range: "0_1" to map from [-1,1] -> [0,1], or "minus1_1" to keep [-1,1]

    Returns:
        x: generated sample [B, C, H, W]
    """
    device = x_cond.device
    model_was_training = model.training
    model.eval()

    # Initial condition: prior at t=1 (standard normal)
    x = torch.randn_like(x_cond)

    # Time grid from 1.0 -> 0.0
    # ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    ts = cosine_timesteps(steps, x_cond.device)

    for k in range(steps, 0, -1):
        t_curr = ts[k]          # scalar tensor
        t_prev = ts[k - 1]
        dt = t_curr - t_prev  # negative step size (since we go 1 -> 0)

        # Embed continuous time
        t_batch = t_curr.expand(x.size(0))
        t_emb = get_time_embedding_continuous(t_batch, EMBED_DIM)

        # with autocast(device_type='cuda'):
        v = model(x, t_emb, x_cond)  # predict velocity

        if solver.lower() == "euler":
            x = x + dt * v

        elif solver.lower() == "heun":
            # Predictor
            x_pred = x + dt * v

            # Corrector: evaluate at (x_pred, t_prev)
            t_batch_prev = t_prev.expand(x.size(0))
            t_emb_prev = get_time_embedding_continuous(t_batch_prev, EMBED_DIM)
            # with autocast(device_type='cuda'):
            v_pred = model(x_pred, t_emb_prev, x_cond)

            x = x + dt * 0.5 * (v + v_pred)

        else:
            raise ValueError("solver must be 'euler' or 'heun'")

    # Restore training mode if needed
    if model_was_training:
        model.train()

    if return_range == "0_1":
        return (x + 1) / 2
    return x


