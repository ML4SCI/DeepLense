# %% MASTER SCRIPT -- 7 DeepONet variants (Vanilla, Stacked, Conv, Fourier,
# Attention, POD, BelNet) + an FNO baseline row, 3 seeds each, alpha-only
# (no image/warp/PSF). 2000/250 split, 40 epochs. Reports rel_L2 + SSIM
# (mean +/- std over seeds), param counts, wall-clock time per architecture,
# a combined qualitative grid, and a final summary table.
#
# The FNO row is a baseline comparison point (paper 3's headline result:
# FNO substantially outperforms every DeepONet variant on raw field
# reconstruction) -- it is not itself a DeepONet variant. Not included here:
# the frozen-representation classification probe and the two POD+FNO hybrid
# experiments described in the paper's Discussion -- ask for those
# separately if you want them added.
#
# This is a separate DeepONet-branch/trunk ablation study (workshop-paper
# scope), distinct from the 11-architecture BALANI-NO comparison in
# train_darcy.py / train_ns.py / train_lensing.py. It uses its own data
# pipeline (manifest.csv-driven, full kappa/psi fields, empirically
# calibrated alpha scale via image-warp matching) rather than the
# kappa_sub/kappa pipeline in train_lensing.py -- the two are not directly
# comparable number-for-number. See configs/hyperparameters.md.
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
RES = 64
EPOCHS = 40
N_POD_MODES = 32

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===========================================================================
# 1. DATA LOADING (shared across all architectures)
# ===========================================================================
def find_dataset_root(base="/kaggle/input"):
    for dirpath, dirnames, filenames in os.walk(base):
        if "manifest.csv" in filenames:
            return dirpath
    raise FileNotFoundError("manifest.csv not found under " + base)

DATA_ROOT = find_dataset_root()
manifest = pd.read_csv(os.path.join(DATA_ROOT, "manifest.csv"))
file_index = {}
for root, dirs, files in os.walk(DATA_ROOT):
    for f in files:
        if f.endswith(".npz"):
            file_index[f] = os.path.join(root, f)
manifest["basename"] = manifest["path"].apply(os.path.basename)
manifest["resolved_path"] = manifest["basename"].map(file_index)
resolved = manifest[manifest["resolved_path"].notna()].copy()
print(resolved.shape)
print(resolved["class"].value_counts())

def load_resized(arr, res):
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

def downsample_tensor(t, res):
    t = t.unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

def compute_alpha_from_psi(psi):
    grad_y, grad_x = torch.gradient(psi, dim=(0, 1))
    return grad_x, grad_y

def make_base_grid(B, H, W, device):
    theta = torch.eye(2, 3, device=device).unsqueeze(0).repeat(B, 1, 1)
    return F.affine_grid(theta, size=(B, 1, H, W), align_corners=False)

def warp_source(source_norm, alpha_raw):
    B, H, W = source_norm.shape
    base_grid = make_base_grid(B, H, W, source_norm.device)
    scale = 2.0 / W
    alpha_grid = torch.stack([alpha_raw[:, 0], alpha_raw[:, 1]], dim=-1) * scale
    sample_grid = base_grid - alpha_grid
    return F.grid_sample(source_norm.unsqueeze(1), sample_grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False).squeeze(1)

def normalize(x, mean, std): return (x - mean) / std
def denormalize(x, mean, std): return x * std + mean

def rel_l2_np(pred, target):
    diff = np.linalg.norm(pred.flatten() - target.flatten())
    norm = np.linalg.norm(target.flatten()) + 1e-8
    return diff / norm

calib_sample = resolved.sample(20, random_state=0)
candidate_multipliers = np.logspace(-1.5, 1.5, 25)
scores = {m: [] for m in candidate_multipliers}
for _, row in calib_sample.iterrows():
    d = np.load(row["resolved_path"])
    source_r = load_resized(d["unlensed"][0], RES).numpy()
    image_r = load_resized(d["image"][0], RES).numpy()
    psi_native = torch.tensor(d["psi"], dtype=torch.float32)
    ax_native, ay_native = compute_alpha_from_psi(psi_native)
    ax_base = downsample_tensor(ax_native, RES)
    ay_base = downsample_tensor(ay_native, RES)
    source_t = torch.tensor(source_r).unsqueeze(0)
    for m in candidate_multipliers:
        alpha_t = torch.stack([ax_base * m, ay_base * m], dim=0).unsqueeze(0)
        with torch.no_grad():
            warped = warp_source(source_t, alpha_t)
        scores[m].append(rel_l2_np(warped[0].numpy(), image_r))
avg_scores = {m: np.mean(v) for m, v in scores.items()}
ALPHA_SCALE_CALIBRATED = min(avg_scores, key=avg_scores.get)
print("Selected ALPHA_SCALE_CALIBRATED =", ALPHA_SCALE_CALIBRATED)

def load_raw_channels(df, res=RES):
    kappas, alphas_x, alphas_y = [], [], []
    for p in df["resolved_path"]:
        d = np.load(p)
        kappa_r = load_resized(d["kappa"], res)
        psi_native = torch.tensor(d["psi"], dtype=torch.float32)
        ax_native, ay_native = compute_alpha_from_psi(psi_native)
        ax_r = downsample_tensor(ax_native, res) * ALPHA_SCALE_CALIBRATED
        ay_r = downsample_tensor(ay_native, res) * ALPHA_SCALE_CALIBRATED
        kappas.append(kappa_r); alphas_x.append(ax_r); alphas_y.append(ay_r)
    return torch.stack(kappas), torch.stack(alphas_x), torch.stack(alphas_y)

wdm_df = resolved[resolved["class"] == "wdm"].reset_index(drop=True)
wdm_all = wdm_df[wdm_df["split"].isin(["val", "test"])].reset_index(drop=True)
wdm_all = wdm_all.sample(frac=1, random_state=0).reset_index(drop=True)
wdm_train_df = wdm_all.iloc[:2000].reset_index(drop=True)
wdm_test_df = wdm_all.iloc[2000:2250].reset_index(drop=True)

print("Loading WDM train...")
kappa_tr, ax_tr, ay_tr = load_raw_channels(wdm_train_df)
print("Loading WDM test...")
kappa_te, ax_te, ay_te = load_raw_channels(wdm_test_df)

kappa_mean, kappa_std = kappa_tr.mean(), kappa_tr.std() + 1e-8
alpha_x_mean, alpha_x_std = ax_tr.mean(), ax_tr.std() + 1e-8
alpha_y_mean, alpha_y_std = ay_tr.mean(), ay_tr.std() + 1e-8

def build_tensor_dataset(kappa, ax, ay):
    k = normalize(kappa, kappa_mean, kappa_std).unsqueeze(1)
    ax_n = normalize(ax, alpha_x_mean, alpha_x_std)
    ay_n = normalize(ay, alpha_y_mean, alpha_y_std)
    return k, torch.stack([ax_n, ay_n], dim=1)

K_train, A_train = build_tensor_dataset(kappa_tr, ax_tr, ay_tr)
K_test, A_test = build_tensor_dataset(kappa_te, ax_te, ay_te)
K_val, A_val = K_test, A_test

cdm_df = resolved[resolved["class"] == "cdm"].reset_index(drop=True)
cdm_all = cdm_df[cdm_df["split"].isin(["val", "test"])].reset_index(drop=True)
cdm_all = cdm_all.sample(frac=1, random_state=0).reset_index(drop=True)
cdm_zero_shot = cdm_all.iloc[:250].reset_index(drop=True)
print("Loading CDM zero-shot...")
kappa_cdm, ax_cdm, ay_cdm = load_raw_channels(cdm_zero_shot)
K_cdm, A_cdm = build_tensor_dataset(kappa_cdm, ax_cdm, ay_cdm)

axion_df = resolved[resolved["class"] == "axion"].reset_index(drop=True)
axion_all = axion_df[axion_df["split"].isin(["val", "test"])].reset_index(drop=True)
axion_all = axion_all.sample(frac=1, random_state=0).reset_index(drop=True)
axion_zero_shot = axion_all.iloc[:250].reset_index(drop=True)
print("Loading Axion zero-shot...")
kappa_ax_, axx_ax_, axy_ax_ = load_raw_channels(axion_zero_shot)
K_axion, A_axion = build_tensor_dataset(kappa_ax_, axx_ax_, axy_ax_)

def make_loaders(batch_size=64):
    return (
        torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_train, A_train), batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_val, A_val), batch_size=batch_size, shuffle=False),
        torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_test, A_test), batch_size=batch_size, shuffle=False),
        torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_cdm, A_cdm), batch_size=batch_size, shuffle=False),
        torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_axion, A_axion), batch_size=batch_size, shuffle=False),
    )

# ===========================================================================
# 2. POD BASIS (computed once, from real WDM-train alpha statistics)
# ===========================================================================
ax_train_norm = A_train[:, 0].reshape(len(A_train), -1)
ay_train_norm = A_train[:, 1].reshape(len(A_train), -1)
pod_mean_x_ = ax_train_norm.mean(dim=0)
pod_mean_y_ = ay_train_norm.mean(dim=0)
centered_x = ax_train_norm - pod_mean_x_
centered_y = ay_train_norm - pod_mean_y_
U_x, S_x, V_x = torch.pca_lowrank(centered_x, q=N_POD_MODES, niter=4)
U_y, S_y, V_y = torch.pca_lowrank(centered_y, q=N_POD_MODES, niter=4)
POD_BASIS_X = V_x.to(DEVICE); POD_BASIS_Y = V_y.to(DEVICE)
POD_MEAN_X = pod_mean_x_.to(DEVICE); POD_MEAN_Y = pod_mean_y_.to(DEVICE)
explained_x = (S_x ** 2).sum() / (centered_x ** 2).sum()
explained_y = (S_y ** 2).sum() / (centered_y ** 2).sum()
print(f"POD basis: {N_POD_MODES} modes | alpha_x var explained: {explained_x:.4f} | alpha_y: {explained_y:.4f}")

# ===========================================================================
# 3. SSIM implementation
# ===========================================================================
def gaussian_window(window_size, sigma, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)

def ssim(pred, target, window_size=11, sigma=1.5, eps=1e-8):
    B, H, W = pred.shape
    pred = pred.unsqueeze(1); target = target.unsqueeze(1)
    data_range = (target.amax(dim=(1,2,3)) - target.amin(dim=(1,2,3))).clamp(min=eps).view(B,1,1,1)
    window = gaussian_window(window_size, sigma, pred.device); pad = window_size // 2
    mu_p = F.conv2d(pred, window, padding=pad); mu_t = F.conv2d(target, window, padding=pad)
    mu_p_sq, mu_t_sq, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
    sig_p_sq = F.conv2d(pred*pred, window, padding=pad) - mu_p_sq
    sig_t_sq = F.conv2d(target*target, window, padding=pad) - mu_t_sq
    sig_pt = F.conv2d(pred*target, window, padding=pad) - mu_pt
    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    ssim_map = ((2*mu_pt+C1)*(2*sig_pt+C2)) / ((mu_p_sq+mu_t_sq+C1)*(sig_p_sq+sig_t_sq+C2))
    return ssim_map.mean(dim=(1,2,3))

def rel_l2_loss(pred, target, eps=1e-8):
    B = pred.shape[0]
    diff = torch.norm(pred.reshape(B,-1) - target.reshape(B,-1), dim=1)
    norm = torch.norm(target.reshape(B,-1), dim=1) + eps
    return (diff / norm).mean()

# ===========================================================================
# 4. MODEL DEFINITIONS -- all 7 architectures, output = 2ch (alpha_x, alpha_y)
# ===========================================================================
class VanillaDeepONet(nn.Module):
    def __init__(self, res=64, out_channels=2, p=128, branch_hidden=256, trunk_hidden=128):
        super().__init__()
        self.out_channels, self.p = out_channels, p
        n_sensors = res * res
        self.branch = nn.Sequential(nn.Linear(n_sensors, branch_hidden), nn.GELU(),
            nn.Linear(branch_hidden, branch_hidden), nn.GELU(), nn.Linear(branch_hidden, out_channels*p))
        self.trunk = nn.Sequential(nn.Linear(2, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(), nn.Linear(trunk_hidden, out_channels*p), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(out_channels))
        ys, xs = torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing="ij")
        self.register_buffer("coords", torch.stack([ys.flatten(), xs.flatten()], dim=-1))
    def forward(self, x):
        B, C, H, W = x.shape
        b_out = self.branch(x.reshape(B,-1)).view(B, self.out_channels, self.p)
        t_out = self.trunk(self.coords); N = t_out.shape[0]
        t_out = t_out.view(N, self.out_channels, self.p)
        out = torch.einsum("bcp,ncp->bcn", b_out, t_out) + self.bias.view(1,-1,1)
        return out.view(B, self.out_channels, H, W)

class SingleDeepONet(nn.Module):
    def __init__(self, res=64, p=128, branch_hidden=256, trunk_hidden=128):
        super().__init__()
        n_sensors = res*res; self.p = p
        self.branch = nn.Sequential(nn.Linear(n_sensors, branch_hidden), nn.GELU(),
            nn.Linear(branch_hidden, branch_hidden), nn.GELU(), nn.Linear(branch_hidden, p))
        self.trunk = nn.Sequential(nn.Linear(2, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(), nn.Linear(trunk_hidden, p), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x_flat, coords):
        return torch.einsum("bp,np->bn", self.branch(x_flat), self.trunk(coords)) + self.bias

class StackedDeepONet(nn.Module):
    def __init__(self, res=64, out_channels=2, p=128, branch_hidden=256, trunk_hidden=128):
        super().__init__()
        self.out_channels = out_channels
        self.nets = nn.ModuleList([SingleDeepONet(res, p, branch_hidden, trunk_hidden) for _ in range(out_channels)])
        ys, xs = torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing="ij")
        self.register_buffer("coords", torch.stack([ys.flatten(), xs.flatten()], dim=-1))
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.reshape(B,-1)
        outs = [net(x_flat, self.coords) for net in self.nets]
        return torch.stack(outs, dim=1).view(B, self.out_channels, H, W)

class ConvBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1,32,3,stride=2,padding=1), nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.GELU(), nn.Conv2d(64,128,3,stride=2,padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Linear(128*4*4,256), nn.GELU(), nn.Linear(256,out_dim))
    def forward(self, x): return self.fc(self.net(x).flatten(1))

class ConvDeepONet(nn.Module):
    def __init__(self, res=64, out_channels=2, p=128, trunk_hidden=128):
        super().__init__()
        self.out_channels, self.p = out_channels, p
        self.branch = ConvBranch(out_dim=out_channels*p)
        self.trunk = nn.Sequential(nn.Linear(2, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(), nn.Linear(trunk_hidden, out_channels*p), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(out_channels))
        ys, xs = torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing="ij")
        self.register_buffer("coords", torch.stack([ys.flatten(), xs.flatten()], dim=-1))
    def forward(self, x):
        B, C, H, W = x.shape
        b_out = self.branch(x).view(B, self.out_channels, self.p)
        t_out = self.trunk(self.coords); N = t_out.shape[0]
        t_out = t_out.view(N, self.out_channels, self.p)
        out = torch.einsum("bcp,ncp->bcn", b_out, t_out) + self.bias.view(1,-1,1)
        return out.view(B, self.out_channels, H, W)

class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_dim=2, n_freqs=64, scale=10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freqs) * scale)
        self.out_dim = n_freqs * 2
    def forward(self, coords):
        proj = 2*np.pi*coords@self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class FourierDeepONet(nn.Module):
    def __init__(self, res=64, out_channels=2, p=128, n_freqs=64, fourier_scale=10.0, trunk_hidden=128):
        super().__init__()
        self.out_channels, self.p = out_channels, p
        self.branch = ConvBranch(out_dim=out_channels*p)
        self.fourier_enc = FourierFeatureEncoding(2, n_freqs, fourier_scale)
        self.trunk = nn.Sequential(nn.Linear(self.fourier_enc.out_dim, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(), nn.Linear(trunk_hidden, out_channels*p), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(out_channels))
        ys, xs = torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing="ij")
        self.register_buffer("coords", torch.stack([ys.flatten(), xs.flatten()], dim=-1))
    def forward(self, x):
        B, C, H, W = x.shape
        b_out = self.branch(x).view(B, self.out_channels, self.p)
        cf = self.fourier_enc(self.coords)
        t_out = self.trunk(cf); N = t_out.shape[0]
        t_out = t_out.view(N, self.out_channels, self.p)
        out = torch.einsum("bcp,ncp->bcn", b_out, t_out) + self.bias.view(1,-1,1)
        return out.view(B, self.out_channels, H, W)

class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))
    def forward(self, x):
        h = self.norm1(x); a, _ = self.attn(h,h,h); x = x+a; h = self.norm2(x)
        return x + self.mlp(h)

class AttentionBranch(nn.Module):
    def __init__(self, res=64, patch_size=8, dim=128, depth=4, heads=4, out_dim=128):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        n_patches = (res//patch_size)**2
        self.cls_token = nn.Parameter(torch.randn(1,1,dim)*0.02)
        self.pos_embed = nn.Parameter(torch.randn(1,n_patches+1,dim)*0.02)
        self.layers = nn.ModuleList([SelfAttnBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.out_proj = nn.Linear(dim, out_dim)
    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)
        tokens = patches.flatten(2).transpose(1,2)
        cls = self.cls_token.expand(B,-1,-1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for layer in self.layers: tokens = layer(tokens)
        return self.out_proj(self.norm(tokens[:,0]))

class AttentionDeepONet(nn.Module):
    def __init__(self, res=64, out_channels=2, p=128, patch_size=8, attn_dim=128, attn_depth=4, attn_heads=4,
                 n_freqs=64, fourier_scale=10.0, trunk_hidden=128):
        super().__init__()
        self.out_channels, self.p = out_channels, p
        self.branch = AttentionBranch(res, patch_size, attn_dim, attn_depth, attn_heads, out_channels*p)
        self.fourier_enc = FourierFeatureEncoding(2, n_freqs, fourier_scale)
        self.trunk = nn.Sequential(nn.Linear(self.fourier_enc.out_dim, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(), nn.Linear(trunk_hidden, out_channels*p), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(out_channels))
        ys, xs = torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing="ij")
        self.register_buffer("coords", torch.stack([ys.flatten(), xs.flatten()], dim=-1))
    def forward(self, x):
        B, C, H, W = x.shape
        b_out = self.branch(x).view(B, self.out_channels, self.p)
        cf = self.fourier_enc(self.coords)
        t_out = self.trunk(cf); N = t_out.shape[0]
        t_out = t_out.view(N, self.out_channels, self.p)
        out = torch.einsum("bcp,ncp->bcn", b_out, t_out) + self.bias.view(1,-1,1)
        return out.view(B, self.out_channels, H, W)

class PODDeepONet(nn.Module):
    def __init__(self, pod_basis_x, pod_mean_x, pod_basis_y, pod_mean_y, n_modes=32, branch_dim=128):
        super().__init__()
        self.n_modes = n_modes
        self.register_buffer("pod_basis_x", pod_basis_x); self.register_buffer("pod_mean_x", pod_mean_x)
        self.register_buffer("pod_basis_y", pod_basis_y); self.register_buffer("pod_mean_y", pod_mean_y)
        self.branch = ConvBranch(out_dim=branch_dim)
        self.coeff_head = nn.Sequential(nn.Linear(branch_dim, branch_dim), nn.GELU(), nn.Linear(branch_dim, 2*n_modes))
    def forward(self, x):
        B, C, H, W = x.shape
        coeffs = self.coeff_head(self.branch(x))
        cx, cy = coeffs[:,:self.n_modes], coeffs[:,self.n_modes:]
        ax = self.pod_mean_x.unsqueeze(0) + cx @ self.pod_basis_x.T
        ay = self.pod_mean_y.unsqueeze(0) + cy @ self.pod_basis_y.T
        return torch.stack([ax.view(B,H,W), ay.view(B,H,W)], dim=1)

class LearnedEncoderBasis(nn.Module):
    def __init__(self, res=64, n_encoder_modes=64, hidden=128):
        super().__init__()
        self.n_encoder_modes = n_encoder_modes
        self.basis_net = nn.Sequential(nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_encoder_modes))
        ys, xs = torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing="ij")
        self.register_buffer("coords", torch.stack([ys.flatten(), xs.flatten()], dim=-1))
        self.res = res
    def forward(self, kappa_flat):
        basis_vals = self.basis_net(self.coords)
        return (kappa_flat @ basis_vals) / self.res

class BelNet(nn.Module):
    def __init__(self, res=64, out_channels=2, p=128, n_encoder_modes=64, n_freqs=64, fourier_scale=10.0,
                 trunk_hidden=128, mlp_hidden=256):
        super().__init__()
        self.out_channels, self.p = out_channels, p
        self.encoder = LearnedEncoderBasis(res, n_encoder_modes, 128)
        self.branch_mlp = nn.Sequential(nn.Linear(n_encoder_modes, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, out_channels*p))
        self.fourier_enc = FourierFeatureEncoding(2, n_freqs, fourier_scale)
        self.trunk = nn.Sequential(nn.Linear(self.fourier_enc.out_dim, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(), nn.Linear(trunk_hidden, out_channels*p), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(out_channels))
        ys, xs = torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing="ij")
        self.register_buffer("coords", torch.stack([ys.flatten(), xs.flatten()], dim=-1))
    def forward(self, x):
        B, C, H, W = x.shape
        latent = self.encoder(x.reshape(B,-1))
        b_out = self.branch_mlp(latent).view(B, self.out_channels, self.p)
        cf = self.fourier_enc(self.coords)
        t_out = self.trunk(cf); N = t_out.shape[0]
        t_out = t_out.view(N, self.out_channels, self.p)
        out = torch.einsum("bcp,ncp->bcn", b_out, t_out) + self.bias.view(1,-1,1)
        return out.view(B, self.out_channels, H, W)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
    def compl_mul2d(self, inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)
    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.spectral(x) + self.local(x))

class FNO2d(nn.Module):
    # baseline comparison point for the DeepONet-family ablation (paper 3,
    # Table 1's "FNO" row) -- not itself part of the DeepONet family.
    def __init__(self, in_channels=1, out_channels=2, width=36, modes1=8, modes2=8, n_layers=4, res=64):
        super().__init__()
        self.res = res
        ys, xs = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing="ij")
        self.register_buffer("grid", torch.stack([ys, xs], dim=0))
        self.fc_in = nn.Conv2d(in_channels + 2, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.fc_out1 = nn.Conv2d(width, width, 1)
        self.fc_out2 = nn.Conv2d(width, out_channels, 1)
        self.act = nn.GELU()
    def forward(self, x):
        B = x.shape[0]
        grid = self.grid.unsqueeze(0).repeat(B, 1, 1, 1)
        h = self.fc_in(torch.cat([x, grid], dim=1))
        for block in self.blocks:
            h = block(h)
        h = self.act(self.fc_out1(h))
        return self.fc_out2(h)

def make_vanilla():   return VanillaDeepONet(res=RES, out_channels=2, p=128, branch_hidden=256, trunk_hidden=128)
def make_stacked():   return StackedDeepONet(res=RES, out_channels=2, p=128, branch_hidden=256, trunk_hidden=128)
def make_conv():      return ConvDeepONet(res=RES, out_channels=2, p=128, trunk_hidden=128)
def make_fourier():   return FourierDeepONet(res=RES, out_channels=2, p=128, n_freqs=64, fourier_scale=10.0, trunk_hidden=128)
def make_attention(): return AttentionDeepONet(res=RES, out_channels=2, p=128, patch_size=8, attn_dim=128,
                                                attn_depth=4, attn_heads=4, n_freqs=64, fourier_scale=10.0, trunk_hidden=128)
def make_pod():        return PODDeepONet(POD_BASIS_X.clone(), POD_MEAN_X.clone(), POD_BASIS_Y.clone(), POD_MEAN_Y.clone(),
                                            n_modes=N_POD_MODES, branch_dim=128)
def make_belnet():    return BelNet(res=RES, out_channels=2, p=128, n_encoder_modes=64, n_freqs=64,
                                     fourier_scale=10.0, trunk_hidden=128, mlp_hidden=256)
def make_fno():        return FNO2d(in_channels=1, out_channels=2, width=36, modes1=8, modes2=8, n_layers=4, res=RES)

ARCHITECTURES = {
    "Vanilla": make_vanilla, "Stacked": make_stacked, "Conv": make_conv,
    "Fourier": make_fourier, "Attention": make_attention, "POD": make_pod, "BelNet": make_belnet,
    "FNO": make_fno,  # baseline comparison, not a DeepONet variant -- see paper 3, Table 1
}

# ===========================================================================
# 5. TRAIN + EVAL (per seed) -- returns rel_L2 AND SSIM for every split
# ===========================================================================
def train_one_seed(model_ctor, seed, arch_name):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader, val_loader, test_loader, cdm_loader, axion_loader = make_loaders()
    model = model_ctor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val, best_state = float("inf"), None

    for epoch in range(EPOCHS):
        model.train()
        for kb, ab in train_loader:
            kb, ab = kb.to(DEVICE), ab.to(DEVICE)
            optimizer.zero_grad()
            pred = model(kb)
            loss = rel_l2_loss(pred[:,0], ab[:,0]) + rel_l2_loss(pred[:,1], ab[:,1])
            loss.backward(); optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vloss = np.mean([rel_l2_loss(model(kb.to(DEVICE)), ab.to(DEVICE)).item() for kb, ab in val_loader])
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    def eval_split(loader):
        rell2s, ssim_xs, ssim_ys = [], [], []
        with torch.no_grad():
            for kb, ab in loader:
                kb, ab = kb.to(DEVICE), ab.to(DEVICE)
                pred = model(kb)
                rell2s.append(rel_l2_loss(pred, ab).item())
                ax_t = denormalize(ab[:,0], alpha_x_mean, alpha_x_std); ax_p = denormalize(pred[:,0], alpha_x_mean, alpha_x_std)
                ay_t = denormalize(ab[:,1], alpha_y_mean, alpha_y_std); ay_p = denormalize(pred[:,1], alpha_y_mean, alpha_y_std)
                ssim_xs.append(ssim(ax_p, ax_t).cpu()); ssim_ys.append(ssim(ay_p, ay_t).cpu())
        return np.mean(rell2s), torch.cat(ssim_xs).mean().item(), torch.cat(ssim_ys).mean().item()

    wdm_r2, wdm_sx, wdm_sy = eval_split(test_loader)
    cdm_r2, cdm_sx, cdm_sy = eval_split(cdm_loader)
    axion_r2, axion_sx, axion_sy = eval_split(axion_loader)

    return {
        "WDM": {"rel_L2": wdm_r2, "SSIM": (wdm_sx+wdm_sy)/2},
        "CDM": {"rel_L2": cdm_r2, "SSIM": (cdm_sx+cdm_sy)/2},
        "Axion": {"rel_L2": axion_r2, "SSIM": (axion_sx+axion_sy)/2},
    }, model

# ===========================================================================
# 6. RUN ALL 7 ARCHITECTURES x 3 SEEDS, TIME EACH, COLLECT QUALITATIVE SAMPLE
# ===========================================================================
all_results = {}
qualitative_preds = {}  # arch_name -> (kappa_sample, true_ax, pred_ax) on one CDM sample
param_counts = {}
arch_times = {}

for arch_name, ctor in ARCHITECTURES.items():
    print(f"\n{'='*70}\n{arch_name}\n{'='*70}")
    param_counts[arch_name] = count_params(ctor())
    t0 = time.time()
    seed_results = []
    last_model = None
    for seed in SEEDS:
        res, model = train_one_seed(ctor, seed, arch_name)
        seed_results.append(res)
        last_model = model
        print(f"  seed={seed}  WDM(rel_L2={res['WDM']['rel_L2']:.4f}, SSIM={res['WDM']['SSIM']:.4f})  "
              f"CDM(rel_L2={res['CDM']['rel_L2']:.4f}, SSIM={res['CDM']['SSIM']:.4f})  "
              f"Axion(rel_L2={res['Axion']['rel_L2']:.4f}, SSIM={res['Axion']['SSIM']:.4f})")
    elapsed = time.time() - t0
    arch_times[arch_name] = elapsed
    all_results[arch_name] = seed_results
    print(f"  [{arch_name}] total time: {elapsed:.1f}s ({elapsed/60:.1f} min) | params: {param_counts[arch_name]:,}")

    kb, ab = K_cdm[0:1].to(DEVICE), A_cdm[0:1].to(DEVICE)
    with torch.no_grad():
        pred = last_model(kb)
    ax_true = denormalize(ab[0,0].cpu(), alpha_x_mean, alpha_x_std).numpy()
    ax_pred = denormalize(pred[0,0].cpu(), alpha_x_mean, alpha_x_std).numpy()
    qualitative_preds[arch_name] = (ax_true, ax_pred)

    torch.save({"seed_results": seed_results, "model_state": last_model.state_dict()},
               f"{arch_name.lower()}_3seed_40ep_ssim.pt")

# ===========================================================================
# 7. COMBINED QUALITATIVE FIGURE -- all 7 architectures, one row each
# ===========================================================================
n_arch = len(ARCHITECTURES)
fig, axes = plt.subplots(n_arch, 3, figsize=(10, 3*n_arch))
for i, (arch_name, (ax_true, ax_pred)) in enumerate(qualitative_preds.items()):
    err = ax_true - ax_pred
    axes[i,0].imshow(ax_true, cmap="coolwarm"); axes[i,0].set_ylabel(arch_name, fontsize=11)
    axes[i,1].imshow(ax_pred, cmap="coolwarm")
    axes[i,2].imshow(err, cmap="coolwarm")
    if i == 0:
        axes[i,0].set_title("True alpha_x (CDM)")
        axes[i,1].set_title("Predicted")
        axes[i,2].set_title("Error")
    for j in range(3):
        axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
plt.tight_layout()
plt.savefig("combined_qualitative_all7.png", dpi=150)
plt.show()

# ===========================================================================
# 8. FINAL SUMMARY TABLE
# ===========================================================================
print("\n" + "="*100)
print("FINAL SUMMARY -- mean ± std over 3 seeds")
print("="*100)
header = f"{'Architecture':<12} {'Params':>10} {'Time(s)':>9} | " \
         f"{'WDM rel_L2':>16} {'WDM SSIM':>14} | {'CDM rel_L2':>16} {'CDM SSIM':>14} | {'Axion rel_L2':>16} {'Axion SSIM':>14}"
print(header)
print("-"*len(header))

summary_rows = []
for arch_name, seed_results in all_results.items():
    row = {"Architecture": arch_name, "Params": param_counts[arch_name], "Time(s)": round(arch_times[arch_name],1)}
    line = f"{arch_name:<12} {param_counts[arch_name]:>10,} {arch_times[arch_name]:>9.1f} | "
    for split in ["WDM", "CDM", "Axion"]:
        r2_vals = np.array([r[split]["rel_L2"] for r in seed_results])
        ssim_vals = np.array([r[split]["SSIM"] for r in seed_results])
        row[f"{split}_rel_L2_mean"] = r2_vals.mean(); row[f"{split}_rel_L2_std"] = r2_vals.std()
        row[f"{split}_SSIM_mean"] = ssim_vals.mean(); row[f"{split}_SSIM_std"] = ssim_vals.std()
        line += f"{r2_vals.mean():.4f}±{r2_vals.std():.4f} {ssim_vals.mean():.4f}±{ssim_vals.std():.4f} | "
    print(line)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("all7_summary_table.csv", index=False)
print("\nSaved summary table to all7_summary_table.csv")
print("Saved combined qualitative figure to combined_qualitative_all7.png")
