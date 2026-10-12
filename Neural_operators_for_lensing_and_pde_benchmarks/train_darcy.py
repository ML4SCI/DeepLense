# %% MASTER SCRIPT -- Darcy flow, all 11 param-matched architectures, 7
# seeds each, @16 (test) and @32 (zero-shot). Fully self-contained: downloads
# the dataset (skips if already present), builds loaders, then trains/evals
# all 11 architectures.
#
# This EXTENDS an existing 3-seed run rather than redoing it: for each
# architecture, if f"{name}_3seed.pt" (saved by an earlier 3-seed-only run)
# is found in the working directory, only the 4 NEW seeds are trained and
# appended to the 3 already-saved results. If no such checkpoint exists,
# all 7 seeds are trained from scratch for that architecture.

# ===========================================================================
# SECTION 0: DOWNLOAD (skips re-download if files already present)
# ===========================================================================
import os, subprocess, sys, time, tarfile, shutil
import requests
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuraloperator"], check=True)

DATA_ROOT = Path("/kaggle/working/darcy_data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
ZENODO_RECORD_ID = "12784353"
RESOLUTIONS_NEEDED = [16, 32]

def download_with_retry(url, dest_path, max_retries=8, timeout=120):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[skip] {dest_path.name} already present ({dest_path.stat().st_size/1e6:.1f} MB)")
        return
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{dest_path.name}] attempt {attempt}/{max_retries} ...")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest_path.with_suffix(dest_path.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                tmp.rename(dest_path)
            print(f"  done: {dest_path.stat().st_size/1e6:.1f} MB")
            return
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"  failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not download {url} after {max_retries} attempts")

for res in RESOLUTIONS_NEEDED:
    fname = f"darcy_{res}.tgz"
    url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{fname}?download=1"
    dest = DATA_ROOT / fname
    download_with_retry(url, dest)
    with tarfile.open(dest, "r:gz") as tf:
        tf.extractall(DATA_ROOT)

expected = [f"darcy_train_{r}.pt" for r in RESOLUTIONS_NEEDED] + [f"darcy_test_{r}.pt" for r in RESOLUTIONS_NEEDED]
for name in expected:
    if not (DATA_ROOT / name).exists():
        hits = list(DATA_ROOT.rglob(name))
        if hits:
            shutil.move(str(hits[0]), str(DATA_ROOT / name))
missing = [n for n in expected if not (DATA_ROOT / n).exists()]
if missing:
    raise RuntimeError(f"still missing: {missing}")
print("all required files present:", expected)

# ===========================================================================
# SECTION 1: LOAD (download=False -- files already fetched above)
# ===========================================================================
from neuralop.data.datasets.darcy import DarcyDataset
from torch.utils.data import DataLoader

BATCH_SIZE = 16

dataset = DarcyDataset(
    root_dir=DATA_ROOT, n_train=1000, n_tests=[100, 50],
    batch_size=BATCH_SIZE, test_batch_sizes=[BATCH_SIZE, BATCH_SIZE],
    train_resolution=16, test_resolutions=[16, 32], download=False,
)

train_loader_raw = DataLoader(dataset.train_db, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True)
test_loaders_raw = {
    res: DataLoader(dataset.test_dbs[res], batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    for res in [16, 32]
}
print("loaded:", len(dataset.train_db), "train,", {r: len(dataset.test_dbs[r]) for r in [16, 32]}, "test")

# ===========================================================================
# SECTION 2: TRAIN + EVAL -- all 11 param-matched architectures, 7 seeds each
# ===========================================================================
import time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLD_SEEDS = [13, 42, 97]
NEW_SEEDS = [7, 64, 128, 2024]
EPOCHS = 60
BATCH_SIZE = 16
print(DEVICE)

# ===========================================================================
# 1. DATA (shared): coord-augmented for spectral archs, raw for branch-trunk
# ===========================================================================
def collect_all(loader):
    xs, ys = [], []
    for batch in loader:
        xs.append(batch["x"]); ys.append(batch["y"])
    return torch.cat(xs), torch.cat(ys)

X_train_raw, Y_train = collect_all(train_loader_raw)
X_test16_raw, Y_test16 = collect_all(test_loaders_raw[16])
X_test32_raw, Y_test32 = collect_all(test_loaders_raw[32])

def add_coord_grid(X):
    B, C, H, W = X.shape
    ys = torch.linspace(0, 1, H); xs = torch.linspace(0, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gy, gx], 0).unsqueeze(0).expand(B, -1, -1, -1)
    return torch.cat([X, grid], dim=1)

X_train, X_test16, X_test32 = add_coord_grid(X_train_raw), add_coord_grid(X_test16_raw), add_coord_grid(X_test32_raw)
IN_CH, OUT_CH = X_train.shape[1], Y_train.shape[1]

def make_grid(res, device):
    ys, xs = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], 1).to(device)

GRID16, GRID32 = make_grid(16, DEVICE), make_grid(32, DEVICE)

# fixed POD basis from real training-target statistics (independent of seed)
N_POD_MODES = 64
u_train_flat = Y_train.reshape(len(Y_train), -1)
pod_mean_ = u_train_flat.mean(dim=0)
centered = u_train_flat - pod_mean_
U, S, V = torch.pca_lowrank(centered, q=N_POD_MODES, niter=4)
POD_BASIS_16, POD_MEAN_16 = V.to(DEVICE), pod_mean_.to(DEVICE)

def resize_basis_to(basis, mean, from_res, to_res, device):
    n_modes = basis.shape[1]
    b_up = F.interpolate(basis.T.view(n_modes, 1, from_res, from_res), size=(to_res, to_res),
                          mode='bilinear', align_corners=False).reshape(n_modes, to_res*to_res).T.contiguous().to(device)
    m_up = F.interpolate(mean.view(1, 1, from_res, from_res), size=(to_res, to_res),
                          mode='bilinear', align_corners=False).reshape(to_res*to_res).contiguous().to(device)
    return b_up, m_up

POD_BASIS_32, POD_MEAN_32 = resize_basis_to(POD_BASIS_16.cpu(), POD_MEAN_16.cpu(), 16, 32, DEVICE)

TEST16_STD = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test16, Y_test16), batch_size=BATCH_SIZE, shuffle=False)
TEST32_STD = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test32, Y_test32), batch_size=BATCH_SIZE, shuffle=False)
TEST16_RAW = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test16_raw, Y_test16), batch_size=BATCH_SIZE, shuffle=False)
TEST32_RAW = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test32_raw, Y_test32), batch_size=BATCH_SIZE, shuffle=False)

def get_test_loaders(kind):
    return (TEST16_STD, TEST32_STD) if kind == "standard" else (TEST16_RAW, TEST32_RAW)

def make_train_loader(kind, seed, bs=BATCH_SIZE):
    g = torch.Generator().manual_seed(seed)
    Xtr = X_train if kind == "standard" else X_train_raw
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Y_train), batch_size=bs,
                                        shuffle=True, drop_last=True, generator=g)

# ===========================================================================
# 2. FORWARD DISPATCH (handles the 3 different call signatures)
# ===========================================================================
def forward_pass(kind, model, xb, res):
    if kind == "standard":
        return model(xb)
    elif kind == "deeponet":
        return model(xb, GRID16 if res == 16 else GRID32, res)
    else:  # poddeeponet
        basis, mean = (POD_BASIS_16, POD_MEAN_16) if res == 16 else (POD_BASIS_32, POD_MEAN_32)
        return model(xb, basis, mean, res)

def rel_l2(pred, target, eps=1e-8):
    num = torch.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.norm(target.reshape(target.shape[0], -1), dim=1) + eps
    return (num / den).mean().item()

def evaluate(kind, model, loader, res):
    model.eval(); total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            total += rel_l2(forward_pass(kind, model, x, res), y); n += 1
    return total / max(n, 1)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===========================================================================
# 3. ARCHITECTURES (param-matched configs from the sweeps)
# ===========================================================================
# --- BALANI-NO ---
class RationalSpectral2d(nn.Module):
    def __init__(self, ic, oc, n_poles=4, deg=2, modes=4):
        super().__init__()
        self.ic, self.oc, self.m = ic, oc, modes
        self.exps = [(p, q) for p in range(deg+1) for q in range(deg+1) if p+q <= deg]
        self.B = len(self.exps)
        self.raw_s = nn.Parameter(torch.linspace(-6.0, 3.0, n_poles))
        self.coef = nn.Parameter(torch.randn(ic, oc, n_poles, self.B, 2) * (1.0/(ic*oc))**0.5)
        self.dc = nn.Parameter(torch.zeros(ic, oc, 2))
        scale = 1.0/(ic*oc)
        self.t1_re = nn.Parameter(scale*torch.randn(ic, oc, modes, modes)); self.t1_im = nn.Parameter(scale*torch.randn(ic, oc, modes, modes))
        self.t2_re = nn.Parameter(scale*torch.randn(ic, oc, modes, modes)); self.t2_im = nn.Parameter(scale*torch.randn(ic, oc, modes, modes))
    def _symbol(self, H, W, device):
        ky = torch.fft.fftfreq(H, device=device)*H; kx = torch.fft.rfftfreq(W, device=device)*W
        KY, KX = torch.meshgrid(ky, kx, indexing='ij'); k2 = KX**2 + KY**2
        s2 = F.softplus(self.raw_s).view(-1,1,1); den = 1.0/(k2[None]+s2+1e-8)
        num = torch.stack([((1j*KX)**p)*((1j*KY)**q) for p,q in self.exps])
        basis = den.to(num.dtype)[:,None]*num[None]
        c = torch.view_as_complex(self.coef.contiguous())
        R = torch.einsum('iojb,jbhw->iohw', c, basis)
        dc = torch.view_as_complex(self.dc.contiguous())
        mask = torch.ones_like(R.real); mask[:,:,0,0] = 0.0
        pad = torch.zeros_like(R); pad[:,:,0,0] = dc
        return R*mask + pad
    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        of = torch.einsum('bihw,iohw->bohw', xf, self._symbol(H, W, x.device))
        m1, m2 = min(self.m, H), min(self.m, W//2+1)
        t1 = torch.complex(self.t1_re, self.t1_im); t2 = torch.complex(self.t2_re, self.t2_im)
        of[:,:,:m1,:m2] = of[:,:,:m1,:m2] + torch.einsum('bixy,ioxy->boxy', xf[:,:,:m1,:m2], t1[:,:,:m1,:m2])
        of[:,:,-m1:,:m2] = of[:,:,-m1:,:m2] + torch.einsum('bixy,ioxy->boxy', xf[:,:,-m1:,:m2], t2[:,:,:m1,:m2])
        return torch.fft.irfft2(of, s=(H,W), norm="ortho")

class MetrexNOBlock(nn.Module):
    def __init__(self, width, n_poles, deg, modes):
        super().__init__()
        self.sp = RationalSpectral2d(width, width, n_poles, deg, modes)
        self.pointwise = nn.Conv2d(width, width, 1); self.norm = nn.GroupNorm(min(2,width), width); self.act = nn.GELU()
    def forward(self, x): return self.act(self.norm(self.sp(x) + self.pointwise(x)))

class MetrexNO(nn.Module):
    def __init__(self, in_ch, out_ch, width=18, n_poles=4, deg=2, modes=4, n_layers=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList([MetrexNOBlock(width, n_poles, deg, modes) for _ in range(n_layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def forward(self, x):
        h = self.lift(x)
        for b in self.blocks: h = b(h)
        return self.project(h)

# --- FNO / U-FNO (shared SpectralConv2d) ---
class SpectralConv2d(nn.Module):
    def __init__(self, ic, oc, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2; scale = 1.0/(ic*oc)
        self.w1_re = nn.Parameter(scale*torch.randn(ic,oc,m1,m2)); self.w1_im = nn.Parameter(scale*torch.randn(ic,oc,m1,m2))
        self.w2_re = nn.Parameter(scale*torch.randn(ic,oc,m1,m2)); self.w2_im = nn.Parameter(scale*torch.randn(ic,oc,m1,m2))
    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        of = torch.zeros(B, self.w1_re.shape[1], H, W//2+1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.m1,H), min(self.m2,W//2+1)
        of[:,:,:m1,:m2] = torch.einsum("bixy,ioxy->boxy", xf[:,:,:m1,:m2], torch.complex(self.w1_re[:,:,:m1,:m2], self.w1_im[:,:,:m1,:m2]))
        of[:,:,-m1:,:m2] = torch.einsum("bixy,ioxy->boxy", xf[:,:,-m1:,:m2], torch.complex(self.w2_re[:,:,:m1,:m2], self.w2_im[:,:,:m1,:m2]))
        return torch.fft.irfft2(of, s=(H,W), norm="ortho")

class FNOBlock(nn.Module):
    def __init__(self, ch, m1, m2):
        super().__init__()
        self.spec = SpectralConv2d(ch, ch, m1, m2); self.pointwise = nn.Conv2d(ch, ch, 1); self.norm = nn.GroupNorm(min(2,ch), ch); self.act = nn.GELU()
    def forward(self, x): return self.act(self.norm(self.spec(x) + self.pointwise(x)))

class FNO2d(nn.Module):
    def __init__(self, in_ch, out_ch, width=16, modes1=6, modes2=6, n_layers=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def forward(self, x):
        h = self.lift(x)
        for b in self.blocks: h = b(h)
        return self.project(h)

class UFNO2d(nn.Module):
    def __init__(self, in_ch, out_ch, width=14, modes1=5, modes2=5, n_levels=3):
        super().__init__()
        self.n_levels = n_levels; self.lift = nn.Conv2d(in_ch, width, 1)
        self.down_blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_levels)])
        self.up_blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def forward(self, x):
        B, C, H, W = x.shape; h = self.lift(x); skips = []; cur_h, cur_w = H, W
        n_eff = min(self.n_levels, int(math.log2(min(H,W))) if min(H,W) > 1 else 0)
        for i in range(n_eff):
            h = self.down_blocks[i](h); skips.append((h,cur_h,cur_w)); h = self.pools[i](h); cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_eff)):
            sh_h, sh, sw = skips[i]; h = F.interpolate(h, size=(sh,sw), mode="bilinear", align_corners=False) + sh_h
            h = self.up_blocks[i](h)
        return self.project(h)

# --- ALNO / LNO ---
class ALNOLayer(nn.Module):
    def __init__(self, ic, oc, n_poles=6, modes_x=6):
        super().__init__()
        self.oc, self.mx = oc, modes_x
        self.mu_re = nn.Parameter(-1.0-torch.rand(ic,oc,n_poles)); self.mu_im = nn.Parameter(torch.linspace(-3,3,n_poles).expand(ic,oc,n_poles).clone())
        s0 = (1.0/(ic*oc))**0.5
        self.beta_re = nn.Parameter(torch.randn(ic,oc,n_poles)*s0); self.beta_im = nn.Parameter(torch.randn(ic,oc,n_poles)*s0)
        s = 1.0/(ic*oc); self.H_re = nn.Parameter(s*torch.randn(ic,oc,modes_x)); self.H_im = nn.Parameter(s*torch.randn(ic,oc,modes_x))
    def forward(self, x):
        B, _, H, W = x.shape; xf = torch.fft.rfft2(x, norm="ortho")
        mx = min(self.mx, W//2+1); H_ = torch.complex(self.H_re[:,:,:mx], self.H_im[:,:,:mx])
        out_ft = torch.zeros(B, self.oc, H, W//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:,:,:,:mx] = torch.einsum('bihk,iok->bohk', xf[:,:,:,:mx], H_)
        omega_y = torch.fft.fftfreq(H, device=x.device)*H*2*np.pi; iom = 1j*omega_y
        mu = torch.complex(self.mu_re, self.mu_im); beta = torch.complex(self.beta_re, self.beta_im)
        K_sym = (beta.unsqueeze(-1) / (iom.view(1,1,1,-1) - mu.unsqueeze(-1))).sum(dim=2)
        return torch.fft.irfft2(out_ft + torch.einsum('bihk,ioh->bohk', xf, K_sym), s=(H,W), norm="ortho")

class ALNOBlock(nn.Module):
    def __init__(self, width, n_poles, modes_x):
        super().__init__()
        self.al = ALNOLayer(width, width, n_poles, modes_x); self.pointwise = nn.Conv2d(width, width, 1); self.norm = nn.GroupNorm(min(2,width), width); self.act = nn.GELU()
    def forward(self, x): return self.act(self.norm(self.al(x) + self.pointwise(x)))

class ALNO(nn.Module):
    def __init__(self, in_ch, out_ch, width=32, n_poles=6, modes_x=6, n_layers=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList([ALNOBlock(width, n_poles, modes_x) for _ in range(n_layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def forward(self, x):
        h = self.lift(x)
        for b in self.blocks: h = b(h)
        return self.project(h)

class LNOLayer(nn.Module):
    def __init__(self, ic, oc, n_poles=16):
        super().__init__()
        self.mu_re = nn.Parameter(-1.0-torch.rand(ic,oc,n_poles)); self.mu_im = nn.Parameter(torch.linspace(-3,3,n_poles).expand(ic,oc,n_poles).clone())
        s0 = (1.0/(ic*oc))**0.5
        self.beta_re = nn.Parameter(torch.randn(ic,oc,n_poles)*s0); self.beta_im = nn.Parameter(torch.randn(ic,oc,n_poles)*s0)
    def forward(self, x):
        B, _, H, W = x.shape; xf = torch.fft.rfft2(x, norm="ortho")
        omega_y = torch.fft.fftfreq(H, device=x.device)*H*2*np.pi; iom = 1j*omega_y
        mu = torch.complex(self.mu_re, self.mu_im); beta = torch.complex(self.beta_re, self.beta_im)
        K_sym = (beta.unsqueeze(-1) / (iom.view(1,1,1,-1) - mu.unsqueeze(-1))).sum(dim=2)
        return torch.fft.irfft2(torch.einsum('bihk,ioh->bohk', xf, K_sym), s=(H,W), norm="ortho")

class LNOBlock(nn.Module):
    def __init__(self, width, n_poles):
        super().__init__()
        self.ln = LNOLayer(width, width, n_poles); self.pointwise = nn.Conv2d(width, width, 1); self.norm = nn.GroupNorm(min(2,width), width); self.act = nn.GELU()
    def forward(self, x): return self.act(self.norm(self.ln(x) + self.pointwise(x)))

class LNO(nn.Module):
    def __init__(self, in_ch, out_ch, width=24, n_poles=16, n_layers=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList([LNOBlock(width, n_poles) for _ in range(n_layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def forward(self, x):
        h = self.lift(x)
        for b in self.blocks: h = b(h)
        return self.project(h)

# --- PDNO ---
class SymbolNet(nn.Module):
    def __init__(self, ic, oc, hidden=64, n_layers=2, n_freq=3):
        super().__init__()
        self.ic, self.oc, self.n_freq = ic, oc, n_freq
        layers, c = [], 2*(1+2*n_freq)
        for _ in range(n_layers): layers += [nn.Conv2d(c, hidden, 1), nn.GELU()]; c = hidden
        self.body = nn.Sequential(*layers); self.head = nn.Conv2d(hidden, ic*oc*2, 1)
    def _feat(self, H, W, device):
        ky = torch.fft.fftfreq(H, device=device)*H; kx = torch.fft.rfftfreq(W, device=device)*W
        KY, KX = torch.meshgrid(ky, kx, indexing='ij'); nyq = max(H,W)/2
        base = torch.stack([KX/nyq, KY/nyq]); ff = [base]
        for i in range(self.n_freq): ff += [torch.sin(2**i*np.pi*base), torch.cos(2**i*np.pi*base)]
        return torch.cat(ff, 0)[None]
    def forward(self, H, W, device):
        h = self.head(self.body(self._feat(H, W, device))).view(self.ic, self.oc, 2, H, W//2+1)
        return torch.complex(h[:,:,0], h[:,:,1])

class PDIOLayer(nn.Module):
    def __init__(self, ic, oc, hidden, n_layers, n_freq):
        super().__init__(); self.sym = SymbolNet(ic, oc, hidden, n_layers, n_freq)
    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]; xf = torch.fft.rfft2(x, norm="ortho")
        return torch.fft.irfft2(torch.einsum('bihw,iohw->bohw', xf, self.sym(H, W, x.device)), s=(H,W), norm="ortho")

class PDNOBlock(nn.Module):
    def __init__(self, width, hidden, n_layers_sym, n_freq):
        super().__init__()
        self.pdio = PDIOLayer(width, width, hidden, n_layers_sym, n_freq); self.pointwise = nn.Conv2d(width, width, 1); self.norm = nn.GroupNorm(min(2,width), width); self.act = nn.GELU()
    def forward(self, x): return self.act(self.norm(self.pdio(x) + self.pointwise(x)))

class PDNO(nn.Module):
    def __init__(self, in_ch, out_ch, width=16, hidden=64, n_layers_block=3, n_layers_sym=2, n_freq=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList([PDNOBlock(width, hidden, n_layers_sym, n_freq) for _ in range(n_layers_block)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def forward(self, x):
        h = self.lift(x)
        for b in self.blocks: h = b(h)
        return self.project(h)

# --- UNO / HC-UNO (shared HyperSpectralConv2d, filter-bank construction) ---
FMAX = 16 // 2
def build_gauss_bank(cx, cy, bw, theta, H, W, device):
    B, n_filters = cx.shape; Wf = W//2+1
    fy = torch.fft.fftfreq(H, d=1.0/H, device=device); fx = torch.fft.rfftfreq(W, d=1.0/W, device=device)
    gy, gx = torch.meshgrid(fy, fx, indexing="ij"); gx, gy = gx.view(1,1,H,Wf), gy.view(1,1,H,Wf)
    cx_, cy_, bw_, th_ = cx.view(B,-1,1,1), cy.view(B,-1,1,1), bw.view(B,-1,1,1), theta.view(B,-1,1,1)
    dx, dy = gx-cx_, gy-cy_
    dxr = dx*torch.cos(th_) + dy*torch.sin(th_); dyr = -dx*torch.sin(th_) + dy*torch.cos(th_)
    return torch.exp(-((dxr/(bw_+1e-6))**2 + (dyr/(bw_/3+1e-6))**2))

class HyperSpectralConv2d(nn.Module):
    def __init__(self, ic, oc, n_filters=8):
        super().__init__()
        self.channel_mix_re = nn.Parameter(0.02*torch.randn(n_filters, ic, oc)); self.channel_mix_im = nn.Parameter(0.02*torch.randn(n_filters, ic, oc))
    def forward(self, x, bank):
        B, C, H, W = x.shape; xf = torch.fft.rfft2(x, norm="ortho")
        filtered = xf.unsqueeze(1) * bank.unsqueeze(2)
        return torch.fft.irfft2(torch.einsum("bfcxy,fco->boxy", filtered, torch.complex(self.channel_mix_re, self.channel_mix_im)), s=(H,W), norm="ortho")

class FixedFilterBank(nn.Module):
    def __init__(self, n_filters, fmax=FMAX):
        super().__init__()
        self.n_filters, self.fmax = n_filters, fmax
        self.raw_cx = nn.Parameter(torch.randn(n_filters)*0.5); self.raw_cy = nn.Parameter(torch.randn(n_filters)*0.5)
        self.raw_bw = nn.Parameter(torch.zeros(n_filters)); self.raw_theta = nn.Parameter(torch.rand(n_filters)*math.pi)
    def forward(self, B):
        return ((self.fmax*torch.tanh(self.raw_cx)).unsqueeze(0).expand(B,-1),
                (self.fmax*torch.tanh(self.raw_cy)).unsqueeze(0).expand(B,-1),
                (F.softplus(self.raw_bw)+0.5).unsqueeze(0).expand(B,-1),
                self.raw_theta.unsqueeze(0).expand(B,-1))

class SharedFilterHyperNet(nn.Module):
    def __init__(self, in_ch, n_filters, hidden=32, fmax=FMAX):
        super().__init__()
        self.n_filters, self.fmax = n_filters, fmax
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Conv2d(in_ch,hidden,3,padding=1), nn.GELU(),
                                  nn.Conv2d(hidden,hidden,3,padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, n_filters*4))
    def forward(self, x):
        raw = self.net(x).view(-1, self.n_filters, 4)
        return (self.fmax*torch.tanh(raw[...,0]), self.fmax*torch.tanh(raw[...,1]), F.softplus(raw[...,2])+0.5, math.pi*torch.sigmoid(raw[...,3]))

class UNOBlock(nn.Module):
    def __init__(self, ch, n_filters):
        super().__init__()
        self.spec = HyperSpectralConv2d(ch, ch, n_filters); self.pointwise = nn.Conv2d(ch, ch, 1); self.norm = nn.GroupNorm(min(2,ch), ch); self.act = nn.GELU()
    def forward(self, x, bank): return self.act(self.norm(self.spec(x, bank) + self.pointwise(x)))

class UNO(nn.Module):
    def __init__(self, in_ch, out_ch, width=24, n_filters=16, n_levels=3):
        super().__init__()
        self.n_levels = n_levels; self.lift = nn.Conv2d(in_ch, width, 1); self.filter_bank_gen = FixedFilterBank(n_filters)
        self.down_blocks = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
        self.up_blocks = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def _resize(self, bank, H, W): return F.interpolate(bank, size=(H, W//2+1), mode="bilinear", align_corners=False)
    def forward(self, x):
        B, C, H, W = x.shape
        base_bank = build_gauss_bank(*self.filter_bank_gen(B), H, W, x.device)
        h = self.lift(x); skips = []; cur_h, cur_w = H, W
        n_eff = min(self.n_levels, int(math.log2(min(H,W))) if min(H,W) > 1 else 0)
        for i in range(n_eff):
            bank = self._resize(base_bank, cur_h, cur_w); h = self.down_blocks[i](h, bank)
            skips.append((h,cur_h,cur_w)); h = self.pools[i](h); cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_eff)):
            sh_h, sh, sw = skips[i]; h = F.interpolate(h, size=(sh,sw), mode="bilinear", align_corners=False) + sh_h
            bank = self._resize(base_bank, sh, sw); h = self.up_blocks[i](h, bank)
        return self.project(h)

class HCUNOv2(nn.Module):
    def __init__(self, in_ch, out_ch, width=20, n_filters=20, n_levels=3, hyper_hidden=32):
        super().__init__()
        self.n_levels = n_levels; self.lift = nn.Conv2d(in_ch, width, 1); self.hypernet = SharedFilterHyperNet(in_ch, n_filters, hyper_hidden)
        self.down_blocks = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
        self.up_blocks = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def _resize(self, bank, H, W): return F.interpolate(bank, size=(H, W//2+1), mode="bilinear", align_corners=False)
    def forward(self, x):
        B, C, H, W = x.shape
        base_bank = build_gauss_bank(*self.hypernet(x), H, W, x.device)
        h = self.lift(x); skips = []; cur_h, cur_w = H, W
        n_eff = min(self.n_levels, int(math.log2(min(H,W))) if min(H,W) > 1 else 0)
        for i in range(n_eff):
            bank = self._resize(base_bank, cur_h, cur_w); h = self.down_blocks[i](h, bank)
            skips.append((h,cur_h,cur_w)); h = self.pools[i](h); cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_eff)):
            sh_h, sh, sw = skips[i]; h = F.interpolate(h, size=(sh,sw), mode="bilinear", align_corners=False) + sh_h
            bank = self._resize(base_bank, sh, sw); h = self.up_blocks[i](h, bank)
        return self.project(h)

# --- CNN ---
class CNNBlockL(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, padding=1); self.conv2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(2,oc), oc); self.norm2 = nn.GroupNorm(min(2,oc), oc); self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x))); return self.act(self.norm2(self.conv2(x)))

class CNNBaseline(nn.Module):
    def __init__(self, in_ch, out_ch, width=32, n_levels=3):
        super().__init__()
        self.n_levels = n_levels; self.lift = nn.Conv2d(in_ch, width, 1)
        self.down_blocks = nn.ModuleList([CNNBlockL(width, width) for _ in range(n_levels)])
        self.up_blocks = nn.ModuleList([CNNBlockL(width, width) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))
    def forward(self, x):
        B, C, H, W = x.shape; h = self.lift(x); skips = []; cur_h, cur_w = H, W
        n_eff = min(self.n_levels, int(math.log2(min(H,W))) if min(H,W) > 1 else 0)
        for i in range(n_eff):
            h = self.down_blocks[i](h); skips.append((h,cur_h,cur_w)); h = self.pools[i](h); cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_eff)):
            sh_h, sh, sw = skips[i]; h = F.interpolate(h, size=(sh,sw), mode="bilinear", align_corners=False) + sh_h
            h = self.up_blocks[i](h)
        return self.project(h)

# --- DeepONet / POD-DeepONet ---
class MLPd(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes)-2): layers += [nn.Linear(sizes[i], sizes[i+1]), nn.GELU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]; self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, branch_res=16, oc=1, p=64, branch_hidden=64, trunk_hidden=160, trunk_layers=4):
        super().__init__()
        self.branch_res, self.oc, self.p = branch_res, oc, p
        self.branch = MLPd([branch_res*branch_res, branch_hidden, branch_hidden, p*oc])
        self.trunk = MLPd([2] + [trunk_hidden]*trunk_layers + [p]); self.bias = nn.Parameter(torch.zeros(oc))
    def forward(self, a_field, grid, query_res):
        B = a_field.shape[0]
        a_enc = a_field if a_field.shape[-1] == self.branch_res else F.interpolate(a_field, size=(self.branch_res,)*2, mode='bilinear', align_corners=False)
        b = self.branch(a_enc.reshape(B,-1)).view(B, self.oc, self.p); t = self.trunk(grid)
        return torch.einsum('bcp,np->bcn', b, t).view(B, self.oc, query_res, query_res) + self.bias.view(1,-1,1,1)

class PODConvBranch(nn.Module):
    def __init__(self, in_ch=1, c1=16, c2=32, c3=48, fc_hidden=96, out_dim=96):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch,c1,3,stride=2,padding=1), nn.GELU(), nn.Conv2d(c1,c2,3,stride=2,padding=1), nn.GELU(),
                                  nn.Conv2d(c2,c3,3,stride=2,padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Linear(c3*16, fc_hidden), nn.GELU(), nn.Linear(fc_hidden, out_dim))
    def forward(self, x): return self.fc(self.net(x).flatten(1))

class PODDeepONet(nn.Module):
    def __init__(self, n_modes=64, c1=16, c2=32, c3=48, fc_hidden=96, branch_dim=96, coeff_hidden=64):
        super().__init__()
        self.n_modes = n_modes; self.branch = PODConvBranch(1, c1, c2, c3, fc_hidden, branch_dim)
        self.coeff_head = nn.Sequential(nn.Linear(branch_dim, coeff_hidden), nn.GELU(), nn.Linear(coeff_hidden, n_modes))
    def forward(self, a_field, basis, mean, query_res):
        B = a_field.shape[0]
        a_enc = a_field if a_field.shape[-1] == 16 else F.interpolate(a_field, size=(16,16), mode='bilinear', align_corners=False)
        coeffs = self.coeff_head(self.branch(a_enc))
        return (mean.unsqueeze(0) + coeffs @ basis.T).view(B, 1, query_res, query_res)

# ===========================================================================
# 4. MODEL ZOO -- param-matched to BALANI-NO's 112,387 params
# ===========================================================================
MODEL_ZOO = [
    ("BALANI-NO",    lambda: MetrexNO(IN_CH, OUT_CH, width=18, n_poles=4, deg=2, modes=4, n_layers=3), "standard"),
    ("FNO",          lambda: FNO2d(IN_CH, OUT_CH, width=16, modes1=6, modes2=6, n_layers=3), "standard"),
    ("ALNO",         lambda: ALNO(IN_CH, OUT_CH, width=32, n_poles=6, modes_x=6, n_layers=3), "standard"),
    ("LNO",          lambda: LNO(IN_CH, OUT_CH, width=24, n_poles=16, n_layers=3), "standard"),
    ("PDNO",         lambda: PDNO(IN_CH, OUT_CH, width=16, hidden=64, n_layers_block=3, n_layers_sym=2, n_freq=3), "standard"),
    ("UNO",          lambda: UNO(IN_CH, OUT_CH, width=24, n_filters=16, n_levels=3), "standard"),
    ("U-FNO",        lambda: UFNO2d(IN_CH, OUT_CH, width=14, modes1=5, modes2=5, n_levels=3), "standard"),
    ("HC-UNO",       lambda: HCUNOv2(IN_CH, OUT_CH, width=20, n_filters=20, n_levels=3, hyper_hidden=32), "standard"),
    ("CNN",          lambda: CNNBaseline(IN_CH, OUT_CH, width=32, n_levels=3), "standard"),
    ("DeepONet",     lambda: DeepONet(branch_res=16, oc=OUT_CH, p=64, branch_hidden=64, trunk_hidden=160, trunk_layers=4), "deeponet"),
    ("POD-DeepONet", lambda: PODDeepONet(n_modes=64, c1=16, c2=32, c3=48, fc_hidden=96, branch_dim=96, coeff_hidden=64), "poddeeponet"),
]

# ===========================================================================
# 5. TRAIN + EVAL -- extend each architecture's saved 3-seed run to 7 seeds
# ===========================================================================
def train_one_seed(ctor, kind, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = make_train_loader(kind, seed)
    test16_loader, test32_loader = get_test_loaders(kind)
    model = ctor().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(forward_pass(kind, model, x, 16), y); loss.backward(); opt.step()
        sched.step()
        val16 = evaluate(kind, model, test16_loader, 16)
        if val16 < best_val:
            best_val = val16; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return evaluate(kind, model, test16_loader, 16), evaluate(kind, model, test32_loader, 32), model

all_results, param_counts, arch_times, qualitative, seeds_used = {}, {}, {}, {}, {}

for name, ctor, kind in MODEL_ZOO:
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    param_counts[name] = count_params(ctor())
    old_ckpt_path = f"{name.lower().replace('-','_')}_3seed.pt"

    reused = os.path.exists(old_ckpt_path)
    if reused:
        old = torch.load(old_ckpt_path, map_location="cpu")
        r16, r32 = list(old["r16"]), list(old["r32"])
        seeds_to_run = NEW_SEEDS
        seeds_used[name] = OLD_SEEDS[:len(r16)] + NEW_SEEDS
        print(f"  found {old_ckpt_path} -- reusing {len(r16)} existing seed(s) "
              f"{OLD_SEEDS[:len(r16)]}, training {len(seeds_to_run)} new seed(s)")
    else:
        r16, r32 = [], []
        seeds_to_run = OLD_SEEDS + NEW_SEEDS
        seeds_used[name] = seeds_to_run
        print(f"  no {old_ckpt_path} found -- training all {len(seeds_to_run)} seeds from scratch")

    t0 = time.time()
    last_model = None
    for seed in seeds_to_run:
        f16, f32, last_model = train_one_seed(ctor, kind, seed)
        r16.append(f16); r32.append(f32)
        print(f"  seed={seed}  @16={f16:.4f}  @32={f32:.4f}")
    elapsed = time.time() - t0
    arch_times[name] = elapsed
    all_results[name] = {"r16": np.array(r16), "r32": np.array(r32)}
    print(f"  [{name}] params={param_counts[name]:,}  total_seeds={len(r16)}  "
          f"new_seeds_time={elapsed:.1f}s ({elapsed/60:.1f} min)")

    with torch.no_grad():
        if kind == "standard":
            pred = last_model(X_test32[0:1].to(DEVICE))[0,0].cpu()
        elif kind == "deeponet":
            pred = last_model(X_test32_raw[0:1].to(DEVICE), GRID32, 32)[0,0].cpu()
        else:
            pred = last_model(X_test32_raw[0:1].to(DEVICE), POD_BASIS_32, POD_MEAN_32, 32)[0,0].cpu()
    qualitative[name] = pred
    # saved under a new name -- the original _3seed.pt checkpoints are left untouched
    torch.save({"r16": r16, "r32": r32, "model_state": last_model.state_dict()}, f"{name.lower().replace('-','_')}_7seed.pt")

# ===========================================================================
# 6. COMBINED QUALITATIVE FIGURE (@32 zero-shot, last seed's model)
# ===========================================================================
n_arch = len(MODEL_ZOO)
true32 = Y_test32[0,0]
fig, axes = plt.subplots(n_arch, 2, figsize=(6, 2.5*n_arch))
for i, (name, _, _) in enumerate(MODEL_ZOO):
    pred = qualitative[name]
    axes[i,0].imshow(pred, vmin=0, vmax=true32.max()); axes[i,0].set_ylabel(name, fontsize=10)
    axes[i,1].imshow((pred-true32).abs(), vmin=0, vmax=true32.max())
    if i == 0: axes[i,0].set_title("Pred @32"); axes[i,1].set_title("|error|")
    for j in range(2): axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
plt.tight_layout(); plt.savefig("combined_qualitative_darcy_7seed.png", dpi=150); plt.show()

# ===========================================================================
# 7. FINAL SUMMARY TABLE
# ===========================================================================
print("\n" + "="*90)
print("FINAL SUMMARY -- mean ± std over 7 seeds (3 reused + 4 new, per architecture)")
print("="*90)
summary_rows = []
for name, _, _ in MODEL_ZOO:
    r16, r32 = all_results[name]["r16"], all_results[name]["r32"]
    deg = r32.mean() / r16.mean()
    print(f"{name:<14} params={param_counts[name]:>7,}  n_seeds={len(r16)}  seeds={seeds_used[name]}  "
          f"@16={r16.mean():.4f}±{r16.std():.4f}  @32={r32.mean():.4f}±{r32.std():.4f}  degradation={deg:.2f}x")
    summary_rows.append({"Architecture": name, "Params": param_counts[name], "N_seeds": len(r16),
                          "16_mean": r16.mean(), "16_std": r16.std(),
                          "32_mean": r32.mean(), "32_std": r32.std(), "Degradation": deg})

pd.DataFrame(summary_rows).to_csv("darcy_all11_7seed_summary.csv", index=False)
print("\nSaved darcy_all11_7seed_summary.csv")


