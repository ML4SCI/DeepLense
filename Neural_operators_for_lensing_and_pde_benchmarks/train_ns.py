# %% MASTER SCRIPT -- Navier-Stokes, all 11 param-matched architectures, 7
# seeds each, @32 (test) and @64 (zero-shot). Fully self-contained:
# downloads the dataset, spectrally downsamples to 32/64, then trains/evals
# all 11 architectures.
#
# DeepONet uses branch_res=16 (not the training resolution, 32): the model
# already bilinearly downsamples whatever field it's given to branch_res
# before the first Linear layer, so this doesn't change what information
# reaches the branch -- it just matches the Darcy config exactly and keeps
# DeepONet param-matched to BALANI-NO's 112,387 (112,833) instead of
# drifting to ~162K if branch_res followed the training resolution.

# ===========================================================================
# SECTION 0: DOWNLOAD (train + test tensors only, streaming extraction)
# ===========================================================================
import os, subprocess, sys, tarfile, shutil
import requests
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuraloperator"], check=True)

DATA_ROOT = Path("/kaggle/working/ns_data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
ZENODO_RECORD_ID = "12825163"

def stream_download_and_extract(url, dest_root, keep_names, max_retries=8, timeout=300):
    if all((dest_root / n).exists() for n in keep_names):
        print(f"[skip] already have {keep_names}")
        return
    for attempt in range(max_retries):
        try:
            print(f"attempt {attempt+1}/{max_retries} ...")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                    for member in tf:
                        if Path(member.name).name in keep_names:
                            fobj = tf.extractfile(member)
                            if fobj is None:
                                continue
                            out_path = dest_root / Path(member.name).name
                            with open(out_path, "wb") as out:
                                shutil.copyfileobj(fobj, out)
                            print(f"  extracted {out_path.name} ({out_path.stat().st_size/1e6:.0f} MB)")
            return
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"  failed ({e}); retrying in {wait}s")
            for n in keep_names:
                p = dest_root / n
                if p.exists():
                    p.unlink()
            import time as _t; _t.sleep(wait)
    raise RuntimeError("download failed after retries")

url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/nsforcing_128.tgz?download=1"
stream_download_and_extract(url, DATA_ROOT, keep_names=["nsforcing_train_128.pt", "nsforcing_test_128.pt"])

# ===========================================================================
# SECTION 1: LOAD native 128x128, spectrally downsample to 32x32 (train) and
# 64x64 (zero-shot test). norm="forward" on BOTH the forward and inverse FFT
# so the reconstructed amplitude is independent of output resolution --
# without this, rfft2/irfft2's default normalization scales the result by
# (128*128)/(res_out**2), which differs at 32 vs 64, silently producing
# training data and zero-shot test data at different, inconsistent
# amplitudes.
# ===========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from neuralop.data.datasets.navier_stokes import NavierStokesDataset

BATCH_SIZE = 16
N_TRAIN, N_TEST = 1000, 200

dataset = NavierStokesDataset(
    root_dir=DATA_ROOT,
    n_train=N_TRAIN,
    n_tests=[N_TEST],
    batch_size=BATCH_SIZE,
    test_batch_sizes=[BATCH_SIZE],
    train_resolution=128,
    test_resolutions=[128],
    download=False,
)

def collect_all(loader):
    xs, ys = [], []
    for batch in loader:
        xs.append(batch["x"]); ys.append(batch["y"])
    return torch.cat(xs), torch.cat(ys)

X_train_128, Y_train_128 = collect_all(DataLoader(dataset.train_db, batch_size=BATCH_SIZE))
X_test_128, Y_test_128 = collect_all(DataLoader(dataset.test_dbs[128], batch_size=BATCH_SIZE, shuffle=False))

def downscale_spectral(t, res_out):
    B, C, H, W = t.shape
    m1 = res_out // 2
    Wf_out = res_out // 2 + 1
    t_ft = torch.fft.rfft2(t, norm="forward")
    out_ft = torch.zeros(B, C, res_out, Wf_out, dtype=t_ft.dtype, device=t.device)
    out_ft[:, :, :m1, :m1] = t_ft[:, :, :m1, :m1]
    out_ft[:, :, -m1:, :m1] = t_ft[:, :, -m1:, :m1]
    return torch.fft.irfft2(out_ft, s=(res_out, res_out), norm="forward")

X_train, Y_train = downscale_spectral(X_train_128, 32), downscale_spectral(Y_train_128, 32)
X_test32, Y_test32 = downscale_spectral(X_test_128, 32), downscale_spectral(Y_test_128, 32)
X_test64, Y_test64 = downscale_spectral(X_test_128, 64), downscale_spectral(Y_test_128, 64)

print(f"Y_train (32x32)  mean abs: {Y_train.abs().mean():.5f}  std: {Y_train.std():.5f}")
print(f"Y_test32 (32x32)  mean abs: {Y_test32.abs().mean():.5f}  std: {Y_test32.std():.5f}")
print(f"Y_test64 (64x64)  mean abs: {Y_test64.abs().mean():.5f}  std: {Y_test64.std():.5f}")
print(f"Y_test_128 (128x128, native) mean abs: {Y_test_128.abs().mean():.5f}  std: {Y_test_128.std():.5f}")

# ===========================================================================
# SECTION 2: add coord grid, build loaders (IN_CH=3, OUT_CH=1, same as Darcy)
# ===========================================================================
def add_coord_grid(X):
    B, C, H, W = X.shape
    ys = torch.linspace(0, 1, H); xs = torch.linspace(0, 1, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gy, gx], 0).unsqueeze(0).expand(B, -1, -1, -1)
    return torch.cat([X, grid], dim=1)

X_train_c = add_coord_grid(X_train)
X_test32_c = add_coord_grid(X_test32)
X_test64_c = add_coord_grid(X_test64)
IN_CH, OUT_CH = X_train_c.shape[1], Y_train.shape[1]

def make_ns_loaders(bs=BATCH_SIZE):
    train_ds = TensorDataset(X_train_c, Y_train)
    test32_ds = TensorDataset(X_test32_c, Y_test32)
    test64_ds = TensorDataset(X_test64_c, Y_test64)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True),
        DataLoader(test32_ds, batch_size=bs, shuffle=False),
        DataLoader(test64_ds, batch_size=bs, shuffle=False),
    )

train_loader, test32_loader, test64_loader = make_ns_loaders()
print(IN_CH, OUT_CH, X_train_c.shape, X_test32_c.shape, X_test64_c.shape)

# ===========================================================================
# SECTION 3: sanity checks -- confirm test32/test64 genuinely differ, and
# quantify how much real spectral energy the @64 zero-shot test actually
# adds over what's already resolvable at 32x32 (see configs/hyperparameters.md
# for why this matters when interpreting the @32 -> @64 degradation numbers)
# ===========================================================================
Y_test32_up = F.interpolate(Y_test32, size=(64,64), mode='bilinear', align_corners=False)
diff = (Y_test32_up - Y_test64).abs()
print(f"mean |Y_test32_upsampled - Y_test64|: {diff.mean():.5f}")
print(f"Y_test64 std: {Y_test64.std():.5f}")

def radial_power_spectrum(field):
    f = torch.fft.rfft2(field, norm="forward")
    power = f.abs() ** 2
    H, W = field.shape[-2], field.shape[-1]
    ky = torch.fft.fftfreq(H) * H
    kx = torch.fft.rfftfreq(W) * W
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')
    k_mag = torch.sqrt(KX**2 + KY**2)
    k_bins = torch.arange(0, H // 2)
    spectrum = torch.stack([power[..., (k_mag >= k) & (k_mag < k + 1)].mean() for k in k_bins])
    return spectrum.numpy()

spectrum = radial_power_spectrum(Y_test64[0])
below16, between, total = spectrum[:16].sum(), spectrum[16:32].sum(), spectrum.sum()
print(f"energy k<16 (resolvable at 32): {below16/total:.1%}")
print(f"energy 16<=k<32 (new at 64):    {between/total:.1%}")

# ===========================================================================
# SECTION 4: TRAIN + EVAL -- all 11 param-matched architectures, 7 seeds each
# ===========================================================================
import time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [13, 42, 97, 7, 64, 128, 2024]
EPOCHS = 60
BATCH_SIZE = 16
print(DEVICE)

# ===========================================================================
# 1. DATA (shared): coord-augmented for spectral archs, raw for branch-trunk
# ===========================================================================
def make_grid(res, device):
    ys, xs = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], 1).to(device)

GRID32, GRID64 = make_grid(32, DEVICE), make_grid(64, DEVICE)

# fixed POD basis from real training-target (w(T)) statistics, independent of seed
N_POD_MODES = 64
w_train_flat = Y_train.reshape(len(Y_train), -1)
pod_mean_ = w_train_flat.mean(dim=0)
centered = w_train_flat - pod_mean_
U, S, V = torch.pca_lowrank(centered, q=N_POD_MODES, niter=4)
POD_BASIS_32, POD_MEAN_32 = V.to(DEVICE), pod_mean_.to(DEVICE)
explained = (S ** 2).sum() / (centered ** 2).sum()
print(f"POD basis: {N_POD_MODES} modes | w(T) var explained: {explained:.4f}")

def resize_basis_to(basis, mean, from_res, to_res, device):
    n_modes = basis.shape[1]
    b_up = F.interpolate(basis.T.view(n_modes, 1, from_res, from_res), size=(to_res, to_res),
                          mode='bilinear', align_corners=False).reshape(n_modes, to_res*to_res).T.contiguous().to(device)
    m_up = F.interpolate(mean.view(1, 1, from_res, from_res), size=(to_res, to_res),
                          mode='bilinear', align_corners=False).reshape(to_res*to_res).contiguous().to(device)
    return b_up, m_up

POD_BASIS_64, POD_MEAN_64 = resize_basis_to(POD_BASIS_32.cpu(), POD_MEAN_32.cpu(), 32, 64, DEVICE)

TEST32_STD = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test32_c, Y_test32), batch_size=BATCH_SIZE, shuffle=False)
TEST64_STD = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test64_c, Y_test64), batch_size=BATCH_SIZE, shuffle=False)
TEST32_RAW = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test32, Y_test32), batch_size=BATCH_SIZE, shuffle=False)
TEST64_RAW = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test64, Y_test64), batch_size=BATCH_SIZE, shuffle=False)

def get_test_loaders(kind):
    return (TEST32_STD, TEST64_STD) if kind == "standard" else (TEST32_RAW, TEST64_RAW)

def make_train_loader(kind, seed, bs=BATCH_SIZE):
    g = torch.Generator().manual_seed(seed)
    Xtr = X_train_c if kind == "standard" else X_train
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Y_train), batch_size=bs,
                                        shuffle=True, drop_last=True, generator=g)

# ===========================================================================
# 2. FORWARD DISPATCH (handles the 3 different call signatures)
# ===========================================================================
def forward_pass(kind, model, xb, res):
    if kind == "standard":
        return model(xb)
    elif kind == "deeponet":
        return model(xb, GRID32 if res == 32 else GRID64, res)
    else:  # poddeeponet
        basis, mean = (POD_BASIS_32, POD_MEAN_32) if res == 32 else (POD_BASIS_64, POD_MEAN_64)
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
# 3. ARCHITECTURES (NS-tuned configs from the single-seed diagnostic cells)
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
    def poles(self):
        return [F.softplus(b.sp.raw_s).sqrt().detach().cpu().numpy().round(4) for b in self.blocks]

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

# --- UNO / HC-UNO (shared filter-bank construction, absolute-frequency,
#     FMAX = Nyquist of NS TRAIN resolution (32) -- this matches what the
#     single-seed NS diagnostic cells (19, 24) actually used and produced
#     real numbers with; NOT the same FMAX as the Darcy master cell (which
#     uses 8, tied to Darcy's train resolution of 16). ---
FMAX = 32 // 2
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
        a_enc = a_field if a_field.shape[-1] == 32 else F.interpolate(a_field, size=(32,32), mode='bilinear', align_corners=False)
        coeffs = self.coeff_head(self.branch(a_enc))
        return (mean.unsqueeze(0) + coeffs @ basis.T).view(B, 1, query_res, query_res)

# ===========================================================================
# 4. MODEL ZOO -- param-matched to BALANI-NO's 112,387 params (NS configs)
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
    ("DeepONet",     lambda: DeepONet(branch_res=16, oc=OUT_CH, p=64, branch_hidden=64, trunk_hidden=160, trunk_layers=4), "deeponet"),  # branch_res fixed 32->16
    ("POD-DeepONet", lambda: PODDeepONet(n_modes=64, c1=16, c2=32, c3=48, fc_hidden=96, branch_dim=96, coeff_hidden=64), "poddeeponet"),
]

# ===========================================================================
# 5. TRAIN + EVAL, 7 SEEDS PER ARCHITECTURE
# ===========================================================================
def train_one_seed(ctor, kind, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = make_train_loader(kind, seed)
    test32_loader, test64_loader = get_test_loaders(kind)
    model = ctor().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(forward_pass(kind, model, x, 32), y); loss.backward(); opt.step()
        sched.step()
        val32 = evaluate(kind, model, test32_loader, 32)
        if val32 < best_val:
            best_val = val32; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return evaluate(kind, model, test32_loader, 32), evaluate(kind, model, test64_loader, 64), model

all_results, param_counts, arch_times, qualitative = {}, {}, {}, {}

for name, ctor, kind in MODEL_ZOO:
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    param_counts[name] = count_params(ctor())
    t0 = time.time()
    r32, r64, last_model = [], [], None
    for seed in SEEDS:
        f32, f64, last_model = train_one_seed(ctor, kind, seed)
        r32.append(f32); r64.append(f64)
        print(f"  seed={seed}  @32={f32:.4f}  @64={f64:.4f}")
    elapsed = time.time() - t0
    arch_times[name] = elapsed
    all_results[name] = {"r32": np.array(r32), "r64": np.array(r64)}
    print(f"  [{name}] params={param_counts[name]:,}  time={elapsed:.1f}s ({elapsed/60:.1f} min)")

    with torch.no_grad():
        if kind == "standard":
            pred = last_model(X_test64_c[0:1].to(DEVICE))[0,0].cpu()
        elif kind == "deeponet":
            pred = last_model(X_test64[0:1].to(DEVICE), GRID64, 64)[0,0].cpu()
        else:
            pred = last_model(X_test64[0:1].to(DEVICE), POD_BASIS_64, POD_MEAN_64, 64)[0,0].cpu()
    qualitative[name] = pred
    torch.save({"r32": r32, "r64": r64, "model_state": last_model.state_dict()}, f"{name.lower().replace('-','_')}_ns_7seed.pt")

# ===========================================================================
# 6. COMBINED QUALITATIVE FIGURE (@64 zero-shot, last seed's model)
# ===========================================================================
n_arch = len(MODEL_ZOO)
true64 = Y_test64[0,0]
fig, axes = plt.subplots(n_arch, 2, figsize=(6, 2.5*n_arch))
for i, (name, _, _) in enumerate(MODEL_ZOO):
    pred = qualitative[name]
    vmax = true64.abs().max()
    axes[i,0].imshow(pred, cmap="RdBu_r", vmin=-vmax, vmax=vmax); axes[i,0].set_ylabel(name, fontsize=10)
    axes[i,1].imshow((pred-true64).abs(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if i == 0: axes[i,0].set_title("Pred @64 (zero-shot)"); axes[i,1].set_title("|error|")
    for j in range(2): axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
plt.tight_layout(); plt.savefig("combined_qualitative_ns.png", dpi=150); plt.show()

# ===========================================================================
# 7. FINAL SUMMARY TABLE
# ===========================================================================
print("\n" + "="*90)
print("FINAL SUMMARY -- mean ± std over 7 seeds")
print("="*90)
summary_rows = []
for name, _, _ in MODEL_ZOO:
    r32, r64 = all_results[name]["r32"], all_results[name]["r64"]
    deg = r64.mean() / r32.mean()
    print(f"{name:<14} params={param_counts[name]:>7,}  @32={r32.mean():.4f}±{r32.std():.4f}  "
          f"@64={r64.mean():.4f}±{r64.std():.4f}  degradation={deg:.2f}x  time={arch_times[name]:.0f}s")
    summary_rows.append({"Architecture": name, "Params": param_counts[name],
                          "32_mean": r32.mean(), "32_std": r32.std(),
                          "64_mean": r64.mean(), "64_std": r64.std(), "Degradation": deg})

pd.DataFrame(summary_rows).to_csv("ns_all11_7seed_summary.csv", index=False)
print("\nSaved ns_all11_7seed_summary.csv")
