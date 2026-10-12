# %% MASTER SCRIPT -- all 11 param-matched architectures, 7 seeds each,
# Lensing. Set LENSING_VARIANT below to switch between the two lensing
# tasks reported in the paper:
#   "full" -> kappa/psi   (full-field convergence/potential)
#   "sub"  -> kappa_sub/psi_sub (subhalo-only convergence/potential)
# Everything downstream (architectures, training loop, evaluation) is
# identical between the two -- only the field keys read from each .npz
# differ. WDM train/test; CDM + Axion are genuine physics zero-shot
# (different dark-matter models), not a resolution zero-shot like Darcy/NS.
#
# LENSING_DATA_ROOT must be set as an environment variable pointing at the
# dataset root (a directory with val/, test/ subfolders, each containing
# per-class subfolders of .npz files). No path is hardcoded here.
#
# Two deliberate departures from the 11 per-architecture cells this is built
# from, worth confirming before trusting the numbers:
#   1. The UNO cell used EPOCHS=40, every other architecture used 60 --
#      standardized to 60 here for a fair multi-seed comparison. If 40 was
#      intentional (UNO overfitting past that point), flag it and this
#      should special-case UNO's epoch count back down.
#   2. FMAX = RES // 4 (UNO, HC-UNO) is kept exactly as in the source cells
#      -- note this is a DIFFERENT formula than the Darcy/NS master cells
#      use (train_res // 2), not just a different constant at the same
#      resolution.
# Optimizer/training loop preserved exactly as validated: plain Adam (not
# AdamW) + grad-norm clipping at 1.0, model selection on WDM (test) loss
# only -- CDM/Axion are never used for checkpointing, they're pure zero-shot.
# eval() also uses the lensing cells' own formula (sum of numerator norms /
# sum of denominator norms over the whole loader), which is NOT the same
# statistic as Darcy/NS's evaluate() (mean of per-sample ratios) -- kept
# as-is since that's what every one of the 11 source cells already used.
import os, time, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [13, 42, 97, 7, 64, 128, 2024]
EPOCHS = 60
BATCH_SIZE = 32
RES = 64
LENSING_VARIANT = "sub"  # "sub" or "full" -- the two lensing tasks reported in the paper
print(DEVICE)

# ===========================================================================
# DATA -- file lists (shared between variants) + field loading (variant-specific)
# ===========================================================================
DATA_ROOT_ENV = "LENSING_DATA_ROOT"
if DATA_ROOT_ENV not in os.environ:
    raise RuntimeError(
        f"Set the {DATA_ROOT_ENV} environment variable to your dataset root "
        f"before running, e.g.: {DATA_ROOT_ENV}=/path/to/modela python train_lensing.py"
    )
ROOT = Path(os.environ[DATA_ROOT_ENV])
random.seed(13)

N_TRAIN = 2000
N_TEST = 250
N_ZEROSHOT_PER_CLASS = 250

def files_for(split_dir_name, class_name):
    split_dir = ROOT / split_dir_name
    return sorted(f for f in split_dir.rglob("*.npz") if f.parent.name == class_name)

wdm_val = files_for("val", "wdm")
wdm_test = files_for("test", "wdm")
wdm_pool = wdm_val + wdm_test
random.shuffle(wdm_pool)
assert len(wdm_pool) >= N_TRAIN + N_TEST, f"only {len(wdm_pool)} wdm files (val+test), need {N_TRAIN+N_TEST}"

train_files = wdm_pool[:N_TRAIN]
test_files_indist = wdm_pool[N_TRAIN:N_TRAIN + N_TEST]

cdm_test_all = files_for("test", "cdm")
axion_test_all = files_for("test", "axion")
assert len(cdm_test_all) >= N_ZEROSHOT_PER_CLASS, f"only {len(cdm_test_all)} cdm test files"
assert len(axion_test_all) >= N_ZEROSHOT_PER_CLASS, f"only {len(axion_test_all)} axion test files"
random.shuffle(cdm_test_all)
random.shuffle(axion_test_all)
zeroshot_files_cdm = cdm_test_all[:N_ZEROSHOT_PER_CLASS]
zeroshot_files_axion = axion_test_all[:N_ZEROSHOT_PER_CLASS]

overlap = set(train_files) & set(test_files_indist)
print(f"train(wdm)={len(train_files)} test(wdm)={len(test_files_indist)} "
      f"zero-shot(cdm)={len(zeroshot_files_cdm)} zero-shot(axion)={len(zeroshot_files_axion)} "
      f"train/test overlap={len(overlap)} (should be 0)")

KAPPA_KEY, PSI_KEY = ("kappa", "psi") if LENSING_VARIANT == "full" else ("kappa_sub", "psi_sub")
print(f"LENSING_VARIANT={LENSING_VARIANT}  fields=({KAPPA_KEY}, {PSI_KEY})")

def load(files):
    Ks, As, f = [], [], 127.0 / RES
    for fn in files:
        d = np.load(fn)
        k = F.interpolate(torch.tensor(d[KAPPA_KEY])[None, None], (RES, RES), mode='bilinear', align_corners=False)[0]
        p = F.interpolate(torch.tensor(d[PSI_KEY])[None, None], (RES, RES), mode='bilinear', align_corners=False)[0, 0]
        ay, ax = torch.gradient(p)
        Ks.append(k); As.append(torch.stack([ax, ay]) / f**2)
    return torch.stack(Ks), torch.stack(As)

K_train, A_train = load(train_files)
K_test, A_test = load(test_files_indist)
K_cdm, A_cdm = load(zeroshot_files_cdm)
K_axion, A_axion = load(zeroshot_files_axion)
print("train (wdm):      ", K_train.shape, A_train.shape)
print("test  (wdm):      ", K_test.shape, A_test.shape)
print("zero-shot (cdm):  ", K_cdm.shape, A_cdm.shape)
print("zero-shot (axion):", K_axion.shape, A_axion.shape)

FMAX = RES // 4

def rel_l2_loss(p, t):
    return (((p-t)**2).sum(dim=(1,2,3)).sqrt() / ((t**2).sum(dim=(1,2,3)).sqrt()+1e-8)).mean()

def evaluate(kind, model, loader, grid=None):
    model.eval(); n = d = 0.
    with torch.no_grad():
        for kb, ab in loader:
            kb, ab = kb.to(DEVICE), ab.to(DEVICE)
            p = model(kb, grid) if kind == "deeponet" else model(kb)
            n += ((p-ab)**2).sum(dim=(1,2,3)).sqrt().sum().item()
            d += (ab**2).sum(dim=(1,2,3)).sqrt().sum().item()
    return n/d

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_grid(res, device):
    ys, xs = torch.meshgrid(torch.linspace(0,1,res), torch.linspace(0,1,res), indexing='ij')
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], 1).to(device)

# ===========================================================================
# ARCHITECTURES (each registers its own coord grid internally at res=RES,
# except DeepONet which needs an external grid, and POD-DeepONet whose POD
# basis is baked in as registered buffers at construction time)
# ===========================================================================
# --- BALANI-NO ---
class RationalSpectral2d(nn.Module):
    def __init__(self, ic, oc, n_poles=6, deg=2, modes=8):
        super().__init__()
        self.ic, self.oc, self.m = ic, oc, modes
        self.exps = [(p, q) for p in range(deg + 1) for q in range(deg + 1) if p + q <= deg]
        self.B = len(self.exps)
        self.raw_s = nn.Parameter(torch.linspace(-6.0, 3.0, n_poles))
        self.coef = nn.Parameter(torch.randn(ic, oc, n_poles, self.B, 2) * (1.0 / (ic * oc)) ** 0.5)
        self.dc = nn.Parameter(torch.zeros(ic, oc, 2))
        s = 1.0 / (ic * oc)
        self.t1 = nn.Parameter(s * torch.rand(ic, oc, modes, modes, dtype=torch.cfloat))
        self.t2 = nn.Parameter(s * torch.rand(ic, oc, modes, modes, dtype=torch.cfloat))
    def _symbol(self, H, W, device):
        ky = torch.fft.fftfreq(H, device=device) * H
        kx = torch.fft.rfftfreq(W, device=device) * W
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        k2 = KX ** 2 + KY ** 2
        s2 = F.softplus(self.raw_s).view(-1, 1, 1)
        den = 1.0 / (k2[None] + s2 + 1e-8)
        num = torch.stack([((1j * KX) ** p) * ((1j * KY) ** q) for p, q in self.exps])
        basis = den.to(num.dtype)[:, None] * num[None]
        c = torch.view_as_complex(self.coef.contiguous())
        R = torch.einsum('iojb,jbhw->iohw', c, basis)
        dc = torch.view_as_complex(self.dc.contiguous())
        mask = torch.ones_like(R.real); mask[:, :, 0, 0] = 0.0
        pad = torch.zeros_like(R); pad[:, :, 0, 0] = dc
        return R * mask + pad
    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        xf = torch.fft.rfft2(x)
        of = torch.zeros(x.shape[0], self.oc, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        of = of + torch.einsum('bihw,iohw->bohw', xf, self._symbol(H, W, x.device))
        m = min(self.m, H // 2, W // 2 + 1)
        of[:, :, :m, :m] = of[:, :, :m, :m] + torch.einsum('bixy,ioxy->boxy', xf[:, :, :m, :m], self.t1[:, :, :m, :m])
        of[:, :, -m:, :m] = of[:, :, -m:, :m] + torch.einsum('bixy,ioxy->boxy', xf[:, :, -m:, :m], self.t2[:, :, :m, :m])
        return torch.fft.irfft2(of, s=(H, W))

class MetrexNOBlock(nn.Module):
    def __init__(self, width, n_poles, deg, modes):
        super().__init__()
        self.sp = RationalSpectral2d(width, width, n_poles, deg, modes)
        self.lc = nn.Conv2d(width, width, 1); self.act = nn.GELU()
    def forward(self, x): return self.act(self.sp(x) + self.lc(x))

class MetrexNO(nn.Module):
    def __init__(self, ic=1, oc=2, width=32, n_poles=6, deg=2, modes=8, n_layers=4, res=64):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(0,1,res), torch.linspace(0,1,res), indexing='ij')
        self.register_buffer('grid', torch.stack([ys,xs],0))
        self.fc_in = nn.Conv2d(ic+2, width, 1)
        self.blocks = nn.ModuleList([MetrexNOBlock(width, n_poles, deg, modes) for _ in range(n_layers)])
        self.fc1 = nn.Conv2d(width, width, 1); self.fc2 = nn.Conv2d(width, oc, 1); self.act = nn.GELU()
    def forward(self, x):
        h = self.fc_in(torch.cat([x, self.grid[None].repeat(x.shape[0],1,1,1)], 1))
        for b in self.blocks: h = b(h)
        return self.fc2(self.act(self.fc1(h)))
    def poles(self):
        return [F.softplus(b.sp.raw_s).sqrt().detach().cpu().numpy().round(4) for b in self.blocks]

# --- FNO / U-FNO ---
class SpectralConv2d(nn.Module):
    def __init__(self, ic, oc, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        s = 1.0/(ic*oc)
        self.w1 = nn.Parameter(s*torch.rand(ic,oc,m1,m2,dtype=torch.cfloat))
        self.w2 = nn.Parameter(s*torch.rand(ic,oc,m1,m2,dtype=torch.cfloat))
    def forward(self, x):
        B=x.shape[0]; xft=torch.fft.rfft2(x)
        o=torch.zeros(B,self.w1.shape[1],x.size(-2),x.size(-1)//2+1,dtype=torch.cfloat,device=x.device)
        m1e, m2e = min(self.m1, x.size(-2)), min(self.m2, x.size(-1)//2+1)
        o[:,:,:m1e,:m2e]=torch.einsum("bixy,ioxy->boxy",xft[:,:,:m1e,:m2e],self.w1[:,:,:m1e,:m2e])
        o[:,:,-m1e:,:m2e]=torch.einsum("bixy,ioxy->boxy",xft[:,:,-m1e:,:m2e],self.w2[:,:,:m1e,:m2e])
        return torch.fft.irfft2(o, s=(x.size(-2),x.size(-1)))

class FNOBlock(nn.Module):
    def __init__(self, w, m1, m2):
        super().__init__()
        self.sp=SpectralConv2d(w,w,m1,m2); self.lc=nn.Conv2d(w,w,1); self.act=nn.GELU()
    def forward(self,x): return self.act(self.sp(x)+self.lc(x))

class FNO2d(nn.Module):
    def __init__(self, ic=1, oc=2, width=56, m1=6, m2=6, n_layers=4, res=64):
        super().__init__()
        ys,xs=torch.meshgrid(torch.linspace(0,1,res),torch.linspace(0,1,res),indexing="ij")
        self.register_buffer("grid", torch.stack([ys,xs],0))
        self.fc_in=nn.Conv2d(ic+2,width,1)
        self.blocks=nn.ModuleList([FNOBlock(width,m1,m2) for _ in range(n_layers)])
        self.fc1=nn.Conv2d(width,width,1); self.fc2=nn.Conv2d(width,oc,1); self.act=nn.GELU()
    def forward(self,x):
        B=x.shape[0]
        h=torch.cat([x,self.grid.unsqueeze(0).repeat(B,1,1,1)],1); h=self.fc_in(h)
        for blk in self.blocks: h=blk(h)
        return self.fc2(self.act(self.fc1(h)))

class UFNOBlock(nn.Module):
    def __init__(self, w, m1, m2):
        super().__init__()
        self.sp = SpectralConv2d(w,w,m1,m2); self.lc = nn.Conv2d(w,w,1)
        self.norm = nn.GroupNorm(min(2,w), w); self.act = nn.GELU()
    def forward(self, x): return self.act(self.norm(self.sp(x)+self.lc(x)))

class UFNO2d(nn.Module):
    def __init__(self, ic=1, oc=2, width=44, m1=6, m2=6, n_levels=3, res=64):
        super().__init__()
        self.n_levels = n_levels
        ys,xs=torch.meshgrid(torch.linspace(0,1,res),torch.linspace(0,1,res),indexing="ij")
        self.register_buffer("grid", torch.stack([ys,xs],0))
        self.fc_in = nn.Conv2d(ic+2, width, 1)
        self.down = nn.ModuleList([UFNOBlock(width, m1, m2) for _ in range(n_levels)])
        self.up = nn.ModuleList([UFNOBlock(width, m1, m2) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.fc1 = nn.Conv2d(width, width, 1); self.fc2 = nn.Conv2d(width, oc, 1); self.act = nn.GELU()
    def forward(self, x):
        B = x.shape[0]
        h = self.fc_in(torch.cat([x, self.grid[None].repeat(B,1,1,1)], 1))
        skips = []
        n_eff = min(self.n_levels, int(math.log2(min(h.shape[-2],h.shape[-1]))) if min(h.shape[-2],h.shape[-1])>1 else 0)
        for i in range(n_eff):
            h = self.down[i](h); skips.append(h); h = self.pools[i](h)
        for i in reversed(range(n_eff)):
            h = F.interpolate(h, size=skips[i].shape[-2:], mode="bilinear", align_corners=False) + skips[i]
            h = self.up[i](h)
        return self.fc2(self.act(self.fc1(h)))

# --- ALNO / LNO ---
class ALNOLayer(nn.Module):
    def __init__(self, ic, oc, n_poles=4, modes_x=8):
        super().__init__()
        self.ic, self.oc, self.N, self.mx = ic, oc, n_poles, modes_x
        self.mu_re = nn.Parameter(-1.0 - torch.rand(ic, oc, n_poles))
        self.mu_im = nn.Parameter(torch.linspace(-3, 3, n_poles).expand(ic, oc, n_poles).clone())
        s0 = (1.0/(ic*oc))**0.5
        self.beta_re = nn.Parameter(torch.randn(ic, oc, n_poles) * s0)
        self.beta_im = nn.Parameter(torch.randn(ic, oc, n_poles) * s0)
        s = 1.0/(ic*oc)
        self.H_re = nn.Parameter(s * torch.rand(ic, oc, modes_x))
        self.H_im = nn.Parameter(s * torch.rand(ic, oc, modes_x))
    def forward(self, x):
        B, _, H, W = x.shape
        xf = torch.fft.rfft2(x)
        mx = min(self.mx, W//2+1)
        H_ = torch.complex(self.H_re[:,:,:mx], self.H_im[:,:,:mx])
        out_ft = torch.zeros(B, self.oc, H, W//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:,:,:,:mx] = torch.einsum('bihk,iok->bohk', xf[:,:,:,:mx], H_)
        omega_y = torch.fft.fftfreq(H, device=x.device) * H * 2 * np.pi
        iom = 1j * omega_y
        mu = torch.complex(self.mu_re, self.mu_im)
        beta = torch.complex(self.beta_re, self.beta_im)
        denom = iom.view(1,1,1,-1) - mu.unsqueeze(-1)
        K_sym = (beta.unsqueeze(-1) / denom).sum(dim=2)
        pole_ft = torch.einsum('bihk,ioh->bohk', xf, K_sym)
        out_ft = out_ft + pole_ft
        return torch.fft.irfft2(out_ft, s=(H, W))

class ALNOBlock(nn.Module):
    def __init__(self, width, n_poles, modes_x):
        super().__init__()
        self.al = ALNOLayer(width, width, n_poles, modes_x)
        self.lc = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.al(x) + self.lc(x))

class ALNO(nn.Module):
    def __init__(self, ic=1, oc=2, width=80, n_poles=4, modes_x=8, n_layers=4, res=64):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(0,1,res), torch.linspace(0,1,res), indexing='ij')
        self.register_buffer('grid', torch.stack([ys,xs],0))
        self.fc_in = nn.Conv2d(ic+2, width, 1)
        self.blocks = nn.ModuleList([ALNOBlock(width, n_poles, modes_x) for _ in range(n_layers)])
        self.fc1 = nn.Conv2d(width, width, 1); self.fc2 = nn.Conv2d(width, oc, 1); self.act = nn.GELU()
    def forward(self, x):
        h = self.fc_in(torch.cat([x, self.grid[None].repeat(x.shape[0],1,1,1)], 1))
        for b in self.blocks: h = b(h)
        return self.fc2(self.act(self.fc1(h)))

class LNOLayer(nn.Module):
    def __init__(self, ic, oc, n_poles=8):
        super().__init__()
        self.ic, self.oc, self.N = ic, oc, n_poles
        self.mu_re = nn.Parameter(-1.0 - torch.rand(ic, oc, n_poles))
        self.mu_im = nn.Parameter(torch.linspace(-3, 3, n_poles).expand(ic, oc, n_poles).clone())
        s0 = (1.0/(ic*oc))**0.5
        self.beta_re = nn.Parameter(torch.randn(ic, oc, n_poles) * s0)
        self.beta_im = nn.Parameter(torch.randn(ic, oc, n_poles) * s0)
    def forward(self, x):
        B, _, H, W = x.shape
        xf = torch.fft.rfft2(x)
        omega_y = torch.fft.fftfreq(H, device=x.device) * H * 2 * np.pi
        iom = 1j * omega_y
        mu   = torch.complex(self.mu_re, self.mu_im)
        beta = torch.complex(self.beta_re, self.beta_im)
        denom = iom.view(1,1,1,-1) - mu.unsqueeze(-1)
        K_sym = (beta.unsqueeze(-1) / denom).sum(dim=2)
        out_ft = torch.einsum('bihk,ioh->bohk', xf, K_sym)
        return torch.fft.irfft2(out_ft, s=(H, W))

class LNOBlock(nn.Module):
    def __init__(self, width, n_poles):
        super().__init__()
        self.ln = LNOLayer(width, width, n_poles)
        self.lc = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.ln(x) + self.lc(x))

class LNO(nn.Module):
    def __init__(self, ic=1, oc=2, width=80, n_poles=8, n_layers=4, res=64):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(0,1,res), torch.linspace(0,1,res), indexing='ij')
        self.register_buffer('grid', torch.stack([ys,xs],0))
        self.fc_in = nn.Conv2d(ic+2, width, 1)
        self.blocks = nn.ModuleList([LNOBlock(width, n_poles) for _ in range(n_layers)])
        self.fc1 = nn.Conv2d(width, width, 1); self.fc2 = nn.Conv2d(width, oc, 1); self.act = nn.GELU()
    def forward(self, x):
        h = self.fc_in(torch.cat([x, self.grid[None].repeat(x.shape[0],1,1,1)], 1))
        for b in self.blocks: h = b(h)
        return self.fc2(self.act(self.fc1(h)))

# --- PDNO ---
class SymbolNet(nn.Module):
    def __init__(self, ic, oc, hidden=64, n_layers=3, n_freq=4):
        super().__init__()
        self.ic, self.oc = ic, oc
        layers, c = [], 2*(1+2*n_freq)
        for _ in range(n_layers):
            layers += [nn.Conv2d(c, hidden, 1), nn.GELU()]; c = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, ic*oc*2, 1)
        self.n_freq = n_freq
    def _feat(self, H, W, device):
        ky = torch.fft.fftfreq(H, device=device)*H
        kx = torch.fft.rfftfreq(W, device=device)*W
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        nyq = max(H, W)/2
        base = torch.stack([KX/nyq, KY/nyq])
        ff = [base]
        for i in range(self.n_freq):
            ff += [torch.sin(2**i*np.pi*base), torch.cos(2**i*np.pi*base)]
        return torch.cat(ff, 0)[None]
    def forward(self, H, W, device):
        h = self.head(self.body(self._feat(H, W, device)))
        h = h.view(self.ic, self.oc, 2, H, W//2+1)
        return torch.complex(h[:,:,0], h[:,:,1])

class PDIOLayer(nn.Module):
    def __init__(self, ic, oc, hidden=64, n_layers=3, n_freq=4):
        super().__init__()
        self.sym = SymbolNet(ic, oc, hidden, n_layers, n_freq)
    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        xf = torch.fft.rfft2(x)
        R = self.sym(H, W, x.device)
        of = torch.einsum('bihw,iohw->bohw', xf, R)
        return torch.fft.irfft2(of, s=(H, W))

class PDNOBlock(nn.Module):
    def __init__(self, width, hidden, n_layers, n_freq):
        super().__init__()
        self.pdio = PDIOLayer(width, width, hidden, n_layers, n_freq)
        self.lc = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.pdio(x) + self.lc(x))

class PDNO(nn.Module):
    def __init__(self, ic=1, oc=2, width=40, hidden=64, n_layers_block=4, n_freq=4, res=64):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(0,1,res), torch.linspace(0,1,res), indexing='ij')
        self.register_buffer('grid', torch.stack([ys,xs],0))
        self.fc_in = nn.Conv2d(ic+2, width, 1)
        self.blocks = nn.ModuleList([PDNOBlock(width, hidden, 3, n_freq) for _ in range(n_layers_block)])
        self.fc1 = nn.Conv2d(width, width, 1)
        self.fc2 = nn.Conv2d(width, oc, 1)
        self.act = nn.GELU()
    def forward(self, x):
        h = self.fc_in(torch.cat([x, self.grid[None].repeat(x.shape[0],1,1,1)], 1))
        for b in self.blocks: h = b(h)
        return self.fc2(self.act(self.fc1(h)))

# --- UNO / HC-UNO ---
def build_gauss_bank(cx, cy, bw, theta, H, W, device):
    B, n_filters = cx.shape; Wf = W//2+1
    fy = torch.fft.fftfreq(H, d=1.0/H, device=device); fx = torch.fft.rfftfreq(W, d=1.0/W, device=device)
    gy, gx = torch.meshgrid(fy, fx, indexing="ij"); gx, gy = gx.view(1,1,H,Wf), gy.view(1,1,H,Wf)
    cx_, cy_, bw_, th_ = cx.view(B,-1,1,1), cy.view(B,-1,1,1), bw.view(B,-1,1,1), theta.view(B,-1,1,1)
    dx, dy = gx-cx_, gy-cy_
    dxr = dx*torch.cos(th_) + dy*torch.sin(th_); dyr = -dx*torch.sin(th_) + dy*torch.cos(th_)
    return torch.exp(-((dxr/(bw_+1e-6))**2 + (dyr/(bw_/3+1e-6))**2))

class FixedFilterBank(nn.Module):
    def __init__(self, n_filters, fmax=FMAX):
        super().__init__()
        self.n_filters, self.fmax = n_filters, fmax
        self.raw_cx = nn.Parameter(torch.randn(n_filters)*0.5)
        self.raw_cy = nn.Parameter(torch.randn(n_filters)*0.5)
        self.raw_bw = nn.Parameter(torch.zeros(n_filters))
        self.raw_theta = nn.Parameter(torch.rand(n_filters)*math.pi)
    def forward(self, B):
        return ((self.fmax*torch.tanh(self.raw_cx)).unsqueeze(0).expand(B,-1),
                (self.fmax*torch.tanh(self.raw_cy)).unsqueeze(0).expand(B,-1),
                (F.softplus(self.raw_bw)+0.5).unsqueeze(0).expand(B,-1),
                self.raw_theta.unsqueeze(0).expand(B,-1))

class SharedFilterHyperNet(nn.Module):
    def __init__(self, ic, n_filters, hidden=32, fmax=FMAX):
        super().__init__()
        self.n_filters, self.fmax = n_filters, fmax
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(8), nn.Conv2d(ic, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, n_filters*4))
    def forward(self, x):
        raw = self.net(x).view(-1, self.n_filters, 4)
        return (self.fmax*torch.tanh(raw[...,0]), self.fmax*torch.tanh(raw[...,1]),
                F.softplus(raw[...,2])+0.5, math.pi*torch.sigmoid(raw[...,3]))

class HyperSpectralConv2d(nn.Module):
    def __init__(self, ic, oc, n_filters=8):
        super().__init__()
        self.channel_mix_re = nn.Parameter(0.02*torch.randn(n_filters, ic, oc))
        self.channel_mix_im = nn.Parameter(0.02*torch.randn(n_filters, ic, oc))
    def forward(self, x, bank):
        B, C, H, W = x.shape; xf = torch.fft.rfft2(x)
        filtered = xf.unsqueeze(1) * bank.unsqueeze(2)
        return torch.fft.irfft2(torch.einsum("bfcxy,fco->boxy", filtered, torch.complex(self.channel_mix_re, self.channel_mix_im)), s=(H,W))

class UNOBlock(nn.Module):
    def __init__(self, ch, n_filters):
        super().__init__()
        self.spec = HyperSpectralConv2d(ch, ch, n_filters); self.lc = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(2,ch), ch); self.act = nn.GELU()
    def forward(self, x, bank): return self.act(self.norm(self.spec(x, bank) + self.lc(x)))

class UNO(nn.Module):
    def __init__(self, ic=1, oc=2, width=64, n_filters=16, n_levels=3, res=64):
        super().__init__()
        self.n_levels = n_levels
        ys,xs=torch.meshgrid(torch.linspace(0,1,res),torch.linspace(0,1,res),indexing="ij")
        self.register_buffer("grid", torch.stack([ys,xs],0))
        self.fc_in = nn.Conv2d(ic+2, width, 1)
        self.filter_bank_gen = FixedFilterBank(n_filters)
        self.down = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
        self.up = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.fc1 = nn.Conv2d(width, width, 1); self.fc2 = nn.Conv2d(width, oc, 1); self.act = nn.GELU()
    def _resize(self, bank, H, W):
        return F.interpolate(bank, size=(H, W//2+1), mode="bilinear", align_corners=False)
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.fc_in(torch.cat([x, self.grid[None].repeat(B,1,1,1)], 1))
        base_bank = build_gauss_bank(*self.filter_bank_gen(B), H, W, x.device)
        skips = []; cur_h, cur_w = h.shape[-2], h.shape[-1]
        n_eff = min(self.n_levels, int(math.log2(min(cur_h,cur_w))) if min(cur_h,cur_w)>1 else 0)
        for i in range(n_eff):
            bank = self._resize(base_bank, cur_h, cur_w)
            h = self.down[i](h, bank); skips.append((h,cur_h,cur_w))
            h = self.pools[i](h); cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_eff)):
            sh_h, sh, sw = skips[i]
            h = F.interpolate(h, size=(sh,sw), mode="bilinear", align_corners=False) + sh_h
            bank = self._resize(base_bank, sh, sw)
            h = self.up[i](h, bank)
        return self.fc2(self.act(self.fc1(h)))

class HCUNOv2(nn.Module):
    def __init__(self, ic=1, oc=2, width=64, n_filters=16, n_levels=3, hyper_hidden=32, res=64):
        super().__init__()
        self.n_levels = n_levels
        ys,xs=torch.meshgrid(torch.linspace(0,1,res),torch.linspace(0,1,res),indexing="ij")
        self.register_buffer("grid", torch.stack([ys,xs],0))
        self.fc_in = nn.Conv2d(ic+2, width, 1)
        self.hypernet = SharedFilterHyperNet(ic+2, n_filters, hyper_hidden)
        self.down = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
        self.up = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.fc1 = nn.Conv2d(width, width, 1); self.fc2 = nn.Conv2d(width, oc, 1); self.act = nn.GELU()
    def _resize(self, bank, H, W):
        return F.interpolate(bank, size=(H, W//2+1), mode="bilinear", align_corners=False)
    def forward(self, x):
        B, C, H, W = x.shape
        x_aug = torch.cat([x, self.grid[None].repeat(B,1,1,1)], 1)
        h = self.fc_in(x_aug)
        base_bank = build_gauss_bank(*self.hypernet(x_aug), H, W, x.device)
        skips = []; cur_h, cur_w = h.shape[-2], h.shape[-1]
        n_eff = min(self.n_levels, int(math.log2(min(cur_h,cur_w))) if min(cur_h,cur_w)>1 else 0)
        for i in range(n_eff):
            bank = self._resize(base_bank, cur_h, cur_w)
            h = self.down[i](h, bank); skips.append((h,cur_h,cur_w))
            h = self.pools[i](h); cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_eff)):
            sh_h, sh, sw = skips[i]
            h = F.interpolate(h, size=(sh,sw), mode="bilinear", align_corners=False) + sh_h
            bank = self._resize(base_bank, sh, sw)
            h = self.up[i](h, bank)
        return self.fc2(self.act(self.fc1(h)))

# --- CNN ---
class CNNBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, padding=1); self.conv2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(2,oc), oc); self.norm2 = nn.GroupNorm(min(2,oc), oc); self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        return self.act(self.norm2(self.conv2(x)))

class CNNBaseline(nn.Module):
    def __init__(self, ic=1, oc=2, width=88, n_levels=3, res=64):
        super().__init__()
        self.n_levels = n_levels
        ys,xs=torch.meshgrid(torch.linspace(0,1,res),torch.linspace(0,1,res),indexing="ij")
        self.register_buffer("grid", torch.stack([ys,xs],0))
        self.fc_in = nn.Conv2d(ic+2, width, 1)
        self.down = nn.ModuleList([CNNBlock(width, width) for _ in range(n_levels)])
        self.up = nn.ModuleList([CNNBlock(width, width) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.fc1 = nn.Conv2d(width, width, 1); self.fc2 = nn.Conv2d(width, oc, 1); self.act = nn.GELU()
    def forward(self, x):
        B = x.shape[0]
        h = self.fc_in(torch.cat([x, self.grid[None].repeat(B,1,1,1)], 1))
        skips = []
        n_eff = min(self.n_levels, int(math.log2(min(h.shape[-2],h.shape[-1]))) if min(h.shape[-2],h.shape[-1])>1 else 0)
        for i in range(n_eff):
            h = self.down[i](h); skips.append(h); h = self.pools[i](h)
        for i in reversed(range(n_eff)):
            h = F.interpolate(h, size=skips[i].shape[-2:], mode="bilinear", align_corners=False) + skips[i]
            h = self.up[i](h)
        return self.fc2(self.act(self.fc1(h)))

# --- DeepONet / POD-DeepONet ---
class MLPd(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes)-2): layers += [nn.Linear(sizes[i], sizes[i+1]), nn.GELU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]; self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, res=64, oc=2, p=96, branch_hidden=160, trunk_hidden=160, trunk_layers=5):
        super().__init__()
        self.res, self.oc, self.p = res, oc, p
        self.branch = MLPd([res*res, branch_hidden, branch_hidden, p*oc])
        self.trunk = MLPd([2] + [trunk_hidden]*trunk_layers + [p])
        self.bias = nn.Parameter(torch.zeros(oc))
    def forward(self, kappa, grid):
        B = kappa.shape[0]
        b = self.branch(kappa.reshape(B, -1)).view(B, self.oc, self.p)
        t = self.trunk(grid)
        return torch.einsum('bcp,np->bcn', b, t).view(B, self.oc, self.res, self.res) + self.bias.view(1,-1,1,1)

class PODConvBranch(nn.Module):
    def __init__(self, ic=1, c1=32, c2=64, c3=128, fc_hidden=320, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, c1, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Sequential(nn.Linear(c3*16, fc_hidden), nn.GELU(), nn.Linear(fc_hidden, out_dim))
    def forward(self, x): return self.fc(self.net(x).flatten(1))

class PODDeepONet(nn.Module):
    def __init__(self, pod_basis_x, pod_mean_x, pod_basis_y, pod_mean_y, n_modes=160,
                 c1=32, c2=64, c3=128, fc_hidden=320, branch_dim=128, coeff_hidden=96):
        super().__init__()
        self.n_modes = n_modes
        self.register_buffer("pod_basis_x", pod_basis_x); self.register_buffer("pod_mean_x", pod_mean_x)
        self.register_buffer("pod_basis_y", pod_basis_y); self.register_buffer("pod_mean_y", pod_mean_y)
        self.branch = PODConvBranch(1, c1, c2, c3, fc_hidden, branch_dim)
        self.coeff_head = nn.Sequential(nn.Linear(branch_dim, coeff_hidden), nn.GELU(), nn.Linear(coeff_hidden, 2*n_modes))
    def forward(self, x):
        B, C, H, W = x.shape
        coeffs = self.coeff_head(self.branch(x))
        cx, cy = coeffs[:,:self.n_modes], coeffs[:,self.n_modes:]
        ax = self.pod_mean_x.unsqueeze(0) + cx @ self.pod_basis_x.T
        ay = self.pod_mean_y.unsqueeze(0) + cy @ self.pod_basis_y.T
        return torch.stack([ax.view(B,H,W), ay.view(B,H,W)], dim=1)

# ===========================================================================
# POD basis for POD-DeepONet -- fixed from real A_train statistics, built
# once, independent of seed (same convention as the Darcy/NS master cells)
# ===========================================================================
N_POD_MODES = 160
ax_train_flat = A_train[:,0].reshape(len(A_train), -1)
ay_train_flat = A_train[:,1].reshape(len(A_train), -1)
pod_mean_x_ = ax_train_flat.mean(dim=0); pod_mean_y_ = ay_train_flat.mean(dim=0)
centered_x = ax_train_flat - pod_mean_x_; centered_y = ay_train_flat - pod_mean_y_
Ux, Sx, Vx = torch.pca_lowrank(centered_x, q=N_POD_MODES, niter=4)
Uy, Sy, Vy = torch.pca_lowrank(centered_y, q=N_POD_MODES, niter=4)
POD_BASIS_X, POD_BASIS_Y = Vx.to(DEVICE), Vy.to(DEVICE)
POD_MEAN_X, POD_MEAN_Y = pod_mean_x_.to(DEVICE), pod_mean_y_.to(DEVICE)
explained_x = (Sx**2).sum() / (centered_x**2).sum()
explained_y = (Sy**2).sum() / (centered_y**2).sum()
print(f"POD basis: {N_POD_MODES} modes | alpha_x var explained: {explained_x:.4f} | alpha_y: {explained_y:.4f}")

GRID = make_grid(RES, DEVICE)  # only used by DeepONet's "deeponet" kind
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_test, A_test), batch_size=BATCH_SIZE)
cdm_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_cdm, A_cdm), batch_size=BATCH_SIZE)
axion_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_axion, A_axion), batch_size=BATCH_SIZE)

# ===========================================================================
# MODEL ZOO
# ===========================================================================
MODEL_ZOO = [
    ("BALANI-NO",    lambda: MetrexNO(width=32, n_poles=6, modes=8, res=RES), "standard"),
    ("FNO",          lambda: FNO2d(width=56, m1=6, m2=6, res=RES), "standard"),
    ("ALNO",         lambda: ALNO(width=80, n_poles=4, modes_x=8, res=RES), "standard"),
    ("LNO",          lambda: LNO(width=80, n_poles=8, res=RES), "standard"),
    ("PDNO",         lambda: PDNO(width=40, hidden=64, n_layers_block=4, n_freq=4, res=RES), "standard"),
    ("UNO",          lambda: UNO(width=64, n_filters=16, n_levels=3, res=RES), "standard"),
    ("U-FNO",        lambda: UFNO2d(width=44, m1=6, m2=6, n_levels=3, res=RES), "standard"),
    ("HC-UNO",       lambda: HCUNOv2(width=64, n_filters=16, n_levels=3, hyper_hidden=32, res=RES), "standard"),
    ("CNN",          lambda: CNNBaseline(width=88, n_levels=3, res=RES), "standard"),
    ("DeepONet",     lambda: DeepONet(res=RES, oc=2, p=96, branch_hidden=160, trunk_hidden=160, trunk_layers=5), "deeponet"),
    ("POD-DeepONet", lambda: PODDeepONet(POD_BASIS_X.clone(), POD_MEAN_X.clone(), POD_BASIS_Y.clone(), POD_MEAN_Y.clone(),
                                          n_modes=N_POD_MODES, c1=32, c2=64, c3=128, fc_hidden=320, branch_dim=128, coeff_hidden=96), "standard"),
]

# ===========================================================================
# TRAIN + EVAL, 7 SEEDS PER ARCHITECTURE
# ===========================================================================
def train_one_seed(ctor, kind, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(K_train, A_train),
                                                batch_size=BATCH_SIZE, shuffle=True, generator=g)
    model = ctor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val, best_state = float("inf"), None
    for ep in range(EPOCHS):
        model.train()
        for kb, ab in train_loader:
            kb, ab = kb.to(DEVICE), ab.to(DEVICE)
            opt.zero_grad()
            pred = model(kb, GRID) if kind == "deeponet" else model(kb)
            rel_l2_loss(pred, ab).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        grid_arg = GRID if kind == "deeponet" else None
        val_wdm = evaluate(kind, model, test_loader, grid_arg)
        if val_wdm < best_val:
            best_val = val_wdm; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    grid_arg = GRID if kind == "deeponet" else None
    final_wdm = evaluate(kind, model, test_loader, grid_arg)
    final_cdm = evaluate(kind, model, cdm_loader, grid_arg)
    final_axion = evaluate(kind, model, axion_loader, grid_arg)
    return final_wdm, final_cdm, final_axion, model

all_results, param_counts, arch_times, qualitative = {}, {}, {}, {}

for name, ctor, kind in MODEL_ZOO:
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    param_counts[name] = count_params(ctor())
    t0 = time.time()
    r_wdm, r_cdm, r_axion, last_model = [], [], [], None
    for seed in SEEDS:
        fw, fc, fa, last_model = train_one_seed(ctor, kind, seed)
        r_wdm.append(fw); r_cdm.append(fc); r_axion.append(fa)
        print(f"  seed={seed}  wdm={fw:.4f}  cdm(0-shot)={fc:.4f}  axion(0-shot)={fa:.4f}")
    elapsed = time.time() - t0
    arch_times[name] = elapsed
    all_results[name] = {"wdm": np.array(r_wdm), "cdm": np.array(r_cdm), "axion": np.array(r_axion)}
    print(f"  [{name}] params={param_counts[name]:,}  time={elapsed:.1f}s ({elapsed/60:.1f} min)")

    with torch.no_grad():
        grid_arg = GRID if kind == "deeponet" else None
        kb = K_axion[0:1].to(DEVICE)
        pred = (last_model(kb, grid_arg) if kind == "deeponet" else last_model(kb))[0].cpu()
    qualitative[name] = pred
    torch.save({"wdm": r_wdm, "cdm": r_cdm, "axion": r_axion, "model_state": last_model.state_dict()},
               f"{name.lower().replace('-','_')}_lensing_{LENSING_VARIANT}_7seed.pt")

# ===========================================================================
# COMBINED QUALITATIVE FIGURE (|alpha| on Axion -- furthest zero-shot)
# ===========================================================================
n_arch = len(MODEL_ZOO)
true_mag = A_axion[0].norm(dim=0)
fig, axes = plt.subplots(n_arch, 2, figsize=(6, 2.5*n_arch))
vmax = true_mag.max().item()
for i, (name, _, _) in enumerate(MODEL_ZOO):
    pred_mag = qualitative[name].norm(dim=0)
    axes[i,0].imshow(pred_mag, vmin=0, vmax=vmax); axes[i,0].set_ylabel(name, fontsize=10)
    axes[i,1].imshow((pred_mag-true_mag).abs(), vmin=0, vmax=vmax)
    if i == 0: axes[i,0].set_title("Pred |alpha| (Axion, 0-shot)"); axes[i,1].set_title("|error|")
    for j in range(2): axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
plt.tight_layout(); plt.savefig(f"combined_qualitative_lensing_{LENSING_VARIANT}.png", dpi=150); plt.show()

# ===========================================================================
# FINAL SUMMARY TABLE
# ===========================================================================
print("\n" + "="*100)
print("FINAL SUMMARY -- mean ± std over 7 seeds")
print("="*100)
summary_rows = []
for name, _, _ in MODEL_ZOO:
    rw, rc, ra = all_results[name]["wdm"], all_results[name]["cdm"], all_results[name]["axion"]
    print(f"{name:<14} params={param_counts[name]:>7,}  wdm={rw.mean():.4f}±{rw.std():.4f}  "
          f"cdm(0-shot)={rc.mean():.4f}±{rc.std():.4f}  axion(0-shot)={ra.mean():.4f}±{ra.std():.4f}  "
          f"time={arch_times[name]:.0f}s")
    summary_rows.append({"Architecture": name, "Params": param_counts[name],
                          "wdm_mean": rw.mean(), "wdm_std": rw.std(),
                          "cdm_mean": rc.mean(), "cdm_std": rc.std(),
                          "axion_mean": ra.mean(), "axion_std": ra.std()})

out_csv = f"lensing_all11_{LENSING_VARIANT}_7seed_summary.csv"
pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")
