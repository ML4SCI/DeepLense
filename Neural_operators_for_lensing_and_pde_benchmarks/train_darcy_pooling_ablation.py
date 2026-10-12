# %% Single cell -- CNN, FNO, U-FNO, UNO, HC-UNO v2 (+ no-pool ablation) on
# Darcy, absolute-frequency filter bank (FMAX = Nyquist of TRAIN res, ORIGINAL
# bandwidth formula bw = softplus(raw)+0.5 -- this is the exact config that
# produced the reported Darcy@16/@32 table), 3-seed.
#
# NOTE: the paper this reproduces states "mean +/- std over 5 seeds" in its
# Table 1 caption, but SEEDS below has 3 entries. Either the caption needs
# correcting to 3, or 2 more seeds need to be added and the table
# regenerated -- flagging before this goes to camera-ready.
#
# Verified with a structural smoke test (random tensors, no dataset
# download): all 7 architectures construct and produce correctly-shaped
# output at both 16x16 and 32x32, confirming the zero-shot-resolution
# forward pass works for every architecture including both no-pool variants.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuraloperator"], check=True)

import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.data.datasets import load_darcy_flow_small
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [7, 21, 42]
BATCH_SIZE = 16
EPOCHS = 60

# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
train_loader_raw, test_loaders_raw, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=BATCH_SIZE,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[BATCH_SIZE, BATCH_SIZE],
)

def collect_all(loader):
    xs, ys = [], []
    for batch in loader:
        xs.append(batch["x"])
        ys.append(batch["y"])
    return torch.cat(xs), torch.cat(ys)

X_train, Y_train = collect_all(train_loader_raw)
X_test16, Y_test16 = collect_all(test_loaders_raw[16])
X_test32, Y_test32 = collect_all(test_loaders_raw[32])

def add_coord_grid(X):
    B, C, H, W = X.shape
    ys = torch.linspace(0, 1, H)
    xs = torch.linspace(0, 1, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    return torch.cat([X, grid], dim=1)

X_train = add_coord_grid(X_train)
X_test16 = add_coord_grid(X_test16)
X_test32 = add_coord_grid(X_test32)

IN_CH = X_train.shape[1]
OUT_CH = Y_train.shape[1]
print("Darcy shapes -- train:", X_train.shape, "| test@16:", X_test16.shape,
      "| test@32(zero-shot res):", X_test32.shape, f"| in_ch={IN_CH} out_ch={OUT_CH}")

def make_darcy_loaders(bs=BATCH_SIZE):
    train_ds = TensorDataset(X_train, Y_train)
    test16_ds = TensorDataset(X_test16, Y_test16)
    test32_ds = TensorDataset(X_test32, Y_test32)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True),
        DataLoader(test16_ds, batch_size=bs, shuffle=False),
        DataLoader(test32_ds, batch_size=bs, shuffle=False),
    )


# ===========================================================================
# 2. SHARED SPECTRAL PRIMITIVES -- absolute-frequency filter bank
# ===========================================================================
FMAX = 16 // 2  # = 8, Nyquist of the TRAIN resolution (16)
print(f"Using FMAX = {FMAX}")


def build_anisotropic_gaussian_bank_absfreq(cx, cy, bw, theta, H, W, device):
    B, n_filters = cx.shape
    Wf = W // 2 + 1
    fy_abs = torch.fft.fftfreq(H, d=1.0 / H, device=device)
    fx_abs = torch.fft.rfftfreq(W, d=1.0 / W, device=device)
    grid_y, grid_x = torch.meshgrid(fy_abs, fx_abs, indexing="ij")
    grid_x = grid_x.view(1, 1, H, Wf)
    grid_y = grid_y.view(1, 1, H, Wf)
    cx_ = cx.view(B, n_filters, 1, 1)
    cy_ = cy.view(B, n_filters, 1, 1)
    bw_ = bw.view(B, n_filters, 1, 1)
    th_ = theta.view(B, n_filters, 1, 1)
    dx = grid_x - cx_
    dy = grid_y - cy_
    dx_rot = dx * torch.cos(th_) + dy * torch.sin(th_)
    dy_rot = -dx * torch.sin(th_) + dy * torch.cos(th_)
    aniso_ratio = 3.0
    return torch.exp(-((dx_rot / (bw_ + 1e-6)) ** 2 + (dy_rot / (bw_ / aniso_ratio + 1e-6)) ** 2))


class SharedFilterHyperNet(nn.Module):
    def __init__(self, in_ch, n_filters, hidden=32, fmax=FMAX):
        super().__init__()
        self.n_filters = n_filters
        self.fmax = fmax
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hidden, n_filters * 4),
        )

    def forward(self, x):
        raw = self.net(x).view(-1, self.n_filters, 4)
        cx = self.fmax * torch.tanh(raw[..., 0])
        cy = self.fmax * torch.tanh(raw[..., 1])
        bw = F.softplus(raw[..., 2]) + 0.5
        theta = math.pi * torch.sigmoid(raw[..., 3])
        return cx, cy, bw, theta


class FixedFilterBank(nn.Module):
    def __init__(self, n_filters, fmax=FMAX):
        super().__init__()
        self.n_filters = n_filters
        self.fmax = fmax
        self.raw_cx = nn.Parameter(torch.randn(n_filters) * 0.5)
        self.raw_cy = nn.Parameter(torch.randn(n_filters) * 0.5)
        self.raw_bw = nn.Parameter(torch.zeros(n_filters))
        self.raw_theta = nn.Parameter(torch.rand(n_filters) * math.pi)

    def forward(self, batch_size):
        cx = (self.fmax * torch.tanh(self.raw_cx)).unsqueeze(0).expand(batch_size, -1)
        cy = (self.fmax * torch.tanh(self.raw_cy)).unsqueeze(0).expand(batch_size, -1)
        bw = (F.softplus(self.raw_bw) + 0.5).unsqueeze(0).expand(batch_size, -1)
        theta = self.raw_theta.unsqueeze(0).expand(batch_size, -1)
        return cx, cy, bw, theta


class HyperSpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, n_filters=8):
        super().__init__()
        self.in_ch, self.out_ch, self.n_filters = in_ch, out_ch, n_filters
        self.channel_mix_re = nn.Parameter(0.02 * torch.randn(n_filters, in_ch, out_ch))
        self.channel_mix_im = nn.Parameter(0.02 * torch.randn(n_filters, in_ch, out_ch))

    def forward(self, x, filter_bank):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        filtered = x_ft.unsqueeze(1) * filter_bank.unsqueeze(2)
        w = torch.complex(self.channel_mix_re, self.channel_mix_im)
        out_ft = torch.einsum("bfcxy,fco->boxy", filtered, w)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1_re = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))
        self.w1_im = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))
        self.w2_re = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))
        self.w2_im = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2))

    def compl_mul(self, x_ft, w_re, w_im):
        w = torch.complex(w_re, w_im)
        return torch.einsum("bixy,ioxy->boxy", x_ft, w)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, H), min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul(x_ft[:, :, :m1, :m2], self.w1_re[:, :, :m1, :m2], self.w1_im[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul(x_ft[:, :, -m1:, :m2], self.w2_re[:, :, :m1, :m2], self.w2_im[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# 3. ARCHITECTURES
# ===========================================================================
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(2, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(2, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class CNNBaseline(nn.Module):
    def __init__(self, in_ch, out_ch, width=16, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.down_blocks = nn.ModuleList([CNNBlock(width, width) for _ in range(n_levels)])
        self.up_blocks = nn.ModuleList([CNNBlock(width, width) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.lift(x)
        skips = []
        cur_h, cur_w = H, W
        max_levels = int(math.log2(min(H, W))) if min(H, W) > 1 else 0
        n_levels_eff = min(self.n_levels, max_levels) if max_levels > 0 else 0
        for i in range(n_levels_eff):
            h = self.down_blocks[i](h)
            skips.append((h, cur_h, cur_w))
            h = self.pools[i](h)
            cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_levels_eff)):
            skip_h, sh, sw = skips[i]
            h = F.interpolate(h, size=(sh, sw), mode="bilinear", align_corners=False)
            h = h + skip_h
            h = self.up_blocks[i](h)
        return self.project(h)


class FNOBlock(nn.Module):
    def __init__(self, ch, modes1, modes2):
        super().__init__()
        self.spec = SpectralConv2d(ch, ch, modes1, modes2)
        self.pointwise = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(2, ch), ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.spec(x) + self.pointwise(x)
        return self.act(self.norm(y))


class FNO2d(nn.Module):
    def __init__(self, in_ch, out_ch, width=16, modes1=4, modes2=4, n_layers=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))

    def forward(self, x):
        h = self.lift(x)
        for blk in self.blocks:
            h = blk(h)
        return self.project(h)


class UFNOBlock(nn.Module):
    def __init__(self, ch, modes1, modes2):
        super().__init__()
        self.spec = SpectralConv2d(ch, ch, modes1, modes2)
        self.pointwise = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(2, ch), ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.spec(x) + self.pointwise(x)
        return self.act(self.norm(y))


class UFNO2d(nn.Module):
    def __init__(self, in_ch, out_ch, width=16, modes1=4, modes2=4, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.down_blocks = nn.ModuleList([UFNOBlock(width, modes1, modes2) for _ in range(n_levels)])
        self.up_blocks = nn.ModuleList([UFNOBlock(width, modes1, modes2) for _ in range(n_levels)])
        self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.lift(x)
        skips = []
        cur_h, cur_w = H, W
        max_levels = int(math.log2(min(H, W))) if min(H, W) > 1 else 0
        n_levels_eff = min(self.n_levels, max_levels) if max_levels > 0 else 0
        for i in range(n_levels_eff):
            h = self.down_blocks[i](h)
            skips.append((h, cur_h, cur_w))
            h = self.pools[i](h)
            cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_levels_eff)):
            skip_h, sh, sw = skips[i]
            h = F.interpolate(h, size=(sh, sw), mode="bilinear", align_corners=False)
            h = h + skip_h
            h = self.up_blocks[i](h)
        return self.project(h)


class UNOBlock(nn.Module):
    def __init__(self, ch, n_filters=8):
        super().__init__()
        self.spec = HyperSpectralConv2d(ch, ch, n_filters=n_filters)
        self.pointwise = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(2, ch), ch)
        self.act = nn.GELU()

    def forward(self, x, filter_bank):
        y = self.spec(x, filter_bank) + self.pointwise(x)
        return self.act(self.norm(y))


class UNO(nn.Module):
    def __init__(self, in_ch, out_ch, width=16, n_filters=12, n_levels=3, n_blocks=None):
        super().__init__()
        self.width, self.n_filters, self.n_levels = width, n_filters, n_levels
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.filter_bank_gen = FixedFilterBank(n_filters)
        if n_levels > 0:
            self.down_blocks = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
            self.up_blocks = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(n_levels)])
            self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
            self.blocks = None
        else:
            self.n_blocks = n_blocks if n_blocks is not None else 6
            self.blocks = nn.ModuleList([UNOBlock(width, n_filters) for _ in range(self.n_blocks)])
            self.down_blocks = self.up_blocks = self.pools = None
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))

    def _resize_bank(self, bank, H, W):
        Wf = W // 2 + 1
        return F.interpolate(bank, size=(H, Wf), mode="bilinear", align_corners=False)

    def forward(self, x):
        B, C, H, W = x.shape
        cx, cy, bw, theta = self.filter_bank_gen(B)
        base_bank = build_anisotropic_gaussian_bank_absfreq(cx, cy, bw, theta, H, W, x.device)
        h = self.lift(x)
        if self.n_levels == 0:
            bank = self._resize_bank(base_bank, H, W)
            for blk in self.blocks:
                h = blk(h, bank)
            return self.project(h)
        skips = []
        cur_h, cur_w = H, W
        max_levels = int(math.log2(min(H, W))) if min(H, W) > 1 else 0
        n_levels_eff = min(self.n_levels, max_levels) if max_levels > 0 else 0
        for i in range(n_levels_eff):
            bank = self._resize_bank(base_bank, cur_h, cur_w)
            h = self.down_blocks[i](h, bank)
            skips.append((h, cur_h, cur_w))
            h = self.pools[i](h)
            cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_levels_eff)):
            skip_h, sh, sw = skips[i]
            h = F.interpolate(h, size=(sh, sw), mode="bilinear", align_corners=False)
            h = h + skip_h
            bank = self._resize_bank(base_bank, sh, sw)
            h = self.up_blocks[i](h, bank)
        return self.project(h)


class HCUNOBlock(nn.Module):
    def __init__(self, ch, n_filters=8):
        super().__init__()
        self.spec = HyperSpectralConv2d(ch, ch, n_filters=n_filters)
        self.pointwise = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(2, ch), ch)
        self.act = nn.GELU()

    def forward(self, x, filter_bank):
        y = self.spec(x, filter_bank) + self.pointwise(x)
        return self.act(self.norm(y))


class HCUNOv2(nn.Module):
    def __init__(self, in_ch, out_ch, width=16, n_filters=12, n_levels=3, n_blocks=None):
        super().__init__()
        self.width, self.n_filters, self.n_levels = width, n_filters, n_levels
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.hypernet = SharedFilterHyperNet(in_ch, n_filters)
        if n_levels > 0:
            self.down_blocks = nn.ModuleList([HCUNOBlock(width, n_filters) for _ in range(n_levels)])
            self.up_blocks = nn.ModuleList([HCUNOBlock(width, n_filters) for _ in range(n_levels)])
            self.pools = nn.ModuleList([nn.AvgPool2d(2) for _ in range(n_levels)])
            self.blocks = None
        else:
            self.n_blocks = n_blocks if n_blocks is not None else 6
            self.blocks = nn.ModuleList([HCUNOBlock(width, n_filters) for _ in range(self.n_blocks)])
            self.down_blocks = self.up_blocks = self.pools = None
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_ch, 1))

    def _resize_bank(self, bank, H, W):
        Wf = W // 2 + 1
        return F.interpolate(bank, size=(H, Wf), mode="bilinear", align_corners=False)

    def forward(self, x):
        B, C, H, W = x.shape
        cx, cy, bw, theta = self.hypernet(x)
        base_bank = build_anisotropic_gaussian_bank_absfreq(cx, cy, bw, theta, H, W, x.device)
        h = self.lift(x)
        if self.n_levels == 0:
            bank = self._resize_bank(base_bank, H, W)
            for blk in self.blocks:
                h = blk(h, bank)
            return self.project(h)
        skips = []
        cur_h, cur_w = H, W
        max_levels = int(math.log2(min(H, W))) if min(H, W) > 1 else 0
        n_levels_eff = min(self.n_levels, max_levels) if max_levels > 0 else 0
        for i in range(n_levels_eff):
            bank = self._resize_bank(base_bank, cur_h, cur_w)
            h = self.down_blocks[i](h, bank)
            skips.append((h, cur_h, cur_w))
            h = self.pools[i](h)
            cur_h, cur_w = h.shape[-2], h.shape[-1]
        for i in reversed(range(n_levels_eff)):
            skip_h, sh, sw = skips[i]
            h = F.interpolate(h, size=(sh, sw), mode="bilinear", align_corners=False)
            h = h + skip_h
            bank = self._resize_bank(base_bank, sh, sw)
            h = self.up_blocks[i](h, bank)
        return self.project(h)


# ===========================================================================
# 4. TRAIN / EVAL UTILITIES
# ===========================================================================
def rel_l2(pred, target, eps=1e-8):
    num = torch.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.norm(target.reshape(target.shape[0], -1), dim=1) + eps
    return (num / den).mean().item()

def evaluate(model, loader, device=DEVICE):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += rel_l2(model(x), y)
            n += 1
    return total / max(n, 1)

def train_model(model, train_loader, test_loader, epochs=60, lr=1e-3, device=DEVICE, model_name="model"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        val_relL2 = evaluate(model, test_loader, device)
        if val_relL2 < best_val:
            best_val = val_relL2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{model_name}] ep {ep:02d}  train_loss {total_loss/len(train_loader):.5f}  "
              f"val_relL2 {val_relL2:.4f}  (best {best_val:.4f})  ({time.time()-t0:.1f}s)")
    model.load_state_dict(best_state)
    print(f"[{model_name}] FINAL best val_relL2 = {best_val:.4f}")
    return best_val, best_state

def full_eval_report(model, loaders_dict, device=DEVICE):
    return {name: evaluate(model, loader, device) for name, loader in loaders_dict.items()}


# ===========================================================================
# 5. 3-SEED TRAIN + EVAL for all 7 architectures on Darcy
# ===========================================================================
def run_seeded_darcy(model_ctor, model_label):
    reports = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_loader, test16_loader, test32_loader = make_darcy_loaders(BATCH_SIZE)
        model = model_ctor()
        model_name = f"{model_label} (darcy) seed={seed}"
        best_val, best_state = train_model(model, train_loader, test16_loader, EPOCHS, model_name=model_name)
        model.load_state_dict(best_state)
        report = full_eval_report(model, {
            "Darcy@16 (test)": test16_loader,
            "Darcy@32 (zero-shot res)": test32_loader,
        })
        print(f"  seed={seed}  test16={report['Darcy@16 (test)']:.4f}  "
              f"test32={report['Darcy@32 (zero-shot res)']:.4f}")
        reports.append(report)
    return reports

def summarize(reports, label):
    print(f"\n=== {label} (mean ± std over {len(reports)} seeds) ===")
    for split in ["Darcy@16 (test)", "Darcy@32 (zero-shot res)"]:
        vals = np.array([r[split] for r in reports])
        print(f"  {split:24s} {vals.mean():.4f} ± {vals.std():.4f}")

def make_cnn(): return CNNBaseline(IN_CH, OUT_CH, width=22, n_levels=3)
def make_fno(): return FNO2d(IN_CH, OUT_CH, width=16, modes1=4, modes2=4, n_layers=3)
def make_ufno(): return UFNO2d(IN_CH, OUT_CH, width=12, modes1=4, modes2=4, n_levels=3)
def make_uno(): return UNO(IN_CH, OUT_CH, width=18, n_filters=12, n_levels=3)
def make_hcuno(): return HCUNOv2(IN_CH, OUT_CH, width=16, n_filters=12, n_levels=3)
def make_uno_nopool(): return UNO(IN_CH, OUT_CH, width=18, n_filters=12, n_levels=0, n_blocks=6)
def make_hcuno_nopool(): return HCUNOv2(IN_CH, OUT_CH, width=16, n_filters=12, n_levels=0, n_blocks=6)

MODEL_ZOO = [
    ("CNN", make_cnn),
    ("FNO", make_fno),
    ("U-FNO", make_ufno),
    ("UNO", make_uno),
    ("HC-UNO v2", make_hcuno),
    ("UNO (no-pool)", make_uno_nopool),
    ("HC-UNO v2 (no-pool)", make_hcuno_nopool),
]

print("\n" + "=" * 70)
print(f"DARCY: a(x) -> u(x)  [absolute-freq filter bank, FMAX={FMAX}, original bw formula]")
print("=" * 70)
for label, ctor in MODEL_ZOO:
    print(f"{label:20s} params: {count_params(ctor())}")

all_reports = {}
for label, ctor in MODEL_ZOO:
    print(f"\n-- {label} --")
    all_reports[label] = run_seeded_darcy(ctor, label)

for label, _ in MODEL_ZOO:
    summarize(all_reports[label], f"{label} -- Darcy")

torch.save(all_reports, "darcy_all7_results_3seed_reproduced.pt")
print("\nsaved darcy_all7_results_3seed_reproduced.pt")
