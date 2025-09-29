
import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter

@dataclass
class NoiseConfig:
    # Background in counts (pre-normalization, preserved DC)
    bg_enable: bool = True
    bg_peak_counts: Union[float, Tuple[float, float]] = (800.0, 1500.0)
    bg_sky_counts:  Union[float, Tuple[float, float]] = (80.0, 300.0)
    bg_read_std_counts: float = 3.0
    bg_prob: float = 1.0

    # Optional post scaling noise (in [0,1])
    post_mode: str = "none"          # "none"|"gaussian"|"poisson"|"poissongauss"
    post_level: Union[float, Tuple[float, float]] = 0.0
    post_read_std: float = 0.0
    post_prob: float = 1.0

class LRNoiseAugment:
    """
    Deterministic per-index LR noise. Same idx -> same noise every epoch.
    """
    def __init__(self, cfg: NoiseConfig, base_seed: int = 12345):
        self.cfg = cfg
        self.base_seed = int(base_seed)

    @staticmethod
    def _u(rng, v):  # sample scalar or (lo,hi)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
            return float(rng.uniform(lo, hi))
        return float(v)

    def _rng(self, idx: int):
        seed = (self.base_seed * 1000003 + idx * 9176) & 0x7fffffff
        return np.random.RandomState(seed)

    def _background_counts_then_scale01(self, lr_np: np.ndarray, rng) -> np.ndarray:
        # robust scale to [0,1] without removing DC
        scale = np.percentile(lr_np, 99.5)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        lr01 = np.clip(lr_np / scale, 0.0, 1.0).astype(np.float32)

        peak = self._u(rng, self.cfg.bg_peak_counts)
        sky  = self._u(rng, self.cfg.bg_sky_counts)

        src  = lr01 * peak
        mean = src + sky
        counts = rng.poisson(mean).astype(np.float32)
        if self.cfg.bg_read_std_counts > 0:
            counts += rng.normal(0.0, self.cfg.bg_read_std_counts, size=counts.shape).astype(np.float32)
        counts = np.clip(counts, 0.0, None, out=counts)

        denom = max(1.0, peak + sky)
        return np.clip(counts / denom, 0.0, 1.0).astype(np.float32)

    def _post(self, img01: np.ndarray, rng) -> np.ndarray:
        m = self.cfg.post_mode.lower()
        if m == "none" or rng.rand() > self.cfg.post_prob:
            return img01
        if m == "gaussian":
            sigma = self._u(rng, self.cfg.post_level)
            out = img01 + rng.normal(0.0, sigma, size=img01.shape).astype(np.float32)
            return np.clip(out, 0.0, 1.0, out=out)
        if m == "poisson":
            peak = max(1.0, self._u(rng, self.cfg.post_level))
            lam = np.clip(img01, 0.0, 1.0) * peak
            counts = rng.poisson(lam).astype(np.float32)
            return np.clip(counts / peak, 0.0, 1.0)
        if m in ("poissongauss", "poisson+gaussian", "pg"):
            peak = max(1.0, self._u(rng, self.cfg.post_level))
            lam = np.clip(img01, 0.0, 1.0) * peak
            counts = rng.poisson(lam).astype(np.float32)
            if self.cfg.post_read_std > 0.0:
                counts += rng.normal(0.0, self.cfg.post_read_std, size=counts.shape).astype(np.float32)
            counts = np.clip(counts, 0.0, None, out=counts)
            return np.clip(counts / peak, 0.0, 1.0)
        return img01

    def __call__(self, lr_np: np.ndarray, idx: int) -> np.ndarray:
        rng = self._rng(idx)
        if self.cfg.bg_enable and rng.rand() <= self.cfg.bg_prob:
            lr01 = self._background_counts_then_scale01(lr_np, rng)
        else:
            # simple per-image min-max fallback
            mn, mx = float(lr_np.min()), float(lr_np.max())
            lr01 = ((lr_np - mn) / (mx - mn + 1e-8)).astype(np.float32)
        return self._post(lr01, rng)



# --- helper to accept (N,H,W), (N,1,H,W) or (N,H,W,1) ---
def _ensure_nhw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return x
    if x.ndim == 4 and x.shape[1] == 1:
        return np.squeeze(x, axis=1)    # (N,1,H,W) -> (N,H,W)
    if x.ndim == 4 and x.shape[-1] == 1:
        return np.squeeze(x, axis=-1)   # (N,H,W,1) -> (N,H,W)
    raise ValueError(f"Expected (N,H,W) or (N,1,H,W)/(N,H,W,1), got {x.shape}")


class PairsDatasetUnified(Dataset):
    """
    Unified HR/LR dataset with shared LR noise (index-deterministic),
    model-specific normalization/resizing/padding, and flexible output range.

    Returns:
      - If out_range == "[-1,1]": tensors in [-1,1]
      - If out_range == "[0,1]":  tensors in [0,1]

    Args:
      norm_preset: "minmax" | "percentile" | "hybrid"  (HR=minmax, LR=percentile when "hybrid")
      hr_norm, lr_norm: optional overrides ("minmax"|"percentile")
      percentile_p: percentile for global LR/HR scaling (default 99.99)
      percentile_from: "lr" or "hr" — which array to compute percentile from
      out_range: "[-1,1]" or "[0,1]"
      resize_to: int or (H,W) or None — if set, interpolate both LR and HR to this size
      resize_mode: "bilinear" | "bicubic" | "nearest" | "area"
      align_corners: bool or None — used for bilinear/bicubic
      pad_to: int side length for square padding (or None for no padding)
      pad_mode: "reflect" | "replicate" | "constant"
      constant_pad_value: used only when pad_mode=="constant"
      channels: 1 or 3 — repeats channels if 3
      noise_aug: LRNoiseAugment or None. If given, used to produce lr01 in [0,1].
      return_order: "HRLR" or "LRHR"
      return_mask: if True and pad_to is not None, also returns a [1, pad_to, pad_to] mask of valid region
    """

    def __init__(self,
                 root_dir: str,
                 hr_name: str = "Mejiro_dataset/hr_all_lsst_20k.npy",
                 lr_name: str = "Mejiro_dataset/lr_all_lsst_20k.npy",
                 mmap: bool = True,

                 # ---- normalization controls ----
                 norm_preset: str = "minmax",
                 hr_norm: Optional[str] = None,
                 lr_norm: Optional[str] = None,
                 percentile_p: float = 99.99,
                 percentile_from: str = "lr",

                 # ---- output / shape controls ----
                 out_range: str = "[-1,1]",                # "[-1,1]" | "[0,1]"
                 resize_to: Optional[Union[int, Tuple[int, int]]] = None,
                 resize_mode: str = "bilinear",
                 align_corners: Optional[bool] = False,
                 pad_to: Optional[int] = 48,
                 pad_mode: str = "reflect",
                 constant_pad_value: float = 0.0,
                 channels: int = 1,

                 # ---- LR noise (index-deterministic) ----
                 noise_aug: Optional[object] = None,

                 # ---- API shape ----
                 return_order: str = "HRLR",
                 return_mask: bool = False):

        # Load arrays
        hr_path = os.path.join(root_dir, hr_name)
        lr_path = os.path.join(root_dir, lr_name)
        if not (os.path.isfile(hr_path) and os.path.isfile(lr_path)):
            raise FileNotFoundError(f"Expected {hr_name} and {lr_name} in {root_dir}")
        self.hr = np.load(hr_path, mmap_mode="r" if mmap else None)
        self.lr = np.load(lr_path, mmap_mode="r" if mmap else None)

        self.hr = _ensure_nhw(self.hr)
        self.lr = _ensure_nhw(self.lr)

        n = min(len(self.hr), len(self.lr))
        self.hr, self.lr = self.hr[:n], self.lr[:n]
        self.N = int(n)
        self.H = int(self.hr.shape[1])
        self.W = int(self.hr.shape[2])

        # Preset → defaults
        preset = norm_preset.lower()
        if preset not in ("minmax", "percentile", "hybrid"):
            raise ValueError("norm_preset must be 'minmax'|'percentile'|'hybrid'")
        preset_map = {
            "minmax":     ("minmax", "minmax"),
            "percentile": ("percentile", "percentile"),
            "hybrid":     ("minmax", "percentile"),  # HR=minmax, LR=percentile
        }
        def_hr, def_lr = preset_map[preset]
        self.hr_norm = (hr_norm or def_hr).lower()
        self.lr_norm = (lr_norm or def_lr).lower()
        if self.hr_norm not in ("minmax", "percentile"): raise ValueError("hr_norm invalid")
        if self.lr_norm not in ("minmax", "percentile"): raise ValueError("lr_norm invalid")

        # Global percentile scale computed once
        self.percentile_p = float(percentile_p)
        src_choice = percentile_from.lower()
        if src_choice not in ("lr", "hr"):
            raise ValueError("percentile_from must be 'lr' or 'hr'")
        src = self.lr if src_choice == "lr" else self.hr
        flat = src.reshape(-1).astype(np.float64)
        vmax = np.percentile(flat, self.percentile_p)
        self.global_scale = float(vmax if np.isfinite(vmax) and vmax > 0 else 1.0)

        # Output / resizing / padding
        self.out_range = out_range
        assert self.out_range in ("[-1,1]", "[0,1]")

        # normalize resize_to + validate mode
        if resize_to is None:
            self.resize_to = None
        elif isinstance(resize_to, int):
            self.resize_to = (int(resize_to), int(resize_to))
        else:
            h, w = resize_to
            self.resize_to = (int(h), int(w))

        self.resize_mode = resize_mode.lower()
        if self.resize_mode not in ("bilinear", "bicubic", "nearest", "area"):
            raise ValueError("resize_mode must be 'bilinear'|'bicubic'|'nearest'|'area'")

        # align_corners only valid for linear/bicubic
        self.align_corners = align_corners if self.resize_mode in ("bilinear", "bicubic") else None

        self.pad_to = pad_to
        self.pad_mode = pad_mode
        self.constant_pad_value = float(constant_pad_value)
        assert channels in (1, 3), "channels must be 1 or 3"
        self.channels = channels

        # Noise & API
        self.noise_aug = noise_aug
        self.return_order = return_order.upper()
        if self.return_order not in ("HRLR", "LRHR"):
            raise ValueError("return_order must be 'HRLR' or 'LRHR'")
        self.return_mask = bool(return_mask)

    def __len__(self) -> int:
        return self.N

    # ---------- helpers ----------
    @staticmethod
    def _minmax01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        mn, mx = float(x.min()), float(x.max())
        return ((x - mn) / (mx - mn + 1e-8)).astype(np.float32) if mx > mn else np.zeros_like(x, np.float32)

    def _percentile01(self, x: np.ndarray) -> np.ndarray:
        s = self.global_scale
        return np.clip(x.astype(np.float32) / (s + 1e-8), 0.0, 1.0)

    @staticmethod
    def _to_tensor(x01: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x01).unsqueeze(0)  # [1,H,W]

    def _pad(self, x: torch.Tensor, side: int) -> torch.Tensor:
        # x: [C,H,W]
        C, H, W = x.shape
        ph, pw = side - H, side - W
        if ph == 0 and pw == 0:
            return x
        l, r = pw // 2, pw - pw // 2
        t, b = ph // 2, ph - ph // 2
        if self.pad_mode == "reflect":
            try:
                return F.pad(x, (l, r, t, b), mode="reflect")
            except RuntimeError:
                return F.pad(x, (l, r, t, b), mode="replicate")
        elif self.pad_mode == "replicate":
            return F.pad(x, (l, r, t, b), mode="replicate")
        elif self.pad_mode == "constant":
            return F.pad(x, (l, r, t, b), mode="constant", value=self.constant_pad_value)
        else:
            raise ValueError("pad_mode must be 'reflect'|'replicate'|'constant'")

    def _map_out_range(self, x01: torch.Tensor) -> torch.Tensor:
        if self.out_range == "[0,1]":
            return x01
        return (x01 - 0.5) * 2.0

    def _repeat_channels(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,H,W] -> [C,H,W]
        return x if self.channels == 1 else x.repeat(self.channels, 1, 1)

    # resizing helper (x is [C,H,W])
    def _resize(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        x4d = x.unsqueeze(0)  # [1,C,H,W]
        kwargs = {"mode": self.resize_mode}
        if self.resize_mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = self.align_corners
        # antialias when available (for downsample in newer PyTorch)
        try:
            x4d = F.interpolate(x4d, size=size_hw, antialias=True, **kwargs)
        except TypeError:
            x4d = F.interpolate(x4d, size=size_hw, **kwargs)
        return x4d.squeeze(0)

    # ---------- main ----------
    def __getitem__(self, idx):
        hr_np = np.array(self.hr[idx], dtype=np.float32)
        lr_np = np.array(self.lr[idx], dtype=np.float32)

        # LR: apply shared noise if provided; else chosen normalization
        if self.noise_aug is not None:
            lr01 = self.noise_aug(lr_np, idx=idx)  # expected to return [0,1]
        else:
            lr01 = self._minmax01(lr_np) if self.lr_norm == "minmax" else self._percentile01(lr_np)

        # HR normalization by selected mode
        hr01 = self._minmax01(hr_np) if self.hr_norm == "minmax" else self._percentile01(hr_np)

        # → tensors [1,H,W]
        HR_t = self._to_tensor(hr01)
        LR_t = self._to_tensor(lr01)

        # Resize both (if requested)
        if self.resize_to is not None:
            HR_t = self._resize(HR_t, self.resize_to)
            LR_t = self._resize(LR_t, self.resize_to)

        # Optional padding (after resize)
        mask = None
        if self.pad_to is not None:
            if self.return_mask:
                _, H, W = HR_t.shape
                mask = torch.zeros((1, self.pad_to, self.pad_to), dtype=torch.float32)
                ph, pw = self.pad_to - H, self.pad_to - W
                l, r = pw // 2, pw - pw // 2
                t, b = ph // 2, ph - ph // 2
                mask[..., t:t+H, l:l+W] = 1.0
            HR_t = self._pad(HR_t, self.pad_to)
            LR_t = self._pad(LR_t, self.pad_to)

        # Map to output range and repeat channels if needed
        HR_t = self._map_out_range(HR_t)
        LR_t = self._map_out_range(LR_t)
        HR_t = self._repeat_channels(HR_t)
        LR_t = self._repeat_channels(LR_t)

        if self.return_order == "HRLR":
            return (HR_t, LR_t, mask) if self.return_mask else (HR_t, LR_t)
        else:
            return (LR_t, HR_t, mask) if self.return_mask else (LR_t, HR_t)
        
        
#---------------------------- Model I --------------------------------------------------------

class PTImageDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path, downsample_size=None):
        self.data = torch.load(pt_path)
        self.data = (self.data - 0.5) * 2  # normalize to [-1, 1]
        self.downsample_size = downsample_size  # e.g., (75, 75)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        img = self.data[idx]  # shape: [1, H, W]
        if self.downsample_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.downsample_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        return img, 0  # dummy label for uncond model



#---------------------------- Model II --------------------------------------------------------

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


class NpyDiffusionDatasetHighLowRes(torch.utils.data.Dataset):
    def __init__(self, root_folder, downsample_size=None, blur_sigma=3):
        self.image_paths = []
        self.class_to_label = {'axion': 0, 'cdm': 1, 'no_sub': 2}
        self.downsample_size = downsample_size
        self.blur_sigma = blur_sigma  # standard deviation for Gaussian blur

        for class_name in self.class_to_label:
            class_folder = os.path.join(root_folder, class_name)
            if not os.path.isdir(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if fname.endswith('.npy'):
                    self.image_paths.append(os.path.join(class_folder, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        data = np.load(path, allow_pickle=True)

        # Handle axion case differently
        if isinstance(data, np.ndarray) and data.shape[0] == 2 and data[0].ndim >= 2:
            img = data[0]
        else:
            img = data

        if not isinstance(img, np.ndarray) or img.ndim < 2:
            raise ValueError(f"Invalid image at {path}, got shape {getattr(img, 'shape', None)}")

        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 1e-8:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # Original (high-res) target
        x0 = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Resize if needed
        if self.downsample_size:
            x0 = F.interpolate(x0.unsqueeze(0), size=self.downsample_size, mode='bilinear', align_corners=False).squeeze(0)

        # Create low-res conditional image by blurring
        img_blur_np = gaussian_filter(img, sigma=self.blur_sigma)
        x_cond = torch.tensor(img_blur_np, dtype=torch.float32).unsqueeze(0)

        # Resize low-res image as well if needed
        if self.downsample_size:
            x_cond = F.interpolate(x_cond.unsqueeze(0), size=self.downsample_size, mode='bilinear', align_corners=False).squeeze(0)

        # Normalize both to [-1, 1]
        x0 = (x0 - 0.5) * 2
        x_cond = (x_cond - 0.5) * 2

        return x0, x_cond  # taret, input


cfg = NoiseConfig(
    bg_enable=True,
    bg_peak_counts=(800.0, 1500.0),
    bg_sky_counts=(80.0*10, 300.0*10),
    bg_read_std_counts=3.0,
    bg_prob=1.0,
    post_mode="none"
)
noise_aug = LRNoiseAugment(cfg, base_seed=42)


