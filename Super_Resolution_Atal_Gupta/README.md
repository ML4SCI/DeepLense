# Super Resolution — Atal Gupta (DeepLense)

Super-resolution of gravitational lensing images using a diffusion-based model.  
This module upscales low-resolution (LR) lensing images to high-resolution (HR) equivalents, improving downstream classification and analysis accuracy.

---

## 📁 Directory Structure

```
Super_Resolution_Atal_Gupta/
├── generate_dataset.py   ← NEW: generates HR/LR pairs from raw data
├── get_data_diff.py      ← data loader (patched — see Issue #135)
├── model.py
├── train.py
├── inference.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision numpy
```

### 2. Download the raw dataset

Download the DeepLense lensing `.npy` files and place them in a folder, e.g. `raw_data/`:

```
raw_data/
    image_00001.npy
    image_00002.npy
    ...
```

> **Dataset source:** The lensing images come from the DeepLense simulation suite.  
> Contact the ML4SCI team or refer to the main repo README for the download link.

### 3. Generate HR / LR pairs ← **Start here**

```bash
python generate_dataset.py \
    --input_dir  raw_data/ \
    --output_dir dataset/ \
    --scale_factor 4 \
    --split 0.8
```

This produces:

```
dataset/
    train/
        HR/   ← 80 % of images (original resolution)
        LR/   ← same images downscaled 4×
    val/
        HR/
        LR/
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | — | Folder with raw `.npy` images |
| `--output_dir` | — | Where to save the dataset |
| `--scale_factor` | `4` | LR downscaling factor |
| `--split` | `0.8` | Fraction of data used for training |

### 4. Train the model

```bash
python train.py --data_dir dataset/
```

### 5. Run inference

```bash
python inference.py --checkpoint checkpoints/best.pth \
                    --lr_image   dataset/val/LR/image_00001.npy
```

---

## 🐛 Bugs Fixed (Issue #135)

| Problem | Fix |
|---------|-----|
| `FileNotFoundError` — no HR/LR directories | Added `generate_dataset.py`; loader now raises a clear error with instructions |
| `IndexError` in `get_data_diff.py` | Added bounds check + clamp; mismatched HR/LR counts handled gracefully |
| `batch_size > dataset size` crash | DataLoader factory clamps batch_size and warns the user |
| Unclear directory structure | This README + script enforce and document the expected layout |

---

## 📖 Citation

Atal Gupta, *Super-Resolution of Gravitational Lensing Images*, GSoC 2022 — ML4SCI / DeepLense.

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b fix/your-fix`)  
3. Commit and push  
4. Open a Pull Request referencing Issue #135