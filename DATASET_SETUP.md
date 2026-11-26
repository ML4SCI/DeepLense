# DeepLense Dataset Setup Guide

> Addresses Issue #180: README lacks clear dataset setup instructions for new users

This guide provides step-by-step instructions for downloading, organizing, and validating the datasets used by DeepLense for dark matter substructure classification.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Descriptions](#dataset-descriptions)
3. [Download Instructions](#download-instructions)
4. [Directory Structure](#directory-structure)
5. [Data Format](#data-format)
6. [Validation](#validation)
7. [Quick Start](#quick-start)
8. [Troubleshooting](#troubleshooting)

---

## Overview

DeepLense uses simulated strong gravitational lensing images to classify dark matter substructure into three categories:

| Class | Label | Description |
|-------|-------|-------------|
| **No Substructure** | 0 | Smooth mass distribution (no dark matter clumps) |
| **Subhalo (CDM)** | 1 | Cold Dark Matter subhalos present |
| **Vortex (Axion)** | 2 | Axion-like vortex substructure |

There are **four Model datasets**, each generated with different simulation parameters:

| Dataset | Lens Model | Image Size | Characteristics |
|---------|-----------|------------|-----------------|
| Model I | SIE | 64×64 | Basic simulations |
| Model II | SIE + Shear | 64×64 | Added external shear |
| Model III | SIE + Shear | 150×150 | Higher resolution |
| Model IV | EPL | 64×64 | Elliptical Power Law lens |

---

## Download Instructions

### Option 1: Direct Download (Recommended)

The datasets are available on Zenodo:

```bash
# Model I
wget https://zenodo.org/record/XXXXX/files/Model_I.zip

# Model II  
wget https://zenodo.org/record/XXXXX/files/Model_II.zip

# Model III
wget https://zenodo.org/record/XXXXX/files/Model_III.zip

# Model IV
wget https://zenodo.org/record/XXXXX/files/Model_IV.zip
```

> **Note:** Replace `XXXXX` with the actual Zenodo record ID. Check the DeepLense README or ask the mentors for the current download links.

### Option 2: Using the Dataset Loader Utility

If you have the `dataset_loader.py` utility installed (see PR #213):

```python
from dataset_loader import DatasetLoader

loader = DatasetLoader(data_dir="./data/Model_I")
images, labels = loader.load()
print(f"Loaded {len(images)} images with shape {images[0].shape}")
```

---

## Directory Structure

After downloading, organize your data as follows:

```
DeepLense/
├── data/
│   ├── Model_I/
│   │   ├── no_substructure/
│   │   │   ├── image_0001.npy
│   │   │   ├── image_0002.npy
│   │   │   └── ...
│   │   ├── subhalo/
│   │   │   ├── image_0001.npy
│   │   │   └── ...
│   │   └── vortex/
│   │       ├── image_0001.npy
│   │       └── ...
│   ├── Model_II/
│   │   └── (same structure)
│   ├── Model_III/
│   │   └── (same structure)
│   └── Model_IV/
│       └── (same structure)
├── DeepLense_Physics_Informed_Super_Resolution_Lensing_Images/
├── DeepLense_Diffusion_Reparameterized/
└── ...
```

### Important Notes

- Each class folder should contain `.npy` (NumPy) files
- Some sub-projects may use `.npy` files with different shapes — the `DatasetLoader` utility handles this automatically
- Keep the folder names consistent: `no_substructure`, `subhalo`, `vortex`

---

## Data Format

### NPY File Format

Each `.npy` file contains a single lensing image as a NumPy array:

```python
import numpy as np

image = np.load("data/Model_I/subhalo/image_0001.npy")
print(f"Shape: {image.shape}")   # e.g., (64, 64) or (1, 64, 64)
print(f"Dtype: {image.dtype}")   # e.g., float32 or float64
print(f"Range: [{image.min():.3f}, {image.max():.3f}]")
```

### Known Format Variations (Issue #178)

Different sub-projects may store images in slightly different formats:

| Variation | Shape | Handling |
|-----------|-------|----------|
| 2D image | `(H, W)` | Add channel dim → `(1, H, W)` |
| 3D with channel | `(1, H, W)` | Use directly |
| 3D channel-last | `(H, W, 1)` | Transpose → `(1, H, W)` |
| Unnormalized | Values > 1.0 | Apply min-max normalization |

The `DatasetLoader` (PR #213) and `dataloader_utils` (PR #214) handle all these variations automatically.

---

## Validation

### Quick Validation Script

Run this to verify your dataset is set up correctly:

```python
import os
import numpy as np

DATA_DIR = "./data/Model_I"
CLASSES = ["no_substructure", "subhalo", "vortex"]

print("Dataset Validation Report")
print("=" * 50)

for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.exists(cls_dir):
        print(f"  MISSING: {cls_dir}")
        continue
    
    files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
    print(f"\n  Class: {cls}")
    print(f"  Files: {len(files)}")
    
    if files:
        sample = np.load(os.path.join(cls_dir, files[0]))
        print(f"  Shape: {sample.shape}")
        print(f"  Dtype: {sample.dtype}")
        print(f"  Range: [{sample.min():.4f}, {sample.max():.4f}]")

print("\n" + "=" * 50)
print("Validation complete!")
```

### Expected Output

```
Dataset Validation Report
==================================================

  Class: no_substructure
  Files: 5000
  Shape: (64, 64)
  Dtype: float32
  Range: [0.0000, 1.0000]

  Class: subhalo
  Files: 5000
  Shape: (64, 64)
  Dtype: float32
  Range: [0.0000, 1.0000]

  Class: vortex
  Files: 5000
  Shape: (64, 64)
  Dtype: float32
  Range: [0.0000, 1.0000]

==================================================
Validation complete!
```

---

## Quick Start

After setting up the data, run a quick test:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load one sample from each class
classes = ["no_substructure", "subhalo", "vortex"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, cls in enumerate(classes):
    files = os.listdir(f"data/Model_I/{cls}")
    img = np.load(f"data/Model_I/{cls}/{files[0]}")
    if img.ndim == 3:
        img = img.squeeze()
    axes[i].imshow(img, cmap="inferno")
    axes[i].set_title(cls.replace("_", " ").title())
    axes[i].axis("off")

plt.suptitle("DeepLense - Model I Samples", fontsize=14)
plt.tight_layout()
plt.savefig("dataset_samples.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` | Check the data directory path and folder names |
| Different `.npy` shapes | Use `DatasetLoader` from PR #213 |
| NaN values in data | Use `safe_normalize` from PR #214 |
| Out of memory | Reduce batch size or use `num_workers=0` |
| Very slow loading | Enable `pin_memory=True` and increase `num_workers` |

### Getting Help

- Open an issue on [GitHub](https://github.com/ML4SCI/DeepLense/issues)
- Contact mentors at: ml4-sci@cern.ch
