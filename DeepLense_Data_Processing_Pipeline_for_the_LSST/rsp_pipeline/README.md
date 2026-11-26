# LSST ↔ DeepLense Data Processing Pipeline

A modular Python pipeline that bridges the **Vera C. Rubin Observatory
Legacy Survey of Space and Time (LSST)** data access tools with the
**DeepLense** machine-learning framework for strong gravitational lens
detection and dark-matter substructure classification.

---

## Motivation

The Rubin Observatory will image ~18,000 deg² of sky to _r_ ≈ 27.5 over
10 years, producing an estimated **~10 million** strong lensing candidates.
DeepLense has demonstrated state-of-the-art classification of simulated
lensing images (Model I/II/III), but no direct interface to Rubin's data
access stack existed.  This pipeline fills that gap.

## Architecture

```
Rubin RSP Catalog (TAP)         Rubin RSP Images (SIA v2)
        │                                  │
        ▼                                  ▼
 RubinTAPClient                   RubinSIAClient
 (candidate selection)            (cutout retrieval)
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              Normaliser  (asinh / minmax / z-score)
              CutoutExtractor  (resize → 64×64)
                       │
                       ▼
              RubinLensDataset  (PyTorch Dataset)
                       │
                       ▼
           DeepLense training loop
       (classification / SR / lens finding)
```

## Key components

| Module | Description |
|--------|-------------|
| `data_access/tap_client.py` | ADQL cone-search against the RSP object catalog |
| `data_access/sia_client.py` | SIA v2 cutout retrieval per band |
| `preprocessing/normalise.py` | Asinh / minmax / z-score pixel normalisation |
| `preprocessing/cutout.py` | Fixed-size extraction and multi-band stacking |
| `preprocessing/transforms.py` | torchvision-compatible augmentation pipeline |
| `dataset/rubin_dataset.py` | PyTorch `Dataset` — drop-in for `LensDataset` |
| `utils/config.py` | YAML config loader |
| `utils/io.py` | Candidate CSV save/load helpers |

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your RSP token

```bash
export RSP_TOKEN=<your-token>   # from https://data.lsst.cloud/
```

### 3. Run the pipeline (online mode)

```python
from lsst_deeplense import RubinTAPClient, RubinSIAClient, RubinLensDataset
from lsst_deeplense.preprocessing import build_deeplense_transforms

# 1. Query candidates
tap = RubinTAPClient()
candidates = tap.query_lens_candidates(ra=150.1, dec=2.2, radius_deg=1.0)

# 2. Build dataset with lazy SIA retrieval + on-disk caching
sia = RubinSIAClient()
transform = build_deeplense_transforms(image_size=64, is_train=True, num_channels=3)

dataset = RubinLensDataset(
    candidates=candidates,
    sia_client=sia,
    transform=transform,
    cache_dir="./cache/rubin_cutouts",
    bands=["g", "r", "i"],
)

# 3. Standard PyTorch DataLoader
import torch
loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
```

### 4. Offline mode (pre-fetched `.npy` files)

```python
dataset = RubinLensDataset.from_npy_dir(
    npy_dir="./cache/rubin_cutouts",
    transform=build_deeplense_transforms(is_train=False, num_channels=3),
    label_csv="./candidates_labelled.csv",
)
```

### 5. Notebook walkthrough

Open `notebooks/01_end_to_end_demo.ipynb` for an interactive walkthrough
covering all pipeline stages, visualisations, and a sanity-check
forward pass through a ResNet-18.

## Normalisation strategy

We default to **asinh stretching** rather than the linear normalisation
used in the existing DeepLense loaders for two reasons:

1. LSST calibrated images span a large dynamic range: the lens galaxy
   nucleus can be 100–1000× brighter than the lensing arcs.  Linear
   normalisation compresses arc structure into the noise floor.
2. The asinh function is the standard in astronomical image display
   (DS9, Lupton et al. 2004) and has been adopted in HSC, DES, and
   KiDS lensing pipelines.

## Compatibility with existing DeepLense code

`RubinLensDataset` implements the same interface as the existing
`LensDataset` (from the refactored `DeepLense_Transformers_*` modules):
- `__len__` and `__getitem__` returning `(tensor, label)` or `tensor`
- Compatible with `build_deeplense_transforms` augmentation pipeline
- Accepts any torchvision-compatible `transform` argument

This means existing classification, super-resolution, and lens-finding
training scripts work with **zero modification** — simply swap the
Dataset constructor.

## Running tests

```bash
cd DeepLense_Data_Processing_Pipeline_for_the_LSST
python -m pytest tests/ -v
```

All tests run offline — no RSP token required.

## Related work

- GSoC 2025 RIPPLe pipeline by @kartikmandar — Butler-based local data
  access for LSST Science Pipelines installations
- DeepLense Model I/II/III classification (Toomey et al. 2022)
- LSST Science Pipelines documentation: https://pipelines.lsst.io

This module is complementary to RIPPLe: RIPPLe targets researchers with
a local LSST stack installation (e.g. on USDF), while this pipeline
targets researchers using the **public Rubin Science Platform**
via standard web APIs (TAP + SIA v2) — no local stack required.
