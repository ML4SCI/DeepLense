# Configuration and Dataset Flow in DeepLense SSL

This document explains how configuration files, dataset paths, and indices are used in the DeepLense self-supervised learning (SSL) pipeline.  
It is intended for new contributors running the code locally for the first time.

---

## 1. Expected Dataset Structure

DeepLense SSL assumes the dataset directory has the following structure:

```text
data_root/
├── lenses/
│   ├── image_001.npy
│   ├── image_002.npy
│   └── ...
└── nonlenses/
    ├── image_101.npy
    ├── image_102.npy
    └── ...
```

Each image is stored as a NumPy (`.npy`) file.
- Images typically contain multiple channels (e.g. g, r, i bands).
- Images are center-cropped to `32 × 32` pixels during training.

The path to `data_root` must be provided in the configuration file under:

```yaml
input:
  data path: /path/to/data_root
```
---
## 2. What is `indices.pkl`?

The file `indices.pkl` defines how the dataset is split into training, validation, and test sets.

It contains:
- Separate indices for **lenses** and **nonlenses**
- Explicit splits for:
  - training
  - validation
  - testing

This file is required so that:
- experiments are reproducible
- all models use the same data splits
- comparisons between methods remain consistent

The path to this file must be provided in the configuration file under:
```yaml
input:
  indices: /path/to/indices.pkl
```
---

## 3. How Configuration Files Are Used

All training behavior in DeepLense SSL is controlled through YAML configuration files, without requiring code changes.


At a high level, the execution flow is:

- config.yaml is parsed into an `args` dictionary
- dataset paths and indices are loaded
- data augmentations are initialized
- SSL model (DINO / SimSiam / iBOT) is constructed
- training loop is executed


Key sections in the config include:
- `experiment`: device, output directory, logging
- `input`: dataset paths and image parameters
- `network`: Vision Transformer architecture details
- `train args`: batch size, epochs, train/validation split
- `ssl augmentation kwargs`: SSL-specific augmentations

---

## 4. Common Setup Issues

New contributors often encounter the following issues:
- Incorrect dataset folder structure
- Missing or mislocated `indices.pkl`
- GPU-only configs being run on CPU machines
- Long training runs failing early due to setup errors

Ensuring paths are correct before running long experiments can save significant time.

---

## 5. Quick Sanity Check Configuration

To help verify that the pipeline is set up correctly, a minimal configuration file is provided:

configs/quick-test-simsiam.yaml



This configuration:
- Runs on **CPU**
- Uses **one training epoch**
- Uses a **small batch size**
- Is intended only for **sanity checks**, not scientific results

New contributors are encouraged to run this configuration first to confirm that:
- dataset paths are correct
- indices load properly
- the training loop executes end-to-end

---

## 6. When to Use Full Training Configs

Once the quick test configuration runs successfully, users can switch to full training configurations (e.g. DINO or iBOT) for actual experiments and result generation.

