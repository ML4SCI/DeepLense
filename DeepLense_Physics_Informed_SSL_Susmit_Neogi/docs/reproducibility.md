# Reproducibility and troubleshooting

## Current script defaults

| Entrypoint | Epochs | Batch size | Optimizer defaults | Seed |
|---|---:|---:|---|---|
| Pretraining | 10 | 32 | SGD, learning rate `1e-3`, weight decay `1e-4` | Not set |
| ResNet classification | 30 | 64 | AdamW, learning rate `1e-3`, weight decay `1e-4` | Not set |
| Frozen-encoder classification | 30 | 64 | AdamW, learning rate `1e-3`, weight decay `1e-4` | Not set |
| Random-feature baseline | 30 | 64 | AdamW, learning rate `1e-3`, weight decay `1e-4` | Not set |
| Einstein-radius regression | 300 | 64 | AdamW, learning rate `1e-4`, weight decay `1e-4` | `42` |
| Arc-feature classification | 30 | 128 | AdamW, learning rate `1e-3`, weight decay `1e-4` | Not set |

These are defaults in the current scripts; command-line options can override many of them. The regression script seeds its PyTorch split and training, but the other entrypoints do not establish a complete deterministic setup. The simulator's seed also does not seed per-image noise; see [Data preparation](data-preparation.md).

For repeatable evaluations, record the exact commands, dependency versions, data split, checkpoint path, hardware, and random seeds. Run multiple seeds and report variation rather than relying on one run. The metrics in the repository's main README are reported values; running these commands does not automatically reproduce or update those tables.

## Troubleshooting

### A dataset directory is reported missing

Run from the repository root and verify the directory structure in the [experiment reference](experiments.md). The default `--root` values are relative to the current working directory.

### No `.npy` files or labeled samples are found

Check that files use the `.npy` suffix and that class-specific datasets place them under `<split>/<class-name>/`. For regression, verify that `labels.csv` contains `filename` and `einstein_radius` columns and that each referenced image exists under `images/`.

### An encoder checkpoint is missing

Run pretraining first, or pass the intended path with `--encoder-checkpoint`. Note that regression and the random-feature baseline have default encoder filenames different from the default pretraining output.

### The pretraining train and validation losses use the same data

The current pretraining script points both dataset objects at `Model_I/val`. Change the training dataset to `split='train'` if a separate training split is intended.

### The simulator cannot import Lenstronomy

Install `lenstronomy` into the active Python environment. The data generator imports it directly, but it is not currently listed in the project's dependency manifests.

### CUDA is unavailable or PyTorch reports a device error

The scripts select CUDA when PyTorch reports it available, otherwise CPU, and most support `--device`. Install a PyTorch build suitable for the local hardware using the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
