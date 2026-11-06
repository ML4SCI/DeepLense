# Physics-Constrained Self-Supervised Representations for Gravitational-Lensing Images

This directory contains shared model and physics utilities, along with experiment entrypoints for self-supervised pretraining, classification, regression, and arc-feature classification.

# Blog

[GSoC Final Submission Blog](https://susmit108.github.io/gsoc/ml4sci/deeplense/2026/10/29/physics-constrained-lensing.html)

## Project structure

```text
.
├── ssl_components/              # Shared encoder, model, and physics utilities
├── experiments/
│   ├── pretraining/              # Self-supervised encoder training
│   ├── classification/           # Frozen encoder, random-feature, and ResNet runs
│   ├── regression/               # Einstein-radius regression
│   └── arc_features/             # Arc-descriptor classification
├── data/                         # Local datasets
├── checkpoints/                  # Model checkpoints
└── results/                      # Experiment outputs
```

The shared architecture and physics helpers are in `ssl_components/`. Experiment-specific loading, training, evaluation, and command-line options are defined in the corresponding files under `experiments/`.

## Architecture

<img src="assets/model_final_preview.png" alt="Model architecture" width="800">

## Environment and installation

The experiments use Python with PyTorch, NumPy, scikit-learn, torchvision, tqdm, and Matplotlib. Install the listed dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

The same unpinned runtime dependencies are declared in `pyproject.toml`. For CUDA-enabled PyTorch, use the installation command recommended by the [PyTorch selector](https://pytorch.org/get-started/locally/) for your platform and hardware.

## Running experiments

Run commands from this directory so relative dataset and checkpoint paths resolve correctly:

```bash
python -m experiments.pretraining.train
python -m experiments.classification.frozen_encoder --help
python -m experiments.classification.resnet_baseline --help
python -m experiments.regression.einstein_radius --help
python -m experiments.arc_features.classifier --help
```

The frozen-encoder classifier and regression experiment require an encoder checkpoint. The random-feature baseline also uses its configured checkpoint path. Use each entrypoint's `--help` output to review available options. By default, scripts write checkpoints under `checkpoints/`.

## Dataset organization

The scripts expect local `.npy` arrays arranged by task. A representative directory layout is:

```text
data/
├── Model_I/
│   ├── train/
│   └── val/
├── Model_III/
│   ├── train/<class-name>/
│   └── val/<class-name>/
├── Model_IV/
│   ├── train/<class-name>/
│   └── val/<class-name>/
└── regression/
    ├── images/
    └── labels.csv
```

The regression CSV should contain `filename` and `einstein_radius` columns. Classification and arc-feature experiments infer class indices from the training directory structure and reuse them for validation. Dataset and checkpoint paths can be adjusted through each script's command-line options where available.

## Reported results

### Lens-class classification

| Model | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| ResNet-152 (supervised) | 70.90% | 0.70 | 0.70 | 0.71 |
| Physics Only | 97.85% | 0.98 | 0.98 | 0.98 |
| SSL + Physics | 97.30% | 0.97 | 0.97 | 0.97 |

### Einstein-radius regression

| Representation | MAE | RMSE | R² |
|---|---:|---:|---:|
| SSL + Physics | 0.37 | 0.47 | 0.63 |
| Physics Only | 0.28 | 0.37 | 0.67 |

Regression metrics are in standardized units. Classification uses three near-balanced classes—axion, cdm, and no_sub—so chance accuracy is 33.3%. Physics Only is marginally stronger than SSL + Physics for both tasks. These results suggest that physical consistency alone is a strong learning signal, while reconstruction provides additional but nonessential structure.

The reported encoder-pretraining setup uses SGD for 10 epochs with learning rate 1e-3, weight decay 1e-4, and batch size 32. Downstream adapters use a frozen encoder and are trained with AdamW for up to 30 epochs. The reported limitations include fixed 64×64 inputs, the need for multi-seed evaluation, and the lack of testing on observed lensing systems. The study also notes a domain shift between simulated Model-I pretraining and simulated Model-III testing; this does not establish transfer to real observations.
