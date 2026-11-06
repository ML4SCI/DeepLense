# Experiment reference

Run these commands from the repository root. Each entrypoint supports `--help` to display its arguments.

## Data layouts

The scripts expect `.npy` images organized as follows:

```text
data/
├── Model_I/
│   ├── train/
│   │   └── <class-name>/*.npy      # ResNet classification
│   └── val/
│       └── <class-name>/*.npy
├── Model_III/
│   ├── train/
│   │   └── <class-name>/*.npy      # Frozen encoder and random-feature classifiers
│   └── val/
│       └── <class-name>/*.npy
├── Model_IV/
│   ├── train/
│   │   └── <class-name>/*.npy      # Arc-feature classifier
│   └── val/
│       └── <class-name>/*.npy
└── regression/
    ├── images/*.npy
    └── labels.csv                  # filename and einstein_radius columns
```

Classification class indices are assigned by sorting the training split's immediate class-directory names; validation reuses that mapping. Keep class directory names consistent between splits.

## Entrypoints

| Task | Command | Default input | Default checkpoint/output |
|---|---|---|---|
| Self-supervised pretraining | `python -m experiments.pretraining.train` | `data/Model_I/{train,val}` | `checkpoints/lens_model_i_ssl.pt`; loss plot at `plots/val_loss_curve.png` |
| ResNet classification | `python -m experiments.classification.resnet_baseline` | `data/Model_I/{train,val}/<class>` | `checkpoints/resnet_model_i.pt` |
| Frozen-encoder classification | `python -m experiments.classification.frozen_encoder` | `data/Model_III/{train,val}/<class>` | Encoder: `checkpoints/lens_model_i_ssl.pt`; classifier: `checkpoints/lens_classifier.pt` |
| Random-feature baseline | `python -m experiments.classification.random_features` | `data/Model_III/{train,val}/<class>` | Encoder: `checkpoints/lens_model_iv.pt`; classifier: `checkpoints/random_lens_classifier.pt` |
| Einstein-radius regression | `python -m experiments.regression.einstein_radius` | `data/regression` | Encoder: `checkpoints/lens_model_i_physics.pt`; regressor: `checkpoints/einstein_radius_regressor.pt` |
| Arc-feature classification | `python -m experiments.arc_features.classifier` | `data/Model_IV/{train,val}/<class>` | `checkpoints/arc_classifier_model_iii.pt` |

Use `--root` to change the input directory and the relevant `--checkpoint`, `--encoder-checkpoint`, or `--classifier-checkpoint` option to change checkpoint paths. Most training entrypoints also expose epoch, batch-size, worker, learning-rate, weight-decay, and device options. The arc-feature entrypoint has an optional `--max-samples` argument for quick checks.

## Task notes

- **Pretraining** consumes unlabeled images recursively from `Model_I` split folders. Images are converted to one channel, resized to 64×64, and preprocessed on load.
- **ResNet classification** trains a supervised ResNet-152 model from scratch on `Model_I`.
- **Frozen-encoder classification** loads a LensModel checkpoint and trains a classifier on `Model_III`. Its default encoder checkpoint is the pretraining output path.
- **Random-feature baseline** trains a classifier on randomly sampled vectors shaped like encoder features. It still loads the configured encoder checkpoint to determine the feature shape; the default checkpoint is `lens_model_iv.pt`, which is not produced by the pretraining command above.
- **Regression** reads `images/` and `labels.csv`, then creates an 80/20 random split. Its default encoder checkpoint is `lens_model_i_physics.pt`, not the checkpoint produced by the default pretraining command; pass `--encoder-checkpoint checkpoints/lens_model_i_ssl.pt` when that is the intended encoder.
- **Arc-feature classification** computes handcrafted arc-brightness descriptors directly from images under `Model_IV`; it does not load the ViT encoder.

The default pretraining script currently constructs both its training and validation datasets with `split='val'`. If separate splits are intended, change the training dataset construction in `experiments/pretraining/train.py` to `split='train'` before running it.

Checkpoints are saved when validation performance improves (or, for pretraining, validation loss decreases). Create or verify the required input directories and encoder checkpoints before launching a run.
