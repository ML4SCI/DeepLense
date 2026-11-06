# Data preparation

The simulator in [`data_preparation/data_construction.py`](../data_preparation/data_construction.py) generates 64×64 single-channel gravitational-lensing images and a CSV of simulation parameters. It uses Lenstronomy to render an SIE lens with a Sersic source.

## Install and run

Install the project's listed dependencies first. The simulator also imports `lenstronomy`, which is not currently listed in `requirements.txt` or `pyproject.toml`; install it separately in the active environment:

```bash
python -m pip install lenstronomy
```

From the repository root, generate a sample dataset:

```bash
python -m data_preparation.data_construction \
  --n 1000 \
  --out data/regression \
  --seed 42
```

Available options:

| Option | Default | Description |
|---|---:|---|
| `--n` | `1000` | Number of simulation attempts. Failed simulations are skipped, so fewer files may be written. |
| `--out` | `data` | Output directory. |
| `--seed` | unset | Seed used to sample lens/source parameters. |
| `--no-progress` | off | Disable progress display. |

The generator creates this structure:

```text
data/regression/
├── images/
│   ├── img_000000.npy
│   └── ...
└── labels.csv
```

Images are saved as `float32` NumPy arrays. The CSV includes `filename`, `einstein_radius`, lens ellipticity and center, and source amplitude, size, index, ellipticity, and center. `einstein_radius` is sampled uniformly from 0.5 to 2.5 arcseconds and is the regression target.

## Using the generated data

The regression experiment accepts this output directly when `--out data/regression` is used:

```bash
python -m experiments.regression.einstein_radius --root data/regression
```

The regression script uses a seeded random 80/20 train/validation split. The simulator does **not** create separate train and validation directories or class labels. To use its images for classification or pretraining, organize the images into the split and class directories described in the [experiment reference](experiments.md).

The `--seed` option controls the NumPy generator used for parameter sampling, but each image's noise is created with a separate unseeded generator in `simulate_one`. A fixed seed therefore does not guarantee bit-for-bit identical output.
