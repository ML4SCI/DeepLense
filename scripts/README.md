# DeepLense Dataset Downloader

This directory contains scripts for automating common tasks in the DeepLense project.

## download_datasets.py

A unified command-line tool to download all DeepLense datasets from Google Drive.

### Installation

Install the required dependencies:

```bash
pip install gdown tqdm
```

### Usage

#### Download a specific model and split

```bash
# Download Model I training data
python scripts/download_datasets.py --model 1 --split train

# Download Model II test data
python scripts/download_datasets.py --model 2 --split test
```

#### Download multiple datasets

```bash
# Download both train and test for Model III
python scripts/download_datasets.py --model 3 --split both

# Download all datasets (Models I, II, III - train and test)
python scripts/download_datasets.py --model all --split both
```

#### Custom output directory

```bash
# Download to a custom location
python scripts/download_datasets.py --model 1 --split train --output-dir ./my_datasets/
```

#### Force re-download

```bash
# Re-download even if file exists
python scripts/download_datasets.py --model 1 --split train --force
```

### Available Datasets

| Model | Split | Size (approx) | Description |
|-------|-------|---------------|-------------|
| Model I | train | ~1.5 GB | 150×150 px, Gaussian PSF, SNR~25, ~30k images |
| Model I | test | ~250 MB | 150×150 px, ~5k images |
| Model II | train | ~300 MB | 64×64 px, Euclid observation, ~30k images |
| Model II | test | ~50 MB | 64×64 px, ~5k images |
| Model III | train | ~300 MB | 64×64 px, HST observation, ~30k images |
| Model III | test | ~50 MB | 64×64 px, ~5k images |

**Total size (all datasets)**: ~2.5 GB

### Dataset Structure

All datasets contain 3 classes of dark matter substructure:
- **Axion** (vortex substructure)
- **CDM** (Cold Dark Matter, point-mass subhalos)  
- **No substructure** (baseline)

### Output Directory Structure

```
data/
├── Model_I/
│   ├── train/
│   │   └── Model_I_train.h5
│   └── test/
│       └── Model_I_test.h5
├── Model_II/
│   ├── train/
│   │   └── Model_II_train.h5
│   └── test/
│       └── Model_II_test.h5
└── Model_III/
    ├── train/
    │   └── Model_III_train.h5
    └── test/
        └── Model_III_test.h5
```

### Command-Line Options

```
usage: download_datasets.py [-h] --model {1,2,3,all} --split {train,test,both}
                            [--output-dir OUTPUT_DIR] [--force]

optional arguments:
  -h, --help            Show this help message and exit
  --model {1,2,3,all}   Model dataset to download (1, 2, 3, or all)
  --split {train,test,both}
                        Dataset split to download (train, test, or both)
  --output-dir OUTPUT_DIR
                        Output directory for datasets (default: ./data)
  --force               Force re-download even if file already exists
```

### Troubleshooting

#### Download interrupted or failed

Try running the same command again. The script will skip already downloaded files and continue from where it left off.

#### "gdown library not found" error

Install gdown:
```bash
pip install gdown
```

#### Permission denied / Access denied

Make sure you have write permissions to the output directory. Try using a different output directory with `--output-dir`.

#### Slow download speeds

This depends on your internet connection and Google Drive's bandwidth limits. Large files may take several minutes to download.

#### File verification

After downloading, you can verify the files:
```python
import h5py

# Load dataset
with h5py.File('data/Model_I/train/Model_I_train.h5', 'r') as f:
    print(f"Keys: {list(f.keys())}")
    print(f"Shape: {f['X'].shape}")
```

### Dataset Sources

All datasets were generated using [Lenstronomy](https://github.com/lenstronomy/lenstronomy) and are hosted on Google Drive. For more information about the datasets, see:

- [Saranga K Mahanta's README](../Updating_the_DeepLense_Pipeline__Saranga_K_Mahanta/README.md)
- [DeepLenseSim Repository](https://github.com/mwt5345/DeepLenseSim)

### Contributing

If you find issues with the downloader or want to add support for more datasets, please submit a pull request or open an issue on GitHub.

---

**Note**: The first time you download a dataset, it may take several minutes depending on your internet connection. Subsequent downloads of the same file will be skipped unless you use the `--force` flag.
