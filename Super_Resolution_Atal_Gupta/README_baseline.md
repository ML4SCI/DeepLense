# Baseline SRCNN for DeepLense Super-Resolution

## Overview

This script provides a minimal, reproducible implementation of SRCNN (Super-Resolution Convolutional Neural Network) for gravitational lensing images. It serves as a baseline for the **DEEPLENSE2** proposal.

**Purpose:** Establish infrastructure for unsupervised super-resolution experiments before integrating real datasets.

## Quick Start

### Basic Usage
```bash
python train_srcnn_minimal.py
```

### With Custom Parameters
```bash
python train_srcnn_minimal.py \
    --epochs 20 \
    --batch-size 32 \
    --img-size 128 \
    --save-model
```

### CPU-Only Mode
```bash
python train_srcnn_minimal.py --cpu
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-samples` | int | 100 | Number of training samples |
| `--img-size` | int | 64 | Image size (square images) |
| `--scale-factor` | int | 2 | Super-resolution scale factor |
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 16 | Training batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--cpu` | flag | False | Force CPU mode (disable CUDA) |
| `--output-dir` | str | 'outputs' | Directory for saved outputs |
| `--save-model` | flag | False | Save trained model weights |

## What It Does

### 1. Data Generation
Generates synthetic gravitational lensing images with Einstein ring patterns:
- Configurable ring radius and width
- Random background galaxies
- Realistic noise modeling
- Automatic downsampling for LR/HR pairs

### 2. Model Architecture
3-layer SRCNN:
- **Layer 1:** Feature extraction (64 filters, 9×9 kernel)
- **Layer 2:** Non-linear mapping (32 filters, 5×5 kernel)
- **Layer 3:** Reconstruction (1 filter, 5×5 kernel)
- Total parameters: ~57,000

### 3. Training
- Loss function: MSE (Mean Squared Error)
- Optimizer: Adam
- Logs loss per epoch
- Optional model saving

### 4. Visualization
Generates comparison images showing:
- Low-resolution input
- Super-resolved output
- High-resolution target

## Output Files

After running with `--save-model`:

```
outputs/
├── srcnn_deeplense.pth    # Trained model weights
└── srcnn_result.png        # Visualization of results
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.10
- NumPy
- Matplotlib

Install with:
```bash
pip install torch torchvision numpy matplotlib
```

## Example Training Session

```bash
$ python train_srcnn_minimal.py --epochs 5 --save-model

======================================================================
SRCNN TRAINING FOR DEEPLENSE
======================================================================

Device: cuda

Creating dataset...
Generating 100 dummy lensing images...
✅ Dummy dataset ready

Initializing model...
Model parameters: 57,281

======================================================================
TRAINING
======================================================================

Epoch [1/5] Loss: 0.107833
Epoch [2/5] Loss: 0.039810
Epoch [3/5] Loss: 0.021186
Epoch [4/5] Loss: 0.012626
Epoch [5/5] Loss: 0.008617

✅ Training complete!
💾 Model saved: outputs\srcnn_deeplense.pth

Creating visualizations...
✅ Visualization saved: outputs\srcnn_result.png

======================================================================
DONE!
======================================================================
```

## Architecture Details

The SRCNN model is designed for simplicity and interpretability:

```python
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
```

## Limitations & Future Work

### Current Limitations
- Uses synthetic data (not real lensing observations)
- Simple supervised approach (not fully unsupervised)
- No domain adaptation for real images
- Fixed scale factor (2×)

### Planned Improvements (DEEPLENSE2)
1. **Real Data Integration:** Use simulated lensing datasets from Model II/III
2. **Unsupervised Learning:** Implement autoencoder-based SR
3. **Sim-to-Real Gap:** Add domain adaptation techniques
4. **Lens Analysis:** Extract physical parameters from super-resolved images

## Related Proposals

- **DEEPLENSE2:** Unsupervised Super-Resolution and Analysis of Real Lensing Images
- **DEEPLENSE1:** Multi-Class Classification
- **DEEPLENSE4:** Regression Tasks

## References

- [SRCNN Paper](https://arxiv.org/abs/1501.00092)
- [DeepLense Project](https://github.com/ML4SCI/DeepLense)

## Contributing

This is a baseline implementation. Contributions welcome for:
- Adding real dataset loaders
- Implementing advanced unsupervised techniques
- Improving model architectures
- Adding evaluation metrics (PSNR, SSIM)

## License

Follows DeepLense repository license.