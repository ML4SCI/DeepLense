# Unsupervised Autoencoder Super-Resolution for DeepLense

## Overview

This implements **unsupervised super-resolution** using autoencoders for gravitational lensing images, directly addressing **DEEPLENSE2 proposal requirements**.

**Key Improvement over PR #1:**
- PR #1: Supervised SRCNN (requires paired LR/HR images)
- **PR #2: Unsupervised Autoencoder (learns from single images)**

This approach is critical for real lensing studies where high-resolution paired data is scarce.

## DEEPLENSE2 Proposal Alignment

| Requirement | Implementation |
|------------|----------------|
| **Unsupervised SR** | ✅ Autoencoder learns without paired data |
| **Autoencoders** | ✅ U-Net style encoder-decoder with skip connections |
| **Simulated images** | ✅ Supports Model I/II/III DeepLense data |
| **Real galaxy sources** | ✅ Can load real simulation .npy files |
| **Task #1: SR of simulated images** | ✅ Directly implemented |
| **Evaluation** | ✅ PSNR and SSIM metrics included |

## Quick Start

### Basic Usage (Synthetic Data)
```bash
python train_autoencoder_sr.py --use-synthetic --epochs 20 --save-model
```

### With Real DeepLense Data
```bash
python train_autoencoder_sr.py \
    --data-path /path/to/Model_I_data.npy \
    --epochs 30 \
    --save-model
```

### Advanced Configuration
```bash
python train_autoencoder_sr.py \
    --data-path data/Model_II/ \
    --epochs 50 \
    --batch-size 32 \
    --base-channels 128 \
    --scale-factor 4 \
    --lr 0.0005 \
    --save-model
```

## Command-Line Arguments

### Data Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-path` | str | None | Path to .npy file or directory with DeepLense data |
| `--use-synthetic` | flag | False | Use synthetic Einstein rings (fallback) |
| `--n-samples` | int | 100 | Number of synthetic samples to generate |
| `--img-size` | int | 64 | Image size (square) |

### Model Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--base-channels` | int | 64 | Base number of filters in autoencoder |
| `--scale-factor` | int | 2 | Super-resolution scale factor |
| `--noise-level` | float | 0.05 | Gaussian noise std for degradation |

### Training Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 20 | Number of training epochs |
| `--batch-size` | int | 16 | Training batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--alpha` | float | 1.0 | Weight for reconstruction loss |
| `--beta` | float | 0.1 | Weight for perceptual loss |

### System Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cpu` | flag | False | Force CPU mode |
| `--output-dir` | str | 'outputs_pr2' | Output directory |
| `--save-model` | flag | False | Save trained model weights |

## Architecture

### Autoencoder Design

```
Input (LR image) → Encoder → Bottleneck → Decoder → Output (SR image)
                      ↓                        ↑
                   Skip Connections (U-Net style)
```

**Encoder:**
- 3 encoder blocks with downsampling
- Each block: Conv → BatchNorm → ReLU → MaxPool
- Progressively increases channels: 64 → 128 → 256

**Bottleneck:**
- 2 convolutional layers
- Highest channel count (512)
- Compressed representation

**Decoder:**
- 3 decoder blocks with upsampling
- Each block: ConvTranspose → Concat(skip) → Conv → BatchNorm → ReLU
- Progressively decreases channels: 256 → 128 → 64

**Total Parameters:** ~2-3 million (depends on `--base-channels`)

### Loss Function

**Perceptual Loss = α × Reconstruction Loss + β × Gradient Loss**

1. **Reconstruction Loss:** Pixel-wise MSE
2. **Gradient Loss:** Preserves edges and structure

This combined loss produces sharper, more realistic super-resolved images than MSE alone.

## Unsupervised Training Process

Unlike supervised methods (SRCNN), this approach doesn't need paired LR/HR images:

1. **Input:** High-resolution image from dataset
2. **Degradation:** Apply downsampling + noise → Create LR version
3. **Forward:** Autoencoder reconstructs HR from LR
4. **Loss:** Compare reconstruction to original HR
5. **Update:** Backpropagate and update weights

This simulates the observation process and allows training on any HR images.

## Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- Measured in decibels (dB)
- Typical range: 20-50 dB
- **Higher is better**
- Measures pixel-wise similarity

### SSIM (Structural Similarity Index)
- Range: -1 to 1
- **Higher is better** (1 = identical)
- Measures perceptual similarity
- Better correlates with human perception than PSNR

## Output Files

After training with `--save-model`:

```
outputs_pr2/
├── autoencoder_sr_deeplense.pth     # Trained model weights
├── autoencoder_sr_result.png        # Visual comparison (LR vs SR vs HR)
└── training_curves.png              # Loss and metrics over epochs
```

## Example Training Session

```bash
$ python train_autoencoder_sr.py --use-synthetic --epochs 20 --save-model

======================================================================
UNSUPERVISED AUTOENCODER SR - DEEPLENSE2 PR #2
======================================================================

🖥  Device: cuda

📊 Loading dataset...
📊 Generating 100 synthetic lensing images...
✅ Dataset ready: torch.Size([100, 1, 64, 64]) (synthetic)

🏗  Building model...
   Parameters: 2,123,713

======================================================================
TRAINING
======================================================================

Epoch 1/20: 100%|████████| 7/7 [00:02<00:00,  3.21it/s, loss=0.0421, psnr=15.23dB]
Epoch [1/20]
  Loss: 0.042145 (Recon: 0.041203, Percep: 0.009421)
  PSNR: 15.23 dB, SSIM: 0.4521

[... training continues ...]

Epoch 20/20: 100%|████████| 7/7 [00:02<00:00,  3.45it/s, loss=0.0053, psnr=28.67dB]
Epoch [20/20]
  Loss: 0.005321 (Recon: 0.005012, Percep: 0.003091)
  PSNR: 28.67 dB, SSIM: 0.8934

✅ Training complete!

💾 Model saved: outputs_pr2/autoencoder_sr_deeplense.pth

📊 Creating visualizations...
✅ Visualizations saved to: outputs_pr2/
   - autoencoder_sr_result.png
   - training_curves.png

======================================================================
DONE! PR #2 READY
======================================================================
```

## Key Differences from PR #1 (SRCNN)

| Feature | PR #1 (SRCNN) | PR #2 (Autoencoder) |
|---------|---------------|---------------------|
| **Learning Type** | Supervised | **Unsupervised** ✅ |
| **Paired Data Required** | Yes | **No** ✅ |
| **Architecture** | 3-layer CNN | U-Net Autoencoder ✅ |
| **Skip Connections** | No | **Yes** ✅ |
| **Loss Function** | MSE only | **Perceptual** ✅ |
| **Evaluation Metrics** | None | **PSNR + SSIM** ✅ |
| **Real Data Support** | No | **Yes** ✅ |
| **Parameters** | ~57K | ~2M |

## Requirements

```bash
pip install torch torchvision numpy matplotlib tqdm
```

Tested with:
- Python ≥ 3.9
- PyTorch ≥ 1.10
- CUDA 11.8+ (optional)

## Data Format

### DeepLense Simulation Format
Expected `.npy` file structure:
- **Shape:** `(N, H, W)` or `(N, 1, H, W)` or `(N, C, H, W)`
- **Type:** float32
- **Range:** Any (will be normalized to [0, 1])

### Supported Models
- **Model I:** No substructure (smooth matter distribution)
- **Model II:** Cold Dark Matter (CDM) subhalos
- **Model III:** Vortex/Axion substructure

All models supported through `--data-path` argument.

## Future Work (DEEPLENSE2 Task #2)

This PR implements **Task #1** (unsupervised SR on simulated images).

**Next steps for full DEEPLENSE2:**
1. Sim-to-real domain adaptation
2. Lens parameter extraction modules
3. Substructure analysis integration
4. Real observation data testing

## References

- DEEPLENSE2 Proposal: Unsupervised Super-Resolution and Analysis of Real Lensing Images
- U-Net Architecture: Ronneberger et al. (2015)
- Perceptual Loss: Johnson et al. (2016)

## Contributing

This is part of the DEEPLENSE2 GSoC 2026 project. Feedback and improvements welcome!

## License

Follows DeepLense repository license.