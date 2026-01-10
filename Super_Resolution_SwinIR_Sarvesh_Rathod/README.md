# DeepLense: Super-Resolution with Swin Transformer (SwinIR)

**Contributor**: Sarvesh Rathod  
**Target**: Google Summer of Code (GSoC) 2026 - ML4SCI

## 1. Project Overview
This project introduces a state-of-the-art Vision Transformer model, **SwinIR**, to the DeepLense repository. The objective is to enhance the resolution of strong gravitational lensing images, reconstructing fine details of arcs and rings that are critical for Dark Matter substructure analysis.

### Scientific Motivation
Gravitational lensing distorts background galaxies into arcs. The subtle variations in these arcs' thickness and intensity can reveal hidden "sub-halos" of Dark Matter. However, telescope images (like Hubble or the upcoming Euclid mission) often suffer from:
1.  **Low Resolution (Pixelation)**: Obscuring micro-structures.
2.  **Instrumental Noise**: Making faint arcs indistinguishable from background.

Traditional CNNs (like SRCNN or ResNet) struggle with **long-range dependencies**—they look at small patches of pixels. Lensing arcs, however, are large, global structures that curve across the entire image.

### The Solution: Swin Transformer
We propose using **SwinIR (Image Restoration Using Swin Transformer)**. Unlike CNNs, Transformers use **Self-Attention** mechanisms. This allows the model to:
*   Understand the global geometry of the lensing arc.
*   "Shift windows" to capture relationships between distant parts of the image.
*   Reconstruct sharper edges and textures than convolutional baselines.

---

## 2. Directory Structure
This folder creates a modular, reproducible pipeline for training and evaluating SwinIR on DeepLense data.

```
Super_Resolution_SwinIR_Sarvesh_Rathod/
├── dataset.py          # PyTorch Dataset loader for Lensing NPY files
├── model.py            # SwinIR Architecture definition
├── train.py            # Training loop with Validation & Checkpointing
├── evaluate.py         # Evaluation script (PSNR/SSIM calculation + Visualization)
├── get_data_diff.py    # Data preprocessing utility (legacy support)
├── requirements.txt    # Python dependencies
└── README.md           # This documentation
```

## 3. Implementation Details

### Dataset
*   **Input**: Low-Resolution (LR) images (64x64), simulated with noise.
*   **Target**: High-Resolution (HR) images (128x128), ground truth.
*   **Source**: "Model 4" Real Galaxy Dataset (from DeepLense simulations).
*   **Simulation Parameters** (defined in `generate_gsoc_pairs.py`):
    *   **Lens Model**: SIE (Singular Isothermal Ellipsoid)
    *   **Velocity Dispersion ($\sigma_v$)**: $260 \pm 20$ km/s
    *   **Lens Redshift ($z_L$)**: 0.5
    *   **Source Redshift ($z_S$)**: 1.0 (with source convention $z=2.5$)
    *   **Source Light**: Real galaxy images (unbarred spirals) from `Galaxy10_DECals` dataset.
    *   **Resolution**:
        *   HR: 0.05 arcsec/pixel (128x128)
        *   LR: 0.10 arcsec/pixel (64x64)


### Model Architecture (SwinIR)
The model consists of three stages:
1.  **Shallow Feature Extraction**: A Convolutional layer maps the input image to a higher-dimensional feature space.
2.  **Deep Feature Extraction**: A stack of **RSTB (Residual Swin Transformer Blocks)**. Each block contains multiple Swin Transformer Layers (STL) with Window-based Multi-head Self Attention (W-MSA).
3.  **HR Reconstruction**: A high-quality upsampling module using **PixelShuffle** to generate the final 128x128 output.

### Training Strategy
*   **Loss Function**: `L1 Loss` (Mean Absolute Error). L1 is preferred over MSE for super-resolution as it encourages sharper edges and less blurring.
*   **Optimizer**: `AdamW` (Adaptive Moment Estimation with Weight Decay) for stable convergence.
*   **Metrics**:
    *   **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level accuracy.
    *   **SSIM (Structural Similarity Index)**: Measures perceptual quality.

---

## 4. How to Run

### Prerequisites
Install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 1: Generate Data
Run the generation script using the virtual environment (where dependencies are installed):
```bash
.\venv\Scripts\python.exe generate_gsoc_pairs.py
```
This will take some time to generate 2500 pairs.

### Step 2: Process Data
Once generation is complete, process the data:
```bash
.\venv\Scripts\python.exe get_data_diff.py
```

### Step 3: Training
Start the training loop:
```bash
.\venv\Scripts\python.exe train.py
```
This generates the formatted `train_HR.npy` and `train_LR.npy` files in `../data_diff/`.

### Step 2: Train the Model
Start the training process:
```bash
python train.py
```
*   Configurable hyperparameters (Epochs, Batch Size, LR) are at the top of `train.py`.
*   Checkpoints will be saved as `swinir_epoch_X.pth`.

### Step 3: Evaluate & Visualize
Run the evaluation script to calculate metrics and generate visual comparisons:
```bash
python evaluate.py
```
This will:
1.  Print the average PSNR and SSIM on the test set.
2.  Save `results.png` showing Side-by-Side comparison (Low Res vs SwinIR vs Ground Truth).

---

## 5. Benchmarks & Expected Results

To achieve state-of-the-art results, the model must be trained for at least **100 epochs** on a GPU.

| Model | Dataset | PSNR (dB) | SSIM | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Bicubic Interpolation** | Galaxy10 (Sim) | ~28.00 | ~0.75 | Baseline |
| **SRResNet** | Galaxy10 (Sim) | ~31.50 | ~0.88 | Standard |
| **SwinIR (Ours)** | Galaxy10 (Sim) | **35.66** (Achieved) | **0.9642** (Achieved) | **Proposed** |

*Note: These results were achieved after training for **10 epochs** on an NVIDIA GPU.*
