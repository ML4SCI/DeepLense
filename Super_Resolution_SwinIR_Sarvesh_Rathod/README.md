# Super Resolution for Gravitational Lensing using SwinIR

An advanced Deep Learning pipeline for recovering high-resolution details from pixelated gravitational lensing images, specifically tailored for the **ML4SCI (DeepLense)** research initiative.

## 🚀 Key Features

*   **SwinIR Transformer Core**: State-of-the-art Window-based Multi-head Self Attention.
*   **Research-Grade Simulation**: Pair generation using `lenstronomy` and real `Galaxy10_DECals` morphological sources.
*   **Accuracy Presets**: Easily switch between **Turbo** (fast debug), **Standard** (balanced), and **Research** (scientific quality) modes.
*   **High-Performance Training**: Fully integrated with **Mixed Precision (AMP)** and **GPU Acceleration (RTX 4050 Verified)**.
*   **Deployment Ready**: Support for **INT8 Quantization**, **Pruning**, and **ONNX Export**.
*   **Dual Implementation**: Available both as Python scripts and interactive Jupyter notebooks.

## 📖 Project Development History

This project was initially developed using standard Python scripts for rapid prototyping and testing. The core functionality was implemented in modular Python files (`generate_gsoc_pairs.py`, `get_data_diff.py`, `train.py`, `evaluate.py`, `model.py`, `dataset.py`) to enable easy debugging and script-based execution. 

After establishing a working pipeline, the codebase was enhanced with **interactive Jupyter notebooks** that provide:
- Step-by-step visualization and explanation
- Interactive parameter tuning
- Real-time training monitoring
- Better documentation and educational value

Both implementations (Python scripts and notebooks) are maintained to provide flexibility for different use cases:
- **Python Scripts**: Best for automated workflows, CI/CD pipelines, and batch processing
- **Jupyter Notebooks**: Best for experimentation, visualization, learning, and interactive development

## 📂 Project Structure

```text
Super_Resolution_SwinIR_Sarvesh_Rathod/
├── notebooks/                                  # Interactive Jupyter notebooks
│   ├── 01_Advanced_Data_Simulation.ipynb      # Lens physics & pair generation
│   ├── 02_Advanced_SwinIR_Training.ipynb      # Hyper-optimized training loop
│   └── 03_Optimization_and_Deployment.ipynb   # Pruning & ONNX export
├── model.py                                    # Core SwinIR architecture
├── dataset.py                                  # PyTorch Lensing dataset
├── generate_gsoc_pairs.py                      # Data generation script (original implementation)
├── get_data_diff.py                            # Data preprocessing script (original implementation)
├── train.py                                    # Training script (original implementation)
├── evaluate.py                                 # Model evaluation script (original implementation)
├── requirements.txt                            # Scientific dependency list
├── .gitignore                                  # Git ignore rules
└── README.md                                   # This document
```

## 🛠️ How to Run

### 1. Environment Setup

Create a virtual environment and install the verified dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Running with Jupyter Notebooks (Recommended for Interactive Use)

#### Data Simulation
Run `01_Advanced_Data_Simulation.ipynb` to generate the training data.
*   **Input**: `Galaxy10_DECals.h5` (expected path: `../../DeepLenseSim/data/Galaxy10_DECals.h5`)
*   **Output**: Normalized HR/LR tensors saved to `data_diff/` directory

#### Advanced Training
Run `02_Advanced_SwinIR_Training.ipynb` for model training.
*   **Kernel Selection**: Ensure you use the appropriate kernel with GPU support if available
*   **Presets**: Choose your mode in the configuration cell:
    *   `PRESET = "TURBO"`: Fast verification (runs in minutes, reduced model size)
    *   `PRESET = "STANDARD"`: Balanced for initial evaluation and hyperparameter tuning (default)
    *   `PRESET = "RESEARCH"`: Maximum capacity SwinIR for best accuracy
*   **Checkpoints**: Automatically saved every 5 epochs as `swinir_advanced_epoch_{epoch}.pth`

#### Optimization and Deployment
Run `03_Optimization_and_Deployment.ipynb` to optimize the model and export it for production.
*   Automatically detects the most recent checkpoint
*   Supports pruning, quantization, and ONNX export
*   Includes benchmarking utilities

### 3. Running with Python Scripts (Recommended for Automated Workflows)

#### Data Generation
```bash
python generate_gsoc_pairs.py    # Generates lensing image pairs
python get_data_diff.py          # Processes and normalizes the data
```

#### Training
```bash
python train.py                  # Trains the SwinIR model
```

#### Evaluation
```bash
python evaluate.py               # Evaluates trained model on test set
```

## ⚖️ Speed vs. Accuracy Trade-off

The project includes three performance presets:

*   **Turbo**: Optimized for local debugging. Uses reduced layers (embed_dim=30, depths=[2,2,2,2]) to provide instant feedback. Perfect for quick validation.
*   **Standard**: Balanced for initial evaluation and hyperparameter tuning. Uses moderate capacity (embed_dim=60, depths=[4,4,4,4]) with perceptual loss. **Recommended starting point.**
*   **Research**: Maximum capacity SwinIR (embed_dim=60, depths=[6,6,6,6]). Eliminates checkerboard artifacts and achieves peak PSNR/SSIM. Best for final submissions and publications.

## 📊 Results Summary

The pipeline has been verified on target hardware (**NVIDIA RTX 4050**) and successfully recovers complex lensing arcs from 64×64 pixelated inputs back to sharp 128×128 high-resolution maps.

### Performance Metrics

The trained model achieves excellent performance on the test set:

| Metric | Value | Assessment |
|--------|-------|------------|
| **PSNR** | 44.13 dB | Excellent (>40 dB is outstanding) |
| **SSIM** | 0.9868 | Excellent (>0.95 is outstanding) |
| **MSE** | 0.0002 | Very low error |
| **MAE** | 0.0060 | Minimal average pixel error |

These results demonstrate state-of-the-art performance for super-resolution tasks in gravitational lensing applications, with exceptional fidelity and structural preservation.

## 🔬 Scientific Contributions

- **Realistic Simulation**: Integration of `lenstronomy` with real `Galaxy10_DECals` galaxy morphologies
- **Advanced Architecture**: SwinIR transformer adapted for gravitational lensing super-resolution
- **Robust Training**: Mixed precision training with gradient clipping and perceptual loss
- **Production Ready**: Model optimization techniques (pruning, quantization) for deployment

## 📝 Notes

- Model checkpoints (`.pth` files) are excluded from git via `.gitignore` to reduce repository size
- Generated data files (`pairs/`, `data_diff/`, `*.npy`, `*.h5`) are also gitignored
- Virtual environment (`venv/`) is excluded from version control
- Ensure sufficient disk space for data generation and model checkpoints

---
**Author**: Sarvesh Rathod  
**Target**: ML4SCI / DeepLense GSoC Submission  
**License**: See repository for license information
