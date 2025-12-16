# Diffusion Model for Gravitational Lensing Simulation

A lightweight Vision Transformer-based diffusion model (NanoDiT) for generating synthetic gravitational lensing images conditioned on dark matter physics.



---

## 🎯 Overview

Gravitational lensing simulations are computationally expensive. This project uses **flow matching** (rectified flow) with a transformer backbone to efficiently generate physically plausible lensing images for three dark matter scenarios:

- **Axion** – Wavelike dark matter candidate
- **CDM** – Standard cold dark matter model  
- **No_sub** – Smooth dark matter (no substructure)

### Key Features

- ⚡ **Fast sampling**: 50 ODE steps vs 1000 diffusion steps
- 🎨 **Conditional generation** with classifier-free guidance (CFG)
- 📊 **Lightweight**: ~5M parameters, single GPU training
- 🔬 **Scientific validation**: FID scores + ResNet18 classifier

---

## 🏗️ Architecture

**NanoDiT** (Nano Diffusion Transformer):
- 6 transformer layers, 8 attention heads
- 512-dimensional hidden states
- Patch size: 2×2, Image size: 64×64×1
- Adaptive layer normalization (AdaLN) for conditioning
- Exponential moving average (EMA) for stable sampling

**Training Method**: Flow matching with linear interpolation
```python
z_t = (1 - t) * x_data + t * x_noise
velocity = model(z_t, t, class_label)
loss = MSE(velocity, x_noise - x_data)
```

---

## 📋 Requirements

### Environment
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 12GB+ GPU RAM recommended (tested on A100)

### Installation

```bash
git clone https://github.com/ML4SCI/DeepLense.git
cd gravitational-lensing-diffusion
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.24.0 \
            matplotlib>=3.7.0 tqdm>=4.65.0 ema-pytorch>=0.3.0 \
            torchmetrics>=1.0.0
```



---

## 📂 Dataset Structure

Organize your `.npy` files (64×64 grayscale arrays) as follows:

```
data/
├── axion/
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── cdm/
│   └── ...
└── no_sub/
    └── ...
```

**Update path in training scripts**:
```python
DATA_DIR = "path/to/your/data"
```

### Expected Data Format
- Shape: `(64, 64)` NumPy array
- Type: `float32`
- Range: Any (will be normalized to [-1, 1])

---

## 🚀 Training

### 1. Train Diffusion Model

```bash
python train/train_diffusion.py
```

**Key hyperparameters** (modify in script):
```python
BATCH_SIZE = 160          # Reduce if OOM
LEARNING_RATE = 1e-4      # Cosine decay to 1e-6
EPOCHS = 600              # ~8 hours on A100
CFG_SCALE = 5.0           # Guidance strength
NUM_ODE_STEPS = 50        # Sampling steps
```

**Training outputs**:
- Checkpoints: `diffusion_training/dit_conditional_ckpts/`
- Sample images: `diffusion_training/dit_conditional_images/`
- Training curves: `diffusion_training/training_metrics.png`

### 2. Train Classifier (Optional)

ResNet18 fine-tuned for evaluating generated samples:

```bash
python train/train_classifier.py
```

Expected accuracy: ~95%+ on real data

---

## 📊 Evaluation

### Compute FID Score

Measures distribution similarity between real and generated images:

```bash
python evaluate/evaluate_fld.py
```

**Good FID scores**:
- Overall: < 50 (lower is better)
- Per-class: < 60

### Classify Generated Samples

Test if model captures class-specific features:

```bash
python evaluate/evaluate_classify.py \
  --npy_path generated_npy/axion/0.npy \
  --model_path classifier_training_resnet18/classifier_ckpts/resnet18_best.pt
```

Expected output:
```
Predicted Class: 'axion' (Index: 0)
Confidence: 0.9234
```

---

## 🎨 Generating Samples

```python
from model import NanoDiT
from evaluate.evaluate_fld import load_model, generate_image

# Load trained model
model = load_model("diffusion_training/dit_conditional_ckpts/dit_conditional_final.pt")

# Generate 4 CDM samples (class_id=1)
images = generate_image(
    model, 
    target_class=1,      # 0=axion, 1=cdm, 2=no_sub
    ode_steps=100,       # More steps = better quality
    cfg_scale=5.0,       # Higher = stronger conditioning
    num_samples=4
)
```

---

## 📁 Code Structure

```
.
├── dataset.py                      # Data loading & transforms
├── models/
│   └── model.py                    # NanoDiT architecture
├── train/
│   ├── train_diffusion.py          # Flow matching training
│   └── train_classifier.py         # ResNet18 fine-tuning
└── evaluate/
    ├── evaluate_fld.py             # FID computation
    └── evaluate_classify.py        # Inference script
```

---

## 🔬 Technical Details

### Why Flow Matching?

Traditional diffusion (DDPM):
- Complex noise schedules
- Requires 1000+ sampling steps
- Sensitive to hyperparameters

Flow matching:
- Simple linear interpolation
- 50-100 steps sufficient
- More stable training

### Classifier-Free Guidance

During training, randomly drop 10% of class labels. At sampling:
```python
pred = pred_uncond + scale * (pred_cond - pred_uncond)
```
Higher `scale` → stronger class conditioning (but less diversity)

### EMA Benefits

Maintains exponentially smoothed model weights:
- Reduces sampling variance
- Improves FID by 10-20%
- No extra computation at training time

---

## 🐛 Troubleshooting

**Out of memory?**
- Reduce `BATCH_SIZE` to 64 or 32
- Use mixed precision: `AMP_DTYPE = torch.float16`

**Poor sample quality?**
- Increase `CFG_SCALE` to 7-10
- Use more ODE steps (100-150)
- Train longer (600+ epochs)

**NaN losses?**
- Reduce learning rate to 5e-5
- Check data normalization
- Increase gradient clipping to 2.0

**Classifier accuracy < 90%?**
- Fine-tune all ResNet layers
- Increase training epochs
- Add data augmentation

---

## 📈 Expected Results

| Metric | Value |
|--------|-------|
| Training time (A100) | ~8 hours |
| Final training loss | ~0.01 |
| Overall FID | 35-45 |
| Classifier accuracy (real) | 95%+ |
| Classifier accuracy (generated) | 85-90% |

---

## 🙏 Acknowledgments

Based on:
- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

