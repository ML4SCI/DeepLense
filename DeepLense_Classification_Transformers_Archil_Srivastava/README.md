# DeepLense: Classification Using Vision Transformers

A PyTorch-based framework for classifying **strong gravitational lensing images** to identify **dark matter substructures** using modern deep learning architectures, especially **Vision Transformers (ViTs)**.

This project benchmarks multiple transformer-based models (including hybrid architectures) using the [timm](https://github.com/rwightman/pytorch-image-models) library and tracks performance metrics (Loss, AUROC, etc.) using [Weights & Biases](https://wandb.ai/site).

> Originally developed as part of **Google Summer of Code (GSoC) 2022**
> [Project Link](https://summerofcode.withgoogle.com/programs/2022/projects/iFKJMj0t)

---

## Problem Statement

Gravitational lensing causes distortion in galaxy images due to massive celestial objects.
This project applies deep learning to:

* Classify lensing images into dark matter categories
* Analyze substructure patterns
* Benchmark modern transformer architectures in astrophysical ML tasks

---

## Key Features

* Image classification using Vision Transformers
* Supports multiple model families (ViT, Swin, CoAtNet, EfficientNet, etc.)
* Experiment tracking with Weights & Biases
* Benchmarking across multiple datasets
* Modular and extensible pipeline

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ML4SCI/DeepLense.git
cd DeepLense/DeepLense_Classification_Transformers_Archil_Srivastava
```

---

### 2. Install Dependencies

```bash
pip install torch torchvision timm wandb numpy matplotlib
```

---

### 3. Train a Model

```bash
python train.py \
--dataset Model_I \
--model_name coatnet_nano_rw_224 \
--pretrained \
--device best
```

---

### 4. Evaluate the Model

```bash
python eval.py \
--run_id <wandb_run_id> \
--device cpu
```

---

## Project Structure

```bash
.
├── data/           # Dataset loading & preprocessing
├── models/         # Model architectures (ViT, etc.)
├── train.py        # Training script
├── eval.py         # Evaluation script
├── utils.py        # Helper utilities
├── constants.py    # Configurations
├── results/        # Output logs and results
```

---

## Datasets

The model is trained on **3 simulated datasets**:

| Dataset   | Image Size | Characteristics               |
| --------- | ---------- | ----------------------------- |
| Model I   | 150×150    | Gaussian PSF + noise          |
| Model II  | 64×64      | Euclid observation simulation |
| Model III | 64×64      | HST observation simulation    |

### Classes:

* Axion (Vortex substructure)
* CDM (Cold Dark Matter)
* No Substructure (baseline class)

> Note: Axion datasets include additional mass-related information.

---

## Training Configuration

Example:

```bash
python train.py \
--dataset Model_I \
--model_name coatnet_nano_rw_224 \
--batch_size 32 \
--lr 0.001 \
--epochs 20 \
--optimizer adam \
--device best
```

### Important Parameters:

| Parameter  | Description                    |
| ---------- | ------------------------------ |
| dataset    | Model_I / Model_II / Model_III |
| model_name | Model from timm library        |
| pretrained | Use pretrained weights         |
| tune       | Fine-tune model                |
| batch_size | Batch size                     |
| lr         | Learning rate                  |
| epochs     | Training epochs                |
| device     | cpu / cuda / best              |

---

## Evaluation

Evaluate trained models using:

```bash
python eval.py --run_id <wandb_run_id>
```

Metrics:

* Accuracy
* AUROC (class-wise & overall)
* Loss

---

## Results & Benchmarks

Models evaluated include:

* EfficientNet
* Vision Transformers (ViT)
* Swin Transformer
* CoAtNet
* CrossViT
* ConViT

Full experiment logs:
https://wandb.ai/_archil/ml4sci_deeplense_final

---

## Suggested Improvements (For Contributors)

This project can be extended by:

* Adding new transformer architectures
* Improving dataset preprocessing
* Optimizing training pipelines
* Adding visualization dashboards
* Improving documentation & usability

---

## Contributing

We welcome contributions!

### How to contribute:

1. Fork the repository
2. Create a new branch
3. Make changes
4. Submit a Pull Request

### Good first contributions:

* Improve documentation
* Add comments to code
* Fix bugs
* Add visualization

---

## Limitations

* Requires WandB account for full tracking
* Dataset setup not fully automated
* Limited documentation for beginners (can be improved)

---

## References

* [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)
* DeepLense GitHub Repository
* GSoC 2022 Project

---

## Summary

This project demonstrates the application of **state-of-the-art transformer architectures** to astrophysical data, specifically for understanding **dark matter substructures through gravitational lensing images**.

---

## Citation

```bibtex
@misc{rw2019timm,
author = {Ross Wightman},
title = {PyTorch Image Models},
year = {2019},
publisher = {GitHub},
journal = {GitHub repository},
doi = {10.5281/zenodo.4414861}
}
```
