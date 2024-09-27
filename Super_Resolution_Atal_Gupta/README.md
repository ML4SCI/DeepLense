# Single Image Super-Resolution with Diffusion Models

This project focuses on enhancing the resolution of gravitational lensing images using diffusion models. The goal is to improve the clarity and detail of these images, aiding in better scientific analysis and discoveries.

## Project Overview

### Gravitational Lensing

Gravitational lensing is a phenomenon predicted by Einstein's theory of general relativity. It occurs when a massive object, like a galaxy cluster or black hole, bends the path of light from a distant source. This can create multiple images, arcs, or rings of the source, depending on the alignment and mass distribution of the lensing object.

![Gravitational Lensing Image](figures/gravitlensing.webp)

*[Image Source](https://www.jpl.nasa.gov/images/pia23641-gravitational-lensing-graphic)*


### Importance of Super-Resolution

Enhancing the resolution of gravitational lensing images is crucial for:

- **Improved Accuracy:** Clearer images allow for more precise measurements of lensing effects, leading to better estimates of the mass and distribution of dark matter.
- **Detecting Faint Sources:** Higher resolution reveals faint, distant galaxies that are magnified by the lensing effect but obscured in lower-quality images.
- **Studying Cosmic Structure:** Enhanced images provide better insights into the structure and evolution of galaxies and galaxy clusters.

## Dataset

The dataset consists of 2,834 pairs of Low Resolution (LR) and High Resolution (HR) images. The LR images are derived from the HR images by adding Gaussian noise and applying blurring.

![Dataset](figures/dataset.webp)

## Results

| Model   | PSNR  | SSIM  | Paper |
|---------|-------|-------|-------|
| SRCNN  | 31.76 | 0.873 | [Link](https://arxiv.org/abs/1501.00092) |
| RCAN    | 32.60 | 0.890 | [Link](https://arxiv.org/abs/1807.02758) |
| SRGAN | 26.52 | 0.566  | [Link](https://arxiv.org/abs/1609.04802) |
| VAESR| 26.00 | 0.594 | [Link](https://arxiv.org/abs/1906.02691) |
| Iterative Auto Encoder  | 33.56| 0.855 | [Link](https://openreview.net/forum?id=k0CWAzK17r) |

*Results on the Test Dataset*


## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
