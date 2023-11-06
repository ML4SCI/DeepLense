<div align="center">

  # Super-Resolution for Strong Gravitational Lensing

![](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)
![](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


**This work was done as part of Google Summer of Code (GSoC) 2023**
</div>

## Project Description

Strong gravitational lensing is a promising probe of the substructure of dark matter to better understand its underlying nature. Deep learning methods have the potential to accurately identify images containing substructure and differentiate WIMP particle dark matter from other well-motivated models, including axions and axion-like particles, cold dark matter, etc. Gravitational lensing data is often collected at low resolution due to the limitations of the instruments or observing conditions. Image super-resolution techniques can be used to enhance the resolution of these images with machine learning, allowing for more precise measurements of the lensing effects and a better understanding of the distribution of matter in the lensing system. This can improve our understanding of the mass distribution of the lensing galaxy and its environment, as well as the properties of the background source being lensed. This project will focus on the development of deep learning-based image super-resolution techniques based on residual networks and conditional diffusion models to enhance the resolution of gravitational lensing data. Furthermore, we will also investigate leveraging the super-resolution models for other downstream tasks, i.e., strong lensing tasks such as regression and lens finding.

## Residual Models

We have explored several residual models for the task of superresolution on the Model-1 dataset. The RCAN model utilizes a channel attention mechanism within its deep residual network to focus on salient features across channels, thereby effectively restoring fine details. The RDN framework harnesses the power of dense connections, allowing it to exploit local features extensively and promote feature reuse for improved image reconstruction. SRResNet leverages residual blocks to enable deeper networks by mitigating the vanishing gradient problem and facilitating the learning of complex mappings between different resolution spaces. EDSR refines this approach by streamlining residual networks for super-resolution and optimizing the network to enhance texture details.

On the faster end of the spectrum, FSRCNN offers a lightweight and efficient alternative, designed for speed without significantly compromising image quality. We have also designed a hybrid FSRCNN, which combines equivariant feature extraction with a convolutional upscaling layer for enhanced performance. Contrasting these deep learning models, bilinear interpolation stands as a basic technique for image scaling, providing a baseline for comparison. Lastly, the equivariant FSRCNN model emphasizes the preservation of geometric transformations from input to output, crucial for maintaining structural integrity in astronomical imaging.

| Model                                                        | MSE     | PSNR    | SSIM    |
|--------------------------------------------------------------|---------|---------|---------|
| RCAN                                                         | 0.00089 | 30.50028| 0.56995 |
| Residual Dense Network (RDN)                                 | 0.0009  | 30.49815| 0.57196 |
| SRResNet (18 Blocks)                                         | 0.0009  | 30.49482| 0.57325 |
| EDSR                                                         | 0.0009  | 30.49347| 0.57424 |
| FSRCNN                                                       | 0.0009  | 30.45184| 0.56641 |
| Hybrid FSRCNN (Feature Extraction Layers - Equivariant, Upscaling Layer - Convolutional) | 0.00091 | 30.42472| 0.57249 |
| Bilinear Interpolation (Baseline)                            | 0.00333 | 24.7818 | 0.30323 |
| Equivariant FSRCNN                                           | 0.08281 | 19.35347| 0.54386 |

<div align="center">
This table provides a comparative analysis of residual models on the Model-1 dataset.
</div>

### Exploring Content Loss

Here we present the performance metrics of the super-resolution models RCAN and RDN, comparing the results of two different loss functions: mean squared error (MSE) alone and a combination of MSE with content loss derived from a pre-trained adversarial encoder (AAE). The content loss is integrated to incorporate high-level feature representations from high-resolution images captured by the AAE into the super-resolution training process. The inclusion of content loss from a pre-trained adversarial autoencoder is having a positive effect on the structural similarity of the super-resolved images.

| Loss Function             | MSE     | PSNR    | SSIM    |
|---------------------------|---------|---------|---------|
| MSE                       | 0.00089 | 30.50028| 0.56995 |
| MSE + AAE Content Loss    | 0.0009  | 30.47946| 0.58035 |

| Loss Function             | MSE     | PSNR    | SSIM    |
|---------------------------|---------|---------|---------|
| MSE                       | 0.0009  | 30.49815| 0.57196 |
| MSE + AAE Content Loss    | 0.0009  | 30.49319| 0.57303 |

### Performance on downstream task

Here we present a comparison of downstream regression task performance using a simple ResNet-18 model across three datasets based on Model-1: original high-resolution images, interpolated low-resolution images, and low-resolution images enhanced by the RDN super-resolution model. The metrics used for evaluation are mean squared error (MSE) and mean absolute error (MAE), with lower values indicating superior performance. The original data serves as the benchmark with the lowest MSE and MAE. Interpolated low-resolution data exhibits higher error rates, reflecting the loss of detail's adverse impact on regression tasks. Notably, the upscaling of low-resolution data via the RDN model yields a marked improvement in both MSE and MAE, approaching the benchmark set by the high-resolution data, thereby underscoring the efficacy of super-resolution processing in enhancing the quality of images for subsequent analytical tasks.

| Trained and Tested On:                   | MSE      | MAE      |
|------------------------------------------|----------|----------|
| Original Data                            | 0.212117 | 0.379851 |
| Interpolated Low Resolution (LR) Data    | 0.244074 | 0.407716 |
| LR Data upscaled by RDN SR Model         | 0.220001 | 0.387331 |

## Conditional Diffusion Model

<img align="left" width="170" height="170" src="https://github.com/pranath-reddy/DeepLenseSR/blob/main/Figures/Demo_Sample.gif"> Diffusion Models operate by methodically degrading training data through the incremental infusion of Gaussian noise, subsequently learning to reconstruct the original data by inverting this noise addition process. In our project, we adapt this principle for the enhancement of astronomical images affected by strong gravitational lensing. We employ high-resolution lensing images as conditions and feed low-resolution images as inputs into the Diffusion Model. During training, the model learns to progressively refine the quality of these inputs by reversing the noise that simulates low resolution. After training, the model can enhance low-resolution data by initiating the process with the degraded images and applying the learned denoising strategy.

<div align="center">
  <img src="https://github.com/pranath-reddy/DeepLenseSR/blob/main/Figures/Diffusion_Samples.png" alt="Phenomenon Description" width="900"/>
</div>
<div align="center">
This visual analysis includes a comparison of low-resolution images, processed outputs from deep learning models, and the original high-resolution samples. The low-resolution images are synthetically generated from the Model-4 dataset by introducing varying levels of Gaussian noise and blur, mimicking the quality typically affected by observational constraints. 
</div>
<br>

## Requirement

``pip install -r requirement.txt``

## Installation & Usage

```
git clone https://github.com/pranath-reddy/DeepLenseSR.git
cd DeepLenseSR/src/
jupyter notebook
```

## To-Do

- [ ] Push work to the main DeepLense repository.
- [ ] Perform a quantitative analysis to evaluate the performance of the diffusion model.
- [ ] Explore few-shot approaches for super-resolution of real-world samples.
- [ ] Document the work in a research article.

## References

~~~
@article{2018arXiv181210477Z,
  title = "{Residual Dense Network for Image Restoration}",
  author = {{Zhang}, Yulun and {Tian}, Yapeng and {Kong}, Yu and {Zhong}, Bineng and {Fu}, Yun},
  journal = {arXiv e-prints arXiv:1812.10477},
  year={2018}
}
~~~

~~~
@article{2020arXiv200611239,
  title = "{Denoising Diffusion Probabilistic Models}",
  author = {{Ho}, Jonathan and {Jain}, Ajay and {Abbeel}, Pieter},
  journal = {arXiv e-prints arXiv:2006.11239},
  year={2020}
}
~~~

## Citation

Citation details to be completed upon publication.

## Acknowledgment

I would like to thank my mentors, Prof. Sergei Gleyzer and Dr. Michael Toomey, for their continued support and guidance.
