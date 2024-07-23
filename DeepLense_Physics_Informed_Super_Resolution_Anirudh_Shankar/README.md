# DeepLense x Physics Informed Super Resolution of Strong Lensing Images: Anirudh Shankar
Hey, I’m Anirudh Shankar. Through my Google Summer of Code (GSoC) project with Machine Learning for Science (ML4Sci) and DeepLense I’ve been working on interesting ways to integrate the Physics of Strong Lensing into Machine Learning models, which for now is through the super-resolution of Lensing images.

This article will guide you through the development, reasoning, implementation and experimentation with the architecture. Below is a list of its contents:

1. For replication: Code and relevant instructions
2. What is gravitational lensing?
3. Motivation and deliverables achieved
4. Implementation
5. Perspectives
6. Appendix for side stories

## 1. For replication: Code and relevant instructions

The Git repository can be accessed from here, as part of the parent ML4Sci repository. 

It contains all the Python Notebooks used in training, the trained model weights, dataset simulation scripts, set-up instructions and some examples.

Use the following to install the required libraries in a virtual environment of choice.

```bash
pip install -r requirements.txt
```

Do find the implementation in Section 4 if you want to skip to it.

The Simulations directory contains the code used to create the dataset, and is adopted from [Michael Toomey’s work,](https://github.com/mwt5345/DeepLenseSim/tree/main/Model_I) one of my project mentors.

## 2. What is gravitational lensing?

Gravitational lensing is the phenomenon of the bending of light in the gravity of a massive celestial object (such as a massive galaxy or a group of galaxies); the object essentially behaving as a cosmic lens. We, as a result see the distorted image(s) of light sources (typically another galaxy) behind it.

![Taken from, [1]](/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/README_resources/lensing_graphic.jpeg)

Taken from, [1]

![Taken from, [2]](/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/README_resources/Gravitational_lens_geometry.png)

Taken from, [2]

with,

$$
\beta = \theta-\frac{D_{ds}}{D_s}\hat{\alpha}(D_d\theta)\\\beta = \theta-\alpha(\theta)
$$

The dynamics of lensing depends on both the composition of the lens and the nature of the source. 

A reasonable simplification of the lens provides a starting point for analysis. These mid-term results use the Single Isothermal Sphere (SIS) lens, whose dynamics can be computed analytically, and whose deflection angle is identical across space.

$$
\alpha = \theta_E\frac{\theta}{|\theta|}
$$

## 3. Motivation and deliverables achieved

The approach devised here is partly inspired from the very interesting work done by H Gao, L Sun and JX Wang (2021) [3]. Two individual models are designed, to handle with a known source profile, and when the source isn’t known. In both cases, the deflection angle is used directly in the models.

![Two_methods.png](/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/README_resources/Two_methods.png)

The goal is to create and train models in a completely unsupervised fashion, which is often the case with real-life lensing images. As an added bonus, the source image: what the source galaxy must have looked like in the absence of the lens is also generated.

Both models are pre-trained with bicubic-interpolated lensing images to give them a starting point and guidance to meaningful convergence.

![Taken from, [4]](/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/README_resources/interpolation.png)

Taken from, [4]

This step is especially necessary because the training here isn’t direct, through the comparison with high-resolution images, and is much more subtle rather. A lower loss score doesn’t necessarily mean we get the results we want, which requires us to use constraints and other methods to ensure meaningful convergence.

## 4. Implementation

### 4.1. LensSR no source

![LensSR_no_source schematic](/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/README_resources/Schema_no_source.png)

LensSR_no_source schematic

This model features two Single Image Super Resolution (SISR) modules that upscale images to a set magnification. They are used to generate a zoomed-in version of the source, and the high-resolution lensing image from low-resolution images as inputs.

The models are trained by imposing the strong lensing equation as a loss, along with intensity constraints as boundary conditions. As a bonus, the source profile is also generated.

Below is an example of the upscaling achieved:

![LensSR_no_source example](/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/README_resources/no_source_example.png)

LensSR_no_source example

Note that the intermediate source profiles resemble the lensing images due to the nature of the SIS lens.

### 4.2. Lens with source

![LensSR_with_source schematic](/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/README_resources/Schema_source.png)

LensSR_with_source schematic

This model features only one Single Image Super Resolution (SISR) module to upscale the lensing images to a set magnification. The source profiles are generated through a Sérsic profile whose physical parameters are obtained from the lensing images themselves.

Lensing is re-performed on the generated Sérsic profile and the image network is trained against such re-lensed images. 

Below is an example of the upscaling achieved:

![with_source_example.png](/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar/README_resources/with_source_example.png)

## 5. Perspectives

1. It is easily noted in Section 4.2’s example that we don’t really need a SISR network to generate upscaled lensing images. The re-lensing exercise already does that. What we could then do, is to use the SISR network to generate the deflection angle, enabling the model to handle any lens type. 
2. In addition, a Sérsic profile can be assumed when the source profile isn’t known as it is a reasonable approximation anyways. These additions to the model will allow it to handle all lensing images unconditionally.
3. The models can be tested on the **Real-Galaxy-Tiny Dataset**, constructed from observed galaxy profiles

## 6. References

1. Scott Dodelson and Fabian Schmidt. Modern Cosmology. Elsevier, 2021.
2. Schneider, P., Kochanek, C., & Wambsganss, J. (2006). Gravitational lensing: strong, weak and micro: Saas-Fee advanced course 33 (Vol. 33). Springer Science & Business Media.
3. Han Gao, Luning Sun, and Jian-Xun Wang. “Super-resolution and denoising of fluid flow using physics-informed convolutional neural networks without high-resolution labels”. In: Physics of Fluids 33.7 (2021).
4. https://en.wikipedia.org/wiki/Interpolation