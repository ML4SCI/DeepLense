# Grid-based strong gravitational lensing for unsupervised super-resolution 
Hey, I'm Anirudh Shankar. This article presents the [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/) project in collaboration with [Machine Learning for Science (ML4Sci)](https://ml4sci.org/), at its conclusion. Over the summer, I've been working on a fast and differentiable grid-based tool ways to perform strong gravitational lensing that I demonstrate here for the task of the unsupervised super-resolution of lensed gravitational images.

This article will guide you through the motivation, and implementation of the architecture. Below is a list of its contents:

1. Code and data access
2. What is gravitational lensing?
3. Motivation and deliverables achieved
4. Implementation
5. Results and discussion
6. Perspectives
7. References

## 1. For replication: Code and relevant instructions
The Git repository can be accessed from [here](https://github.com/ML4SCI/DeepLense), in the parent ML4Sci repository.

It contains all the Python Notebooks used in training, the trained model weights, set-up instructions and some examples.

Use the following to install the required libraries in a virtual environment of choice.

`pip install -r requirements.txt`

Do find the implementation in Section 4 if you want to skip to it.

Observed galaxy images of the [Galaxy10 DECaLS Dataset](https://astronn.readthedocs.io/en/latest/galaxy10.html) are used in the analyses of this work.

The strong lensing images used in the demonstration are obtained from Michael W. Toomey's [work](https://github.com/mwt5345/DeepLenseSim/tree/main/Model_IV).

## 2. What is gravitational lensing?
The phenomenon of gravitational lensing was predicted by Einstein's general theory of relativity. The path of light curves due to the curvature of spacetime, often caused by the presence of massive (groupes of) objects in the line-of-sight between the the distorted source and the observer.

![Lensing graphic](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/lensing_graphic.jpg)
Lensing graphic, taken from Scott Dodelson and Fabian Schmidt.
Modern Cosmology. Elsevier, 2021.

The dynamics of the lensing process and the characteristics of the observed image both depend on the intensity distribution of the source, the spacetime curvature given by the mass distribution producing it (the lens), the distances and line-of-sight displacements between the source, the lens and the observer, and sometimes, the presence of interstellar media.

We restrict our study to the phenomenon of strong lensing, where the source, the lens and the observer are nearly perfectly aligned along their mutual axis, and the lens is sufficiently massive. This produces directly observable features such as multiple images of Einstein rings and elongated arcs.

An approximation we then make to simplify the study of the system is to say that almost all of the lens mass can be assumed to be concentrated at a single point, the thin-lens approximation. This allows for the study of lensing through the principles of Optics.

![Lensing diagram](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/lensing_diagram.png)
Lensing diagram, taken from Wikipedia

The positions of intensities of the source β are translated to produce the intensity positions of the observed image θ through a quantity called the deflection angle α, given by the lensing equation.
![Lensing equation](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/lensing_equation.png)

With these concepts in mind, we can proceed to the project.
## 3. Motivation
The studying of gravitational lensing can be very effective in probing the sub-structure of dark matter composing the lens. This requires images of high quality for a faithful study, which are often scarce. While super-resolution techniques exist, they traditionally require low-high resolution image pairs that are near impossible to obtain for the same source at the same conditions. The goal of this project is to use the physics of strong gravitational lensing to direct the unsupervised super-resolution of observed images, which in theory should outperform traditional unsupervised super-resolution, as a result of being informed by the physics.

### Handling degeneracies
We fix the lens model and conduct a posterior study of lensing. This is because the lensing system is degenerate, as we only have one of the two unknowns in the lensing equation, the observed image. Another way of saying this is that for a particular observed image, there can be many possible lens and source combinations that produce it. Another complication is that in reality, a single source is often lensed multiple times by different lenses to give multiple sets of images. Assuming the lens model thus eliminates both these degeneracies, at the cost of the unfaithful reconstruction of the source. While this could be handled to an extent through source specific deflection angle constructions, we skip this step as it is not essential for our task of super-resolution, and leave it for a future perspective. For all images, a single isothermal sphere lens model ψ producing a uniform radial deflection is assumed.
## 4. Formulation of the task

![Formulation of the task](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/formulation.png)

## 5. Implementation
### Why grid-based lensing
Over a given deflection field, one can one can easily compute a pixel-wise deflection angle through a pixel to arcsecond conversion. They can then be used to displace pixels to achieve lensing in a very inexpensive way. Why this isn't done in this work is two-fold:
- Pixel area density isn't conserved. When deflecting a source to produce a lensed image for example, pixels are scattered to a larger volume, and the inverse is true while reproducing the source. This could be fixed with interpolation,  but that would be an approximation, undesirably adding an error. An illustration follows-

![Pixel lensing illustration](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/pixel_scattering.png)

- Lensed pixel positions will be in decimal pixel indices. Realignment with the pixel grid would require rounding off or interpolation, which would introduce over- and under-densities. Below is an extract of the said over-densities-

![Lensing artefact](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/lensing_artefact.png)

### Grid-based lensing
One way to deal with the drawbacks presented above is to use a grid-based lensing method. The vertices of the pixels are projected onto a grid that itself is lensed to polygons in which the intensities are scattered. An illustration of the grid pre- and post-lensing follows-

![Lensing grid](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/lensing_grid.png)

The lensed grid must also be recast to an axis-aligned square grid to be able to be broadcast as an image. This is done by firstly cropping the polygon grid onto the square grid and then refilling intensities while conserving intensity.

### The resolution problem
Lensing of a point source by an axis-symmetric lens distributes the intensity to a ring called the Einstein radius, that is determined by the properties of the lens and the lensing system's geometry. As a result, all the intensity of the central pixel is distributed to all pixels within the Einstein radius. This creates an incredibly steep difference in information density, as a function of radius, i.e., pixels toward the center of the image have much fewer incoming pixels. The artefact thus created is illustrated here-

![Resolution artefact](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/resolution_artefact.png)

Dealing with this is straightforward- increase pixel density in the center of the image to offset the pixel density difference brought by lensing. A double logarithmic grid is therefore used in the rest of this work to perform all lensing operations, as shown here-

![Log grid](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/loggrid.png)
![Log forward grid](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/logforward.png)

Its results are much more promising-

![Example](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/log_grid_examples.png)

### Lensing systems as matrices
The result of the grid operations is the availability of matrices that can be loaded on a GPU to quickly perform lensing as follows-
1. The low resolution observed lensed images are firstly projected onto the double-logarithmic grid.
2. They are back-lensed to produce low resolution source reconstructions.
3. The low resolution source reconstructions are upscaled through the CNN to produce super resolved source reconstructions.
4. The super resolved source reconstructions are then forward-lensed to produce super resolved lensed images that we require after projecting them back from the double-logarithmic grid to the axis-aligned square grid.
5. As a final step, the instrumental effects of the observing telescope are synthetically applied to the  super resolved lensed images, as detailed later.

Additionally, lensing systems with multiple lenses can be handled very easily through the addition of the relevant matrices to the product chain, whose matrix multiplications can be accelerated with CUDA. 
### Formulation of the super-resolution task
Now that we have all the pieces forming the lensing pipeline in hand, we can set the super-resolution pipeline up to demonstrate the pipeline's use. The schematic is presented below-

![Super-resolution schematic](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/schematic.png)

The idea is to train against images upscaled using interpolation, while constraining the loss through lensing. The data flow is as follows-
1. A low resolution source is reconstructed via the pipeline and upscaled via interpolation. 
2. A CNN based autoencoder reproduces the upscaled reconstructed sources that are relensed via the pipeline. 
3. The final image is downsampled again via interpolation and trained to match the input LR images, along with the learned source to match the reconstructed LR source. 

A full description of the losses is presented shortly.
### Convergence maps
The CNN is encouraged to only fill intensity in destinations allowed by the lensing geometry. Lensing a uniform intensity image through the same grid-based method gives a convergence maps of pixel intensities that agrees with the lensing system. These maps are used to weight the soon to be explained pixel-wise loss, but before that, here's a look at it-

![Convergence map](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/convergence_maps.png)

### PSF convolution
The observed lensed systems are subject to the instrumental response of the telescope. Requiring the CNN to recreate this would be cruel and unnecessary, as one could smooth the images produced by the CNN using a matching Point Spread Function (PSF). This is exactly what is done. Here's the PSF used-

![PSF](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/psf.png)

### Weighed pixel-wise loss
A MSE weighted by the convergence maps in either direction is employed to compare the downsampled sources and images. These act to constrain the otherwise interpolated images.
## 6. Results and discussion

![Result example](/Grid_based_strong_lensing_for_unsupervised_super_resolution_Anirudh_Shankar/Readme_requirements/first_results.png)

It's seen that the pipeline upscales the lensing images reasonably well. The limitations of the interpolation method are however apparent whose potential handling is suggested in the perspectives.
## 7. Perspectives
### 7.1 SR convergence improvement
The absolute difference between adjacent pixels can be computed for the true high resolution lensed images, and enforced on the super resolved ones. A few of these should suffice to learn the overall trend to be used on the entire dataset. This mapping can also be learned with an additional CNN, an autoencoder for example.
### 7.2 Adaptive logarithmic grids
The super-resolution demonstrated here sufficed with an axis-symmetric deflection field. For such a field, the double-logarithmic grid can adequately combated the grid resolution issue described earlier. For an arbitrary deflection field however, an adaptive logarithmic grid would be required, as done by Vegetti and Koopmans (2018).
## 8. References
Vegetti, S., & Koopmans, L. V. (2009). Bayesian strong gravitational-lens modelling on adaptive grids: objective detection of mass substructure in Galaxies. Monthly Notices of the Royal Astronomical Society, 392(3), 945–963.