# Physics Informed Neural Network for Dark Matter Morphology

![Image](https://upload.wikimedia.org/wikipedia/commons/e/e3/DeepLense_simulation.png)

## Introduction

Greetings, everyone! I am very excited to share the progress of my project, "Physics-Guided Machine Learning" for Google Summer of Code 2024. Over the past two months, I have been working diligently with ML4SCI on this project.

## Project Overview

### DeepLense: Physics-Guided Machine Learning

DeepLense is a deep learning framework designed for searching for particle dark matter through strong gravitational lensing.

### Gravitational Lensing

Gravitational Lensing is a phenomenon that occurs when a massive celestial body, such as a galaxy cluster, causes sufficient curvature of spacetime, bending the path of light around it as if by a lens.

### Strong Gravitational Lensing

Strong Gravitational Lensing occurs when the gravitational field of a massive galaxy or other celestial object is strong enough to produce multiple images of a source galaxy.

### Dark Matter

Dark matter doesn't emit or absorb light, making it invisible. However, we can detect its presence through its gravitational effects, such as observing more gravitational lensing than what we would expect from visible matter alone.

## Project Goals

This project aims to develop a physics-informed neural network framework to infer the properties and distribution of dark matter in the lensing galaxy responsible for strong gravitational lensing.

## Data

The data includes gravitational lensing images categorized into three classes representing the substructures of the lensing galaxy:
- Axion
- Cold Dark Matter
- No Substructure

## Methodology

### Physics Informed Preprocessing

We analyze the abnormal distribution of the intensity of the map, which depends on the lensing galaxy and its substructures.

### Image Reconstruction

Using the Gravitational Lensing Equation, we perform lensing inversion to reconstruct the source image.

### Encoder

The encoder encodes the magnitude of the angular deviation, which is the product of the Angular Diameter Distance and Einstein's Radius.

### CNN as Decoder

After obtaining the source image and the distorted image, we train the model to learn from the differences between these images.

### GradCAM Visualization

Grad-CAM is used to interpret the model by highlighting the regions of an image where the model is focusing to make its decision.

## Results

We tested and compared three models:
- Resnet18 (11.68M)
- Lensiformer (15.78M)
- New Model (14.43M)

### Performance

The new model converged faster and achieved a lower Cross Entropy Loss compared to Lensiformer and ResNet18.

### F1 Score

The new model achieved an F1 score of 0.997, the highest among the three models.

### ROC Curve

The new model achieved an ROC AUC score of 1, demonstrating its capability to differentiate between the classes across various decision thresholds.

## Future Goals

- Test these models on a harder dataset.
- Implement a physics-informed loss function to guide the model towards correct convergence and better reconstruction of the source galaxy.

## Acknowledgements

Special thanks to my incredible mentors: Sergei Gleyzer, Pranath Reddy, Anna Parul, PG Guan, and Mike Toomey.

## Repository

You can find my code, results, and all the plots on [GitHub](https://github.com/your-repo-link).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
