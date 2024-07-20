# Physics Informed Neural Network for Dark Matter Morphology

![image](https://github.com/user-attachments/assets/4c2f33d4-8ba2-46bc-83b9-eef6364b0cc4)


## Introduction

Greetings, everyone! I am very excited to share the progress of my project, "Physics-Guided Machine Learning" for Google Summer of Code 2024. Over the past two months, I have been working diligently with ML4SCI on this project.

## Project Overview

Deep learning is one of the most crucial tools today, as many advancements are driven by neural networks. This project aims to develop a physics-informed neural network framework to infer the properties and distribution of dark matter in the lensing galaxy responsible for strong gravitational lensing.

## Data

The data includes gravitational lensing images categorized into three classes representing the substructures of the lensing galaxy:
- Axion
- Cold Dark Matter
- No Substructure

All the images are single-channel and have 64 x 64 dimensions. The dataset can be found at the [DeepLense Model II simulated dataset](https://github.com/mwt5345/DeepLenseSim/tree/main/Model_II).

![image](https://github.com/user-attachments/assets/72ae3445-e491-4b10-bfd0-dd16ae931b8e)


## Methodology

### Physics Informed Preprocessing

The goal of this project is to study dark matter through strong gravitational lensing. To achieve this, we use a fundamental approach called Learning from Difference, analyzing the abnormal distribution of the intensity map as this distribution depends on the lensing galaxy and its substructures causing the bending of light.

![image](https://github.com/user-attachments/assets/0e6279d2-db08-41c3-afc8-3d73da441e2d)

The intuition behind using (Imax/I) is for contrast enhancement. The logarithm function is applied to compress and filter the dynamic range of intensities. Squaring is used to ensure all intensities are positive, as negative intensities are not possible. Finally, normalization (optional) is achieved using the tanh function to map values to a desired range. Performing the above transformation on each image results in:

![image](https://github.com/user-attachments/assets/ba01b771-fec1-4493-98b0-8709ed60eab5)

### Image Reconstruction

Using the Gravitational Lensing Equation, we perform lensing inversion to reconstruct the source image. For the Singular Isothermal Sphere (SIS) model, the angular deviation is proportional to Einstein's angle and is directed along the angular coordinate theta.

![image](https://github.com/user-attachments/assets/cb41845e-4888-428a-8e66-fc764ea8967b)

![image](https://github.com/user-attachments/assets/ed66b40f-26c5-4038-94a2-8058f671984c)

### GradCAM Visualization

Grad-CAM is used to interpret the model by highlighting the regions of an image where the model is focusing to make its decision. Below are some results for the GradCAM Visualization corresponding to the ResNet18 model.

![image](https://github.com/user-attachments/assets/91924180-b121-4723-9da9-9834cd7a8606)
![image](https://github.com/user-attachments/assets/3a963852-7837-4243-a6df-7ad9e584f676)
![image](https://github.com/user-attachments/assets/f15e6b51-d738-4a64-acad-65a71c0d810c)

## Results

I tested and compared three models:
- Resnet18 (11.68M)
- Lensiformer (15.78M)
- New Model (14.43M)

Below is the plot for comparing the ROC curve of the three models on the test dataset.

![image](https://github.com/user-attachments/assets/edd24833-8d81-4e4f-94dc-c1e046c95c7a)

## Future Goals

- Grad-CAM and other explainable AI tools can be valuable for studying dark matter, as they help visualize and interpret model decisions. However, Grad-CAM is sensitive because it relies on gradients and weights that are randomly initialized, affecting its robustness.
- Testing these models on a harder dataset.
- Implementing a physics-informed loss function to guide the model towards correct convergence and better reconstruction of the source galaxy. This is achieved by incorporating physical principles and source profile information into the loss function.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
