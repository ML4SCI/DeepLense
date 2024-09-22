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

## Performance

### Large Dataset Results
The performance of different models on a large dataset is compared in the table below.

**Table 1: Performance Metrics of Different Models (Best in bold, Previous best underlined)**

| Model Name        | Parameters (Millions) | Micro F1 Score | No Subs. | CDM  | Axion |
|-------------------|-----------------------|----------------|---------------|----------|------|
| ResNet18          | 11.17                 | 0.910          | 0.98          | 0.90     | 0.92 |       
| ViT               | 13.7                  | 0.902          | 0.99          | 0.90     | 0.90 |       
| CaiT              | 13.7                  | 0.936          | 0.98          | 0.92     | 0.95 |       
| ViTSD             | 13.7                  | 0.911          | 0.99          | 0.91     | 0.90 |       
| Lensiformer       | 15.7                  | 0.976          | 1.00          | 0.97     | 0.98 |       
| LensCoAt_small    | 7.04                  | 0.994          | 1.00          | 0.99     | 0.99 |       
| LensCoAt_large    | 14.43                 | **0.999**      | **1.00**      | **1.00** | **1.00** |

### Small Dataset Results
For the small dataset case with 3000 images per class, the performance of the models was as follows.

**Table 2: Performance of Different Models (Parameters in millions; Best in bold, 2nd best underlined)**

| Model Name        | Parameters (Millions) | Accuracy | Micro F1 Score | No Subs. | CDM  | Axion |
|-------------------|-----------------------|----------|----------------|---------------|----------|------|
| ResNet18          | 11.17                 | 0.818    | 0.817          | 0.97          | 0.85     | 0.95 |       
| ViT               | 13.72                 | 0.863    | 0.864          | 0.99          | 0.66     | 0.96 |       
| CaiT              | 13.76                 | 0.878    | 0.871          | 0.99          | 0.64     | 0.96 |       
| ViTSD             | 13.73                 | 0.867    | 0.868          | 1.00          | 0.66     | 0.97 |       
| Lensformer        | 15.7                  | 0.957    | 0.959          | 1.00          | 0.99     | 0.99 |       
| LensPINN_small    | 7.17                  | 0.956    | 0.957          | 1.00          | 0.99     | 0.99 |       
| LensPINN_large    | 14.56                 | **0.996** | **0.996**      | **1.00**      | **1.00** | **1.00** |



## Future Goals

- Grad-CAM and other explainable AI tools can be valuable for studying dark matter, as they help visualize and interpret model decisions. However, Grad-CAM is sensitive because it relies on gradients and weights that are randomly initialized, affecting its robustness.
- Testing these models on a harder dataset.
- Implementing a physics-informed loss function to guide the model towards correct convergence and better reconstruction of the source galaxy. This is achieved by incorporating physical principles and source profile information into the loss function.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
