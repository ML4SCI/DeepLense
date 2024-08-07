# ML4Sci x DeepLense

## 1. Background

We at DeepLense explore cutting-edge Machine Learning techniques for the study of Strong Gravitational Lensing and Dark Matter Sub-structure. We use both simulated and real lensing images, for a variety of tasks that you can find [here](#3-projects).
1. Find below a description of [gravitational lensing](#11-gravitational-lensing) and [dark matter sub-structure](#12-dark-matter-and-sub-structure).
2. [Section 2](#2-models) contains a detailed description of the datasets used in the various projects
3. [Section 3](#3-projects) beins with a short description followed by an expansion on the various projects conducted at DeepLense

### 1.1 Gravitational Lensing

Gravitational lensing is the phenomenon of the bending of light in the gravity of a massive celestial object (such as a massive galaxy or a group of galaxies); the object essentially behaving as a cosmic lens. We, as a result see the distorted image(s) of light sources (typically another galaxy) behind it. 

![Lensing Schematic](https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Gravitational_lens-full.jpg/330px-Gravitational_lens-full.jpg)

The dynamics of lensing depends on both the composition of the lens and the nature of the source. We explore different lens models and source light profiles including real galaxy images in DeepLense, that you can find [here](#2-datasets).
### 1.2 Dark Matter and Sub-structure 

In DeepLense, we’re mainly dealing with three kinds of simulated Dark Matter:
* **Axion Dark Matter (Vortex)**: 
* **Cold Dark Matter (Subhalo)**: This model suggests that dark matter consists of slow-moving particles. In the CDM paradigm, smaller clusters of dark matter, known as subhalos, are approximated as “point masses.” This simplification facilitates computational modeling by treating these subhalos as singular points in the overall distribution of dark matter.
* **No-Substructure Dark Matter**: Unlike the CDM model, the “no-substructure” approach assumes that dark matter is evenly spread out, devoid of any smaller-scale clusters or sub-halos. This stands in stark contrast to the hierarchical structuring and layering of sub-halos within larger halos as predicted by CDM models.

## 2. Datasets

All datasets are constructed using Lenstronomy, by Michael W. Toomey, as presented in their [repository](https://github.com/mwt5345/DeepLenseSim/tree/main/).

| Dataset | Lens model | Light Profile | Modelling strategy |
|---|---|---|---|
|Model 1 dataset|Sheared Isothermal Elliptical lens| Sérsic light profile | Gaussian point spread function, Gaussian and Poissonian noise for SNR ~ 25, Axion DM and CDM substructure appended to base halo to create 3 sub-structure classes
|Model 2 dataset|Sheared Isothermal Elliptical lens | Sérsic light profile | Euclid's observation characteristics, Axion DM and CDM substructure appended to base halo to create 3 sub-structure classes
|Model 3 dataset|Sheared Isothermal Elliptical lens | Sérsic light profile | HST's observation characteristics, Axion DM and CDM substructure appended to base halo to create 3 sub-structure classes
|Model 4 dataset|Two Isothermal Elliptical lenses | Three-channel **real galaxy** images | Euclid's observation characteristics, Axion DM and CDM substructure appended to base halo to create 3 sub-structure classes

## 3. Projects

![Project compositions](/Images_for_README/DeepLense%20project%20composition.jpeg)
![Techniques compositions](/Images_for_README/DeepLense%20ML%20Techniques.jpeg)

| Project Name | Contributor | Task | ML Techniques | Repository Link |
|---|---|---|---|---|
|[DeepLense Classification Using Vision Transformers](#311-deeplense-classification-using-vision-transformers)|Archil Srivastava|Sub-structure classification|Vision Transformer Variants|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_Classification_Transformers_Archil_Srivastava)|
|[Unsupervised Domain Adaptation](#312-unsupervised-domain-adaptation)|Mriganka Nath|Sub-structure classification|Unsupervised Domain Adaptation|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_Gravitational_Lensing_Mriganka_Nath)|
|[Lensiformer: A Relativistic Physics-Informed Vision Transformer Architecture for Dark Matter Morphology in Gravitational Lensing](#313-lensiformer-a-relativistic-physics-informed-vision-transformer-architecture-for-dark-matter-morphology-in-gravitational-lensing)|Lucas Jose|Sub-structure classification|Physics-Informed Transformer|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Physics_Informed_Transformers_For_Dark-Matter_Morphology_Lucas_Jose)|
|[Physics Informed Neural Network for Dark Matter Morphology](#314-physics-informed-neural-network-for-dark-matter-morphology)|Ashutosh Ojha|Sub-structure classification|Physics-Informed Transformer|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_Physics_Informed_Neural_Network_for_Dark_Matter_Morphology_Ashutosh_Ojha)|
|[Learning Representation Through Self-Supervised Learning on Real Gravitational Lensing Images](#315-learning-representation-through-self-supervised-learning-on-real-gravitational-lensing-images)|Sreehari Iyer|Sub-structure classification|Self-supervised Learning|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_SSL_from_real_dataset_Sreehari_Iyer)|
|[Contrastive Learning vs BYOL](#316-contrastive-learning-vs-byol)|Yashwardhan Deshmukh|Sub-structure classification|Self-supervised Learning|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Deeplens_Self_Supervised_Learning_Yashwardhan_Deshmukh)|
|[Domain Adaptation for Simulation-Based Dark Matter Searches Using Strong Gravitational Lensing](#317-domain-adaptation-for-simulation-based-dark-matter-searches-using-strong-gravitational-lensing)|Marcos Tidball|Sub-structure classification|Unsupervised Domain Adaptation|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Domain_Adaptation_for_DeepLense_Marcos_Tidball)|
|[Equivariant Neural Networks for Dark Matter Morphology with Strong Gravitational Lensing](#318-equivariant-neural-networks-for-dark-matter-morphology-with-strong-gravitational-lensing)|Apoorva Singh|Sub-structure classification|Equivariant Neural Networks|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Equivariant_Neural_Networks_for_DeepLense_Apoorva_Singh)|
|[Equivariant Neural Networks for Dark Matter Morphology with Strong Gravitational Lensing](#318-equivariant-neural-networks-for-dark-matter-morphology-with-strong-gravitational-lensing)|Geo Jolly|Sub-structure classification|Equivariant Neural Networks|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Equivariant_Neural_Networks_for_DeepLense_GEO)|
|[Self-Supervised & Supervised Learning](#319-classification--self-supervised--supervised-learning)|Kartik Sachdev|Sub-structure classification|Self-Supervised Learning|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Physics_Informed_Transformers_For_Dark-Matter_Morphology_Lucas_Jose)|
|[Updating the DeepLense Pipeline](#3110-updating-the-deeplense-pipeline)|Saranga K Mahanta|Sub-structure classification, Anomaly Detection and Regression|Autoencoder Variants|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Updating_the_DeepLense_Pipeline__Saranga_K_Mahanta)|
|[DeepLense Regression](#32-dark-matter-property-estimation-through-regression)|Yurii Halychanskyi|Property estimation through regression|Residual Networks, Transformer Variants|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_Regression_Yurii_Halychanskyi)|
|[DeepLense Regression](#32-dark-matter-property-estimation-through-regression)|Zhongchao Guan|Property estimation through regression|Residual Networks, Transformer Variants|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_Regression_Zhongchao_Guan)|
|[Single Image Super-Resolution with Diffusion Models](#331-single-image-super-resolution-with-diffusion-models)|Atal Gupta|Super-Resolution|Diffusion Models|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Super_Resolution_Atal_Gupta)|
|[Super-Resolution for Strong Gravitational Lensing](#332-super-resolution-for-strong-gravitational-lensing)|Pranath Reddy|Super-Resolution|Residual Networks, Conditional Diffusion Models|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/Super_Resolution_Pranath_Reddy)|
|[Physics-Informed Unsupervised Super-Resolution of Strong Lensing Images](#333-physics-informed-unsupervised-super-resolution-of-strong-lensing-images)|Anirudh Shankar|Super-Resolution|Physics-Informed SISRs|[Click Here](https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_Physics_Informed_Super_Resolution_Anirudh_Shankar)|

### 3.1 Dark matter sub-structure classification

Classification of lensing images into their intrinsic dark matter sub-structure can reveal information about their characteristics and ways to study them. DeepLense presents an extensive array of techniques to study dark matter substructure involving both the simulated and real galaxy datasets.  

#### 3.1.1 DeepLense Classification Using Vision Transformers

**Archil Srivastava** explores the potency of variants of Vision Transformers (EfficientNet, ViT, ConViT, CrossViT, Bottleneck Transformers, EfficientFormer, CoaT, CoAtNet and Swin) on the classification of dark matter substructure on the three datasets, Model 1, 2 and 3.

#### 3.1.2 Unsupervised Domain Adaptation 

**Mriganka Nath** employs unsupervised domain adaptation techniques such as ADDA, AdaMatch and self-ensembling from simulated lensing images to classify dark matter substructure of real lensing images from the Hyper Suprime-Cam (HSC) Subaru Strategic Program Public Data Release 3.

#### 3.1.3 Lensiformer: A Relativistic Physics-Informed Vision Transformer Architecture for Dark Matter Morphology in Gravitational Lensing 
Lucas Jose

#### 3.1.4 Physics Informed Neural Network for Dark Matter Morphology
Ashutosh Ojha

#### 3.1.5 Learning Representation Through Self-Supervised Learning on Real Gravitational Lensing Images

**Sreehari Iyer** evaluates Transformer-based self-supervised learning techniques utilizing the real-world strong gravitational lensing dataset, Model 4. ViT-S and ViT-B are selected as the backbone and SimSiam, DINO and iBOT for self supervised training.

#### 3.1.6 Contrastive Learning vs BYOL

**Yashwardhan Deshmukh** compares the performance of the self-supervised learning techniques, Contranstive Learning and Bootstrap Your Own Latent (BYOL) on the three datasets, Model 1, 2 and 3.

#### 3.1.7 Domain Adaptation for Simulation-Based Dark Matter Searches Using Strong Gravitational Lensing

**Marcos Tidball** performs unsupervised domain adaptation to mitigate the poor generalization of models trained on simulated data to real lensing images using ADDA, Self-Ensemble, CGDM and AdaMatch.
Their work has been published as a [paper](https://iopscience.iop.org/article/10.3847/1538-4357/acdfc7/meta 
) in the Astrophysical Journal.
#### 3.1.8 Equivariant Neural Networks for Dark Matter Morphology with Strong Gravitational Lensing
**Apoorva Singh** and **Geo Jolly** exploit the inherent symmetries present in the strong lensing system (such as rotations and reflections) using equivariant neural networks, to extract dark matter sub-structural information.

#### 3.1.9 Classification : Self-Supervised & Supervised Learning
**Kartik Sachdev** performs benchmarking of an extensive variety of vision transformer architectures, and contrasting of ten supervised and two self-supervised learning frameworks on the classification of dark matter substructure. They use the three datasets, Model 1, 2 and 3.

#### 3.1.10 Updating the DeepLense Pipeline
**Saranga K Mahanta** conducts a study on strong lensing through a variety of tasks on strong lensing images including classification, anomaly detection and regression using several different neural network architectures, on the three datasets, Model 1, 2 and 3. 

### 3.2 Dark matter property estimation through regression

Another means of dark matter study through strong lensing is through the approximation of their properties. **Yurii Halychanskyi** and **Zhongchao Guan** approximate the mass density of vortex substructure of dark matter condensates on the three datasets, Model 1, 2 and 3. 
Yurii uses the ResNet18Hybrid and CmtTi architectures, while Zhongchao demonstres with ResNet18, ViT, CNN-T, MobileNet V2 and CvT-13.

### 3.3 Super-resolution of lensing images

Finally, DeepLense help combat the problem of noisy and low-resolution of real lensing images through various super-resolution techniques. Denoising and upscaling of lensing images can help make their study more accurate.

#### 3.3.1 Single Image Super-Resolution with Diffusion Models
**Atal Gupta** achieves super-resolution of the real-galaxy lensing dataset, Model 4 using a variety of Diffusion Models (DDPM, SR3, SRDiff, ResShift and CG-DPM).

#### 3.3.2 Super-Resolution for Strong Gravitational Lensing
**Pranath Reddy** performs a comparative study of the super-resolution of strong lensing images using Residual Models with Content Loss and Conditional Diffusion Models, on the Model 1 dataset.

#### 3.3.3 Physics-Informed Unsupervised Super-Resolution of Strong Lensing Images
**Anirudh Shankar** explores the unsupervised super-resolution of strong lensing images through a Physics-Informed approach, built to handle sparse datasets. They use custom datasets using different lens models and light profiles. 