
# Lensiformer: a Relativistic Physics-Informed Vision Transformer Architecture for Dark Matter Morphology.


![Google_Summer_of_Code_logo_(2021) svg](https://github.com/SVJLucas/Physics-Informed-Features-For-Dark-Matter-Morphology/assets/60625769/ac669c97-02ef-48cb-877c-1d794a506476)

The Lensiformer architecture, developed as part of the international **Google Summer of Code** program in collaboration with **Machine Learning for Science (ML4SCI)**, serves as a transformer specifically designed for studies in relativistic gravitational lensing.

# Problem Description

The identification of dark matter through the analysis of dark matter halos using strong gravitational lensing is a promising approach in cosmology. However, due to the limited available data, simulations trained on machine learning models offer a potential solution.

However, even in simulations, in many instances, it becomes complex to differentiate between the potential different gravitational anomalies. In Machine Learning for Science (ML4SCI), we're dealing with two mainly kinds of simulated Dark Matter:

* **Cold Dark Matter (CDM)**: This model suggests that dark matter consists of slow-moving particles. In the CDM paradigm, smaller clusters of dark matter, known as subhalos, are approximated as "point masses." This simplification facilitates computational modeling by treating these subhalos as singular points in the overall distribution of dark matter.

* **No-Substructure Dark Matter**: Unlike the CDM model, the "no-substructure" approach assumes that dark matter is evenly spread out, devoid of any smaller-scale clusters or sub-halos. This stands in stark contrast to the hierarchical structuring and layering of sub-halos within larger halos as predicted by CDM models.

The observable distortions of distant galaxies, known as gravitational lensing, provide an intriguing connection between the types of dark matter and the roles of different galaxies. This phenomenon serves as an illustrative example of how different types of dark matter, despite their elusive nature, can exert gravitational influence and leave noticeable imprints on our observations. **Through gravitational lensing, dark matter influences the light path from the source galaxy, causing it to bend around the lensing galaxy.** This effect underscores the crucial role of dark matter in determining the large-scale structure of the universe. A detailed illustration that facilitates a deeper understanding of the phenomenon can be found below:


<br>


<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/260b4df2-414a-4825-94c5-cba4ffa02742" alt="Phenomenon Description" width="900"/>
</div>
<div align="center">
Gravitational Lensing Phenomenon Description: In this figure, the distorted observation of the galaxy serves as an example of an Einstein Ring. This phenomenon occurs when the gravitational field of a massive object bends the light emanating from a more distant object. Source: Bell Labs and the Hubble Space Telescope. 
</div>
<br>

Consequently, given that the type of dark matter present in the galaxy serving as the lens plays a crucial role in shaping the resulting image, **the primary motivation for this research is to simulate the phenomenon and develop Deep Learning models capable of capturing the differences between No-Substructure Dark Matter and Cold Dark Matter**.**Beyond that, the overarching objective is to establish a novel problem architecture that is able to incorporate domain-specific knowledge as well as the relevant relativistic equations**.


<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/e0911866-0fe8-4567-8038-4b164c3139f9" alt="Description" width="750"/>
</div>
<div align="center">
Simulations for both No-Substructure Dark Matter and Cold Dark Matter are conducted for the same source galaxy, positioned identically and at the same angle relative to the observer.
</div>
<br>


# Real-Galaxy-Tiny Dataset: Using a True Galaxy as Source Galaxy.

Instead of using a Sersic profile, as we previously did in Machine Learning for Science (ML4SCI), **we propose a more realistic approach that utilizes actual galaxies as the source**. To accomplish this, we will make use of the [Galaxy10 DECals Dataset](https://github.com/henrysky/Galaxy10), an extensive compilation of high-resolution galaxy images.


To create the **Real-Galaxy-Tiny Dataset**, we can do:

- **1º**  We select a random image of a galaxy from a pre-determined subset of the Galaxy10 DECals Dataset.
- **2º** We extract the red band of the galaxy, as we aim to work with a two-dimensional image.
- **3º**  We resize and randomly rotate the galaxy image. Subsequently, we adjust the image quality and noise levels to closely emulate the characteristics of captures taken by the Hubble telescope when imaging distant galaxies, which inherently results in a significant reduction in image quality.
- **4º**  Utilizing the red band of the galaxy, we simulate the relativistic lens equation, taking into account both dark matter patterns and the characteristics of the lensing galaxy.
- **5º**  The labels for each image are assigned as [1,0] for the No-Substructure Dark Matter type and [0,1] for the Cold Dark Matter type.

The Real Galaxy Dataset Pipeline process is as follow:

![Real Galaxy Dataset Pipeline](https://github.com/SVJLucas/DeepLense/assets/60625769/87fd7c93-7189-46a7-ae56-2b67e7a5ab29)


To construct the dataset, we perform **1,000** training simulations (comprising 50% No-Substructure Dark Matter and 50% Cold Dark Matter) and **1,000** testing simulations (also split evenly between No-Substructure Dark Matter and Cold Dark Matter).

The Real-Galaxy-Tiny Dataset is constrained in size due to the computational rigor required for simulating realistic galaxies and dark matter. Additionally, the dataset seeks to emulate the real-world complexity of manually identifying gravitational distortions in astrophysical images, a task that naturally comes with a limited number of cataloged examples. Hence, the dataset's limited size is a deliberate reflection of these multifaceted challenges, ensuring both high-quality data and alignment with real-world conditions.

You can generate the dataset using the notebook **create_dataset.ipynb** or [download the pre-generated dataset](https://drive.google.com/file/d/1ZNZE4pLcAsY8C-cp4efNNHsyDrNYNMaB/view?usp=sharing).

# Lensiformer

The exploration of dark matter and its interaction with the universe is one of the most intriguing and challenging subjects in modern astrophysics. Traditional machine learning models have been used to varying degrees of success to analyze complex morphologies and data patterns. However, these models often **lack the ability to incorporate domain-specific knowledge**, particularly the principles of relativistic physics that govern the universe's behavior.

Lensiformer aims to bridge this gap by offering a novel approach to the study of dark matter morphologies through gravitational lensing. Leveraging a transformer architecture, Lensiformer incorporates principles of relativistic physics directly into its design, creating a more coherent and accurate representation of the underlying phenomena.



![Vit-ViTSD-Lensiformer](https://github.com/SVJLucas/DeepLense/assets/60625769/4cfff8d6-cf56-4e7e-a3b3-6833b4bedf88)

The Lensiformer architecture brings together the best of both worlds: the physics-informed rigor of gravitational lensing and the machine learning prowess of [Vision Transformers designed for small datasets](https://arxiv.org/abs/2112.13492). This is achieved through a two-pronged approach consisting of a specialized encoder and decoder.

## Relativistic Physics-Informed Encoder


The encoder is rooted in the principles of relativistic physics. It employs the **relativistic lens equation in conjunction with a Singular Isothermal Spherical (SIS) model as an ansatz**. This approach is used to approximate the gravitational potential exerted by the galaxy acting as the lens, as well as by the dark matter. Subsequently, this information is used to **reconstruct the source galaxy that is being lensed**.


<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/02a4dad4-f6f4-40a2-85ce-0eaffe12c718" alt="Encoder" width="350"/>
</div>


In the Inverse Lens Layer, we use the equation for gravitational lensing, in its dimensionless form, that is given by the following relation:

$$
\begin{align}
\vec{\mathcal{S}} = \vec{\mathcal{I}} - \vec{\nabla} \Psi (\vec{\mathcal{I}})
\end{align}
$$

In this equation, $\vec{\mathcal{S}}=(x_s,y_s)$ represents the dimensionless source vector position in the source plane, which corresponds to the position of the source galaxy. On the other hand, $\vec{\mathcal{I}}=(x_i,y_i)$ represents the dimensionless image vector position in the image plane, which corresponds to the image we observe. Finally, $\vec{\nabla} \Psi (\vec{\mathcal{I}}) = \big(\Psi_x(x_i,y_i),\Psi_y(x_i,y_i)\big)$ represents the gradient of the dimensionless gravitational potential produced by the lens, which in our case, includes both the lensing galaxy and the possible dark matter.

Observe that this equation involves three unknowns: the source position $\vec{\mathcal{S}}$, the image position $\vec{\mathcal{I}}$, and the gravitational potential of the system $\Psi(x_i,y_i)$. Yet, we only have knowledge of the produced image $\vec{\mathcal{I}}$.

In order to estimate the position of the source galaxy $\vec{\mathcal{S}}$, we need to make several assumptions about the potential of the system (i.e., the lensing galaxy plus dark matter). These are as follows:

The gravitational potential can be computed by:

$$
\begin{align}
Ψ(x_i,y_i) = Ψ_{Galaxy}(x_i,y_i) + Ψ_{DarkMatter}(x_i,y_i)
\end{align}
$$

The distortions caused by dark matter are localized and can be ignored in most of the formed image. Therefore, we will approximate:

$$
\begin{align}
Ψ(x_i,y_i) \approx Ψ_{Galaxy}(x_i,y_i)
\end{align}
$$

Given that we don't know the profile of the galaxy, we will assume a Singular Isothermal Sphere (SIS) model, with a proportionality parameter $k$ to correct potential distortions.

$$
\begin{align}
Ψ_{Galaxy}(x_i,y_i) \approx k \cdot \sqrt{x_i^2+y_i^2}
\end{align}
$$

Hence:

$$
\begin{align}
Ψ(x_i,y_i) \approx k \cdot \sqrt{x_i^2+y_i^2}
\end{align}
$$

By imposing a potential profile, we can now estimate the source position. Instead of using a single correction term $k$ for the entire image, we found that better convergence was achieved when the value of $k$ was localized, leading to a non-uniform distribution. This approach helps capture the localized gravitational distortions of dark matter that we had initially neglected. Then:

$$
\begin{align}
\boxed{Ψ(x_i,y_i) \approx k(x_i,y_i) \cdot \sqrt{x_i^2+y_i^2}}
\end{align}
$$

To predict the values of $k$, we use the image of the lensed source galaxy and employ the architecture of a ViTSD (Vision Transformer for Small Datasets). We predict a corresponding $k_{ij}$ value for each pixel $(i, j)$, as follow:



![k model](https://github.com/SVJLucas/DeepLense/assets/60625769/8742970e-b50e-4132-b6df-596dc0a3a3ba)


During experiments, we observed that calibrating the values of $k_{ij}$ to fall within a range of **20%** more or less than the values predicted by the Singular Isothermal Sphere (SIS) model facilitated faster convergence. Consequently, we opted for a saturation layer to ensure that the correction values remained within this specified range.

With the obtained values $k_{ij}$, we can estimate the potential $\Psi(x_i, y_i)$. Alongside the image, this allows us to solve the gravitational lens equation and generate an estimated image of the source galaxy. This image will then undergo **Shift Patch Tokenization (SPT)** and serve as the input for the decoder.






## Visual Transformers for Small Datasets as Decoder
For the decoder part of the architecture, Lensiformer leverages advances from the paper "Vision Transformer for Small-Size Datasets" by Seung Hoon Lee, Seunghyun Lee, and Byung Cheol Song from Inha University. This paper outlines techniques like Shifted Patch Tokenization (SPT) and Locality Self-Attention (LSA) that effectively counter the Vision Transformer's inherent limitations on small datasets. When applied to Lensiformer, these techniques enable the architecture to learn effectively even from smaller astrophysical datasets.

## Integration of Encoder and Decoder
In Lensiformer, the original lensed image is used as the "query" and "key" for the transformer architecture. This allows the decoder to generate a more accurate and physics-consistent representation of the lensing phenomena. The application of Shifted Patch Tokenization and Locality Self-Attention not only improves the model's performance but also makes it more adaptable to various types of data.


# Experiment

To rigorously evaluate the performance of the Lensiformer model, we plan to train a selection of state-of-the-art vision transformer models on the **Real-Galaxy-Tiny Dataset**. These selected models will be standardized for a fair comparison by ensuring they have the same number of parameters and transformer blocks. The chosen models for this comparative analysis are leading architectures in the realm of computer vision, and they originate from diverse research entities. Below is a brief overview of the selected models:

* **ViT (Vision Transformer):** Conceived by **Google**, the Vision Transformer (ViT) model employs a pure Transformer architecture that directly processes sequences of image patches. This approach bypasses the need for traditional Convolutional Neural Networks (CNNs) and has been proven to be especially effective when pre-trained on large datasets.

* **ViTSD (Vision Transformer for Small Datasets):** Developed by **Inha University in South Korea**, the ViTSD model is an enhancement of the original ViT architecture. It incorporates Shifted Patch Tokenization and Locality Self-Attention, features that are particularly beneficial for improving performance on datasets of smaller sizes.

* **CvT (Convolutional Vision Transformer):** Created by **Microsoft**, the CvT model blends the advantages of CNNs and Transformers by introducing convolutional components to the standard Vision Transformer architecture. This hybrid approach yields state-of-the-art performance metrics while requiring fewer computational parameters, thereby improving efficiency.

* **CaiT (Class-Attention in Image Transformers):** Originating from **Facebook AI**, the CaiT model focuses on optimizing deep Transformer architectures specifically for the task of image classification. It not only achieves state-of-the-art performance on the ImageNet benchmark but also does so with fewer Floating Point Operations Per Second (FLOPs) and parameters, making it computationally efficient.


To provide a more comprehensive understanding of each model's specifications and settings, we have compiled the following table. It outlines the number of parameters, the number of transformer blocks used, the learning rate, and the optimizer for each model.

  
|      Model Name      | Number of Parameters | Transformer Blocks | Num Epochs | Learning Rate | Optimizer |
|:--------------------:|:------------------------:|:------------------:|:----------:|:-------------:|:---------:|
| Class-Attention in Image Transformers (Facebook AI) | 13M |         2         |    3000    |     5e-7     |   AdamW   |
| Convolutional Vision Transformer (Microsoft)       | 13M |         2         |    3000    |     5e-7     |   AdamW   |
| Vision Transformer (Google)                       | 13M |         2         |    3000    |     5e-7     |   AdamW   |
| Vision Transformer for Small Datasets (Inha University in South Korea) | 13M | 2  | 3000       |   5e-7     |   AdamW   |
| Lensiformer (Ours)                                | 13M |         2         |    3000    |     5e-7     |   AdamW   |

To access specific parameters for each model, refer to the notebook **training_and_compare_models.ipynb**, where the models are defined and trained. The implementation of the various Lensiformer models was carried out with the assistance of the [vit-pytorch project](https://github.com/lucidrains/vit-pytorch).


# Results

To delineate the strengths of the Lensiformer model in comparison to other state-of-the-art vision transformer architectures, we conducted an extensive evaluation on the Real-Galaxy-Tiny Dataset. Our analysis is predicated on a set of performance metrics that are important for classification tasks in the realm of astrophysics, namely Accuracy, ROC AUC, and F1 Score. Each model was subjected to the same training regimen to ensure an unbiased comparison. The ensuing table elucidates the performance metrics for Lensiformer alongside other leading models in the field.


|      Model Name      | Accuracy | ROC AUC | F1 (No Substructure Dark Matter) | F1 (Cold Dark Matter) |
|:--------------------:|:--------:|:-------:|:-------------------------------:|:---------------------:|
| Class-Attention in Image Transformers (Facebook AI) | 0.71 | 0.775 | 0.76 | 0.63 |
| Convolutional Vision Transformer (Microsoft)       | 0.73 | 0.796 | 0.73 | 0.72 |
| Vision Transformer (Google)                       | 0.78 | 0.841 | 0.78 | 0.77 |
| Vision Transformer for Small Datasets (Inha University in South Korea) | 0.75 | 0.819 | 0.76 | 0.74 |
| Lensiformer (Ours)                                | **0.80** | **0.865** | **0.80**| **0.79** |

Table above provides a comprehensive summary of the performance metrics, including F1 scores for each class ('No Substructure Dark Matter' and 'Cold Dark Matter').  you wish, you may [download the weights of the various models here](https://drive.google.com/file/d/1bWU8bMalSkLN9GAhjl0bcP0d0uie7cw0/view?usp=sharing).



## Training Loss

Upon examining the training loss in the graph bellow, distinct convergence patterns emerge for each model. The Class-Attention in Image Transformers (CaiT) model shows a relatively slow reduction in loss, which is consistent with its lower performance metrics. In contrast, the Convolutional Vision Transformer (CvT) model converges rapidly, with its loss reaching near-zero levels by the 2500th epoch.

The Lensiformer model starts with a loss similar to other models but shows a more gradual and consistent decline. This suggests that the model is learning effectively, albeit at a slower pace, which is corroborated by its higher accuracy and ROC AUC values in the final evaluation.

<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/56cf2e61-7061-43c9-9417-14f6e28448ea" alt="Loss" width="500"/>
</div>

The Vision Transformer (ViT) and Vision Transformer for Small Datasets (ViTSD) both show fast convergence with significant reductions in loss by the 2500th epoch, aligning with their competitive performance metrics.

Overall, the Lensiformer model's slower but consistent loss reduction may be indicative of its ability to generalize better, especially in scenarios with limited data. This is likely a result of its domain-specific knowledge, which becomes increasingly valuable in such contexts.

## Accuracy

The Lensiformer model outperforms its competitors with an accuracy of 80%. While Vision Transformer (Google) and Convolutional Vision Transformer (Microsoft) exhibit competitive accuracies of 78% and 73% respectively, Lensiformer's superior accuracy indicates its efficacy in correctly identifying the two classes of dark matter morphologies.

<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/8c33a0f8-8719-4102-ad8d-a793c9dda475" alt="accuracy" width="500"/>
</div>

In the graph, it is evident that up to approximately 700 epochs, the Vision Transformer (ViT), Convolutional Vision Transformer (CvT), and Lensiformer models all exhibit competitive performance with closely matched accuracy levels. In contrast, the Class-Attention in Image Transformers (CaiT) model lags behind in this metric. However, beyond the 700-epoch mark, Lensiformer experiences a rapid acceleration in learning. This enhanced rate of learning is attributable to Lensiformer's domain-specific knowledge, which proves to be particularly valuable in scenarios with limited sample sizes.


## ROC-Curve

Furthermore, Lensiformer achieves an ROC AUC score of 0.87, the highest among all models. The ROC AUC score is particularly important in this context as it demonstrates the model's capability to differentiate between the classes across various decision thresholds. This suggests that Lensiformer possesses a robust discriminatory power, which is imperative for applications in astrophysics where false positives can have significant implications.

<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/a514ada0-7485-492e-a63c-88759aeb22cd" alt="auc" width="500"/>
</div>

Besides, The F1 score for both classes ('No Substructure Dark Matter' and 'Cold Dark Matter') is 0.80 and 0.79, respectively. These scores further affirm the model's balanced performance across both classes, which is crucial for ensuring that the model does not exhibit a bias towards a particular class.

## __Citation__

* [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929v2)

  ```bibtex
  @misc{dosovitskiy2021image,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale}, 
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2021},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
  ```


* [Vision Transformer for Small-Size Datasets (ViTSD)](https://arxiv.org/abs/2112.13492)

  ```bibtex
  @misc{lee2021vision,
      title={Vision Transformer for Small-Size Datasets}, 
      author={Seung Hoon Lee and Seunghyun Lee and Byung Cheol Song},
      year={2021},
      eprint={2112.13492},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
  ```

* [Convolutional Transformer (CvT)](https://arxiv.org/abs/2103.15808)
  ```bibtex
  @article{wu2021cvt,
    title={Cvt: Introducing convolutions to vision transformers},
    author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},
    journal={arXiv preprint arXiv:2103.15808},
    year={2021}
  }
  ```

* [Class-Attention in Image Transformers (CaiT)](https://arxiv.org/abs/2103.17239)
  ```bibtex
  @misc{touvron2021going,
      title   = {Going deeper with Image Transformers}, 
      author  = {Hugo Touvron and Matthieu Cord and Alexandre Sablayrolles and Gabriel Synnaeve and Hervé Jégou},
      year    = {2021},
      eprint  = {2103.17239},
      archivePrefix = {arXiv},
      primaryClass = {cs.CV}
  }
  ```





  
