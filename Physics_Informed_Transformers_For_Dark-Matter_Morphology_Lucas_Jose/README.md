
# Lensiformer: a Relativistic Physics-Informed Vision Transformer Architecture for Dark Matter Morphology.


![Google_Summer_of_Code_logo_(2021) svg](https://github.com/SVJLucas/Physics-Informed-Features-For-Dark-Matter-Morphology/assets/60625769/ac669c97-02ef-48cb-877c-1d794a506476)

The Lensiformer architecture, developed as part of the international **Google Summer of Code** program in collaboration with **Machine Learning for Science (ML4SCI)**, serves as a transformer specifically designed for studies in relativistic gravitational lensing.

# Problem Description

The identification of dark matter through the analysis of dark matter halos using strong gravitational lensing is a promising approach in cosmology. However, due to the limited available data, simulations trained on machine learning models offer a potential solution.

However, even in simulations, in many instances, it becomes complex to differentiate between the potential different gravitational anomalies. In Machine Learning for Science, we're dealing with two mainly kinds of simulated Dark Matter:

* **Cold Dark Matter (CDM)**: This model suggests that dark matter consists of slow-moving particles. In the CDM paradigm, smaller clusters of dark matter, known as subhalos, are approximated as "point masses." This simplification facilitates computational modeling by treating these subhalos as singular points in the overall distribution of dark matter.

* **No-Substructure Dark Matter**: Unlike the CDM model, the "no-substructure" approach assumes that dark matter is evenly spread out, devoid of any smaller-scale clusters or sub-halos. This stands in stark contrast to the hierarchical structuring and layering of sub-halos within larger halos as predicted by CDM models.

The observable distortions of distant galaxies, known as gravitational lensing, provide an intriguing connection between the types of dark matter and the roles of different galaxies. This phenomenon serves as an illustrative example of how different types of dark matter, despite their elusive nature, can exert gravitational influence and leave noticeable imprints on our observations. Through gravitational lensing, dark matter influences the light path from the source galaxy, causing it to bend around the lensing galaxy. This effect underscores the crucial role of dark matter in determining the large-scale structure of the universe.


# Real-Galaxy-Tiny Dataset: Using a True Galaxy as Source Galaxy.

Instead of using a Sersic profile, as we previously did in Machine Learning for Science (ML4SCI), we propose a more realistic approach that utilizes actual galaxies as the source. To accomplish this, we will make use of the [Galaxy10 DECals Dataset](https://github.com/henrysky/Galaxy10), an extensive compilation of high-resolution galaxy images.


To create the **Real-Galaxy-Tiny Dataset**, we can do:

- **1º**  We select a random image of a galaxy from a pre-determined subset of the Galaxy10 DECals Dataset.
- **2º** We extract the red band of the galaxy, as we aim to work with a two-dimensional image.
- **3º**  We rescale and randomly rotate the galaxy; thereafter, the image quality and noise levels are adjusted to resemble Hubble telescope captures closely.
- **4º**  Utilizing the red band of the galaxy, we simulate the relativistic lens equation, taking into account both dark matter patterns and the characteristics of the lensing galaxy.
- **5º**  The labels for each image are assigned as [1,0] for the No-Substructure Dark Matter type and [0,1] for the Cold Dark Matter type.

The Real Galaxy Dataset Pipeline process is as follow:


![Real Galaxy Dataset Pipeline](https://github.com/SVJLucas/DeepLense/assets/60625769/2413a133-2fe5-42d0-bd48-74e071e12183)


To construct the dataset, we perform **1,000** training simulations (comprising 50% No-Substructure Dark Matter and 50% Cold Dark Matter) and **1,000** testing simulations (also split evenly between No-Substructure Dark Matter and Cold Dark Matter).



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


The equation for gravitational lensing, in its dimensionless form, can be given by the following relation:

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


During experiments, we observed that calibrating the values of $k_{ij}$ to fall within a range of 20% more or less than the values predicted by the Singular Isothermal Sphere (SIS) model facilitated faster convergence. Consequently, we opted for a saturation layer to ensure that the correction values remained within this specified range.

With the obtained values $k_{ij}$, we can estimate the potential $\Psi(x_i, y_i)$. Alongside the image, this allows us to solve the gravitational lens equation and generate an estimated image of the source galaxy. This image will then undergo **Shift Patch Tokenization (SPT)** and serve as the input for the decoder.






## Visual Transformers for Small Datasets as Decoder
For the decoder part of the architecture, Lensiformer leverages advances from the paper "Vision Transformer for Small-Size Datasets" by Seung Hoon Lee, Seunghyun Lee, and Byung Cheol Song from Inha University. This paper outlines techniques like Shifted Patch Tokenization (SPT) and Locality Self-Attention (LSA) that effectively counter the Vision Transformer's inherent limitations on small datasets. When applied to Lensiformer, these techniques enable the architecture to learn effectively even from smaller astrophysical datasets.

## Integration of Encoder and Decoder
In Lensiformer, the original lensed image is used as the "query" and "key" for the transformer architecture. This allows the decoder to generate a more accurate and physics-consistent representation of the lensing phenomena. The application of Shifted Patch Tokenization and Locality Self-Attention not only improves the model's performance but also makes it more adaptable to various types of data.


# Experiment

To rigorously evaluate the performance of the Lensiformer model, we plan to train a selection of state-of-the-art vision transformer models on the Real-Galaxy-Tiny Dataset. These selected models will be standardized for a fair comparison by ensuring they have the same number of parameters and transformer blocks. The chosen models for this comparative analysis are leading architectures in the realm of computer vision, and they originate from diverse research entities. Below is a brief overview of the selected models:

* **ViT (Vision Transformer):** Conceived by **Google**, the Vision Transformer (ViT) model employs a pure Transformer architecture that directly processes sequences of image patches. This approach bypasses the need for traditional Convolutional Neural Networks (CNNs) and has been proven to be especially effective when pre-trained on large datasets.

* **ViTSD (Vision Transformer for Small Datasets):** Developed by **Inha University in South Korea**, the ViTSD model is an enhancement of the original ViT architecture. It incorporates Shifted Patch Tokenization and Locality Self-Attention, features that are particularly beneficial for improving performance on datasets of smaller sizes.

* **CvT (Convolutional Vision Transformer):** Created by **Microsoft**, the CvT model blends the advantages of CNNs and Transformers by introducing convolutional components to the standard Vision Transformer architecture. This hybrid approach yields state-of-the-art performance metrics while requiring fewer computational parameters, thereby improving efficiency.

* **CaiT (Class-Attention in Image Transformers):** Originating from **Facebook AI**, the CaiT model focuses on optimizing deep Transformer architectures specifically for the task of image classification. It not only achieves state-of-the-art performance on the ImageNet benchmark but also does so with fewer Floating Point Operations Per Second (FLOPs) and parameters, making it computationally efficient.

  


# Results



<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/97377c06-ae2a-4de7-a657-c3977e98bab1" alt="results" width="2000"/>
</div>


## Training Loss
<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/56cf2e61-7061-43c9-9417-14f6e28448ea" alt="Loss" width="500"/>
</div>

## Accuracy
<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/8c33a0f8-8719-4102-ad8d-a793c9dda475" alt="accuracy" width="500"/>
</div>

## ROC-Curve
<div align="center">
  <img src="https://github.com/SVJLucas/DeepLense/assets/60625769/a514ada0-7485-492e-a63c-88759aeb22cd" alt="auc" width="500"/>
</div>


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





  
