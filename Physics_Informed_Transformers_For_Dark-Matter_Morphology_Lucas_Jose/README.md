
# Lensiformer: a Relativistic Physics-Informed Vision Transformer Architecture for Dark Matter Morphology.


![Google_Summer_of_Code_logo_(2021) svg](https://github.com/SVJLucas/Physics-Informed-Features-For-Dark-Matter-Morphology/assets/60625769/ac669c97-02ef-48cb-877c-1d794a506476)

The Lensiformer architecture, developed as part of the international **Google Summer of Code** program in collaboration with **Machine Learning for Science (ML4SCI)**, serves as a transformer specifically designed for studies in relativistic gravitational lensing.

# Problem Description

The identification of dark matter through the analysis of dark matter halos using strong gravitational lensing is a promising approach in cosmology. However, due to the limited available data, simulations trained on machine learning models offer a potential solution.

However, even in simulations, in many instances, it becomes complex to differentiate between the potential different gravitational anomalies. In this tutorial, we will gain a better understanding of the physics behind this phenomenon, and from it, obtain better features to train artificial intelligence models.

In Machine Learning for Science, we're dealing with two mainly kinds of simulated Dark Matter:

* **Cold Dark Matter (CDM)**: This model suggests that dark matter consists of slow-moving particles. In the CDM paradigm, smaller clusters of dark matter, known as subhalos, are approximated as "point masses." This simplification facilitates computational modeling by treating these subhalos as singular points in the overall distribution of dark matter.

* **No-Substructure Dark Matter**: Unlike the CDM model, the "no-substructure" approach assumes that dark matter is evenly spread out, devoid of any smaller-scale clusters or sub-halos. This stands in stark contrast to the hierarchical structuring and layering of sub-halos within larger halos as predicted by CDM models.

The observable distortions of distant galaxies, known as gravitational lensing, provide an intriguing connection between the types of dark matter and the roles of different galaxies. This phenomenon serves as an illustrative example of how different types of dark matter, despite their elusive nature, can exert gravitational influence and leave noticeable imprints on our observations. Through gravitational lensing, dark matter influences the light path from the source galaxy, causing it to bend around the lensing galaxy. This effect underscores the crucial role of dark matter in determining the large-scale structure of the universe.

# Dataset

In this way, let's simulate 2 cases of Gravitational Lensing:

## Sersic-Tiny Dataset: Approximate the Source Galaxy as the Sersic Profile

The Sersic profile is a mathematical function that describes how the intensity I of the light emitted by a galaxy varies with the distance R from its center. It is widely used in astronomy to characterize the radial brightness profiles of galaxies.

The Sersic profile, for the Source, is given by the formula:

$$
\begin{align}
I_S(x_s,y_s) = I_0 \cdot \exp \left( -b_n \cdot \left(\frac{R(x_s,y_s)}{R_{ser}}\right)^{1/n} \right)
\end{align}
$$

Here, $I_S$ is the intensity at each point $(x_s, y_s)$ based on the Sersic profile of the Source Galaxy, $n$ is the Sersic index of the Galaxy, $R_{ser}$ is the Sersic radius of the Galaxy, $I_0$ is the central surface brightness, $b_n$ related to the  Sersic index $n$ and it is given by:

$$
\begin{align}
b_n \approx 1.999 \cdot n - 0.327
\end{align}
$$

Besides, $R$ is the radii from the center of the ellipse given by:

$$
\begin{align}
R(x_s,y_s) = \frac{1}{q}\sqrt{\frac{{(\cos(\theta) \cdot (x_s-x_0) + \sin(\theta) \cdot (y_s-y_0))^2}}{q^2} + \big({\sin(\theta) \cdot (x_s-x_0) - \cos(\theta) \cdot (y_s-y_0)}\big)^2}
\end{align}
$$

Where  $x_0$ and $y_0$ are the center of the ellipse, $\theta$ is the rotation angle, the axis ratio $q$ (eccentricity).


* **First: A Galaxy as Gravitational Lensing and No-Substructure Dark Matter.**
* **Second: A Galaxy and Cold Dark Matter (CDM) as Gravitational Lensing.**


## Real-Galaxy-Tiny: Using a True Galaxy as Source

![download](https://github.com/SVJLucas/DeepLense/assets/60625769/b37f19e5-c226-4812-86cd-4131d8838488)








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
\boxed{Ψ(x_i,y_i) \approx k \cdot \sqrt{x_i^2+y_i^2}}
\end{align}
$$

By imposing a potential profile, we can now estimate the source position. Instead of using a single correction term $k$ for the entire image, we found that better convergence was achieved when the value of $k$ was localized, leading to a non-uniform distribution. This approach helps capture the localized gravitational distortions of dark matter that we had initially neglected.

To predict the values of $k$, we use the image of the lensed source galaxy and employ the architecture of a ViTSD (Vision Transformer for Small Datasets). We predict a corresponding $k_{ij}$ value for each pixel $(i, j)$, as follow:



![k model](https://github.com/SVJLucas/DeepLense/assets/60625769/8742970e-b50e-4132-b6df-596dc0a3a3ba)


During experiments, we observed that calibrating the values of $k_{ij}$ to fall within a range of 20% more or less than the values predicted by the Singular Isothermal Sphere (SIS) model facilitated faster convergence. Consequently, we opted for a saturation layer to ensure that the correction values remained within this specified range.

With the obtained values $k_{ij}$, we can estimate the potential $\Psi(x_i, y_i)$. Alongside the image, this allows us to solve the gravitational lens equation and generate an estimated image of the source galaxy. This image will then undergo **Shift Patch Tokenization (SPT)** and serve as the input for the decoder.






## Visual Transformers for Small Datasets as Decoder
For the decoder part of the architecture, Lensiformer leverages advances from the paper "Vision Transformer for Small-Size Datasets" by Seung Hoon Lee, Seunghyun Lee, and Byung Cheol Song from Inha University. This paper outlines techniques like Shifted Patch Tokenization (SPT) and Locality Self-Attention (LSA) that effectively counter the Vision Transformer's inherent limitations on small datasets. When applied to Lensiformer, these techniques enable the architecture to learn effectively even from smaller astrophysical datasets.

## Integration of Encoder and Decoder
In Lensiformer, the original lensed image is used as the "query" and "key" for the transformer architecture. This allows the decoder to generate a more accurate and physics-consistent representation of the lensing phenomena. The application of Shifted Patch Tokenization and Locality Self-Attention not only improves the model's performance but also makes it more adaptable to various types of data.


# Results

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




  
