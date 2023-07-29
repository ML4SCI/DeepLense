# Physics Informed Features For Dark Matter Morphology



![Google_Summer_of_Code_logo_(2021) svg](https://github.com/SVJLucas/Physics-Informed-Features-For-Dark-Matter-Morphology/assets/60625769/ac669c97-02ef-48cb-877c-1d794a506476)




The Physics Informed Features module, created under the international **Google Summer of Code program**, in partnership with the institution **Machine Learning for Science (ML4SCI)**, is a complex tool designed for the generation and manipulation of data in relativistic gravitational lensing studies.


# Problem Description

The identification of dark matter through the analysis of dark matter halos using strong gravitational lensing is a promising approach in cosmology. However, due to the limited available data, simulations trained on machine learning models offer a potential solution.

However, even in simulations, in many instances, it becomes complex to differentiate between the potential different gravitational anomalies. In this tutorial, we will gain a better understanding of the physics behind this phenomenon, and from it, obtain better features to train artificial intelligence models.

In Machine Learning for Science, we're dealing with two mainly kinds of simulated Substructure of Dark Matter:

* **Axion Dark Matter**: Axions are hypothetical particles suggested as a potential form of dark matter. Unlike typical dark matter candidates, axions are extremely light and weakly interactive with ordinary matter and radiation. They can produce observable effects in the universe like vortices, rotating solutions of field equations.

* **Cold Dark Matter (CDM)**: is a type of slow-moving dark matter proposed to explain various cosmic phenomena. Within the CDM framework, subhalos, or small dense concentrations of dark matter, are treated as "point masses". This simplification allows easier computation by treating them as singularities in the dark matter distribution.


The observable distortions of distant galaxies, known as gravitational lensing, provide an intriguing connection between the types of dark matter and the roles of different galaxies. This phenomenon serves as an illustrative example of how different types of dark matter, despite their elusive nature, can exert gravitational influence and leave noticeable imprints on our observations. Through gravitational lensing, dark matter influences the light path from the source galaxy, causing it to bend around the lensing galaxy. This effect underscores the crucial role of dark matter in determining the large-scale structure of the universe.

In this way, let's simulate 3 cases of Gravitational Lensing:

* **First: Only a Galaxy as Gravitational Lensing and no Dark Matter.**
* **Second: A Galaxy and Cold Dark Matter (CDM) as Gravitational Lensing.**
* **Third: A Galaxy and Axion Dark Matter as Gravitational Lensing.**

Let's see the results:

![gravitationallensing](https://github.com/SVJLucas/Physics-Informed-Features-For-Dark-Matter-Morphology/assets/60625769/180f5ef9-eac4-4989-9d24-8b5e9f90b31a)

As can be seen, distinguishing among different types of dark matter, or even between the presence or absence of dark matter in observations, can be a complex task, even for a trained eye. Let's see how the PhysicsInformedFeatures can help us to solve this problem.


# Physics Informed Features

Here we introduce the new PhysicsInformedFeatures Module, designed to extract new features from images of astronomical simulations, thereby simplifying the classification process.

Now, the new module performs the following tasks:

* **Reconstructs the source galaxy image**: This task is based on the hypothesis that the gravitational effects of dark matter are negligible in most regions of space, and that the lensing galaxy can be approximated by a Singular Isothermal Sphere.

* **Obtains the approximate parameters of the Sersic Profile of the source galaxy**: Upon reconstructing the source galaxy, an attempt is made to approximate it with a Sersic profile, and the parameters of this distribution are obtained.

* **Gravitational Anomalies Reconstruction**: Using the source galaxy profile and the Sersic approximation, we can derive the profile of gravitational distortions based on the differences in intensity and the estimated Sersic profile.

* **Resize the images**: The module does interpolation to resize the given images to get better resolution.

* **Plotting of Results**: The module itself comes equipped with functions to facilitate the plotting of results.

To understand how to use these features, please refer to the Jupyter notebook 'example.ipynb' in the "examples" Folder which provides various examples of usage.

# Results

First, we may remember the following:

- **No Substructure**: The distortion will be centered on the galaxy acting as a lens. Therefore, once we are aligned, it will be near the center of the image.

- **Cold Dark Matter (CDM)**: Small dense concentrations of dark matter are treated as "point masses," producing gravitational distortions in different locations on the image and in a circular shape.

- **Axion Dark Matter**: They can produce observable effects such as vortices, which result in gravitational distortions appearing as "lines".

We can use then the **Physics Informed Features** module:

![nosubs](https://github.com/SVJLucas/Physics-Informed-Features-For-Dark-Matter-Morphology/assets/60625769/9affecce-ff7f-4597-aabc-710cc84c423e)

![cdms](https://github.com/SVJLucas/Physics-Informed-Features-For-Dark-Matter-Morphology/assets/60625769/abd1f36f-bba9-498e-bc2e-eeef8226d4d7)

![axions](https://github.com/SVJLucas/Physics-Informed-Features-For-Dark-Matter-Morphology/assets/60625769/6a548325-7263-4eb7-949c-351368611877)


Now, considering the characteristics of each type of dark matter, the images can be easily distinguished from one another due to the development of the feature, Gravitational Distortion. This feature necessitates the existence of other features for its creation.

# How it works?

## Reconstruction of the Source Galaxy

The equation for gravitational lensing, in its dimensionless form, can be given by the following relation:

$$
\begin{align}
\mathcal{S}(x_s,y_s) = \mathcal{I}(x_i,y_i) - \nabla Ψ(x_i,y_i)
\end{align}
$$

In this equation, $\mathcal{S}(x_s,y_s)$ represents the dimensionless source vector position in the source plane, which corresponds to the position of the source galaxy. On the other hand, $\mathcal{I}(x_i,y_i)$ represents the dimensionless image vector position in the image plane, which corresponds to the image we observe. Finally, $\nabla Ψ(x_i,y_i)$ represents the gradient of the dimensionless gravitational potential produced by the lens, which in our case, includes both the lensing galaxy and the possible dark matter.

Observe that this equation involves three unknowns: the source position $\mathcal{S}$, the image position $\mathcal{I}$, and the gravitational potential of the system $\nabla Ψ(x_i,y_i)$. Yet, we only have knowledge of the produced image $\mathcal{I}$.

In order to estimate the position of the source galaxy, we need to make several assumptions about the potential of the system (i.e., the lensing galaxy plus dark matter). These are as follows:

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

By imposing a potential profile, we can now estimate the source position. We must then estimate the value of the constant $k$. For the simulation (DeepLens Models I, II, III), a value of $k \approx 1$ works well.

Following this process, we are able to obtain a reconstructed image $\hat{I}_{\hat{\mathcal{S}}}(x_s,y_s)$ of the source.

## Approximate the Source Galaxy as the Sersic Profile

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

To find these parameters, we can do first:

$$
\begin{align}
I_0 = max_{x_s,y_s}\Big( I_S(x_s,y_s) \Big)
\end{align}
$$

Then, we can use the reconstructed galaxy and minimize the error of the model and the galaxy itself:


The values  are determined by minimizing:

$$
\begin{align}
x_0^{\*},y_0^{\*},\theta^{\*},q^{\*},n^{\*},R_{ser}^{\*} = \text{minimize}\ \Sigma_{x_s,y_s} ||sersic_{model}(x_s,y_s|x_0,y_0,\theta,q,n,R_{ser},I_0) - sersic_{observed}(x_s,y_s)||^2
\end{align}
$$

And then:

$$
\begin{align}
\boxed{x_0^{\*},y_0^{\*},\theta^{\*},q^{\*},n^{\*},R_{ser}^{\*} = \text{minimize}\ \Sigma_{x_s,y_s} ||I_S(x_s,y_s|x_0,y_0,\theta,q,n,R_{ser},I_0) - \hat{I}_{\hat{\mathcal{S}}}(x_s,y_s)||^2}
\end{align}
$$


## Gravitational Anomalies Reconstruction

Now that we have an estimated clean version of the source galaxy, we can use the obtained parameters to find an **Anisotropic Eikonal Partial Differential Equation** that allows us to obtain the potential:

$$
\begin{align}
∇P(x_i,y_i)^T Z(\theta^{\*}, q^{\*}) ∇P(x_i,y_i) = \frac{R_{ser}^{\* 2}}{b_n^{\* 2n\*}}ln^{2n*}\Bigg(\frac{I_0}{I_{\mathcal{I}}(x_i,y_i)}\Bigg)
\end{align}
$$

With the boundary condition of:

$$
\begin{align}
P(0,0) = 0
\end{align}
$$

Where $Z(\theta, q) \in \mathbb{R}^{2\times2}$ is a positive definite matrix that only depends on $\theta$ and $q$ and $I_{\mathcal{I}}(x_i,y_i)$ is the light intensity of the formed image, and, then, the image itself.

Moreover, we also have:

$$
\begin{align}
\Psi(x_i,y_i) = \frac{x_i^2+y_i^2}{2}-P(x_i,y_i)-(x_0x_i+y_0y_i)
\end{align}
$$

To avoid solving the partial differential equation, since we only wish to get a glimpse of the distortions, we can do:

$$
\begin{align}
∇\Psi(\vec{x_i}) = \vec{x_i}-∇P(\vec{x_i})-\vec{x_0}
\end{align}
$$

Which means:

$$
\begin{align}
∇\Psi(\vec{x_i}) + ∇P(\vec{x_i})= \vec{x_i}-\vec{x_0}
\end{align}
$$

Therefore, if the gradient $∇\Psi(\vec{x_i})$ changes abruptly (and therefore near a dark matter singularity) near a given point $\vec{x_i}$, then $∇P(\vec{x_i})$ will also change abruptly, in the opposite direction, to mantain the equality $∇\Psi(\vec{x_i}) + ∇P(\vec{x_i})= \vec{x_i}-\vec{x_0}$, which is approximately constant near $\vec{x_i}$.

The product, therefore, $∇P(x_i,y_i)^T Z(\theta, q) ∇P(x_i,y_i)$ should also change abruptly, once $∇P(x_i,y_i)$ itself change abruptly,  which leads to the function:

$$
\begin{align}
\boxed{\mathcal{F}(x_i,y_i) = \frac{R_{ser}^{\* 2}}{b_n^{\* 2n\*}}ln^{2n*}\Bigg(\frac{I_0}{I_{\mathcal{I}}(x_i,y_i)}\Bigg)}
\end{align}
$$

Changing abruptly as well. To find gravitational distortion points, we simply find the derivatives of the function $\mathcal{F}$ on both the x and y axes. Once our image has very low resolution, we'll use fifth-order finite difference to approximate the derivative:

$$
\begin{align}
df_i \approx \frac{-25f_i + 48f_{i+1} - 36f_{i+2} + 16f_{i+3} - 3f_{i+4}}{12h}
\end{align}
$$


Thus, the anomaly image can be obtained by:

$$
\begin{align}
\mathcal{A}_{raw}(x_i,y_i) = \frac{ \partial^2 \mathcal{F}}{\partial x \partial y}(x_i,y_i)
\end{align}
$$

However, to better visualize the final result, avoiding numerical errors due to the high absolute values of the potential near the singularities, as well as numerical errors in the calculation of the derivatives, we will make a post procees by doing the following transformation:

$$
\begin{align}
\boxed{\mathcal{A}(x_i,y_i) = \Bigg|Tanh\Bigg(\frac{1}{F_Q}\frac{ \partial^2 \mathcal{F}}{\partial x \partial y}(x_i,y_i)\Bigg)\Bigg|}
\end{align}
$$

Where $F_Q$ is a normalization constant.

You can get any of $\mathcal{A}_{raw}(x_i,y_i)$ or $\mathcal{A}(x_i,y_i)$ just changing the boolean *post_process* in one of the module functions.

# Next Steps

* Implement the Fast Marching method, analogous to Dijkstra's algorithm, to solve the non-linear **Anisotropic Eikonal Partial Differential Equation**: $∇P(x_i,y_i)^T Z(\theta, q) ∇P(x_i,y_i) = \frac{R_{ser}^2}{b_n^{2n}}ln^{2n}\Bigg(\frac{I_0}{I_{\mathcal{I}}(x_i,y_i)}\Bigg)$, with the initial condition $P(0,0) = 0$. This approach aims to enhance the resolution of distortions in scenarios where it is challenging to distinguish between the presence of Axion Dark Matter and the absence of Dark Matter.
  
* Refactor the created module, ensuring that all functions are not confined within a single class. Instead, distribute them across different files to facilitate reusability and intelligent code versioning.
  
