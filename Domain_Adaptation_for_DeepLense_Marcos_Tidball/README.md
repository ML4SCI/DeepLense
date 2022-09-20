# DeepLense Domain Adaptation
A PyTorch-based collection of Unsupervised Domain Adaptation methods applied to strong gravitational lenses.

This project was created for Google Summer of Code 2021 under the Machine Learning for Science (ML4Sci) umbrella organization.

### Paper
A paper describing our methodology was published: [Domain Adaptation for Simulation-Based Dark Matter Searches Using Strong Gravitational Lensing](https://arxiv.org/abs/2112.12121).

### Description
A promising means to identify the nature of dark matter is to study it through dark matter halos, and strong gravitational lenses have seen encouraging results in detecting the existence of dark matter substructure. Unfortunately, there isn't a lot of data of strong gravitational lenses available, which means that, if we want to train a machine learning model to identify the different kinds of dark matter substructure, we'd need to use simulations. The problem though, is that a model trained on simulated data does not generalize well to real data, having a very bad performance. This project aims to fix this problem by using Unsupervised Domain Adaptation (UDA) techniques to adapt a model trained on simulated data to real data!

### Blog post
For more about the motivation behind the project and also my Google Summer of Code experience check out [this blog post](https://medium.com/@marcostidball/gsoc-2021-with-ml4sci-domain-adaptation-for-decoding-dark-matter-bf0380898aed).

# Installation
The implementations can be acessed through the `deeplense_domain_adaptation` package. To install it simply do:
```shell
pip install --upgrade deeplense_domain_adaptation
```

# Data
The data loading pipeline implemented here expects the image data to be in the form of a four dimensional numpy array of shape: `(number_of_images, 1, height, width)`. Label data is expected to have a shape: `(number_of_images, 1)`.

The paper's Model A is our source dataset (less complex simulations) and the paper's Model B is our target dataset (more complex simulations). We have three distinct classes: no dark matter substructure, spherical dark matter substructure and vortex dar matter substructure. On our training sets we have 30'000 images for the source and 30'000 images for the target; in both cases there are 10'000 images per class. On our test sets we have 7'500 images for the source and 7'500 images for the target; in both cases there are 2'500 images per class.

# How to use `deeplense_domain_adaptation`
For a tutorial on how to use the `deeplense_domain_adaptation` package check out `tutorial.ipynb`. If the file isn't loading properly on GitHub you can also check the Jupyter Notebook on nbviewer [here](https://nbviewer.org/github/ML4SCI/DeepLense/blob/main/Domain_Adaptation_for_DeepLense_Marcos_Tidball/tutorial.ipynb). For more information on specific functions/classes check out the documentation available on the functions/classes definitions.

# Before and after UDA
### Equivariant Network model
- Supervised training on source inferring on **source**: accuracy = 97.09; AUROC = 0.996.
- Supervised training on source inferring on **target**: accuracy = 67.53; AUROC = 0.856.

- Applying UDA and inferring on target:

| Algorithm |  ADDA | AdaMatch | Self-Ensemble |
|-----------|:-----:|:--------:|:-------------:|
| Accuracy  | 91.47 |   85.81  |     80.09     |
| AUROC     | 0.980 |   0.960  |     0.939     |

### ResNet-18
- Supervised training on source inferring on **source**: accuracy = 96.84; AUROC = 0.996.
- Supervised training on source inferring on **target**: accuracy = 59.19; AUROC = 0.880.

- Applying UDA and inferring on target:

| Algorithm |  ADDA | AdaMatch | Self-Ensemble |
|-----------|:-----:|:--------:|:-------------:|
| Accuracy  | 85.84 |   75.55  |     76.71     |
| AUROC     | 0.955 |   0.919  |     0.917     |