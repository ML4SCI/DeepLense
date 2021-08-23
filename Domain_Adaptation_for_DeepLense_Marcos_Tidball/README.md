# DeepLense Domain Adaptation
A PyTorch-based collection of Unsupervised Domain Adaptation methods applied to strong gravitational lenses!

This project was created for Google Summer of Code 2021 under the Machine Learning for Science (ML4Sci) umbrella organization.

### Description
A promising means to identify the nature of dark matter is to study it through dark matter halos, and strong gravitational lenses have seen encouraging results in detecting the existence of dark matter substructure. Unfortunately, there isn't a lot of data of strong gravitational lenses available, which means that, if we want to train a machine learning model to identify the different kinds of dark matter substructure, we'd need to use simulations. The problem though, is that a model trained on simulated data does not generalize well to real data, having a very bad performance. This project aims to fix this problem by using Unsupervised Domain Adaptation (UDA) techniques to adapt a model trained on simulated data to real data!

### Blog post
For more about the motivation behind the project and also the Google Summer of Code experience check out [this blog post](https://medium.com/@marcostidball/gsoc-2021-with-ml4sci-domain-adaptation-for-decoding-dark-matter-bf0380898aed).

# Algorithms
There are currently four different UDA algorithms supported:
- ADDA
- Self-Ensemble
- CGDM
- AdaMatch

There is also a normal supervised training algorithm

# Installation
This repo's code can be acessed through the `deeplense_domain_adaptation` package. To install it simply do:
```shell
pip install --upgrade deeplense_domain_adaptation
```

# Data
The data loading pipeline implemented here expects the image data to be in the form of a four dimensional numpy array of shape: `(number_of_images, number_of_channels, height, width)`. Label data is expected to have a shape: `(number_of_images, 1)`.

The dataset used for training and inference will be made available as soon as possible!

# Tutorial
For a tutorial on how to use the `deeplense_domain_adaptation` package check out the tutorial available ate `tutorial.ipynb`. For a more user friendly experience, you can also check the Jupyter Notebook on nbviewer [here](https://nbviewer.jupyter.org/github/zysymu/DeepLense-Domain-Adaptation/blob/main/tutorial.ipynb).

# Current results (ResNet18 model)
Training on source and testing on target:
- Accuracy = 0.6326
- AUROC = 0.8114

After applying UDA algorithms:

| Algorithm |  ADDA  | Self-Ensemble |  CGDM  | AdaMatch |
|-----------|:------:|:-------------:|:------:|:--------:|
| Accuracy  | 0.8449 |     0.7480    | 0.7132 |  0.7300  |
| AUROC     | 0.9442 |     0.8816    | 0.8931 |  0.9179  |