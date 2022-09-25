# Updating the DeepLense Pipeline

Strong gravitational lensing is a potential way to learn more about dark matter's basic nature by probing its substructure.
The DeepLense pipeline combines state-of-the-art deep learning models with strong lensing simulations based on lenstronomy. The focus of this project is updating the previous results of DeepLense with new dark matter simulations (Models I, II, III).

 This is a PyTorch-based library for performing Regression, Classification, and Anomaly Detection on the new dark matter simulation models. It is one of the projects under __Google Summer of Code 2022__. For more info on the project [Click Here](https://summerofcode.withgoogle.com/programs/2022/projects/iFKJMj0t)
<br>

## Datasets
The models are trained on mainly 3 datasets consisting of ~30,000 images (single channel) per class. The test sets contain 5000 images. All the dataset consists of 3 classes : 

- axion (vortex)
- no_sub (No substructure)
- cdm (point mass subhalos)

*Note: The Axion files have an extra datapoint corresponding to mass of axion used in simulation. This mass is used for the Regression task on the axion class of the Models.*

All the data are available [here](https://github.com/mwt5345/DeepLenseSim).

### __Model_I__
- 150 x 150 dimensional images
- Modeled with a Gaussian point spread function
- Added background and noise for SNR of around 25

### __Model_II__
- 64 x 64 dimensional images
- Modeled after Euclid observation characteristics as done by default in lenstronomy
- Modeled with simple Sersic light profile

### __Model_III__
- 64 x 64 dimensional images.
- Modeled after HST observation characteristics as done by default in lenstronomy.
- Modeled with simple Sersic light profile

<br>

## Setting up the environment

Clone the repository and enter the project folder.

```
cd https://github.com/ML4SCI/DeepLense.git
cd DeepLense
cd Updating_the_DeepLense_Pipeline__Saranga_K_Mahanta
```

Create a python virtual environment.

```
python -m venv /path/to/new/virtual/environment
```

Activate virtual environment and install the necessary packages from the provided requirements.txt.

```
pip install -r /path/to/requirements.txt
```

You're ready!
More details and instructions to train/test the Model data of specific tasks can be found in their respective folders.
