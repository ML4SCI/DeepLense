model_grav.py - contains the NN architecture that is trained to denoise, as well as the noise schedule and the sampling function to generate images using the trained model
train_grav.ipynb - is used to train the NN that is used for denoising steps
sampling.ipynb - is used to sample images based on the 

classiers.py - contains different NN architectures that are used to classify the images into 3 classes depending on the assumed dark matter structure
train_classifier.ipynb - is used to train the classifier, it is used to calculate the FD and assess the performance of the diffusion model

evaluation.ipynb - loads images from the original dataset and the generated ones to compare them and evaluate the FD
