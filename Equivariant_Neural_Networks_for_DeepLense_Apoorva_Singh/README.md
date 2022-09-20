[![](https://img.shields.io/badge/license-MIT-green)](https://https://github.com/Apoorva99/GSOC-Equivariant-Network/blob/main/LICENSE.md)


# Equivariant Neural Networks for Dark Matter Morphology with Strong Gravitational Lensing

* Current convolutional neural networks are only capable of translational equivariance. However, in a number of application (including ours), a larger groups of symmetries, including rotations and reflections are present in the data as well that needs to be exploited. This gives rise to the notion of Equivariant Convolutional Networks.
* E2-CNNs (proposed by Weiler et. al) propose a solution to this by guaranteeing a specified transformation behavior of their feature spaces under transformations of their input.
* E2-CNNs are equivariant under all isometries E(2) of the image plane i.e. under translations, rotations and reflections.
* We design a network that uses equivariant convolutional layers instead of a vanilla convolutional layer to capture the symmetries present in the data.

# Datasets

* We use two different datasets to train and test our network:
  * **Model F** (corresponding to model A in the paper) 
  * **Model J** (corresponding to model B in the paper) 
* The paper mentioned above refers to Alexander, Stephon, et al. "[Deep Learning the Morphology of Dark Matter Substructure](https://arxiv.org/abs/1909.07346)." The Astrophysical Journal 893.1 (2020): 15.
* Both model F and J contain 75000 images for training and 7500 images for testing. The images are evenly divided with each of the three class having 25000 training images and 2500 testing images.

# File Description

* **create_env.sh:** Bash file to create a conda environment and install all the dependencies needed.
* **Example.ipynb:** A sample notebook for the code of equivariant network.
* **dataset.py:** Dataloader to load the data for model_f and model_j.
* **main.py:** Script for loading the data, initialising the model and training the network.
* **model.py:** Contains the models to be used (Equivariant Network).
* **trainer.py:** Script for training the model and saving the results in the generated directory.
* **resnet_model.py:** Script containing the code for ResNet-18 model. 

# Training the Network

```
$ python main.py [ARGS]
```
You can set the parameters of the model using the command line parameters. Description of the parameters used is shown in the table below:

| Argument | Description | Options | Default |
| --- | --- | --- | --- |
| `--data_dir` | Path of the dataset folder to be used. | string | images_f |
| `--sym_group` | Specify the group symmetry to be used if equivariant network is to be used | string <ul><li>Circular</li><li>Dihyderal</li> | Circular |
| `--N` | Order of the symmetry group | int | 4 |
| `--use_CNN` | True, if use simple convolution (ResNet18) | Bool:- True/False | False |
| `--epochs` | Number of epochs to train the network | int | 30 |
|`--batch_size` | Batch size to be used while training the network | int | 64 |
| `--lr` | Learning rate for training | Float | 5e - 5 |
| `--mode` | Specify training or testing mode | string <ul><li>Train</li><li>Test</li> | Train |
| `--test_time` | Name of the folder containing the model with timestamp (if testing with pretrained weights)| string |  |
| `--use_cuda` | True, if use GPU | Bool:- True/False | True |
|`--cuda_idx` | Which cuda device index to use | int | 0 |
  
Sample of a command to run Dihyderal symmetry group of 8 order on model J dataset:
 
```
$ python main.py --data_dir images_j --sym_group Dihyderal --N 8
```
 
**NOTE:** You can download the pre-trained models from [here](https://drive.google.com/drive/folders/1HTz0fO2Vtlsq1QZesGUOi84QQzy5KhDM?usp=sharing)
 
# Results
A folder name [dataset name]_[First letter of symmtry group]_[order of symmetry group] will be created. It will contain a folder having the name as the timestamp at which the code was run. The results will be located inside this folder as:

| Folder/File | Description |
| --- | ---|
| `log_[timstamp at which code was run]` | Contains loss and accuracy at each epoch | 
| `code.py` | Copy of the python script executed | 
| `best-model-parameters.pt` | Saved model with best performance on testing data | 
| `train_loss.npy` | Numpy file containing the training losses |
| `test_loss.npy` | Numpy file containing the testing losses |
| `train_accuracy.npy` | Numpy file containing the training accuracies |
| `test_accuracy.npy` | Numpy file containing the testing accuracies |
| `accuracy.png` | Image showing the training and testing accuracies at different epochs |
| `loss.png` | Image showing the training and testing losses at different epochs |
| `ROC.png` | Image of the ROC curve |
| `CM.png` | Image of the confusion matrix |

## Cite

This project implements the ideas from the work [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251).
 
```
@inproceedings{e2cnn,
    title={{General E(2)-Equivariant Steerable CNNs}},
    author={Weiler, Maurice and Cesa, Gabriele},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2019},
}
```
This project is the follow up work of the previous attempts to study and determine the morphology of dark matter substructure using deep learning based approaches. It is recommended to go through the following works for a clearer understanding of the problem formulation and the attempts made for substructure studies in the past.

```
@article{alexander2020deep,
  title={Deep Learning the Morphology of Dark Matter Substructure},
  author={Alexander, Stephon and Gleyzer, Sergei and McDonough, Evan and Toomey, Michael W and Usai, Emanuele},
  journal={The Astrophysical Journal},
  volume={893},
  number={1},
  pages={15},
  year={2020},
  publisher={IOP Publishing}
}
```
```
@article{alexander2020decoding,
  title={Decoding Dark Matter Substructure without Supervision},
  author={Alexander, Stephon and Gleyzer, Sergei and Parul, Hanna and Reddy, Pranath and Toomey, Michael W and Usai, Emanuele and Von Klar, Ryker},
  journal={arXiv preprint arXiv:2008.12731},
  year={2020}
}
```




 
