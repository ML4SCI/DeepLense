# __DeepLense Classification Using Vision Transformers__
  
 PyTorch-based library for performing image classification of the simulated strong lensing images to predict substructures of dark matter halos. The project involves implementation and benchmarking of various versions of Vision Transformers from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and logging metrics like loss and AUROC (class-wise and overall) scores on [Weights and Biases](https://wandb.ai/site).

This is an ongoing __Google Summer of Code (GSoC) 2022__ project. For more info on the project [Click Here](https://summerofcode.withgoogle.com/programs/2022/projects/iFKJMj0t) <br>
<br>

# __Datasets__
The models are tested on mainly 3 datasets consisting of 30,000 images (single channel) per class. All the dataset consists of 3 classes namely: 
- No substructure
- Axion (vortex)
- CDM (point mass subhalos)

___Note__: Axion files have extra data corresponding to mass of axion used in simulation._

## __Model_I__
- Images are 150 x 150 pixels
- Modeled with a Gaussian point spread function
- Added background and noise for SNR of around 25

## __Model_II__
- Images are 64 x 64 pixels
- Modeled after Euclid observation characteristics as done by default in lenstronomy
- Modeled with simple Sersic light profile

## __Model_III__
- Images are 64 x 64 pixels
- Modeled after HST observation characteristics as done by default in lenstronomy.
- Modeled with simple Sersic light profile

<br>

# __Training__

Use the train.py script to train a particular model. The script will ask for a WandB login key, hence a WandB account is needed. Example: 
```bash
python3 train.py \
--dataset Model_I \
--model_source timm \
--model_name convit_small \
--pretrained 1 \
--tune 1 \
--device cuda
```
| Arguments | Description |
| :---  | :--- | 
| dataset | Name of dataset i.e. Model_I, Model_II or Model_III |
| model_source | Where to pick the model from. Currently supported values are "baseline" and "timm" |
| model_name | Name of the model from pytorch-image-models |
| complex | 0 if use model from pytorch-image-models directly, 1 if add some additional layers at the end of the model |
| pretrained | Picked pretrained weights or train from scratch |
| tune | 0 if only tune the last layers of the model, 1 if tune all layers |
| batch_size | Batch Size |
| lr | Learning Rate |
| dropout | Dropout Rate |
| optimizer | Optimizer name |
| decay_lr | 0 if use constant LR, 1 if use CosineAnnealingWarmRestarts |
| epochs | Number of epochs |
| random_zoom | Random zoom for augmentation |
| random_rotation | Random rotation for augmentation (in degreees) |
| log_interval | Log interval for logging to weights and biases |
| device | Device: cuda or mps or cpu |
| seed | Random seed |

# __Evaluation__

Run evaluation of trained model on test sets using eval.py script. Pass the run_id of the train run from WandB to pick the proper configuration. Example: 
```bash
python3 eval.py \
--run_id 1g9hi3n6 \
--device cuda
```

<br>

# __Results__

So far, a baseline CNN model and 3 variants of vision transformers (along with 2 subvariants in 2 of them) have been tested. Results are as follows:

### __[Simple CNN Baseline__

  | Dataset | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | 
  | Model I   |  0.9923  | 0.9856 | 0.9971 |
  | Model II  | 0.9988  | 0.9978 | 0.9997 |
  | Model III |  1.0000  | 1.0000 | 1.0000 |

### __[ConViT (Tiny version)](https://arxiv.org/abs/2103.10697)__

  | Dataset | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | 
  | Model I   |  0.9522  | 0.9216 | 0.9909 |
  | Model II  | 0.9445  | 0.8475 | 0.9617 |
  | Model III |  0.9910  | 0.9668 | 0.9856 |
  
### __[ConViT (Small version)](https://arxiv.org/abs/2103.10697)__

  | Dataset | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | 
  | Model I   |  0.9633  | 0.9221 | 0.9892 |
  | Model II  | 0.9407  | 0.7223 | 0.9452 |
  | Model III |  0.9901  | 0.9582 | 0.9876 |

### __ViT-ResNet Hybrid (Tiny version)__

  | Dataset | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | 
  | Model I   |  0.9447  | 0.8863 | 0.9881 |
  | Model II  | 0.9410  | 0.8391 | 0.9500 |
  | Model III |  0.9888  | 0.9695 | 0.9912 |

### __ViT-ResNet Hybrid (Small version)__

  | Dataset | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | 
  | Model I   |  0.9781  | 0.9680 | 0.9978 |
  | Model II  | 0.9553  | 0.8714 | 0.9633 |
  | Model III |  0.9991  | 0.9908 | 0.9952 |

### __[Bottleneck Transformers](https://arxiv.org/abs/2101.11605)__

  | Dataset | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | 
  | Model I   |  0.9911  | 0.9845 | 0.9995 |
  | Model II  | 0.9607  | 0.9043 | 0.9772 |
  | Model III |  0.9992  | 0.9927 | 0.9976 |

<br>

## __Citation__

* [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

  ```bibtex
  @misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
  }
  ```

* [ConViT (Soft Convolutional Inductive Biases Vision Transformers)](https://arxiv.org/abs/2103.10697)

  ```bibtex
  @misc{https://doi.org/10.48550/arxiv.2103.10697,
  doi = {10.48550/ARXIV.2103.10697},
  url = {https://arxiv.org/abs/2103.10697},
  author = {d'Ascoli, Stéphane and Touvron, Hugo and Leavitt, Matthew and Morcos, Ari and Biroli, Giulio and Sagun, Levent},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
  }
  ```

* [Bottleneck Transformers](https://arxiv.org/abs/2101.11605)

  ```bibtex
  @misc{https://doi.org/10.48550/arxiv.2101.11605,
  doi = {10.48550/ARXIV.2101.11605},
  url = {https://arxiv.org/abs/2101.11605},
  author = {Srinivas, Aravind and Lin, Tsung-Yi and Parmar, Niki and Shlens, Jonathon and Abbeel, Pieter and Vaswani, Ashish},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Bottleneck Transformers for Visual Recognition},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
  }
  ```
  
* Apoorva Singh, Yurii Halychanskyi, Marcos Tidball, DeepLense, (2021), GitHub repository, https://github.com/ML4SCI/DeepLense
