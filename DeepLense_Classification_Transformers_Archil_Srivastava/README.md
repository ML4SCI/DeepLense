# __DeepLense Classification Using Vision Transformers__
  
 PyTorch-based library for performing image classification of the strong lensing images to predict the type of dark matter substructure. The code contains implementation and benchmarking of various versions of Vision Transformers (especially hybrid ones) from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and logging metrics like loss and AUROC (class-wise and overall) scores on [Weights and Biases](https://wandb.ai/site).

This was a __Google Summer of Code (GSoC) 2022__ project. For more info on the project [Click Here](https://summerofcode.withgoogle.com/programs/2022/projects/iFKJMj0t) <br>
<br>

# __Datasets__
The models are trained on 3 datasets -- namely Model I, Model II, and Model III -- consisting of ~30,000 train and ~5000 test images in each dataset. All images are single channel images with the size being 150x150 for Model I and 64x64 for Model II and Model III both. All dataasets consis of 3 classes, namely:
- Axion (Vortex substructure)
- CDM (Cold Dark Matter, point mass subhalos)
- No substructure (this doesn’t occur observationally as there is always a substructure in reality, but we use this simulated class as a baseline)

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

Use the train.py script to train a particular model (using timm model name). The script will ask for a WandB login key, hence a WandB account is needed. Example: 
```bash
python3 train.py \
--dataset Model_I \
--model_source timm \
--model_name efficientformer_l3 \
--pretrained 1 \
--tune 1 \
--device cuda \
--project ml4sci_deeplense_final
```
| Arguments | Description |
| :---  | :--- |
| dataset | Name of dataset i.e. Model_I, Model_II or Model_III |
| model_name | Name of the model from pytorch-image-models |
| complex | 0 if use model from pytorch-image-models directly, 1 if add some additional layers at the end of the model |
| pretrained | Picked pretrained weights or train from scratch |
| tune | Whether to further tune (1) pretrained model (if any) or freeze the pretrained weights (0) |
| batch_size | Batch Size |
| lr | Learning Rate |
| dropout | Dropout Rate |
| optimizer | Optimizer name |
| decay_lr | 0 if use constant LR, 1 if use CosineAnnealingWarmRestarts |
| epochs | Number of epochs |
| random_zoom | Random zoom for augmentation |
| random_rotation | Random rotation for augmentation (in degreees) |
| log_interval | Log interval for logging to weights and biases |
| project | Project name in Weight and Biases
| device | Device: cuda or mps or cpu or best |
| seed | Random seed |

# __Evaluation__

Run evaluation of trained model on test sets using eval.py script. Pass the run_id of the train run from WandB to pick the proper configuration. Example: 
```bash
python3 eval.py \
--run_id 1g9hi3n6 \
--device cuda \
-- project ml4sci_deeplense_final
```

<br>

# __Results__

So far, around 9 model families (including EfficientNet as baseline and 8 transformer families). Different variants of models from the same families were tested and the results are as follows:

## __[EfficientNet](https://arxiv.org/abs/1905.11946)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/efficientnet_b1__complex.png?raw=true)

### Model II
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_II/efficientnet_b1__complex.png?raw=true)

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/efficientnet_b1__complex.png?raw=true)

## __[ViT](https://arxiv.org/abs/2010.11929)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/vit_tiny_r_s16_p8_224.png?raw=true)

### Model II

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/vit_tiny_r_s16_p8_224__complex.png?raw=true)

## __[ConViT](https://arxiv.org/abs/2103.10697)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/convit_tiny__complex.png?raw=true)

### Model II
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_II/convit_tiny.png?raw=true)

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/convit_tiny__complex.png?raw=true)

## __[CrossViT](https://arxiv.org/abs/2103.14899)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/crossvit_small_240.png?raw=true)

### Model II
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_II/crossvit_small_240__complex.png?raw=true)

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/crossvit_small_240.png?raw=true)

## __[Bottleneck Transformers](https://arxiv.org/abs/2101.11605)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/botnet_26t_256.png?raw=true)

### Model II
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_II/botnet_26t_256.png?raw=true)

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/botnet_26t_256.png?raw=true)

## __[EfficientFormer](https://arxiv.org/abs/2206.01191)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/efficientformer_l3.png?raw=true)

### Model II
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_II/efficientformer_l3.png?raw=true)

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/efficientformer_l3.png?raw=true)

## __[CoaT](https://arxiv.org/abs/2104.06399)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/coat_lite_small__complex.png?raw=true)

### Model II
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_II/coat_lite_small.png?raw=true)

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/coat_lite_small.png?raw=true)

## __[CoAtNet](https://arxiv.org/abs/2106.04803)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/coatnet_nano_rw_224.png?raw=true)

### Model II

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/coatnet_nano_rw_224.png?raw=true)

## __[Swin](https://arxiv.org/abs/2103.14030)__

### Model I
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_I/swinv2_tiny_window8_256%20.png?raw=true)

### Model II
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_II/swinv2_tiny_window8_256.png?raw=true)

### Model III
![Alt text](https://github.com/archilk/ml4sci-gsoc22/blob/main/deeplense/results/Model_III/swinv2_tiny_window8_256%20.png?raw=true)

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
  
* Apoorva Singh, Yurii Halychanskyi, Marcos Tidball, DeepLense, (2021), GitHub repository, https://github.com/ML4SCI/DeepLense
