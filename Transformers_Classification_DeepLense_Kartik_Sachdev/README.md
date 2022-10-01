# __DeepLense Classification__
  
 PyTorch-based library for performing image classification of the simulated strong lensing images to predict substructures of dark matter halos. The project involves implementation and benchmarking of various versions of Vision Transformers to achieve a robust architecture with high metrics for classification namely Validation Accuracy, ROC and AUC scores.

It is one of __Google Summer of Code (GSoC) 2022__ project in association with [__Machine Learning for Science (ML4Sci)__](https://ml4sci.org/). More information about the project can be found [here](https://summerofcode.withgoogle.com/programs/2022/projects/L557jFPL) <br>

A brief summary of the project along with technical background, and key takeaways can be found in the [Medium Blog Post](https://medium.com/@sachdev.kartik25/benchmarking-vision-transformers-for-classification-of-dark-matter-substructure-gsoc-2022-with-6ec7711cc32d)
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

# __Installation__
To install locally, using pip:
```bash
git clone https://github.com/ML4SCI/DeepLense.git
cd DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev
git checkout PR_kartik_transformers
pip3 install --upgrade -r requirements.txt
```

<br>

# __Training__

### __Train with custom config__
Modify the configuration of the chosen model and training scheme from e.g. `TwinsSVT_CONFIG` in the [config file](/config/twinssvt_config.py). Then, the script can be run locally, for example:

```bash
cd DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev
python3 -u main.py \
  --num_workers 20 \
  --dataset_name Model_II \
  --train_config TwinsSVT \
  --cuda   
```

| Arguments | Description |
| :---  | :--- | 
| num_workers | Number of workers available for training |
| dataset_name | Name of the dataset type for DeepLense project |
| save | Path where the dataset is stored |
| train_config | Transformer config: [CvT, CCT, TwinsSVT, LeViT, CaiT, CrossViT, PiT, Swin, T2TViT, CrossFormer] |
| cuda | Use cuda |
| no-cuda | Not use cuda |

<br>

### __Hyperparameter optimization__
Modify the configuration of the model and training scheme from e.g. `TwinsSVT_RAY_CONFIG` in the [config file](/config/twinssvt_config.py). Then, the script can be run, for example: 

```bash
cd DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev
python3 main_ray.py \
  --num_workers 20 \
  --dataset_name Model_II \
  --train_config TwinsSVT \
  --cuda \
  --num_samples 10
```
| Arguments | Description |
| :---  | :--- | 
| num_workers | Number of workers available for training |
| dataset_name | Name of the dataset type for DeepLense project |
| save | Path where the dataset is stored |
| train_config | Transformer config: [CvT, CCT, TwinsSVT, LeViT, CaiT, CrossViT, PiT, Swin, T2TViT, CrossFormer] |
| cuda | Use cuda |
| no-cuda | Not use cuda |
| num_samples | Number of samples for [ASHA scheduler](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html)  |


<br>

### __Jupyterfile__

Run the [example file](example.ipynb)  

___Note__: To view the dataset, ROC curve and confusion matrix in the jupyter file, please comment out: `matplotlib.use("Agg")` from  [`utils/inference.py`](utils/inference.py) file._

<br>

### __Cluster__
Modify the file `jobscript.sh` as per the system and user specifics and then, train using __SLURM__. Example: 
```bash
sbatch < jobscript.sh
```
<br>

# __Results__

So, far 9 different versions of Vision Transformers have been tested. Results are as follows:


### __[CvT](https://arxiv.org/pdf/2103.15808.pdf)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  91.54  | 0.9714  | 0.9657 | 0.9991 |
  | Model II  | 99.41  | 0.9987  | 0.9990 | 1.0000 |
  | Model III |  99.04   | 0.9986  | 0.9992 | 1.0000 |


### __[CrossFormer](https://arxiv.org/pdf/2203.13387.pdf)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  91.20  | 0.9508  | 0.9511 | 0.9996 |
  | Model II  | 97.42  | 0.9856  | 0.6041 | 0.9998 |
  | Model III |  98.13   | 0.9782  | 0.9096 | 0.9998 |


### __[LeViT](https://openaccess.thecvf.com/content/ICCV2021/papers/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.pdf)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   | 91.64   |  0.9720 | 0.8612 | 0.9997 |
  | Model II  | 97.12  | 0.9936  | 0.9800 | 1.0000 |
  | Model III | 97.97    | 0.9981  | 0.9973 | 1.0000 |


### __[TwinsSVT](https://arxiv.org/abs/2104.13840)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  91.08   | 0.6678  | 0.9179 | 0.9993 |
  | Model II  |   97.44  | 0.9812  | 0.6572 | 0.9996 |
  | Model III | 98.48    | 0.9942  | 0.5183 | 0.9999 |
  

### __[CCT](https://arxiv.org/abs/2104.05704v4)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  89.71  | 0.9632  | 0.9462 | 0.9988 |
  | Model II  |  69.68 | 0.9566  | 0.8216 | 0.9061 |
  | Model III | 99.48    | 0.9999  | 0.9998 | 1.0000 |



### __[CrossViT](https://arxiv.org/abs/2103.14899)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  84.20   | 0.9261  | 0.8816 | 0.9988 |
  | Model II  |  91.33    | 0.9816  | 0.9386 | 0.9959 |
  | Model III | 81.29    | 0.5192  | 0.5106 | 0.5419 |


### __[CaiT](https://arxiv.org/abs/2103.17239)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |   67.93  | 0.6726  | 0.5640 | 0.7497 |
  | Model II  |  62.97   | 0.9318  | 0.6548 | 0.7833 |
  | Model III | 69.38    | 0.5160  | 0.4986 | 0.5195 |


### __[T2TViT](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.html)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |   88.12 | 0.9706  | 0.6502 | 0.9902 |
  | Model II  |  94.33  | 0.9883  | 0.9095 | 0.9983 |
  | Model III |  77.29   | 0.6515  | 0.9737 | 0.8980 |


### __[PiT](https://arxiv.org/abs/2103.16302)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  40.63   | 0.5981  | 0.5584 | 0.6580|
  | Model II  |   33.60  | 0.5351 | 0.5022 | 0.5378 |
  | Model III | 34.18    | 0.5176  | 0.5153 | 0.5294 |

### __[Swin](https://arxiv.org/pdf/2103.14030.pdf)__

  | Dataset | Acc (%) | AUC (axion) | AUC (cdm) | AUC (no_sub) 
  | :---:  | :---: | :---: | :---: | :---: | 
  | Model I   |  34.16   | 0.5066  | 0.4886 | 0.4945 |
  | Model II  |   33.72  | 0.5056 | 0.4922 | 0.5022 |
  | Model III | 33.56    | 0.4357  | 0.6116 | 0.5909 |

<br>

# __Trained Models__
Trained models and logs can be found here: https://mega.nz/fm/tHtGERDY

<br>

## __Citation__

* [CvT](https://arxiv.org/abs/2103.15808)


  ```bibtex
  @article{wu2021cvt,
    title={Cvt: Introducing convolutions to vision transformers},
    author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},
    journal={arXiv preprint arXiv:2103.15808},
    year={2021}
  }
  ```

*  [CrossFormer](https://arxiv.org/pdf/2203.13387.pdf)
    
    ```bibtex
    @misc{wang2021crossformer,
        title   = {CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention}, 
        author  = {Wenxiao Wang and Lu Yao and Long Chen and Binbin Lin and Deng Cai and Xiaofei He and Wei Liu},
        year    = {2021},
        eprint  = {2108.00154},
        archivePrefix = {arXiv},
        primaryClass = {cs.CV}
    }
    ```
 * [LeViT](https://openaccess.thecvf.com/content/ICCV2021/papers/Graham_LeViT_A_Vision_Transformer_in_ConvNets_Clothing_for_Faster_Inference_ICCV_2021_paper.pdf)

    ```bibtex
    @misc{graham2021levit,
        title   = {LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference},
        author  = {Ben Graham and Alaaeldin El-Nouby and Hugo Touvron and Pierre Stock and Armand Joulin and Hervé Jégou and Matthijs Douze},
        year    = {2021},
        eprint  = {2104.01136},
        archivePrefix = {arXiv},
        primaryClass = {cs.CV}
    }
    ```
* [CrossViT](https://arxiv.org/abs/2103.14899)

    ```bibtex
    @misc{chen2021crossvit,
        title   = {CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification},
        author  = {Chun-Fu Chen and Quanfu Fan and Rameswar Panda},
        year    = {2021},
        eprint  = {2103.14899},
        archivePrefix = {arXiv},
        primaryClass = {cs.CV}
    }
    ```
* [PiT](https://arxiv.org/abs/2103.16302)
    ```bibtex
    @misc{heo2021rethinking,
        title   = {Rethinking Spatial Dimensions of Vision Transformers}, 
        author  = {Byeongho Heo and Sangdoo Yun and Dongyoon Han and Sanghyuk Chun and Junsuk Choe and Seong Joon Oh},
        year    = {2021},
        eprint  = {2103.16302},
        archivePrefix = {arXiv},
        primaryClass = {cs.CV}
    }
  ```
* [TwinsSVT](https://arxiv.org/abs/2104.13840)
    ```bibtex
  @misc{chu2021twins,
      title   = {Twins: Revisiting Spatial Attention Design in Vision Transformers},
      author  = {Xiangxiang Chu and Zhi Tian and Yuqing Wang and Bo Zhang and Haibing Ren and Xiaolin Wei and Huaxia Xia and Chunhua Shen},
      year    = {2021},
      eprint  = {2104.13840},
      archivePrefix = {arXiv},
      primaryClass = {cs.CV}
  }
  ```
* [CCT](https://arxiv.org/abs/2104.05704v4)
  ```bibtex
  @article{hassani2021escaping,
      title   = {Escaping the Big Data Paradigm with Compact Transformers},
      author  = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
      year    = 2021,
      url     = {https://arxiv.org/abs/2104.05704},
      eprint  = {2104.05704},
      archiveprefix = {arXiv},
      primaryclass = {cs.CV}
  }
  ```
* [CaiT](https://arxiv.org/abs/2103.17239)
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
* [T2TViT](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.html)

  ```bibtex
  @misc{yuan2021tokenstotoken,
      title   = {Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet},
      author  = {Li Yuan and Yunpeng Chen and Tao Wang and Weihao Yu and Yujun Shi and Francis EH Tay and Jiashi Feng and Shuicheng Yan},
      year    = {2021},
      eprint  = {2101.11986},
      archivePrefix = {arXiv},
      primaryClass = {cs.CV}
  }
  ```

* [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)


    ```bibtex
    @article{touvron2020deit,
      title={Training data-efficient image transformers & distillation through attention},
      author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
      journal={arXiv preprint arXiv:2012.12877},
      year={2020}
    }
    ```

* [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251)
 
    ```bibtex
    @inproceedings{e2cnn,
        title={{General E(2)-Equivariant Steerable CNNs}},
        author={Weiler, Maurice and Cesa, Gabriele},
        booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
        year={2019},
    }
    ```

* [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf)

    ```bibtex
    @inproceedings{liu2021Swin,
      title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
      author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      year={2021}
    }
    ```

* Apoorva Singh, Yurii Halychanskyi, Marcos Tidball, DeepLense, (2021), GitHub repository, https://github.com/ML4SCI/DeepLense


