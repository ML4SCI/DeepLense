# Learning Representation Through Self-Supervised Learning on Real Gravitational Lensing Images
This repository contains the code for my ongoing project with <a href = "https://ml4sci.org"> Machine Learning for Science (ML4Sci)</a> as part of the <a href = "https://summerofcode.withgoogle.com/programs/2024">Google Summer of Code (GSoC) 2024</a>. The work done as part of this project is summarized in this <a href="https://iyersreehari.github.io/gsoc24-blog-deeplense-ssl/">blog post</a>

This project focuses on evaluating of self-supervised learning techniques with Transformers utilizing real-world strong gravitational lensing dataset. The learned representations are then evaluated on the downstream task to classify lens and non-lens images. <br>

Before training, download the lenses dataset from <a href = "https://drive.google.com/drive/folders/1JHEQFgyGedSm0pVfYH66cHmYOqlqm992?usp=sharing"> drive </a> and the nonlenses dataset from <a href = "https://drive.google.com/drive/folders/11vdOCZKp3tt-Ls-1d8xIfoXgyuLmL9S9?usp=sharing"> drive </a> and place them in lenses and nonlenses subdirectories respectively. <br>
The train dataset contains 2333 lens images and 1530 non-lens images. The validation dataset contains 259 lens images and 170 non-lens images. The test dataset contains 458 lens images and 300 non-lens images. Each image has 3 channels, g, r and i, corresponding to green, red and infrared filters respectively. Each image has 3 channels, g, r and i, corresponding to green, red and infrared filters respectively. The images are center cropped to 32 × 32 pixel as this empirically resulted in better prediction accuracy. The models are evaluated for the downstream task of classifying images into lenses and non-lenses on the held out test split of the dataset.<br>
To understand how well SSL works with different fractions of labelled and unlabelled data, the model is pre-trained through self supervision on the entire train data and then finetuned on the labelled fraction of the train data and compared with supervised baseline trained only on that labeled fraction. This simulates the real world scenario where only a fraction of dataset may have associated labels.  <br>

# Supervised Learning Baseline
Following is the evaluation results for supervised baselines computed over a held-out test dataset. 
|Backbone | # labelled data for <br> training/fine-tuning | Accuracy | AUC |
|:---:|:----------:|:----------:|:----------:|
|ViT-S (patch size: 8)|300 | 89.7098% | 0.9547 |
|ViT-B (patch size: 8)|300 | 90.7652% | 0.9534 |
|ViT-S (patch size: 8)|600 | 92.2164% | 0.9445 |
|ViT-B (patch size: 8)|600 | 92.6121% | 0.9579 |
|ViT-S (patch size: 8)|1200 | 93.1398% | 0.9780 |
|ViT-B (patch size: 8)|1200 | 93.4037% | 0.9678 |
|ViT-S (patch size: 8)|3863 | 94.9868% | 0.9861 |
|ViT-B (patch size: 8)|3863 | 94.9868% | 0.9843 |

# Self-Supervised Learning 

To train self supervised backbone, clone this repository and create a config file with the experiment parameters. The config file is expected to be a .yaml file. <br>
Examples of the config files are provided in the <a href="https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_SSL_from_real_dataset_Sreehari_Iyer/configs"> configs </a> folder.
Update the `data path` field in the config file with the path to the dataset downloaded previously and the `indices` field with the path to the indices.pkl file. The indices.pkl file is present in <a href="https://github.com/ML4SCI/DeepLense/tree/main/DeepLense_SSL_from_real_dataset_Sreehari_Iyer/input"> input </a> folder.

Currently, ViT-S and ViT-B for the backbone and DINO, iBOT and SimSima algorithms for self supervised training have been implemented

Inside the directory, install the required packages.

        pip install -r requirements.txt

The specified network can then be trained through self-supervision as follows:

        python ./ssltrain.py /path/to/config/file

For self-supervised learning, the ViT backbone is trained on the training dataset without the label information. The trained models are saved in the output directory specified in the config file. For evaluating the learned network, the backbone followed by a linear classifier is finetuned on the labelled fraction of the train dataset and then evaluated over the held-out test dataset.<br>
The indices used for the same is provided in the <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/input"> input </a> folder.
The notebooks in the <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/notebooks"> notebooks </a> folder provide examples for fine-tuning the pre-trained models.<br>
<br>
Following is the evaluation results computed over a held-out test dataset for fine-tuning of **DINO** pre-trained models.

|Backbone | # labelled data for <br> training/fine-tuning | Accuracy | AUC |
|:---:|:----------:|:----------:|:----------:|
|ViT-S (patch size: 8)|300 | 91.1609% | 0.9689 |
|ViT-B (patch size: 8)|300 | 91.5567% | 0.9703 |
|ViT-S (patch size: 8)|600 | 92.6121% | 0.9653 |
|ViT-B (patch size: 8)|600 | 92.2164% | 0.9667 |
|ViT-S (patch size: 8)|1200 | 93.7995% | 0.9858 |
|ViT-B (patch size: 8)|1200 | 93.2718% | 0.9798 |
|ViT-S (patch size: 8)|3256 | 94.9868% | 0.9865 |
|ViT-B (patch size: 8)|3256 | 94.1953% | 0.9876 |

Following is the evaluation results computed over a held-out test dataset for fine-tuning of **Simsiam** pre-trained models.

|Backbone | # labelled data for <br> training/fine-tuning | Accuracy | AUC |
|:---:|:----------:|:----------:|:----------:|
|ViT-S (patch size: 8)|300 | 89.8417% | 0.9540 |
|ViT-B (patch size: 8)|300 | 90.3694% | 0.9539 |
|ViT-S (patch size: 8)|600 | 92.2164% | 0.9691 |
|ViT-B (patch size: 8)|600 | 92.0844% | 0.9589 |
|ViT-S (patch size: 8)|1200 | 94.9868% | 0.9843 |
|ViT-B (patch size: 8)|1200 | 93.5356% | 0.9817 |
|ViT-S (patch size: 8)|3256 | 95.1187% | 0.9864 |
|ViT-B (patch size: 8)|3256 | 94.8549% | 0.9832 |


Following is the evaluation results computed over a held-out test dataset for fine-tuning of **iBot** pre-trained models.

|Backbone | # labelled data for <br> training/fine-tuning | Accuracy | AUC |
|:---:|:----------:|:----------:|:----------:|
|ViT-S (patch size: 8)|300 | 91.4248% | 0.9669 |
|ViT-B (patch size: 8)|300 | 91.9525% | 0.9668 |
|ViT-S (patch size: 8)|600 | 92.2164% | 0.9631 |
|ViT-B (patch size: 8)|600 | 92.6121% | 0.9696 |
|ViT-S (patch size: 8)|1200 | 93.7995% | 0.9844 |
|ViT-B (patch size: 8)|1200 | 93.1398% | 0.9814 |
|ViT-S (patch size: 8)|3256 | 95.6464% | 0.9906 |
|ViT-B (patch size: 8)|3256 | 94.4591% | 0.9885 |

<br>

# References

- Dosovitskiy, Alexey, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. International Conference on Learning Representations (ICLR). (2020).
- Caron, Mathilde, et al. Emerging properties in self-supervised vision transformers. Proceedings of the IEEE/CVF international conference on computer vision (ICCV). (2021).
- <a href="https://github.com/facebookresearch/dino"> DINO github repository </a>
- X. Chen and K. He. Exploring Simple Siamese Representation Learning. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021).
- <a href="https://github.com/facebookresearch/simsiam"> Simsiam github repository </a>
- Zhou, Jinghao, et al. ibot: Image bert pre-training with online tokenizer. International Conference on Learning Representations (ICLR) (2022).
- <a href="https://github.com/bytedance/ibot"> iBot github repository </a>


