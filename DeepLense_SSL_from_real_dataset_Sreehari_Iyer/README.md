# Learning Representation Through Self-Supervised Learning on Real Gravitational Lensing Images
This repository contains the code for my ongoing project with <a href = "https://ml4sci.org"> Machine Learning for Science (ML4Sci)</a> as part of the <a href = "https://summerofcode.withgoogle.com/programs/2024">Google Summer of Code (GSoC) 2024</a>. The work done as part of this project is summarized in this <a href="https://iyersreehari.github.io/gsoc24-blog-deeplense-ssl/">blog post</a>

This project focuses on evaluating of self-supervised learning techniques with Transformers utilizing real-world strong gravitational lensing dataset. The learned representations are then evaluated on the downstream task to classify lens and non-lens images. <br>

Before training, download the lenses dataset from <a href = "https://drive.google.com/drive/folders/1JHEQFgyGedSm0pVfYH66cHmYOqlqm992?usp=sharing"> drive </a> and the nonlenses dataset from <a href = "https://drive.google.com/drive/folders/11vdOCZKp3tt-Ls-1d8xIfoXgyuLmL9S9?usp=sharing"> drive </a> and place them in lenses and nonlenses subdirectories respectively. <br>
The dataset contains 1949 lens images and 2000 non-lens images. Each image has 3 channels, g, r and i, corresponding to green, red and infrared filters respectively. The images are center cropped to 32 × 32 pixel as this empirically resulted in better prediction accuracy. The models are evaluated on a 15% held out test split of the dataset.<br>
To understand how well SSL works with different fractions of labelled and unlabelled data, the models are pre-trained through self supervision on the entire data and then finetuned on the labelled fraction and compared with supervised baseline trained only on that labeled fraction. <br>

# Supervised Learning Baseline
Following is the evaluation results for supervised baselines computed over a held-out test dataset. 
|Backbone | # labelled data for <br> training/fine-tuning | Accuracy | AUC |
|:---:|:----------:|:----------:|:----------:|
|ViT-S (patch size: 8)|300 | 86.0034% | 0.9235 |
|ViT-B (patch size: 8)|300 | 86.6779% | 0.9339 |
|ViT-S (patch size: 8)|600 | 88.0270% | 0.9210 |
|ViT-B (patch size: 8)|600 | 88.3642% | 0.9371 |
|ViT-S (patch size: 8)|1200 | 89.5447% | 0.9663 |
|ViT-B (patch size: 8)|1200 | 90.7251% | 0.9698 |
|ViT-S (patch size: 8)|3256 | 93.7605% | 0.9786 |
|ViT-B (patch size: 8)|3256 | 93.0860% | 0.9786 |

# Self-Supervised Learning 

To run self supervised training, clone this repository and create a config file with the experiment parameters. The config file is expected to be a .yaml file. <br>
Examples of the config files are provided in the <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/configs"> configs </a> folder.
Update the `data path` field in the config file with the path to the dataset downloaded previously.
Currently, ViT-S and ViT-B for the backbone and DINO and iBOT algorithms for self supervised training have been implemented

Inside the directory, install the required packages.

        pip install -r requirements.txt

The specified network can then be trained through self-supervision as follows:

        python ./ssltrain.py /path/to/config/file

For self-supervised learning, the ViT backbone is trained on the training dataset without the label information. The hyperparameters are chosen such that the K-NN accuracy computed over the representations obtained for the validation dataset is maximized. For evaluating the learned network, the backbone followed by a linear classifier is finetuned on the train dataset (with label information) and then evaluated over the held-out test dataset. The trained models are saved in the output directory specified in the config file.<br>

The pretrained models can then be fine-tuned with the labeled dataset. The indices list for different count of labeled data is provided in the <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/input"> input </a> folder.
The notebooks in the <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/notebooks"> notebooks </a> folder provide examples for fine-tuning the pre-trained models.<br>
<br>
Following is the evaluation results computed over a held-out test dataset for fine-tuning of **DINO** pre-trained models.

|Backbone | # labelled data for <br> training/fine-tuning | Accuracy | AUC |
|:---:|:----------:|:----------:|:----------:|
|ViT-S (patch size: 8)|300 | 88.1956% | 0.9382 |
|ViT-B (patch size: 8)|300 | 89.5447% | 0.9487 |
|ViT-S (patch size: 8)|600 | 89.5447% | 0.9361 |
|ViT-B (patch size: 8)|600 | 89.5447% | 0.9376 |
|ViT-S (patch size: 8)|1200 | 90.7251% | 0.9657 |
|ViT-B (patch size: 8)|1200 | 92.5801% | 0.9768 |
|ViT-S (patch size: 8)|3256 | 92.4115% | 0.9806 |
|ViT-B (patch size: 8)|3256 | 94.6037% | 0.9810 |


Following is the evaluation results computed over a held-out test dataset for fine-tuning of **iBot** pre-trained models.

|Backbone | # labelled data for <br> training/fine-tuning | Accuracy | AUC |
|:---:|:----------:|:----------:|:----------:|
|ViT-S (patch size: 8)|300 | 88.5329% | 0.9367 |
|ViT-B (patch size: 8)|300 | 86.6779% | 0.9339 |
|ViT-S (patch size: 8)|600 | 89.2074% | 0.9471 |
|ViT-B (patch size: 8)|600 | 88.3642% | 0.9371 |
|ViT-S (patch size: 8)|1200 | 90.7251% | 0.9675 |
|ViT-B (patch size: 8)|1200 | 90.7251% | 0.9698 |
|ViT-S (patch size: 8)|3256 | 92.9174% | 0.9798 |
|ViT-B (patch size: 8)|3256 | 93.0860% | 0.9786 |

<br>

# References

- Dosovitskiy, Alexey, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. International Conference on Learning Representations. 2020.
- Caron, Mathilde, et al. Emerging properties in self-supervised vision transformers. Proceedings of the IEEE/CVF international conference on computer vision. 2021.
- <a href="https://github.com/facebookresearch/dino"> DINO github repository </a>
- Zhou, Jinghao, et al. ibot: Image bert pre-training with online tokenizer. International Conference on Learning Representations (ICLR) (2022).
- <a href="https://github.com/bytedance/ibot"> iBot github repository </a>


