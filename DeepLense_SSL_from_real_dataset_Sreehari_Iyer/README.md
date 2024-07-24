# Learning Representation Through Self-Supervised Learning on Real Gravitational Lensing Images
This repository contains the code for my ongoing project with <a href = "https://ml4sci.org"> Machine Learning for Science (ML4Sci)</a> as part of the <a href = "https://summerofcode.withgoogle.com/programs/2024">Google Summer of Code (GSoC) 2024</a>. The work done as part of this project is summarized in this <a href="https://iyersreehari.github.io/gsoc24-blog-deeplense-ssl/">blog post</a>

This project focuses on evaluating of self-supervised learning techniques with Transformers utilizing real-world strong gravitational lensing dataset.

# Supervised Learning Baseline
The supervised learning baselines trained can be found in this <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/working/supervised"> folder </a> <br>
Following is the evaluation results computed over a held-out test dataset. 
| Backbone | # Parameters | Accuracy | AUC Score |
|----------|----------|----------|----------|
| ViT-S (patch size: 16)    | 22M   | 91.3997 %  | 0.9780   |
| ViT-B (patch size: 16)    | 86M   | 92.2428 %  | 0.9762    |

# Self-Supervised Learning 

To run self supervised training, clone this repository and create a config file with the experiment parameters. The config file is expected to be a .yaml file. <br>
Examples of the config files are provided in the <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/configs"> configs </a> folder.
Currently, ViT-S and ViT-B for the backbone and SimSiam, DINO and iBOT algorithms for self supervised training have been implemented

While inside the directory, install the required packages.

        pip install -r requirements.txt

The specified network can then be trained through self-supervision as follows:

        python ./ssl/ssltrain.py /path/to/config/file

For self-supervised learning, the ViT backbone is trained on the training dataset without the label information. The hyperparameters are chosen such that the K-NN accuracy computed over the representations obtained for the validation dataset is maximized. For evaluating the learned network, the backbone followed by a linear classifier is finetuned on the train dataset (with label information) and then evaluated over the held-out test dataset.<br>

| SSL algorithm / Backbone | # Parameters | Accuracy | AUC Score |
|----------|----------|----------|----------|
|<a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/working/simsiam_vit_small_patch_16"> SimSiam ViT-S (patch size: 16) </a>   | 22M   | 91.5683 %  | 0.9703   |
| <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/working/simsiam_vit_base_patch_16"> SimSiam ViT-B (patch size: 16) </a>   | 86M   | 91.5683 %  | 0.9651   |
|<a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/working/dino_vit_base-20240716-193405"> DINO ViT-S (patch size: 16) </a>   | 22M   | 93.9292 %  |   0.9782   |
| <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/working/dino_vit_small-20240716-005817"> DINO ViT-B (patch size: 16) </a>   | 86M   | 92.9174 %  |   0.9785   |
|<a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/working/ibot_vit_small_patch_16"> iBOT ViT-S (patch size: 16) </a>   | 22M   | 91.7369 %  |   0.9720   |
| <a href="https://github.com/iyersreehari/DeepLense_SSL_Sreehari_Iyer/tree/main/working/ibot_vit_base_patch_16"> iBOT ViT-B (patch size: 16) </a>   | 86M   | 91.0624 %  |   0.9689   |


