#!/bin/bash
export CONDA_ALWAYS_YES="true"

conda create -n deeplense python=3.7
conda activate deeplense
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch

# install apt-get packages
sudo apt-get update && apt-get -y install \
    python3-pip \
    unzip

# install pip packages
pip3 install --upgrade matplotlib>=3.2.2 \
    numpy>=1.18.5 \
    opencv-python>=4.1.1 \
    Pillow>=8.2.0 \
    PyYAML>=5.3.1 \
    requests>=2.25.1 \
    scipy>=1.4.1   \
    vit-pytorch==0.27.0 \
    torchinfo \
    tqdm>=4.41.0 \
    timm>=0.5.4 \
    transformers>=4.18.0 \
    e2cnn==0.1.9 \
    ray[tune] \
    ray[air] \
    tensorboard>=2.4.1 \
    wandb \
    pandas>=1.1.4 \
    seaborn>=0.11.0 \
    scikit-learn \
    ipython   \
    psutil   \
    thop  \
    albumentations>=1.0.3 \
    gdown \
    split-folders \
    ipywidgets \
    einops \

