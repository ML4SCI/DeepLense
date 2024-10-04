#!/bin/bash

PREFIX=$1
MODEL_CONFIG=$2
DATA_CONFIG=$3
PATH_TO_SAMPLES=$4
WORKDIR=`pwd`

# Download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_install.sh
bash miniconda_install.sh -b -p ${WORKDIR}/miniconda
export PATH=$WORKDIR/miniconda/bin:$PATH
pip install numpy pandas scikit-learn scipy matplotlib tqdm PyYAML
pip install uproot3 awkward0 lz4 xxhash
pip install tables
pip install onnxruntime-gpu
pip install tensorboard
pip install torch

# CUDA environment setup
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/lib64

# Clone weaver-benchmark
git clone --recursive https://github.com/colizz/weaver-benchmark.git
ln -s ../top_tagging weaver-benchmark/weaver/top_tagging
cd weaver-benchmark/weaver/
mkdir output

# Training, using 1 GPU
python train.py \
 --data-train ${PATH_TO_SAMPLES}'/prep/top_train_*.root' \
 --data-val ${PATH_TO_SAMPLES}'/prep/top_val_*.root' \
 --fetch-by-file --fetch-step 1 --num-workers 3 \
 --data-config top_tagging/data/${DATA_CONFIG} \
 --network-config top_tagging/networks/${MODEL_CONFIG} \
 --model-prefix output/${PREFIX} \
 --gpus 0 --batch-size 1024 --start-lr 5e-3 --num-epochs 1 --optimizer ranger \
 --log output/${PREFIX}.train.log

# Predicting score, using 1 GPU
python train.py --predict \
 --data-test ${PATH_TO_SAMPLES}'/prep/top_test_*.root' \
 --num-workers 3 \
 --data-config top_tagging/data/${DATA_CONFIG} \
 --network-config top_tagging/networks/${MODEL_CONFIG} \
 --model-prefix output/${PREFIX}_best_epoch_state.pt \
 --gpus 0 --batch-size 1024 \
 --predict-output output/${PREFIX}_predict.root

[ -d "runs/" ] && tar -caf output.tar output/ runs/ || tar -caf output.tar output/