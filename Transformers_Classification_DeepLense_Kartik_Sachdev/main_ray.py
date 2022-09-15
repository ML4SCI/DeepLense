from __future__ import print_function
import os
from os import listdir
from os.path import join
import random
import logging
import time
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchinfo import summary
from sklearn.metrics import roc_curve, auc, confusion_matrix

# from models.cvt import CvT, EqCvT
from typing import *
from utils.util import (
    make_directories,
    seed_everything,
    get_device,
    init_logging_handler,
)
from utils.dataset import download_dataset, DeepLenseDataset, visualize_samples
from utils.train_ray import train
from utils.inference import Inference
from argparse import ArgumentParser
from config.data_config import DATASET
from config.eqcvt_config import EQCVT_CONFIG
from config.pretrained_config import PRETRAINED_CONFIG
from utils.augmentation import get_transform_test, get_transform_train
from torch.utils.data import DataLoader
import timm
from torchvision import models
from models.cnn_zoo import Model, ConViT
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, Dataset, random_split
from models.transformer_zoo import (
    GetCrossFormer,
    GetTwinsSVT,
    GetLeViT,
    GetPiT,
    GetCCT,
    GetT2TViT,
    TransformerModels,
)

from config.cct_config import CCT_CONFIG
from config.twinssvt_config import TWINSSVT_CONFIG
from config.levit_config import LEVIT_CONFIG
from config.cait_config import CAIT_CONFIG
from config.crossvit_config import CROSSVIT_CONFIG
from config.pit_config import PIT_CONFIG

import json

import wandb

from ray import tune
from ray.tune import CLIReporter

# from ray.tune.integration.wandb import WandbLogger
from ray.tune.schedulers import ASHAScheduler

parser = ArgumentParser()
parser.add_argument(
    "--dataset_name",
    metavar="Model_X",
    type=str,
    default="Model_I",
    choices=["Model_I", "Model_II", "Model_III", "Model_IV"],
    help="dataset type for DeepLense project",
)
parser.add_argument(
    "--save", metavar="XXX/YYY", type=str, default="data", help="destination of dataset"
)

parser.add_argument(
    "--num_workers", metavar="1", type=int, default=1, help="number of workers"
)

parser.add_argument(
    "--train_config",
    type=str,
    default="CCT",
    help="transformer config",
    choices=["CCT", "TwinsSVT", "LeViT", "CaiT", "CrossViT", "PiT"],
)

parser.add_argument("--cuda", action="store_true", help="whether to use cuda")
parser.add_argument(
    "--no-cuda", dest="cuda", action="store_false", help="not to use cuda"
)
parser.set_defaults(cuda=True)

args = parser.parse_args()


def main():
    dataset_name = args.dataset_name
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save)
    use_cuda = args.cuda
    num_workers = args.num_workers
    train_config_name = args.train_config

    classes = DATASET[f"{dataset_name}"]["classes"]

    if train_config_name == "CCT":
        train_config = CCT_CONFIG
    elif train_config_name == "TwinsSVT":
        train_config = TWINSSVT_CONFIG
    elif train_config_name == "LeViT":
        train_config = LEVIT_CONFIG
    elif train_config_name == "CaiT":
        train_config = CAIT_CONFIG
    elif train_config_name == "CrossViT":
        train_config = CROSSVIT_CONFIG
    elif train_config_name == "PiT":
        train_config = PIT_CONFIG
    else:
        train_config = CCT_CONFIG  # temporary

    network_type = train_config["network_type"]
    network_config = train_config["network_config"]
    image_size = train_config["image_size"]
    optimizer_config = train_config["optimizer_config"]
    lr_schedule_config = train_config["lr_schedule_config"]

    make_directories([dataset_dir])

    trainset = DeepLenseDataset(
        dataset_dir,
        "train",
        dataset_name,
        transform=get_transform_train(
            upsample_size=387,
            final_size=train_config["image_size"],
            channels=train_config["channels"],
        ),
        download=True,
        channels=train_config["channels"],
    )

    split_ratio = 0.25
    valid_len = int(split_ratio * len(trainset))
    train_len = len(trainset) - valid_len
    trainset, testset = random_split(trainset, [train_len, valid_len])

    seed_everything(seed=42)
    device = get_device(use_cuda=use_cuda, cuda_idx=0)

    # logging
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir_base = "logger"
    log_dir = f"{log_dir_base}/{current_time}"
    init_logging_handler(log_dir_base, current_time)

    PATH = os.path.join(
        f"{log_dir}/checkpoint", f"{network_type}_{dataset_name}_{current_time}.pt"
    )

    num_classes = len(classes)  # number of classes to be classified
    # image size (129x129)
    print(num_classes)
    print(f"Train Data: {len(trainset)}")
    print(f"Val Data: {len(testset)}")

    # Transformer model
    model = TransformerModels(
        transformer_type=train_config["network_type"],
        num_channels=train_config["channels"],
        num_classes=num_classes,
        img_size=image_size,
        **train_config["network_config"],
    )

    summary(model, input_size=(train_config["batch_size"], 1, image_size, image_size))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Parameter count:", count_parameters(model))

    # loss function
    criterion = nn.CrossEntropyLoss()
    epochs = train_config["num_epochs"]

    train_config["optimizer_config"]["lr"] = tune.sample_from(
        lambda spec: 10 ** (-10 * np.random.rand())
    )
    train_config["optimizer_config"]["momentum"] = tune.uniform(0.7, 0.99)
    train_config["wandb"] = {
        "project": "HPO-trial-1",
        "api_key": "0eab39620668aed6d80d5cc8e58407d2509af0eb",  # os.environ["WANDB_KEY"]
    }

    scheduler = ASHAScheduler(
        metric="best_accuracy",
        max_t=epochs,
        grace_period=1,
        reduction_factor=2,
        mode="max",
    )

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["best_accuracy"]
    )

    num_samples = 10

    trainable = tune.with_parameters(
        train,
        # epochs=epochs,
        model=model,
        device=device,
        trainset=trainset,
        testset=testset,
        criterion=criterion,
        path=PATH,
        log_dir=log_dir,
        log_freq=20,
        dataset_name=dataset_name,
        num_workers=num_workers,
    )

    result = tune.run(
        trainable,
        name=f"{train_config_name}_{current_time}",
        config=train_config,
        scheduler=scheduler,
        # search_alg=algo,
        resources_per_trial={"cpu": num_workers, "gpu": 1},
        stop={"training_iteration": epochs,},
        verbose=1,
        num_samples=num_samples,
        # reuse_actors=True,  # keep to true to check how training progresses
        # fail_fast=True,  # fail on first error
        # keep_checkpoints_num=2,
        # progress_reporter=reporter,
        # checkpoint_score_attr="best_accuracy",
        # local_dir="Tune-Best-Test",
        # loggers=[WandbLogger],
    )

    best_trial = result.get_best_trial("best_accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["best_accuracy"]
        )
    )

    # train(
    #     epochs=train_config["num_epochs"],
    #     model=model,
    #     device=device,
    #     train_loader=train_loader,
    #     valid_loader=test_loader,  # change to val-loader
    #     criterion=criterion,
    #     path=PATH,
    #     log_dir=log_dir,
    #     log_freq=20,
    #     config=train_config,
    #     dataset_name=dataset_name,
    # )

    # infer_obj = Inference(
    #     model,
    #     test_loader,
    #     device,
    #     num_classes,
    #     testset,
    #     dataset_name,
    #     labels_map=classes,
    #     image_size=image_size,
    #     channels=train_config["channels"],
    #     destination_dir="data",
    #     log_dir=log_dir,  # log_dir
    # )
    # infer_obj.infer_plot_roc()
    # infer_obj.generate_plot_confusion_matrix()


if __name__ == "__main__":
    main()

