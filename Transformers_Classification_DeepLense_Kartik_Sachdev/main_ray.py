from __future__ import print_function
import os
import time

import torch.nn as nn
from typing import *
from utils.util import (
    make_directories,
    seed_everything,
    get_device,
    init_logging_handler,
)
from utils.dataset import DeepLenseDataset
from utils.train_ray import train
from argparse import ArgumentParser
from config.data_config import DATASET
from utils.augmentation import get_transform_train
from torch.utils.data import random_split

from config.twinssvt_config import TWINSSVT_RAY_CONFIG
from config.cct_config import CCT_RAY_CONFIG
from config.levit_config import LEVIT_RAY_CONFIG
from config.cait_config import CAIT_RAY_CONFIG
from config.crossvit_config import CROSSVIT_RAY_CONFIG
from config.pit_config import PIT_RAY_CONFIG
from config.swin_config import SWIN_RAY_CONFIG
from config.t2tvit_config import T2TViT_RAY_CONFIG
from config.crossformer_config import CROSSFORMER_RAY_CONFIG
from config.cvt_config import CvT_RAY_CONFIG


from ray import tune
from ray.tune import CLIReporter
import ray

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
    "--num_samples",
    metavar="10",
    type=int,
    default=10,
    help="number of combinations for tuning",
)

parser.add_argument(
    "--train_config",
    type=str,
    default="CvT",
    help="transformer config",
    choices=[
        "CvT",
        "CCT",
        "TwinsSVT",
        "LeViT",
        "CaiT",
        "CrossViT",
        "PiT",
        "Swin",
        "T2TViT",
        "CrossFormer",
    ],
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
        train_config = CCT_RAY_CONFIG
    elif train_config_name == "TwinsSVT":
        train_config = TWINSSVT_RAY_CONFIG
    elif train_config_name == "LeViT":
        train_config = LEVIT_RAY_CONFIG
    elif train_config_name == "CaiT":
        train_config = CAIT_RAY_CONFIG
    elif train_config_name == "CrossViT":
        train_config = CROSSVIT_RAY_CONFIG
    elif train_config_name == "PiT":
        train_config = PIT_RAY_CONFIG
    elif train_config_name == "Swin":
        train_config = SWIN_RAY_CONFIG
    elif train_config_name == "T2TViT":
        train_config = T2TViT_RAY_CONFIG
    elif train_config_name == "CrossFormer":
        train_config = CROSSFORMER_RAY_CONFIG
    else:
        train_config = CvT_RAY_CONFIG

    network_type = train_config["network_type"]
    image_size = train_config["image_size"]

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
    init_logging_handler(log_dir_base, current_time, use_ray=True)

    PATH = os.path.join(
        f"{log_dir}/checkpoint", f"{network_type}_{dataset_name}_{current_time}.pt"
    )

    num_classes = len(classes)

    print(num_classes)
    print(f"Train Data: {len(trainset)}")
    print(f"Val Data: {len(testset)}")

    # loss function
    criterion = nn.CrossEntropyLoss()
    epochs = train_config["num_epochs"]

    train_config["wandb"] = {
        "project": f"{network_type}_{dataset_name}_hpo",
        "api_key": os.environ["WANDB_KEY"],
    }

    train_config["labels_map"] = classes
    scheduler = ASHAScheduler(
        metric="best_accuracy",
        max_t=epochs,
        grace_period=1,
        reduction_factor=2,
        mode="max",
    )

    reporter = CLIReporter(metric_columns=["best_accuracy"])

    num_samples = args.num_samples

    trainable = tune.with_parameters(
        train,
        device=device,
        trainset=trainset,
        testset=testset,
        criterion=criterion,
        path=PATH,
        log_dir=log_dir,
        log_freq=20,
        dataset_name=dataset_name,
        num_workers=num_workers,
        num_classes=num_classes,
        image_size=image_size,
    )

    ray.init()
    result = tune.run(
        trainable,
        name=f"{train_config_name}_{current_time}",
        config=train_config,
        scheduler=scheduler,
        resources_per_trial={"cpu": num_workers, "gpu": 1},
        stop={"training_iteration": epochs,},
        verbose=1,
        num_samples=num_samples,
        keep_checkpoints_num=2,
        progress_reporter=reporter,
        checkpoint_score_attr="best_accuracy",
        local_dir="Tune-Best-Test",
    )

    ray.shutdown()


if __name__ == "__main__":
    main()

