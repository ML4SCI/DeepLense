from __future__ import print_function

import json
import logging
import math
import os
import time
from typing import *
from argparse import ArgumentParser
import math

import torch.nn as nn
import torch.optim as optim
from config.cvt_config import CvT_CONFIG
from config.data_config import DATASET
from lightly.loss import NegativeCosineSimilarity
from lightly.transforms import SimSiamTransform
from models.finetune_classifier_transformer import FinetuneClassifierTransformer
from models.transformer_zoo import TransformerModels
from self_supervised.losses.contrastive_loss import NegativeCosineSimilarity
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from transformers import get_cosine_schedule_with_warmup
from utils.inference import InferenceSSL
from utils.util import *
from config import *
from utils.transforms.simsiam_transform import SimSiamTransform
from utils.dataset import visualize_samples_ssl, DeepLenseDatasetSSL
from models.self_supervised.simsiam import SimSiamTransformer
from utils.trainer.simsiam_train import simsiam_train
from utils.trainer.finetune import finetune


def main(args):
    dataset_name = args.dataset_name
    dataset_dir = args.save
    batch_size = args.batch_size
    epochs_pretrain = args.epochs_pretrain
    epochs_finetune = args.epochs_finetune
    train_config_name = args.train_config
    use_cuda = args.cuda
    num_workers = args.num_workers

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
    elif train_config_name == "Swin":
        train_config = SWIN_CONFIG
    elif train_config_name == "T2TViT":
        train_config = T2TViT_CONFIG
    elif train_config_name == "CrossFormer":
        train_config = CROSSFORMER_CONFIG
    else:
        train_config = CvT_CONFIG

    network_type = train_config["network_type"]
    num_channels = train_config["channels"]
    network_config = train_config["network_config"]
    image_size = train_config["image_size"]
    optimizer_config = train_config["optimizer_config"]
    optimizer_finetune_config = train_config["optimizer_finetune_config"]
    lr_schedule_config = train_config["lr_schedule_config"]

    log_dir_base = "logger"
    classes = DATASET[f"{dataset_name}"]["classes"]
    num_classes = len(classes)

    make_directories([dataset_dir])
    seed_everything(seed=42)
    device = get_device(use_cuda=use_cuda, cuda_idx=0)

    # logging
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_dir = f"{log_dir_base}/{current_time}"
    init_logging_handler(log_dir_base, current_time)

    # paths
    model_path_pretrained = os.path.join(
        f"{log_dir}/checkpoint",
        f"{network_type}_pretrained_{dataset_name}_{current_time}.pt",
    )

    finetuned_model_path = os.path.join(
        f"{log_dir}/checkpoint",
        f"{network_type}_finetune_{dataset_name}_{current_time}.pt",
    )

    # trainset
    dino_transform = SimSiamTransform()
    train_transforms = dino_transform.get_transforms()
    train_dataset = DeepLenseDatasetSSL(
        destination_dir=dataset_dir,
        transforms=train_transforms,
        mode="train",
        dataset_name=dataset_name,
        download=True,
        channels=1,
    )
    logging.debug(f"train data: {len(train_dataset)}")
    visualize_samples_ssl(
        train_dataset, labels_map=classes, num_rows_inner=1, num_cols_inner=2
    )

    # split in train and valid set
    split_ratio = 0.25  # 0.25
    valid_len = int(split_ratio * len(train_dataset))
    train_len = len(train_dataset) - valid_len

    train_dataset, val_set = random_split(train_dataset, [train_len, valid_len])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_dataset_dir = dataset_dir
    test_dataset = DeepLenseDatasetSSL(
        destination_dir=test_dataset_dir,
        transforms=train_transforms,
        mode="test",
        dataset_name=dataset_name,
        download=True,
        channels=num_channels,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # size check
    sample = next(iter(train_loader))
    logging.debug("num of classes: ", num_classes)
    logging.debug(sample[0].shape)

    # Transformer model
    out_features = train_config["out_features"]
    backbone = TransformerModels(
        transformer_type=train_config["network_type"],
        num_channels=train_config["channels"],
        num_classes=train_config["out_features"],
        img_size=image_size,
        **network_config,  # **train_config["network_config"]
    )

    model = SimSiamTransformer(backbone, input_dim=out_features)
    model.to(device)
    summary(model, input_size=(2, 1, 224, 224), device=device)

    # optimizer
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]
    warmup_epochs = optimizer_config["warmup_epoch"]

    optimizer_pretrain = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    warmup_epochs = warmup_epochs
    num_train_steps = math.ceil(len(train_loader))
    num_warmup_steps = num_train_steps * warmup_epochs
    num_training_steps = int(num_train_steps * epochs_pretrain)

    # learning rate scheduler
    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer_pretrain,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = NegativeCosineSimilarity()

    # dump config in logger
    with open(f"{log_dir}/config.json", "w") as fp:
        json.dump(train_config, fp)

    simsiam_train(
        epochs=epochs_pretrain,
        model=model,
        device=device,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer_pretrain,
        use_lr_schedule=True,
        scheduler=cosine_scheduler,
        path=model_path_pretrained,
        log_freq=100,
        ci=False,
    )

    # load model
    backbone = nn.Sequential(model.backbone, model.projection_head)
    classification_head = nn.Sequential(
        nn.Linear(out_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, num_classes),
    )

    model = FinetuneClassifierTransformer(
        backbone, classification_head
    )  # num_ftrs_dict["resnet34"]
    # model.load_state_dict(torch.load(model_path_pretrained))
    summary(model, input_size=(2, 1, 224, 224), device=device)

    lr = optimizer_finetune_config["lr"]  # 3e-4
    weight_decay = optimizer_finetune_config["weight_decay"]
    warmup_epochs = optimizer_finetune_config["warmup_epoch"]

    # optimizer
    optimizer_finetune = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_train_steps = math.ceil(len(train_loader))
    num_warmup_steps = num_train_steps * warmup_epochs
    num_training_steps = int(num_train_steps * epochs_finetune)

    # learning rate scheduler
    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer_finetune,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion_finetune = nn.CrossEntropyLoss()

    # Finetune
    finetune(
        epochs_finetune,
        model,
        device,
        train_loader,
        criterion_finetune,
        optimizer_finetune,
        finetuned_model_path,
        valid_loader=val_loader,
        scheduler=cosine_scheduler,
        ci=False,
    )

    # Infer on test data
    infer_obj = InferenceSSL(
        model,
        test_loader,
        device,
        num_classes,
        test_dataset,
        dataset_name,
        labels_map=classes,
        image_size=image_size,
        channels=num_channels,
        destination_dir="data",
        log_dir=log_dir,
    )

    infer_obj.infer_plot_roc()
    infer_obj.generate_plot_confusion_matrix()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        metavar="Model_X",
        type=str,
        default="Model_II",
        choices=["Model_I", "Model_II", "Model_III", "Model_IV"],
        help="dataset type for DeepLense project",
    )
    parser.add_argument(
        "--save",
        metavar="XXX/YYY",
        type=str,
        default="data",
        help="destination of dataset",
    )

    parser.add_argument(
        "--num_workers", metavar="5", type=int, default=1, help="number of workers"
    )

    parser.add_argument(
        "--batch_size", metavar="5", type=int, default=64, help="batch size"
    )

    parser.add_argument(
        "--epochs_pretrain",
        metavar="5",
        type=int,
        default=15,
        help="pretraining epochs",
    )

    parser.add_argument(
        "--epochs_finetune", metavar="5", type=int, default=20, help="finetuning epochs"
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

    main(args)
