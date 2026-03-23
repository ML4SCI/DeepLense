from __future__ import print_function

import os
import argparse
import yaml

from utils.dataset import DefaultDatasetSetupSSL
from utils.inference import InferenceSSL
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from models.byol import BYOLSingleChannel, FinetuneModelByol
import torchvision
from torchsummary import summary


def parse_args():
    parser = argparse.ArgumentParser(description="BYOL Inference for DeepLense")
    parser.add_argument(
        "--config",
        type=str,
        default="config/yaml/inference_config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument("--log_dir",            type=str, default=None, help="Path to log directory")
    parser.add_argument("--finetune_model_path", type=str, default=None, help="Path to finetuned model .pt file")
    parser.add_argument("--dataset_name",       type=str, default=None, help="Dataset name e.g. Model_II")
    parser.add_argument("--device",             type=str, default=None, help="cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # CLI args override YAML values
    device              = args.device              or config.get("device",              "cuda")
    num_classes         = config.get("num_classes",  3)
    dataset_name        = args.dataset_name        or config.get("dataset_name",        "Model_II")
    labels_map          = config.get("labels_map",   {0: "axion", 1: "cdm", 2: "no_sub"})
    image_size          = config.get("image_size",   224)
    channels            = config.get("channels",     1)
    batch_size          = config.get("batch_size",   512)
    num_workers         = config.get("num_workers",  8)
    log_dir             = args.log_dir             or config.get("log_dir",             None)
    finetune_model_path = args.finetune_model_path or config.get("finetune_model_path", None)

    # Validate required paths
    if not log_dir:
        raise ValueError(
            "log_dir is required.\n"
            "Set it in config/yaml/inference_config.yaml or pass --log_dir <path>"
        )
    if not finetune_model_path:
        raise ValueError(
            "finetune_model_path is required.\n"
            "Set it in config/yaml/inference_config.yaml or pass --finetune_model_path <path>"
        )
    if not os.path.exists(log_dir):
        raise FileNotFoundError(
            f"log_dir not found: {log_dir}\n"
            "Please check your config/yaml/inference_config.yaml or --log_dir argument."
        )
    if not os.path.exists(finetune_model_path):
        raise FileNotFoundError(
            f"finetune_model_path not found: {finetune_model_path}\n"
            "Please check your config/yaml/inference_config.yaml or --finetune_model_path argument."
        )

    # Load pretrained model and add head
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = BYOLSingleChannel(backbone, num_ftrs=512)
    model.to(device)
    input_feature = 256
    finetune_head = nn.Sequential(
        nn.Linear(input_feature, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, input_feature),
        nn.BatchNorm1d(input_feature),
        nn.ReLU(),
        nn.Linear(input_feature, num_classes),
    )
    finetune_model = FinetuneModelByol(backbone=model, head=finetune_head)
    finetune_model.to(device=device)
    finetune_model.load_state_dict(torch.load(finetune_model_path))
    print(">>>> Keys matched")
    summary(finetune_model, input_size=(10, 1, 224, 224), device=device)

    # setup default dataset
    default_dataset_setup = DefaultDatasetSetupSSL()
    default_dataset_setup.setup(dataset_name=dataset_name)
    default_dataset_setup.setup_transforms(image_size=image_size)

    # trainset
    train_dataset = default_dataset_setup.get_dataset(mode="train")
    default_dataset_setup.visualize_dataset(train_dataset)

    # split in train and valid set
    split_ratio = 0.05
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

    infer_obj = InferenceSSL(
        finetune_model,
        val_loader,
        device,
        num_classes,
        val_set,
        dataset_name,
        labels_map=labels_map,
        image_size=image_size,
        channels=channels,
        destination_dir="data",
        log_dir=log_dir,
    )

    infer_obj.infer_plot_roc()
    infer_obj.generate_plot_confusion_matrix()


if __name__ == "__main__":
    main()