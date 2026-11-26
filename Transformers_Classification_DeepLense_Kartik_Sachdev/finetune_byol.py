import os
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.dataset import DeepLenseDatasetSSL, DefaultDatasetSetupSSL
from models.cnn_zoo import CustomResNet
from utils.losses.contrastive_loss import ContrastiveLossEuclidean
from utils.train import train_simplistic
from utils.util import (
    load_model_add_head,
    get_second_last_layer,
    get_last_layer_features,
)
from torchsummary import summary
from models.byol import BYOLSingleChannel, FinetuneModelByol
import torchvision
from models.utils.finetune_model import FinetuneModel


# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="BYOL Finetuning for DeepLense")
parser.add_argument(
    "--config",
    type=str,
    default="config/yaml/finetune_byol_config.yaml",
    help="Path to YAML config file"
)
parser.add_argument("--saved_model_path", type=str, default=None, help="Path to pretrained BYOL .pt file")
parser.add_argument("--device",           type=str, default=None, help="cuda or cpu")
args = parser.parse_args()

# ── Load YAML config ───────────────────────────────────────────────────────────
config = {}
if os.path.exists(args.config):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

# ── Settings (CLI overrides