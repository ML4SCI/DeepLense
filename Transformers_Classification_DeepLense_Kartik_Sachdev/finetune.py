import os
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import DefaultDatasetSetup
from models.cnn_zoo import CustomResNet
from utils.losses.contrastive_loss import ContrastiveLossEuclidean
from utils.train import train_simplistic
from utils.util import load_model_add_head
from torchsummary import summary


# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Contrastive Finetuning for DeepLense")
parser.add_argument(
    "--config",
    type=str,
    default="config/yaml/finetune_config.yaml",
    help="Path to YAML config file"
)
parser.add_argument("--saved_model_path", type=str, default=None, help="Path to pretrained model .pth file")
parser.add_argument("--device",           type=str, default=None, help="cuda or cpu")
args = parser.parse_args()

# ── Load YAML config ───────────────────────────────────────────────────────────
config = {}
if os.path.exists(args.config):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

# ── Settings (CLI overrides YAML) ──────────────────────────────────────────────
device           = args.device           or config.get("dev