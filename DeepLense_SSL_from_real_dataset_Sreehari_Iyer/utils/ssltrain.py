import os
import sys
import logging
import glob
import pickle
from datetime import datetime
from typing import List, Dict, Union, Callable
from yaml import safe_load, safe_dump

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as Transforms
from torch.utils.data import DataLoader, Dataset

# Internal DeepLense imports
from selfsupervised.utils import get_system_info, set_seed
from selfsupervised.ssltraining.dino import TrainDINO
from selfsupervised.ssltraining.ibot import TrainIBOT
from selfsupervised.ssltraining.simsiam import TrainSIMSIAM
from selfsupervised.models import Backbone
from selfsupervised.augmentations import get_dino_augmentations, get_simsiam_augmentations

# --- NEW: VALIDATION UTILITY BLOCK ---
def validate_ssl_dataset(data_path: str, lenses_folder: str, nonlenses_folder: str, logger: logging.Logger):
    """
    Standardizes input validation for DeepLense SSL pipelines.
    Ensures that data exists before starting expensive operations.
    """
    path_a = os.path.join(data_path, lenses_folder)
    path_b = os.path.join(data_path, nonlenses_folder)

    for p in [path_a, path_b]:
        if not os.path.exists(p):
            logger.error(f"Required data folder missing: {p}")
            raise FileNotFoundError(f"Missing directory: {p}")
            
    files_a = glob.glob(os.path.join(path_a, "*.npy"))
    files_b = glob.glob(os.path.join(path_b, "*.npy"))

    if len(files_a) == 0 or len(files_b) == 0:
        logger.error(f"No .npy files found in {path_a} or {path_b}")
        raise ValueError("Dataset empty or path incorrect. No .npy files detected.")
    
    logger.info(f"✅ Dataset Validated: {len(files_a)} lenses and {len(files_b)} nonlenses found.")
    return True

# Utility function to update config yaml from default
def update_dict(args, config_args):
    for key in config_args:
        if key in args:
            if isinstance(args[key], dict):
                update_dict(args[key], config_args[key])
            else:
                args[key] = config_args[key]
        else:
            args[key] = config_args[key]

def npy_loader(path):
    # Added robust error handling for file loading
    try:
        sample = torch.from_numpy(np.load(path))
        return sample
    except Exception as e:
        raise RuntimeError(f"Failed to load .npy file at {path}: {e}")

class ImageDataset(Dataset):
    def __init__(
            self, 
            image_paths: List[str],
            labels: List[int],
            loader: Callable=npy_loader, 
            transform=None):
        self.image_paths = image_paths
        self.label = labels
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Failure-aware check
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = self.loader(img_path)
        
        if self.transform:
            image = self.transform(image)

        return image, self.label[idx]

def main():
    if len(sys.argv) not in [2,3]:
        print("Usage: python ssltrain.py <config_file> <optional_default_config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    args = None
    if len(sys.argv) == 2:
        par = os.path.dirname(os.path.realpath(__file__))
        args = safe_load(open(os.path.join(par, "configs", "defaults.yaml"), "r"))
    else:
        args = safe_load(open(sys.argv[2], "r"))
    
    config_args = safe_load(open(config_file, "r"))
    update_dict(args, config_args)

    # 1. SETUP OUTPUT AND LOGGER
    args["experiment"]["output_dir"] = f"{args['experiment']['output_dir']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not os.path.exists(args["experiment"]["output_dir"]):
        os.makedirs(args["experiment"]["output_dir"])

    log_file = os.path.join(args["experiment"]["output_dir"], 'logs.txt')
    logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG,
                        format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')
    logger = logging.getLogger()
    
    # 2. DATASET VALIDATION (Added step for GSoC standardization)
    assert args["input"]["data path"] is not None, "Input data path cannot be None"
    try:
        validate_ssl_dataset(args["input"]["data path"], "lenses", "nonlenses", logger)
    except Exception as e:
        print(f"Validation Error: {e}")
        sys.exit(1)

    set_seed(args["experiment"]["seed"], args["experiment"]["device"])
    system_info = get_system_info()
    safe_dump(system_info, open(os.path.join(args["experiment"]["output_dir"], "sysinfo.yaml"), "w"))

    # 3. BACKBONE INITIALIZATION
    backbone = args["network"]["backbone"].lower()
    kwargs = {
        "arch": backbone,
        "image_size": args["input"]["image size"],
        "input_channels": args["input"]["channels"],
        "patch_size": args["network"]["patch_size"],
        "use_dense_prediction": args["network"].get("use_dense_prediction"),
        "return_all_tokens": args["network"].get("return_all_tokens"),
        "masked_im_modeling": args["network"].get("masked_im_modeling"),
    }
    student_backbone = Backbone(**kwargs)
    teacher_backbone = None
    if args["experiment"]["ssl_training"].lower() not in ["swav", "simsiam"]:
        teacher_backbone = Backbone(**kwargs)
    logger.info("Backbones initialized successfully.")

    # 4. DATASET PREPARATION & NORMALIZATION CALCULATION
    data_path = args["input"]["data path"]
    with open(args["input"]["indices"], "rb") as f:
        indices = pickle.load(f)
    
    train_paths = [os.path.join(data_path, "lenses", img) for img in np.array(indices["train"]["lenses"])] + \
                  [os.path.join(data_path, "nonlenses", img) for img in np.array(indices["train"]["nonlenses"])]
    train_labels = [0]*len(indices["train"]["lenses"]) + [1]*len(indices["train"]["nonlenses"])
    
    pre_transform = Transforms.Compose([
        Transforms.CenterCrop(args["ssl augmentation kwargs"]["center_crop"]),
    ])
    
    norm_loader = DataLoader(
        dataset=ImageDataset(train_paths, train_labels, transform=pre_transform),
        batch_size=64, num_workers=0, shuffle=False
    )

    # Compute Statistics
    mean = torch.zeros(args["input"]["channels"])
    std = torch.zeros(args["input"]["channels"])
    nb_samples = 0
    
    logger.info("Calculating dataset statistics...")
    for data, _ in norm_loader:
        if len(data.shape) == 3: data = data.unsqueeze(1)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(-1).sum(0)
        std += data.std(-1).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    args["ssl augmentation kwargs"]["dataset_mean"] = mean.tolist()
    args["ssl augmentation kwargs"]["dataset_std"] = std.tolist()

    # 5. TRANSFORMS & TRAINING OBJECT
    if args["experiment"]["ssl_training"].lower() in ["dino", "ibot"]:
        data_augmentation_transforms = get_dino_augmentations(**args["ssl augmentation kwargs"])
    elif args["experiment"]["ssl_training"].lower() == "simsiam":
        data_augmentation_transforms = get_simsiam_augmentations(**args["ssl augmentation kwargs"])
    
    eval_transforms = Transforms.Compose([
        Transforms.CenterCrop(args["ssl augmentation kwargs"]["center_crop"]),
        Transforms.Normalize(args["ssl augmentation kwargs"]["dataset_mean"], args["ssl augmentation kwargs"]["dataset_std"]),
    ])

    # Select SSL Method (DINO/SIMSIAM/IBOT)
    method = args["experiment"]["ssl_training"].lower()
    if method == "dino":
        ssl_training = TrainDINO(
            output_dir=args["experiment"]["output_dir"],
            expt_name=args["experiment"]["expt_name"],
            logger=logger,
            student_backbone=student_backbone,
            teacher_backbone=teacher_backbone,
            data_path=data_path,
            train_test_indices=args["input"]["indices"],
            data_augmentation_transforms=data_augmentation_transforms,
            eval_transforms=eval_transforms,
            num_classes=args["input"]["num classes"],
            batch_size=args["train args"]["batch_size"],
            num_epochs=args["train args"]["num_epochs"],
            device=args["experiment"]["device"],
            # ... [Pass other args as required by TrainDINO]
        )
    # [Additional elif blocks for SIMSIAM and IBOT would follow the same pattern]
    else:
        logger.error(f"Method {method} not implemented.")
        sys.exit(1)

    # 6. EXECUTE TRAINING
    logger.info(f"Starting {method} training session.")
    ssl_training.train()
    
    # Save Model
    final_path = os.path.join(args["experiment"]["output_dir"], 'representation_network.pth')
    torch.save(ssl_training.student.backbone, final_path)
    logger.info(f"Model saved to {final_path}")

if __name__ == "__main__":
    main()