import os
import sys
import logging
from yaml import safe_load, safe_dump
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as Transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from selfsupervised.utils import get_system_info, set_seed
from selfsupervised.ssltraining.dino import TrainDINO
from selfsupervised.ssltraining.ibot import TrainIBOT
from selfsupervised.ssltraining.simsiam import TrainSIMSIAM
from selfsupervised.models import Backbone
from datetime import datetime
from selfsupervised.augmentations import get_dino_augmentations, get_simsiam_augmentations
from functools import partial

from typing import List, Dict, Union, Callable 
import pickle


# utility function to update config yaml from default
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
    sample = torch.from_numpy(np.load(path))
    return sample

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
        image = self.loader(img_path)
        
        if self.transform:
            image = self.transform(image)

        return image, self.label[idx]

def main():
    # Ensure a config file is provided as a command-line argument
    if len(sys.argv) not in [2,3]:
        print("Usage: python main.py <config_file> <optional_default_config_file>")
        sys.exit(1)

    config_file = sys.argv[1] # dict with default args
    args = None
    if len(sys.argv) == 2:
        par = os.path.dirname(os.path.realpath(__file__))
        args = safe_load(open(os.path.join(par, *["configs", "defaults.yaml"]), "r"))
    else:
        args = safe_load(open(sys.argv[2], "r"))
    
    # will be updated from the parsed config yaml file
    config_args = safe_load(open(config_file, "r"))
    update_dict(args, config_args)

    assert args["input"]["data path"] is not None, "Input data path cannot be `None`"
    args["experiment"]["output_dir"] = f"{args['experiment']['output_dir']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not os.path.exists(args["experiment"]["output_dir"]):
        os.makedirs(args["experiment"]["output_dir"])

    set_seed(args["experiment"]["seed"], args["experiment"]["device"])

    # Retrieve system information
    system_info = get_system_info()
    safe_dump(system_info, open(os.path.join(args["experiment"]["output_dir"], "sysinfo.yaml"), "w"))

    # create logger
    log_file = os.path.join(args["experiment"]["output_dir"], 'logs.txt')
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        force=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info(f"logger initialized")

    #--------------------------------------------------------------------------------------------------------
    # backbone for ssl
    student_backbone = None
    teacher_backbone = None
    backbone = args["network"]["backbone"].lower()
    kwargs = {
        "arch": backbone,
        "image_size": args["input"]["image size"],
        "input_channels": args["input"]["channels"],
        "patch_size": args["network"]["patch_size"],
        "use_dense_prediction": args["network"].get("use_dense_prediction", None),
        "return_all_tokens": args["network"].get("return_all_tokens", None),
        "masked_im_modeling": args["network"].get("masked_im_modeling", None),
        "window_size": args["network"].get("window_size", None),
        "ape": args["network"].get("ape", None),
    }
    student_backbone = Backbone(**kwargs)
    if args["experiment"]["ssl_training"].lower() not in ["swav", "simsiam"]:
        teacher_backbone = Backbone(**kwargs)
    logger.info(f"student and teacher networks initialized")
    #--------------------------------------------------------------------------------------------------------

    
    #--------------------------------------------------------------------------------------------------------
    # compute mean and std based on the training dataset
    data_path = args["input"]["data path"]
    indices = None
    with open(args["input"]["indices"], "rb") as f:
        indices = pickle.load(f)
    train_paths = [os.path.join(data_path, *["lenses", img]) for img in np.array(indices["train"]["lenses"])] +\
                        [os.path.join(data_path, *["nonlenses", img]) for img in np.array(indices["train"]["nonlenses"])]
    train_labels = [0]*len(indices["train"]["lenses"]) + [1]*len(indices["train"]["nonlenses"])
    transform = Transforms.Compose([
        # Transforms.ToTensor(), # npy loader returns torch.Tensor
        Transforms.CenterCrop(args["ssl augmentation kwargs"]["center_crop"]),
    ])
    loader = DataLoader(
        dataset = ImageDataset(
            image_paths=train_paths,
            labels=train_labels,
            loader=npy_loader,
            transform=transform
        ),
        batch_size=64,
        num_workers=0,
        shuffle=False
    )
    mean = torch.zeros(3) if args["input"]["channels"] == 3 else torch.zeros(1)
    std = torch.zeros(3) if args["input"]["channels"] == 3 else torch.zeros(1)
    maximum = 0
    nb_samples = 0
    dtype = None
    # Iterate over the dataset
    for data, _ in loader:
        if len(data.shape) == 3:
            data = data.unsqueeze(1)
        batch_samples = data.size(0)  # batch size (the number of images in the current batch)
        data = data.view(batch_samples, data.size(1), -1)  # reshape to (batch_size, channels, H*W)
        mean += data.mean(-1).sum(0)  # sum the means over all pixels in each channel
        std += data.std(-1).sum(0)    # sum the standard deviations over all pixels in each channel
        maximum = max(maximum, data.max()) 
        dtype = str(data.dtype)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    args["ssl augmentation kwargs"]["dataset_mean"] = mean.tolist()
    args["ssl augmentation kwargs"]["dataset_std"] = std.tolist()
    logger.info(f"Dataset mean: {args['ssl augmentation kwargs']['dataset_mean']}\nDataset std: {args['ssl augmentation kwargs']['dataset_std']}, {maximum}, {dtype}")

    assert torch.all(torch.isfinite(mean)), "Mean is not finite"
    assert torch.all(torch.isfinite(std)), "Std is not finite"
    #--------------------------------------------------------------------------------------------------------
    

    #--------------------------------------------------------------------------------------------------------
    # initialize data transforms for train and eval
    # train transforms
    data_augmentation_transforms = None 
    if args["experiment"]["ssl_training"].lower() == "dino" or args["experiment"]["ssl_training"].lower() == "ibot":
        data_augmentation_transforms = get_dino_augmentations(**args["ssl augmentation kwargs"]) 
    elif args["experiment"]["ssl_training"].lower() == "simsiam":
        data_augmentation_transforms = get_simsiam_augmentations(**args["ssl augmentation kwargs"])
    # eval transforms
    eval_transforms = Transforms.Compose([
        # Transforms.ToTensor(), # npy loader returns torch.Tensor
        Transforms.CenterCrop(args["ssl augmentation kwargs"]["center_crop"]),
        Transforms.Normalize(args["ssl augmentation kwargs"]["dataset_mean"], \
                             args["ssl augmentation kwargs"]["dataset_std"]),
    ])
    #--------------------------------------------------------------------------------------------------------

        
    # initialize ssl training object
    ssl_training = None
    if args["experiment"]["ssl_training"] is None:
        print("`ssl_training` which specifies the training method cannot be `None`. Exiting.")
        sys.exit(1)
    elif args["experiment"]["ssl_training"].lower() == "dino":
        ssl_training = TrainDINO(
            output_dir = args["experiment"]["output_dir"],
            expt_name = args["experiment"]["expt_name"],
            logger = logger,
            student_backbone = student_backbone,
            teacher_backbone = teacher_backbone,
            data_path = args["input"]["data path"],
            train_test_indices = args["input"]["indices"],
            data_augmentation_transforms = data_augmentation_transforms,
            eval_transforms = eval_transforms,
            num_classes = args["input"]["num classes"],
            train_val_split = tuple(args["train args"]["train_val_split"]),
            batch_size = args["train args"]["batch_size"],
            #physical_batch_size = args["train args"]["physical_batch_size"],
            embed_dim = student_backbone.embed_dim,
            num_local_crops = args["ssl augmentation kwargs"]["num_local_crops"],
            scheduler_warmup_epochs = args["optimizer"]["scheduler_warmup_epochs"],
            warmup_teacher_temp = args["optimizer"]["warmup_teacher_temp"],
            teacher_temp = args["optimizer"]["teacher_temp"],
            warmup_teacher_temp_epochs = args["optimizer"]["warmup_teacher_temp_epochs"],
            momentum_teacher = args["optimizer"]["momentum_teacher"],
            num_epochs = args["train args"]["num_epochs"],
            head_output_dim = args["network"]["head_output_dim"],
            head_hidden_dim = args["network"]["head_hidden_dim"],
            head_bottleneck_dim = args["network"]["head_bottleneck_dim"],
            restore_from_ckpt = args["restore"]["restore"],
            restore_ckpt_path = args["restore"]["ckpt_path"],
            lr = args["optimizer"]["init_lr"],
            final_lr = args["optimizer"]["final_lr"],
            weight_decay = args["optimizer"]["init_wd"],
            final_weight_decay = args["optimizer"]["final_wd"],
            clip_grad_magnitude = args["optimizer"]["clip_grad_magnitude"],
            head_use_bn = args["network"]["head_use_bn"],
            head_norm_last_layer = args["network"]["head_norm_last_layer"],
            head_nlayers = args["network"]["head_nlayers"],
            optimizer = args["optimizer"]["optimizer"],
            log_freq = args["experiment"]["log_freq"],
            device = args["experiment"]["device"],
            use_mixed_precision = args["experiment"]["use_mixed_precision"],
            freeze_last_layer = args["train args"]["freeze_last_layer"],
            knn_neighbours = args["train args"]["knn_neighbours"],
            use_dense_prediction = args["network"]["use_dense_prediction"],
            use_L1 = args["network"]["use_L1"]
        )
    elif args["experiment"]["ssl_training"].lower() == "simsiam":
        ssl_training = TrainSIMSIAM(
            output_dir = args["experiment"]["output_dir"],
            expt_name = args["experiment"]["expt_name"],
            logger = logger,
            student_backbone = student_backbone,
            data_path = args["input"]["data path"],
            train_test_indices = args["input"]["indices"],
            data_augmentation_transforms = data_augmentation_transforms,
            eval_transforms = eval_transforms,
            num_classes = args["input"]["num classes"],
            train_val_split = tuple(args["train args"]["train_val_split"]),
            batch_size = args["train args"]["batch_size"],
            
            embed_dim = student_backbone.embed_dim,
            projector_head_hidden_dim = args["network"]["projector_head_hidden_dim"],
            head_output_dim = args["network"]["head_output_dim"],
            projector_use_bn = args["network"]["projector_use_bn"], 
            projector_nlayers = args["network"]["projector_head_nlayers"],

            predictor_head_hidden_dim = args["network"]["predictor_head_hidden_dim"],

            lr = args["optimizer"]["init_lr"],
            final_lr = args["optimizer"]["final_lr"],
            weight_decay = args["optimizer"]["init_wd"],
            final_weight_decay = args["optimizer"]["final_wd"],
            
            scheduler_warmup_epochs = args["optimizer"]["scheduler_warmup_epochs"],
            
            num_epochs = args["train args"]["num_epochs"],
            
            restore_from_ckpt = args["restore"]["restore"],
            restore_ckpt_path = args["restore"]["ckpt_path"],
            
            clip_grad_magnitude = args["optimizer"]["clip_grad_magnitude"],
            
            optimizer = args["optimizer"]["optimizer"],
            log_freq = args["experiment"]["log_freq"],
            device = args["experiment"]["device"],
            use_mixed_precision = args["experiment"]["use_mixed_precision"],
            freeze_last_layer = args["train args"]["freeze_last_layer"],
            knn_neighbours = args["train args"]["knn_neighbours"],
        )
    elif args["experiment"]["ssl_training"].lower() == "ibot":
        ssl_training = TrainIBOT(
            output_dir = args["experiment"]["output_dir"],
            expt_name = args["experiment"]["expt_name"],
            logger = logger,
            student_backbone = student_backbone,
            teacher_backbone = teacher_backbone,
            data_path = args["input"]["data path"],
            train_test_indices = args["input"]["indices"],
            data_augmentation_transforms = data_augmentation_transforms,
            eval_transforms = eval_transforms,
            pred_size = args["network"]["pred_size"],
            num_classes = args["input"]["num classes"],
            train_val_split = tuple(args["train args"]["train_val_split"]),
            batch_size = args["train args"]["batch_size"],
            #physical_batch_size = args["train args"]["physical_batch_size"],
            embed_dim = student_backbone.embed_dim,
            num_local_crops = args["ssl augmentation kwargs"]["num_local_crops"],
            scheduler_warmup_epochs = args["optimizer"]["scheduler_warmup_epochs"],
            warmup_teacher_temp = args["optimizer"]["warmup_teacher_temp"],
            teacher_temp = args["optimizer"]["teacher_temp"],
            teacher_patch_temp = args["optimizer"]["teacher_patch_temp"],
            lambda1 = args["optimizer"]["lambda1"], 
            lambda2 = args["optimizer"]["lambda2"], 
            warmup_teacher_temp_epochs = args["optimizer"]["warmup_teacher_temp_epochs"],
            warmup_teacher_patch_temp = args["optimizer"]["warmup_teacher_patch_temp"],
            momentum_teacher = args["optimizer"]["momentum_teacher"],
            num_epochs = args["train args"]["num_epochs"],
            head_output_dim = args["network"]["head_output_dim"],
            head_hidden_dim = args["network"]["head_hidden_dim"],
            head_bottleneck_dim = args["network"]["head_bottleneck_dim"],
            patch_size = args["network"]["patch_size"],
            pred_ratio = args["network"]["pred_ratio"],
            pred_ratio_var = args["network"]["pred_ratio_var"],
            pred_aspect_ratio = args["network"]["pred_aspect_ratio"],
            pred_shape = args["network"]["pred_shape"],
            pred_start_epoch = args["network"]["pred_start_epoch"],
            shared_head = args["network"]["shared_head"],
            shared_head_teacher = args["network"]["shared_head_teacher"],
            patch_out_dim = args["network"]["patch_out_dim"],
            restore_from_ckpt = args["restore"]["restore"],
            restore_ckpt_path = args["restore"]["ckpt_path"],
            lr = args["optimizer"]["init_lr"],
            final_lr = args["optimizer"]["final_lr"],
            weight_decay = args["optimizer"]["init_wd"],
            final_weight_decay = args["optimizer"]["final_wd"],
            clip_grad_magnitude = args["optimizer"]["clip_grad_magnitude"],
            head_use_bn = args["network"]["head_use_bn"],
            head_norm_last_layer = args["network"]["head_norm_last_layer"],
            head_nlayers = args["network"]["head_nlayers"],
            optimizer = args["optimizer"]["optimizer"],
            log_freq = args["experiment"]["log_freq"],
            device = args["experiment"]["device"],
            use_mixed_precision = args["experiment"]["use_mixed_precision"],
            freeze_last_layer = args["train args"]["freeze_last_layer"],
            knn_neighbours = args["train args"]["knn_neighbours"],
            use_dense_prediction = args["network"]["use_dense_prediction"],
            use_L1 = args["network"]["use_L1"]
        )
    else:
        print(f"Specified `ssl_training` method {args['experiment']['ssl_training']} not implemented. Exiting.")
        sys.exit(0)

    # train
    ssl_training.train()

    # is redundant
    # save student backbone model
    torch.save(ssl_training.student.backbone, os.path.join(args["experiment"]["output_dir"], 'representation_network.pth'))
    


if __name__ == "__main__":
    main()

    
    
    
    
    
