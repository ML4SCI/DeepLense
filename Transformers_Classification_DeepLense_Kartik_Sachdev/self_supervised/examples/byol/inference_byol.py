from __future__ import print_function

from turtle import down

from utils.dataset import DefaultDatasetSetupSSL
from utils.inference import InferenceSSL
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from models.byol import BYOLSingleChannel, FinetuneModelByol
from models.byol import BYOL, BYOLSingleChannel, BYOLTransformer
import torchvision

from self_supervised.config.crossformer_network import CROSSFORMER_CONFIG
from models.transformer_zoo import TransformerModels

import torchvision
from torchsummary import summary


def get_resnet_finetuned(finetune_model_path, num_classes, device):
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

    return finetune_model


def add_head_byol(model, finetune_head, device):
    finetune_model = FinetuneModelByol(backbone=model, head=finetune_head)
    finetune_model.to(device=device)
    summary(finetune_model, input_size=(10, 1, 224, 224), device=device)

    return finetune_model


def get_transfomer_finetuned(saved_model_path, device, num_classes) -> nn.Module:
    # Create pretrain model
    transformer_config = CROSSFORMER_CONFIG
    image_size = 224

    in_features = 128
    # Transformer model
    transformer = TransformerModels(
        transformer_type=transformer_config["network_type"],
        num_channels=1,
        num_classes=in_features,
        img_size=image_size,
        **transformer_config["network_config"],
    )

    model = BYOLTransformer(
        transformer, num_ftrs=in_features, hidden_dim=512, out_dim=256
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model(
        torch.randn(10, 1, 224, 224).to(device), torch.randn(10, 1, 224, 224).to(device)
    )


    model.to(device)

    in_features_projection = 256
    finetune_head = nn.Sequential(
        nn.Linear(in_features_projection, in_features_projection),
        nn.BatchNorm1d(in_features_projection),
        nn.ReLU(),
        nn.Linear(in_features_projection, num_classes),
    )

    finetune_model = add_head_byol(
        model=model, finetune_head=finetune_head, device=device
    )

    finetune_model = FinetuneModelByol(backbone=model, head=finetune_head)
    finetune_model.to(device=device)

    finetune_model.load_state_dict(torch.load(saved_model_path))
    print(">>>>>> Keys matched. Model loaded")

    return finetune_model


def main():
    device = "cuda"
    num_classes = 3
    dataset_name = "Model_II"
    labels_map = {0: "axion", 1: "cdm", 2: "no_sub"}
    image_size = 224
    channels = 1
    log_dir = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/2023-07-25-06-27-13"
    finetune_model_path = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/2023-07-25-06-27-13/checkpoint/CrossFormer_finetuned_Model_II_2023-07-25-06-27-13.pt"
    batch_size = 512
    num_workers = 8

    # testset
    # setup default dataset
    default_dataset_setup = DefaultDatasetSetupSSL()
    default_dataset_setup.setup(dataset_name=dataset_name)
    default_dataset_setup.setup_transforms(image_size=image_size)

    # trainset
    train_dataset = default_dataset_setup.get_dataset(mode="train")
    default_dataset_setup.visualize_dataset(train_dataset)

    # split in train and valid set
    split_ratio = 0.1  # 0.25
    valid_len = int(split_ratio * len(train_dataset))
    train_len = len(train_dataset) - valid_len

    train_dataset, val_set = random_split(train_dataset, [train_len, valid_len])

    val_loader = DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # finetune_model = get_resnet_finetuned(finetune_model_path, num_classes, device)
    finetune_model = get_transfomer_finetuned(finetune_model_path, device, num_classes)


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
        log_dir=log_dir,  # log_dir
    )

    infer_obj.infer_plot_roc()
    infer_obj.generate_plot_confusion_matrix()


if __name__ == "__main__":
    main()
