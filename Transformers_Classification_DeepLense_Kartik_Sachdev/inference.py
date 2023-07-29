from __future__ import print_function

from turtle import down

from utils.dataset import DefaultDatasetSetupSSL
from utils.inference import InferenceSSL
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from models.byol import BYOLSingleChannel, FinetuneModelByol
import torchvision
from torchsummary import summary


def main():
    device = "cuda"
    num_classes = 3
    dataset_name = "Model_II"
    labels_map = {0: "axion", 1: "cdm", 2: "no_sub"}
    image_size = 224
    channels = 1
    log_dir = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/2023-07-23-13-30-24"
    finetune_model_path = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/2023-07-23-13-30-24/checkpoint/Resnet_finetune_Model_II.pt"
    batch_size = 512
    num_workers = 8

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

    # testset
    # setup default dataset
    default_dataset_setup = DefaultDatasetSetupSSL()
    default_dataset_setup.setup(dataset_name=dataset_name)
    default_dataset_setup.setup_transforms(image_size=image_size)

    # trainset
    train_dataset = default_dataset_setup.get_dataset(mode="train")
    default_dataset_setup.visualize_dataset(train_dataset)

    # split in train and valid set
    split_ratio = 0.05  # 0.25
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
        log_dir=log_dir,  # log_dir
    )

    infer_obj.infer_plot_roc()
    infer_obj.generate_plot_confusion_matrix()


if __name__ == "__main__":
    main()
