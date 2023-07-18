from __future__ import print_function

from turtle import down

from utils.dataset import DefaultDatasetSetup
from utils.inference import Inference, InferenceSSL
from torch.utils.data import DataLoader
from models.cnn_zoo import CustomResNet
import torch.nn as nn
from utils.util import load_model_add_head, load_dummy_model_with_head
import torch


def main():
    device = "cuda"
    num_classes = 2
    dataset_name = "Base Model"
    labels_map = {0: "No Sub", 1: "Sub"}
    image_size = 224
    channels = 1
    log_dir = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger"
    saved_model_path = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/pretrained_contrastive_embedding.pth"
    batch_size = 32

    # Load pretrained model and add head
    pretrain_model = CustomResNet(channels)
    in_features = pretrain_model.get_last_layer_features()
    finetune_head = nn.Sequential(
        nn.Linear(in_features, 256), nn.ReLU(), nn.Linear(256, num_classes)
    )

    model = load_dummy_model_with_head(backbone=pretrain_model, head=finetune_head,)
    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)

    # testset
    default_dataset_setup = DefaultDatasetSetup()
    testset = default_dataset_setup.get_default_testset_ssl()
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True,)

    infer_obj = InferenceSSL(
        model,
        test_loader,
        device,
        num_classes,
        testset,
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
