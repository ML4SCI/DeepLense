import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import DefaultDatasetSetupSSL
from torch.utils.data import DataLoader, random_split

from models.cnn_zoo import CustomResNet
from utils.train import train_simplistic

from torchsummary import summary
from models.byol import BYOLSingleChannel, FinetuneModelByol
from models.byol import BYOL, BYOLSingleChannel, BYOLTransformer
import torchvision

from self_supervised.config.crossformer_network import CROSSFORMER_CONFIG
from models.transformer_zoo import TransformerModels


def get_resnet_finetuned(saved_model_path, device, num_classes) -> nn.Module:
    # Load pretrained model and add head
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = BYOLSingleChannel(backbone, num_ftrs=512)

    model.load_state_dict(torch.load(saved_model_path))
    print(">>>>>> Keys matched. Model loaded")

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

    finetune_model = add_head_byol(
        model=model, finetune_head=finetune_head, device=device
    )

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

    model.load_state_dict(torch.load(saved_model_path))
    print(">>>>>> Keys matched. Model loaded")

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
    summary(finetune_model, input_size=(10, 1, 224, 224), device=device)

    return finetune_model


def main():
    # Set device
    device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_path = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/2023-07-25-06-27-13/checkpoint/CrossFormer_pretrained_Model_II_2023-07-25-06-27-13.pt"
    finetuned_model_path = "/home/kartik/git/DeepLense/Transformers_Classification_DeepLense_Kartik_Sachdev/logger/2023-07-25-06-27-13/checkpoint/CrossFormer_finetuned_Model_II_2023-07-25-06-27-13.pt"

    # Set hyperparameters
    batch_size = 512
    epochs = 10
    learning_rate = 0.0001
    weight_decay = 0.01
    margin = 1.0
    num_channels = 1
    temperature = 0.5
    num_classes = 3

    # Load dataset
    default_dataset_setup = DefaultDatasetSetupSSL(dataset_name="Model_II")
    train_dataset = default_dataset_setup.get_dataset()

    # split in train and valid set
    split_ratio = 0.25  # 0.25
    valid_len = int(split_ratio * len(train_dataset))
    train_len = len(train_dataset) - valid_len

    train_dataset, val_set = random_split(train_dataset, [train_len, valid_len])

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    val_loader = DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=8
    )

    finetune_model = get_transfomer_finetuned(
        pretrained_model_path, device, num_classes
    )

    # Define optimizer and loss function
    optimizer_adam = optim.Adam(finetune_model.parameters(), lr=learning_rate)
    optimizer_adamw = optim.AdamW(
        finetune_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    train_simplistic(
        epochs,
        finetune_model,
        device,
        train_loader,
        criterion,
        optimizer_adam,
        finetuned_model_path,
        valid_loader=val_loader,
    )


if __name__ == "__main__":
    main()
