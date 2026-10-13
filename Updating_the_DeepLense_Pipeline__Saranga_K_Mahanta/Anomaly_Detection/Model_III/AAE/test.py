import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch import nn
from tqdm import tqdm

sys.path.append("../")
from config import BATCH_SIZE, FULL_TEST_DATA_PATH, MODEL_PATH
from model import Decoder, Discriminator, Encoder
from test_dataloaders import create_full_test_dataloader
from utils import device, set_seed, test_transforms


def final_test_epoch(model, dataloader, criterion):

    encoder, decoder = model["enc"], model["dec"]
    encoder.eval()
    decoder.eval()

    losses = []
    truth_list = []

    with torch.no_grad():
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for _batch_idx, (img_batch, class_ids) in loop:
            for img, class_id in zip(img_batch, class_ids, strict=False):
                if class_id == class_map["no_sub"]:
                    truth_list.append(0)
                else:
                    truth_list.append(1)
                img = img.unsqueeze(0)
                X = img.to(device)

                encoded_space = encoder(X)
                recon = decoder(encoded_space)

                loss = criterion(recon, X).item()
                losses.append(loss)

    return np.asarray(losses), np.asarray(truth_list), np.mean(losses)


if __name__ == "__main__":
    set_seed(7)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    disc = Discriminator().to(device)

    if device != "cpu":
        encoder.load_state_dict(torch.load(MODEL_PATH + "encoder.pth"))
        decoder.load_state_dict(torch.load(MODEL_PATH + "decoder.pth"))
        disc.load_state_dict(torch.load(MODEL_PATH + "discriminator.pth"))
    else:
        encoder.load_state_dict(
            torch.load(MODEL_PATH + "encoder.pth", map_location=torch.device("cpu"))
        )
        decoder.load_state_dict(
            torch.load(MODEL_PATH + "decoder.pth", map_location=torch.device("cpu"))
        )
        disc.load_state_dict(
            torch.load(
                MODEL_PATH + "discriminator.pth", map_location=torch.device("cpu")
            )
        )

    model = {"enc": encoder, "dec": decoder, "disc": disc}

    full_test_loader, class_map = create_full_test_dataloader(
        FULL_TEST_DATA_PATH, test_transforms, BATCH_SIZE
    )
    criterion = nn.MSELoss()
    loss_list, truth_list, mean_loss = final_test_epoch(
        model, full_test_loader, criterion
    )
    print(f"\nMean loss: {mean_loss}")

    fpr, tpr, thresholds = roc_curve(truth_list, loss_list)
    roc_auc = auc(fpr, tpr)

    # Plot the AUC
    plt.figure(figsize=(19, 12))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.5f})"
    )
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Binary Classification")
    plt.legend(loc="lower right", prop={"size": 10})
    plt.savefig("ROC-AUC.png", dpi=80)
    plt.show()
