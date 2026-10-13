import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch import nn
from tqdm import tqdm

sys.path.append("../")
from config import BATCH_SIZE, FULL_TEST_DATA_PATH, MODEL_PATH
from model import Autoencoder
from test_dataloaders import create_full_test_dataloader
from utils import device, set_seed, test_transforms


def final_test_epoch(model, dataloader, criterion):
    model.eval()
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
                recon = model(X)
                loss = criterion(recon, X)
                losses.append(loss.detach().cpu().numpy())

    return np.asarray(losses), np.asarray(truth_list), np.mean(losses)


if __name__ == "__main__":
    set_seed(7)

    model = Autoencoder().to(device)
    if device != "cpu":
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

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
