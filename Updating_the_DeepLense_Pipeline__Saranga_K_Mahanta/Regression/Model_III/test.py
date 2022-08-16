import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import Regressor
from dataloader import create_dataloaders
from train import test_epoch
from config import  MODEL_PATH, BATCH_SIZE, TRAIN_DATA_PATH, TEST_DATA_PATH
from utils import device, set_seed, train_transforms, test_transforms

def flatten_list(x):
    flattened_list = []
    for i in x:
        for j in i:
            flattened_list.append(j)

    return flattened_list

if __name__ == "__main__":
    set_seed(7)
    criterion = nn.MSELoss()
    model = Regressor().to(device)
    if device != 'cpu':
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location = torch.device('cpu')))

    train_loader, val_loader, test_loader = create_dataloaders(
        TRAIN_DATA_PATH, TEST_DATA_PATH, 
        train_transforms, test_transforms, 
        BATCH_SIZE
        )

    _, _, train_loss = test_epoch(model, train_loader, criterion)
    print("Train set MSE:" + str(train_loss))
    _, _, val_loss = test_epoch(model, val_loader, criterion)
    print("Val set MSE:" + str(val_loss))
    y_pred_list, y_truth_list, test_loss = test_epoch(model, test_loader, criterion)
    print("Test set MSE:" + str(test_loss))

    y_pred_list_flattened = flatten_list(y_pred_list)
    y_truth_list_flattened = flatten_list(y_truth_list)

    plt.figure(figsize=(12,9))
    plt.scatter(y_truth_list_flattened, y_pred_list_flattened)
    plt.xlabel('Observed mass')
    plt.ylabel('Predicted mass')
    plt.savefig("test_plot.png", dpi=80)  
    plt.show()