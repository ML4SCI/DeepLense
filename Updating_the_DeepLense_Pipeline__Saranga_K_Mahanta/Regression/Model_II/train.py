import gc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from ranger import Ranger
from utils import device, set_seed, train_transforms, test_transforms
from model import Regressor
from dataloader import create_dataloaders
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE, SAVE_MODEL, MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH


def train_epoch(model, dataloader, criterion, optimizer, scheduler, example_ct):
    
    model.train()
    train_loss = []

    loop=tqdm(enumerate(dataloader),total = len(dataloader))

    for batch_idx, (img_batch,labels) in loop:

        X = img_batch.to(device)
        y_truth = labels.to(device)
        example_ct += len(img_batch)
        
        #forward prop
        y_pred = model(X)
        y_pred = y_pred.view(-1)
        #loss calculation
        loss = criterion(y_pred, y_truth)

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        #batch loss
        train_loss.append(loss.detach().cpu().numpy())

    return model, np.mean(train_loss), example_ct


def test_epoch(model, dataloader,criterion):

    model.eval()
    losses = []
    
    y_pred_list = []
    y_truth_list = []

    with torch.no_grad():

        loop=tqdm(enumerate(dataloader),total=len(dataloader))
        
        for batch_idx, (img_batch,masses) in loop:
            X = img_batch.to(device)
            y_truth = masses.to(device)
            y_truth_list.append(y_truth.detach().cpu().numpy())

            #forward prop
            y_pred = model(X)
            y_pred = y_pred.view(-1)
            
            y_pred_list.append(y_pred.detach().cpu().numpy())

            #loss calculation
            loss = criterion(y_pred, y_truth)
            losses.append(loss.detach().cpu().numpy())

    return y_pred_list, y_truth_list, np.mean(losses)


def plot_results(model, dataloader, criterion, epoch):
    y_pred_list, y_truth_list, test_loss = test_epoch(model, test_loader, criterion)
    
    def flatten_list(x):
        flattened_list = []
        for i in x:
            for j in i:
                flattened_list.append(j)

        return flattened_list
    
    y_pred_list_flattened = flatten_list(y_pred_list)
    y_truth_list_flattened = flatten_list(y_truth_list)
    
    plt.figure(figsize=(9,9))
    plt.scatter(y_truth_list_flattened, y_pred_list_flattened)
    plt.xlabel('Observed mass')
    plt.ylabel('Predicted mass')
    plt.draw()
    

def fit_model(model, checkpoint_path):
    
        optimizer = Ranger(model.parameters(), lr = LEARNING_RATE)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, LEARNING_RATE, epochs = EPOCHS, steps_per_epoch = len(train_loader), verbose = False)
        criterion = nn.MSELoss()

        loss_dict = {'train_loss':[],'val_loss':[]}
        example_ct = 0  # number of examples seen
        min_val_loss = 999 #high value to initlialize

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}:")
            model, train_loss, example_ct = train_epoch(model, train_loader, criterion, optimizer, scheduler, example_ct)
            _, _, val_loss = test_epoch(model, val_loader, criterion)
            
            if SAVE_MODEL:
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    print("New lower val loss. Saving model checkpoint!")
                    torch.save(model.state_dict(), checkpoint_path)

            print(f'Train loss:{train_loss}, Val loss:{val_loss}')

            loss_dict['train_loss'].append(train_loss)
            loss_dict['val_loss'].append(val_loss)

            if epoch % 3 == 0 or epoch + 1 == EPOCHS:
                plot_results(model, val_loader, criterion, epoch)

        return model, loss_dict
    

    
if __name__ == "__main__":
    set_seed(7)
    train_loader, val_loader, test_loader = create_dataloaders(
        TRAIN_DATA_PATH, TEST_DATA_PATH, 
        train_transforms, test_transforms, 
        BATCH_SIZE
        )

    model = Regressor().to(device)
    model, loss_dict = fit_model(model, checkpoint_path = MODEL_PATH)

    # Plot losses
    plt.figure(figsize=(19,12))
    plt.semilogy(loss_dict['train_loss'], label='Train')
    plt.semilogy(loss_dict['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    #plt.grid()
    plt.legend()
    plt.title('Loss history')
    plt.savefig('Loss_history.png')
    plt.show()





