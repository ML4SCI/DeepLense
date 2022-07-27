
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from utils import device, calculate_accuracy
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from dataloader import create_data_loaders
from model import EffNetB1_backbone_model

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    train_loss = []
    train_accuracy = []

    loop=tqdm(enumerate(dataloader),total = len(dataloader))

    for batch_idx, (img_batch,labels) in loop:

        X = img_batch.to(device)
        y_truth = labels.to(device)
        
        #forward prop
        y_pred = model(X)
        
        #loss and accuracy calculation
        loss = criterion(y_pred, y_truth)
        accuracy = calculate_accuracy(y_pred, y_truth)
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #batch loss and accuracy
        # print(f'Partial train loss: {loss.data}')
        train_loss.append(loss.detach().cpu().numpy())
        train_accuracy.append(accuracy.detach().cpu().numpy())

    return model, np.mean(train_loss), np.mean(train_accuracy)

def val_epoch(model, dataloader,criterion):
    model.eval()
    val_loss = []
    val_accuracy = []

    with torch.no_grad():

        loop=tqdm(enumerate(dataloader),total=len(dataloader))
        
        for batch_idx, (img_batch,labels) in loop:
            X = img_batch.to(device)
            y_truth = labels.to(device)

            #forward prop
            y_pred = model(X)

            #loss and accuracy calculation
            loss = criterion(y_pred, y_truth)
            accuracy = calculate_accuracy(y_pred, y_truth)


            #batch loss and accuracy
            # print(f'Partial train loss: {loss.data}')
            val_loss.append(loss.detach().cpu().numpy())
            val_accuracy.append(accuracy.detach().cpu().numpy())
            
    return np.mean(val_loss), np.mean(val_accuracy)

def fit_model(model, criterion, optimizer):
    loss_dict = {'train_loss':[],'val_loss':[]}
    acc_dict = {'train_accuracy':[],'val_accuracy':[]}
    
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = 15, T_mult = 1,eta_min = 1e-6, verbose=True)
    
#     scheduler = ReduceLROnPlateau(optimizer, 'min',patience=2,factor=0.3,verbose=True)


    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        model, train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = val_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f'Train loss:{train_loss}, Val loss:{val_loss}')
        loss_dict['train_loss'].append(train_loss)
        loss_dict['val_loss'].append(val_loss)
        print(f'Train accuracy: {train_accuracy}, Val accuracy:{val_accuracy}')
        acc_dict['train_accuracy'].append(train_accuracy)
        acc_dict['val_accuracy'].append(val_accuracy)


    return model, loss_dict, acc_dict

if __name__ == '__main__':

    #pre-processing transformation
    transforms = A.Compose(
            [
                A.CenterCrop(height = 50, width = 50, p=1.0),
                ToTensorV2()
            ]
        )

    train_loader, val_loader, _ = create_data_loaders(TRAIN_DATA_PATH, TEST_DATA_PATH, 
                                                                val_split = 0.2, batch_size = BATCH_SIZE,
                                                                transforms = transforms)

    model = EffNetB1_backbone_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)

    model, loss_dict, acc_dict = fit_model(model,criterion,optimizer)
    torch.save(model.state_dict(), MODEL_PATH)

    # Plot losses
    plt.figure(figsize=(9,7))
    plt.semilogy(loss_dict['train_loss'], label='Train')
    plt.semilogy(loss_dict['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(9,7))
    plt.semilogy(acc_dict['train_accuracy'], label='Train')
    plt.semilogy(acc_dict['val_accuracy'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    plt.show()




