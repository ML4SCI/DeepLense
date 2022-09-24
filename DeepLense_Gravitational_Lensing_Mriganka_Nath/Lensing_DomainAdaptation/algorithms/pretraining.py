### This script is used for the first part of ADDA, that is pretraining the model with source data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(train_loader,encoder,classifier,device,optimizer,criterion,e,epochs,scheduler):
    '''Trains the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''

    losses = AverageMeter()
    scores1 = AverageMeter()
    scores2 = AverageMeter()
    encoder.train()
    classifier.train()
    global_step = 0
    loop = tqdm(enumerate(train_loader),total = len(train_loader))
    
    for _,(image,labels) in loop:
        image = image.to(device)
        labels = labels.unsqueeze(1)
        labels= labels.to(device)
        
        output = classifier(encoder(image))
        batch_size = labels.size(0)
        
        loss = criterion(output,labels.float())
        
        out = F.sigmoid(output)
        outputs = out.cpu().detach().numpy()
        targets = labels.cpu().detach().numpy()
        try:
            auc = roc_auc_score(targets, outputs)
            accuracy = accuracy_score(targets, outputs > 0.5)
            losses.update(loss.item(), batch_size)
            scores1.update(auc.item(), batch_size)
            scores2.update(accuracy.item(), batch_size)
            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step() 
            
            global_step += 1
        
            loop.set_description(f"Epoch {e+1}/{epochs}")
            loop.set_postfix(loss = loss.item(), auc = auc.item(), acc = accuracy.item() ,stage = 'train')
        
            
        except ValueError:
            pass     
    return losses.avg,scores1.avg,scores2.avg

def val_one_epoch(loader,encoder,classifier,device,criterion):
    '''Validates the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''
    losses = AverageMeter()
    scores1 = AverageMeter()
    scores2 = AverageMeter()
    encoder.eval()
    classifier.eval()
    global_step = 0
    loop = tqdm(enumerate(loader),total = len(loader))
    
    for _,(image,labels) in loop:
        image = image.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.no_grad():
            output = classifier(encoder(image))
        loss =criterion(output,labels.float()) 
        out = F.sigmoid(output)
        outputs = out.cpu().detach().numpy()
        targets = labels.cpu().detach().numpy()
        try:
            auc = roc_auc_score(targets, outputs)
            accuracy = accuracy_score(targets, outputs > 0.5)
            losses.update(loss.item(), batch_size)
            scores1.update(auc.item(), batch_size)
            scores2.update(accuracy.item(), batch_size)
            loop.set_postfix(loss = loss.item(), auc = auc.item(), acc = accuracy.item() ,stage = 'val')
            global_step += 1
        except ValueError:
            pass

    return losses.avg,scores1.avg,scores2.avg

def fit(encoder,classifier,device,t_loader , v_loader, hpms , OUTPUT_DIR):
    ''' Learning loop including training and validation of the models'''

    T_AUC = []
    T_LOSS = []
    V_AUC = []
    V_LOSS = []
   
    criterion1= nn.BCEWithLogitsLoss() # Loss function
    optimizer = optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr= hpms.pretraining_learning_rate , weight_decay = hpms.pretraining_weight_decay ) 
    
    epochs = hpms.pretraining_epochs
    warmup_epochs = hpms.pretraining_warmup_epochs
    
    num_train_steps = math.ceil(len(t_loader))
    num_warmup_steps= num_train_steps * warmup_epochs
    num_training_steps=int(num_train_steps * epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps = num_warmup_steps,num_training_steps =num_training_steps)
    
    loop = range(epochs)
    for e in loop:
      
        train_loss,train_auc,tacc = train_one_epoch(t_loader,encoder,classifier,device,optimizer,criterion1,e,epochs,scheduler )

        print(f'For epoch {e+1}/{epochs}')
        print(f'average train_loss {train_loss}')
        print(f'average train_auc {train_auc}' )
        print(f'average train_acc {tacc}' )
        T_AUC.append(train_auc)
        T_LOSS.append(train_loss)     
        
        val_loss,val_auc,vacc = val_one_epoch(v_loader,encoder,classifier,device,criterion1)
        
        print(f'avarage val_loss { val_loss }')
        print(f'avarage val_auc {val_auc}')
        print(f'avarage val_acc {vacc}')
        V_AUC.append(val_auc)
        V_LOSS.append(val_loss)
        torch.save(encoder.state_dict(),OUTPUT_DIR+ f' Encoder_val_auc {val_auc}.pth')
        torch.save(classifier.state_dict(),OUTPUT_DIR+ f' Classiifier_val_auc {val_auc}.pth')
    
    return T_AUC, T_LOSS , V_AUC, V_LOSS

def plot_train_metrics(ta , tl , va , vl):
    fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=200)
    axs[0].plot(np.arange(0, len(ta)), tl, color='r', label='Train_loss')
    axs[0].plot(np.arange(0, len(ta)), vl, color='g', label='Val_loss')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss Curve")
    axs[0].legend()
    
    axs[1].plot(np.arange(0, len(ta)), ta, color='r', label='Train_AUC')
    axs[1].plot(np.arange(0, len(ta)), va, color='g', label='Val_AUC')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("AUC_Score")
    axs[1].set_title("AUC")
    axs[1].legend()
    
    plt.show()

def inference_func(encoder ,  classifier,test_loader , device, e_path, c_path ):

    encoder.load_state_dict(torch.load(e_path))
    classifier.load_state_dict(torch.load(c_path))
    encoder.eval()
    classifier.eval()
    bar = tqdm(test_loader)

    PREDS = []
    TARGET = []
    
    with torch.no_grad():
        for batch_idx, (images,t) in enumerate(bar):
            img = images.to(device)
            output = classifier(encoder(img))
            output = output.sigmoid()
            PREDS += [output.detach().cpu()]
            TARGET += [t]
        PREDS = torch.cat(PREDS).cpu().numpy()
        TARGET =  torch.cat(TARGET).cpu().numpy()
        
    return PREDS , TARGET

def binarize(x):
    ''' function to binarize( 1/0 ) the final outputs'''
    if(x >= 0.5):
        return 1.
    else:
        return 0.

def plot_test_metrics(PREDS , TARGET):     
    _, axs = plt.subplots(1, 2, figsize=(12,5), dpi=200)
    c_map = sns.color_palette("mako", as_cmap=True)
    fpr, tpr, _ = roc_curve(TARGET, PREDS)
    roc_auc = auc(fpr, tpr)
    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].plot(fpr, tpr, label='Model (ROC-AUC = {:.3f})'.format(roc_auc))
    axs[0].set_xlabel('False positive rate')
    axs[0].set_ylabel('True positive rate')
    axs[0].set_title('ROC curve')
    axs[0].legend(loc='best')
    
    axs[1].set_title('Confusion Matrix')
    cm = confusion_matrix(TARGET,[ binarize(x) for x in PREDS ])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm )
    disp.plot( cmap = c_map ,ax=axs[1])

    plt.show()


class PreTraining_Train():
    '''
    Arguments:
    _________

    encoder: Neural Network model outputting 
        a vector when input given an image

    classifier: Neural Network model outputting 
        a single logit when input given a vector
    
    device : where the model will be trained

    t_loader: Pytorch dataloader for 
        data (for training)

    v_loader : Pytorch dataloader for  
        data (for validation)

    hpms: Hyperparemetrs for training the network

    OUTPUT_DIR: Where trained models per epoch 
        will be saved
    
    plot_metrics : to plot the training 
        metric or not
    
    Returns:
    ________

    Differnt metrics for training and validation

    '''
    def __init__(self, encoder,classifier,device,t_loader , v_loader, hpms , OUTPUT_DIR,plot_metrics = True):
        
        self.encoder = encoder
        self.classifier =classifier
        self.device = device
        self.t_loader = t_loader
        self.v_loader = v_loader
        self.plot = plot_metrics
        self.hpms = hpms
        self.op = OUTPUT_DIR
    
    def train(self):
        ta , tl , va , vl = fit(self.encoder ,  self.classifier , self.device ,self.t_loader , self.v_loader,self.hpms,self.op )
        if(self.plot):
            plot_train_metrics(ta , tl , va , vl)


class PreTraining_Test():
    def __init__(self, encoder,classifier,device,test_loader , e_path, c_path):
        '''
    Arguments:
    _________

    encoder: Neural Network model outputting 
        a vector when input given an image

    classifier: Neural Network model outputting 
        a single logit when input given a vector
    
    device : where the model will be trained

    test_loader: Pytorch dataloader for 
        data (for test)
    
    e_path : path to the best encoder
        weights
    
    c_path : path to the best classifier
        weights
    
    Returns:
    ________

    Differnt metrics for testing

    '''
        
        self.encoder = encoder
        self.classifier =classifier
        self.device = device
        self.test_loader  = test_loader
        self.e_path  = e_path
        self.c_path  = c_path

    def test(self):
        pred, label = inference_func(self.encoder ,  self.classifier,self.test_loader , self.device, self.e_path, self.c_path )
        plot_test_metrics(pred, label)

