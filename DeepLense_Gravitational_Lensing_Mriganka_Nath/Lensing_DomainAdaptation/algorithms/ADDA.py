import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm.notebook import tqdm
from sklearn import model_selection
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
from sklearn import  model_selection

class AverageMeter(object):
    #Computes and stores the average and current value
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

def train_one_epoch(s_loader,t_loader , s_encoder , t_encoder , discriminator , optimizer_discriminative,optimizer_target,criterion,e,epochs,scheduler1 , scheduler2, device ):
    '''Trains the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    
    t_encoder.train()
    discriminator.train()
    s_encoder.eval()

    loop = tqdm(zip(s_loader , t_loader),total = min(len(s_loader) , len(t_loader)) )
    
    for (s_image,s_labels), (t_image,t_labels) in loop:
        
        batch_size = t_labels.size(0)
        s_image = s_image.to(device)
        t_image = t_image.to(device)
        
        s_labels = s_labels.unsqueeze(1)
        t_labels = t_labels.unsqueeze(1)
        s_labels= s_labels.to(device)
        t_labels= t_labels.to(device)
        
        optimizer_discriminative.zero_grad()
        
        s_feat = s_encoder(s_image)
        t_feat = t_encoder(t_image)
        
        s_op = discriminator(s_feat)
        t_op = discriminator(t_feat)
        
        loss1 = criterion(s_op , s_labels.float())
        loss2 = criterion(t_op , t_labels.float())
        
        discriminative_loss = loss1 + loss2
        discriminative_loss.backward()
        optimizer_discriminative.step()
        scheduler1.step() 
        
        optimizer_discriminative.zero_grad()
        optimizer_target.zero_grad()
        
        new_t_labels =  torch.zeros(batch_size, dtype=torch.long).to(device)
        t_feat = t_encoder(t_image)
        t_op = discriminator(t_feat)
        
     
        target_loss = criterion(t_op , new_t_labels.unsqueeze(1).float())
        target_loss.backward()
        optimizer_target.step()
        scheduler2.step() 

        losses1.update(discriminative_loss.item(), batch_size)
        losses2.update(target_loss.item(), batch_size)


        loop.set_description(f"Epoch {e+1}/{epochs}")
        loop.set_postfix(discriminative_loss = discriminative_loss.item(), target_loss = target_loss.item()  ,stage = 'train')
        
    
    return losses1.avg , losses2.avg 

def val_one_epoch(loader,target_encoder,classifier,criterion,device):
    '''Validates the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''
    losses = AverageMeter()
    scores1 = AverageMeter()
    scores2 = AverageMeter()
    
    target_encoder.eval()
    classifier.eval()
    
    global_step = 0
    loop = tqdm(enumerate(loader),total = len(loader))
    
    for step,(image,labels) in loop:
        image = image.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.no_grad():
            output = classifier(target_encoder(image))
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

def fit(s_loader,t_loader ,tv_loader, s_encoder , t_encoder , discriminator,classifier, hpms , OUTPUT_DIR,device):
    
    D_LOSS = []
    T_LOSS = []
    V_LOSS = []
    V_AUC = []
   
    criterion = nn.BCEWithLogitsLoss() # Loss function
    dis_optimizer = optim.AdamW(discriminator.parameters(), lr=hpms.discriminator_learning_rate , weight_decay = hpms.discriminator_weight_decay ) 
    tar_optimizer = optim.AdamW(t_encoder.parameters(), lr=hpms.target_learning_rate , weight_decay = hpms.targetweight_decay ) 
    
    epochs = hpms.adversarial_epochs
    warmup_epochs = hpms.adversarial_warmup_epochs
    
    num_train_steps = math.ceil(len(t_loader))
    num_warmup_steps= num_train_steps * warmup_epochs
    num_training_steps=int(num_train_steps * epochs)
    scheduler1 = get_cosine_schedule_with_warmup(dis_optimizer,num_warmup_steps = num_warmup_steps,num_training_steps =num_training_steps)
    scheduler2 = get_cosine_schedule_with_warmup(tar_optimizer,num_warmup_steps = num_warmup_steps,num_training_steps =num_training_steps)
    
    loop = range(epochs)
    for e in loop:
      
        loss1,loss2 = train_one_epoch(s_loader,t_loader , s_encoder , t_encoder , discriminator , 
                    dis_optimizer,tar_optimizer,criterion,e,epochs,scheduler1 , scheduler2,device  )

        print(f'For epoch {e+1}/{epochs}')
        print(f'average discriminative train_loss {loss1}')
        print(f'average target_loss {loss2}')
        
        D_LOSS.append(loss1)
        T_LOSS.append(loss2)
        
        val_loss,val_auc,vacc = val_one_epoch(tv_loader,t_encoder,classifier,criterion,device)
        
        print(f'avarage val_loss { val_loss }')
        print(f'avarage val_auc {val_auc}')
        print(f'avarage val_acc {vacc}')
        
        V_LOSS.append(val_loss)
        V_AUC.append(val_auc)

        torch.save(discriminator.state_dict(),OUTPUT_DIR+ f'Discriminator_val_auc {val_auc}.pth')
        torch.save(t_encoder.state_dict(),OUTPUT_DIR+ f'Target_Encoder_val_auc {val_auc}.pth')

    return D_LOSS,T_LOSS , V_LOSS , V_AUC

def test_func(t_encoder ,  classifier,test_loader , device, e_path, c_path ):

    t_encoder.load_state_dict(torch.load(e_path))
    classifier.load_state_dict(torch.load(c_path))
    t_encoder.eval()
    classifier.eval()
    bar = tqdm(test_loader)

    PREDS = []
    TARGET = []
    
    with torch.no_grad():
        for batch_idx, (images,t) in enumerate(bar):
            img = images.to(device)
            output = classifier(t_encoder(img))
            output = output.sigmoid()
            PREDS += [output.detach().cpu()]
            TARGET += [t]
        PREDS = torch.cat(PREDS).cpu().numpy()
        TARGET =  torch.cat(TARGET).cpu().numpy()
        
    return PREDS , TARGET

def binarize(x):
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

def plot_train_metrics(D_LOSS,T_LOSS , V_LOSS , V_AUC):
    fig, axs = plt.subplots(3, 1, figsize=(10,15), dpi=200)

    axs[0].plot(np.arange(0, len(D_LOSS)), D_LOSS, color='r', label='Discriminator_loss')
    axs[0].plot(np.arange(0, len(D_LOSS)), T_LOSS, color='g', label='Target_loss')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss Curve (Training)")
    axs[0].legend()

    axs[1].plot(np.arange(0, len(D_LOSS)), V_LOSS, color='b', label='Val_Loss')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Loss Curve")
    axs[1].legend()

    axs[2].plot(np.arange(0, len(D_LOSS)), V_AUC, color='y', label='Val_AUC')
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("AUC score")
    axs[2].set_title("AUC score Curve")
    axs[2].legend()

    plt.show()


class ADDA_Train():

    '''
    Arguments:
    _________

    s_loader: Pytorch dataloader for source 
        data (for training)

    t_loader : Pytorch dataloader for Target 
        data (dor training)

    tv_loader: Pytorch dataloader for validation 
        data(contains target data)

    s_encoder: Neural Network model outputting 
        a vector when input given an 
        image (for source images)
    
    t_encoder: Neural Network model outputting 
        a vector when input given an 
        image (for target images)

    discriminator: Neural Network model outputting 
        a single logit when input given a vector

    classifier: Neural Network model outputting 
        a single logit when input given a vector

    hpms: Hyperparemetrs for training the network

    OUTPUT_DIR: Where trained models per epoch will be
        saved
    
    device: the devicec where the model will
        be trained on
    
    Returns:
    ________

    Differnt metrics for training and validation

    '''
    def __init__(self, s_loader,t_loader ,tv_loader, s_encoder , t_encoder , discriminator,classifier,hpms,OUTPUT_DIR,device):
        
        self.s_encoder = s_encoder
        self.t_encoder = t_encoder
        self.classifier =classifier
        self.discriminator = discriminator
        self.t_loader = t_loader
        self.s_loader = s_loader
        self.tv_loader = tv_loader
        self.device = device
        self.hpms = hpms
        self.op = OUTPUT_DIR

        for param in self.s_encoder.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = False

    
    def train(self):
        D_LOSS,T_LOSS , V_LOSS , V_AUC = fit(self.s_loader,self.t_loader ,self.tv_loader, self.s_encoder ,
                                             self.t_encoder , self.discriminator,self.classifier,self.hpms,self.op,self.device)
        plot_train_metrics(D_LOSS,T_LOSS , V_LOSS , V_AUC )

