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

class EMA(object):
    """
    Exponential moving average weight optimizer for mean teacher model.
    """
    def __init__(self, student_parameters, teacher_parameters, alpha=0.999):        
        # get network parameters (weights)
        self.student_parameters = student_parameters
        self.teacher_parameters = teacher_parameters
        self.alpha = alpha

    def step(self):
        one_minus_alpha = 1.0 - self.alpha

        for student_p, teacher_p in zip(self.student_parameters, self.teacher_parameters):
            tmp = student_p.clone().detach()
            tmp1 = teacher_p.clone()
            tmp.mul_(one_minus_alpha)

            tmp1.mul_(self.alpha)
            tmp1.add_(tmp)

def train_one_epoch(s_loader,t_loader , s_encoder , t_encoder ,t_classifier,s_classifier, optimizer1,optimizer2,e,epochs,scheduler1,scheduler2 , device):
    '''Trains the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()
    
    t_encoder.train()
    s_encoder.train()
    t_classifier.train()
    s_classifier.train()


    loop = tqdm(zip(s_loader , t_loader),total = min(len(t_loader) , len(s_loader)))
    
    for (s_image,s_labels), (t_image1,t_image2) in loop:
        
        batch_size = s_labels.size(0)
        s_image = s_image.to(device)
        t_image1 = t_image1.to(device)
        t_image2 = t_image2.to(device)
        
        s_labels = s_labels.unsqueeze(1)
        s_labels= s_labels.to(device)

        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        #Step1
        output1 = s_classifier(s_encoder(s_image))
        loss1 = nn.BCEWithLogitsLoss()(output1 , s_labels.float())
        
        #step@
        output_x = s_classifier(s_encoder(t_image1))
        output_y = t_classifier(t_encoder(t_image2))
        
        loss2 = nn.MSELoss()(output_x , output_y)
        
        loss = loss1 + loss2
        
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        
        #scheduler1.step() 
        #scheduler2.step() 

        losses1.update(loss1.item(), batch_size)
        losses2.update(loss2.item(), batch_size)
        losses3.update(loss.item(), batch_size)


        loop.set_description(f"Epoch {e+1}/{epochs}")
        loop.set_postfix(Source_loss = loss1.item(), Gap_loss = loss2.item(),Total_loss = loss.item()  ,stage = 'train')
        
    
    return losses1.avg , losses2.avg ,losses3.avg 

def val_one_epoch(loader,t_encoder,t_classifier , device):
    '''Validates the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''
    losses = AverageMeter()
    scores1 = AverageMeter()
    scores2 = AverageMeter()
    
    t_encoder.eval()
    t_classifier.eval()
    
    global_step = 0
    loop = tqdm(enumerate(loader),total = len(loader))
    
    for step,(image,labels) in loop:
        image = image.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.no_grad():
            output = t_classifier(t_encoder(image))
        loss = nn.BCEWithLogitsLoss()(output,labels.float()) 
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

def fit(s_loader,t_loader ,tv_loader, s_encoder , t_encoder , t_classifier,s_classifier,hpms, OUTPUT_DIR,device):

    TRAIN_LOSS = []
    V_LOSS = []
    V_AUC = []
   
    optimizer1 = optim.AdamW(list(s_encoder.parameters()) + list(s_classifier.parameters()), lr= hpms.source_learning_rate , weight_decay = hpms.source_weight_decay ) 
    optimizer2 = optim.AdamW(list(t_encoder.parameters()) + list(t_classifier.parameters()), lr= hpms.target_learning_rate, weight_decay = hpms.target_weight_decay ) 
    
    epochs = hpms.epochs
    warmup_epochs = hpms.warmup_epochs
    
    num_train_steps = math.ceil(len(t_loader))
    num_warmup_steps= num_train_steps * warmup_epochs
    num_training_steps= int(num_train_steps * epochs)
    scheduler1 = get_cosine_schedule_with_warmup(optimizer1,num_warmup_steps = num_warmup_steps,num_training_steps =num_training_steps)
    scheduler2 = get_cosine_schedule_with_warmup(optimizer2,num_warmup_steps = num_warmup_steps,num_training_steps =num_training_steps)
    
    loop = range(epochs)
    for e in loop:
      
        loss1,loss2, loss3 = train_one_epoch(s_loader,t_loader , s_encoder , t_encoder ,t_classifier,s_classifier, 
                                        optimizer1,optimizer2,e,epochs,scheduler1,scheduler2 , device)

        print(f'For epoch {e+1}/{epochs}')
        print(f'loss1 {loss1}')
        print(f'loss2 {loss2}')
        print(f'Total loss {loss3}')
        
        TRAIN_LOSS.append(loss3)
        
        val_loss,val_auc,vacc = val_one_epoch(tv_loader,t_encoder,t_classifier , device)
        
        print(f'avarage val_loss { val_loss }')
        print(f'avarage val_auc {val_auc}')
        print(f'avarage val_acc {vacc}')
        
        V_LOSS.append(val_loss)
        V_AUC.append(val_auc)

        torch.save(t_encoder.state_dict(),OUTPUT_DIR+ f'Encoder_val_auc {val_auc}.pth')
        torch.save(t_classifier.state_dict(),OUTPUT_DIR+ f'Classifier_val_auc {val_auc}.pth')

    return TRAIN_LOSS , V_LOSS , V_AUC

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

def plot_train_metrics(T_LOSS , V_LOSS , V_AUC):
    fig, axs = plt.subplots(3, 1, figsize=(10,15), dpi=200)

    axs[0].plot(np.arange(0, len(T_LOSS)), T_LOSS, color='g', label='Training_loss')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss Curve (Training)")
    axs[0].legend()

    axs[1].plot(np.arange(0, len(T_LOSS)), V_LOSS, color='b', label='Val_Loss')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Loss Curve")
    axs[1].legend()

    axs[2].plot(np.arange(0, len(T_LOSS)), V_AUC, color='y', label='Val_AUC')
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("AUC score")
    axs[2].set_title("AUC score Curve")
    axs[2].legend()

    plt.show()

class SE_Train():
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

    s_classifier: Neural Network model outputting 
        a single logit when input given 
        a vector (for source images)
    
    t_classifier: Neural Network model outputting 
        a single logit when input given a 
        vector (for target images)

    hpms: Hyperparemetrs for training the network

    OUTPUT_DIR: Where trained models per epoch will be
        saved
    
    device: the devicec where the model will
        be trained on
    
    Returns:
    ________

    Differnt metrics for training and validation

    '''
    def __init__(self, s_loader,t_loader ,tv_loader, s_encoder , t_encoder , t_classifier,s_classifier, hpms,OUTPUT_DIR,device):
        
        self.s_encoder = s_encoder
        self.t_encoder = t_encoder
        self.t_classifier =t_classifier
        self.s_classifier =s_classifier
        self.t_loader = t_loader
        self.s_loader = s_loader
        self.tv_loader = tv_loader
        self.device = device
        self.hpms = hpms
        self.op = OUTPUT_DIR
    
    def train(self):
        T_LOSS , V_LOSS , V_AUC = fit(self.s_loader,self.t_loader ,self.tv_loader, self.s_encoder ,
                                             self.t_encoder ,self.t_classifier,self.s_classifier,self.hpms,self.op,self.device)
        plot_train_metrics(T_LOSS , V_LOSS , V_AUC )
