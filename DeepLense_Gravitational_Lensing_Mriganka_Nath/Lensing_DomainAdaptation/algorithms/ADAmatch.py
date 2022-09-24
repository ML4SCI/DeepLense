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

def deactivate_batchnorm(m):
    # deactivate batchnorm tracking for the model
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False
        
def activate_batchnorm(m):
    # activates batchnorm tracking for the model
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True


def train_one_epoch(encoder,classifier , s_loader , t_loader , tau , optimizer , scheduler  ,e,epochs,device ):
    '''Trains the model for a single epoch and returns the differnt Loss that epoch'''

    losses1 = AverageMeter() #source_loss
    losses2 = AverageMeter() #target_loss
    losses3 = AverageMeter() #total_loss
    current_step = 0
    
    iters = min(len(s_loader), len(t_loader))
    steps_per_epoch = iters
    total_steps = epochs * steps_per_epoch 
    
    encoder.train()
    classifier.eval()

    loop = tqdm(zip(s_loader , t_loader),total = min(len(t_loader) , len(s_loader)) )
    
    for (s_weak,s_strong, labels), (t_weak,t_strong, _) in loop:
        
        batch_size = labels.size(0)    
        s_weak = s_weak.to(device)
        s_strong = s_strong.to(device)
        t_weak = t_weak.to(device)
        t_strong = t_strong.to(device)
        labels = labels.to(device)
        
        s_combined = torch.cat([s_weak,s_strong],0)
        all_combined = torch.cat([s_weak,s_strong , t_weak , t_strong],0)
        
        s_size = s_combined.size(0)
        
        optimizer.zero_grad()
        
        all_logits = classifier(encoder(all_combined))
        s_logits1 = all_logits[:s_size]
        
        # In my experiments enabling/disabling batchnorm was not working so I commented it out
        
        #disable_batch_norm
        #encoder.apply(deactivate_batchnorm)
        
        s_logits2 = classifier(encoder(s_combined))
        
        #enable batch_norm
        #encoder.apply(activate_batchnorm)

        
        # perform random logit interpolation
        lambd = torch.rand_like(s_logits1).to(device)
        final_logits_source = (lambd * s_logits1) + ((1-lambd) * s_logits2)
        
         # distribution allignment
        s_weak_logits = final_logits_source[:s_weak.size(0)]
        t_logits = all_logits[s_size:]
        t_weak_logits = t_logits[:t_weak.size(0)]
        
        sigmoid_t_weak_logits = torch.sigmoid(t_weak_logits)
        sigmoid_s_weak_logits =  torch.sigmoid(s_weak_logits)
        
        # Find psedolabels for the weak augmented target images
        expectation_ratio = (1e-6 + torch.mean(sigmoid_s_weak_logits)) / (1e-6 + torch.mean(sigmoid_t_weak_logits))
        final_pseudolabels = torch.sigmoid(sigmoid_t_weak_logits * expectation_ratio)
        
        # relative confidence thresholding
        row_wise_max, _ = torch.max(sigmoid_t_weak_logits , dim=1)
        final_sum = torch.mean(row_wise_max, 0)
        
        c_tau = tau * final_sum
        max_values, _ = torch.max(final_pseudolabels, dim=1)
        mask = (max_values >= c_tau).float()
    
        # source loss
        source_loss1 = nn.BCEWithLogitsLoss()(s_weak_logits , labels.unsqueeze(1))
        source_loss2 = nn.BCEWithLogitsLoss()(final_logits_source[s_weak.size(0):] ,labels.unsqueeze(1))
        source_loss = (source_loss1 + source_loss2)/2.

        # target loss (between psedolabels and predicted logits)
        t_strong_logits = t_logits[t_weak.size(0):]
        target_loss = (nn.BCEWithLogitsLoss()((final_pseudolabels.detach()>0.5).float()  ,t_strong_logits ))*mask.mean()
        
        pi = torch.tensor(np.pi, dtype=torch.float).to(device)
        step = torch.tensor(current_step, dtype=torch.float).to(device)
        mu = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / total_steps)) / 2
        current_step += 1
        
        # total loss
        loss = source_loss + (mu * target_loss)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses1.update(source_loss.item(), batch_size)
        losses2.update(target_loss.item(), batch_size)
        losses3.update(loss.item(), batch_size)

        loop.set_description(f"Epoch {e+1}/{epochs}")
        loop.set_postfix(source_loss = source_loss.item(), target_loss = target_loss.item()  , loss =loss.item(),stage = 'train')

    return losses1.avg , losses2.avg , losses3.avg

def val_one_epoch(loader,encoder,classifier,device):
    '''Validates the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''
    losses = AverageMeter()
    scores1 = AverageMeter()
    scores2 = AverageMeter()
    
    encoder.eval()
    classifier.eval()
    
    global_step = 0
    loop = tqdm(enumerate(loader),total = len(loader))
    
    for step,(image,labels) in loop:
        image = image.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.to(device)

        batch_size = labels.size(0)
        with torch.no_grad():
            output = classifier(encoder(image))
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

def fit(s_loader,t_loader ,tv_loader, encoder  ,classifier,hpms, OUTPUT_DIR,device):
    ''' The Training/Validation loop '''
    optimizer = optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr= hpms.learning_rate , weight_decay = hpms.weight_decay ) 
    tau= hpms.tau
    epochs = hpms.epochs
    warmup_epochs = hpms.warmup_epochs
    num_train_steps = math.ceil(len(t_loader))
    num_warmup_steps= num_train_steps * warmup_epochs
    num_training_steps=int(num_train_steps * epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps = num_warmup_steps,num_training_steps =num_training_steps)

    TRAIN_LOSS = []
    V_LOSS = []
    V_AUC = []

    loop = range(epochs)
    for e in loop:

        loss1,loss2,loss3 = train_one_epoch(encoder,classifier , s_loader , t_loader , tau , optimizer , scheduler ,e,epochs,device)

        print(f'For epoch {e+1}/{epochs}')
        print(f'loss1 {loss1}')
        print(f'loss2 {loss2}')
        print(f'Total loss {loss3}')

        TRAIN_LOSS.append(loss3)

        val_loss,val_auc,vacc = val_one_epoch(tv_loader,encoder,classifier,device)

        print(f'avarage val_loss { val_loss }')
        print(f'avarage val_auc {val_auc}')
        print(f'avarage val_acc {vacc}')

        V_LOSS.append(val_loss)
        V_AUC.append(val_auc)
        
        torch.save(encoder.state_dict(),OUTPUT_DIR+ f'Encoder_val_auc {val_auc}.pth')
        torch.save(classifier.state_dict(),OUTPUT_DIR+ f'Classifier_val_auc {val_auc}.pth')

    return TRAIN_LOSS , V_LOSS , V_AUC

def test_func(t_encoder ,  classifier,test_loader , device, e_path, c_path ):
    '''Used to find the final predictions on the test set '''
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
    '''required for plotting the graphs'''
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

class Adamatch_Train():
    '''
    Arguments:
    _________

    s_loader: Pytorch dataloader for source 
        data (for training)

    t_loader : Pytorch dataloader for Target 
        data (dor training)

    tv_loader: Pytorch dataloader for validation 
        data(contains target data)

    encoder: Neural Network model outputting 
        a vector when input given an image

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
    def __init__(self, s_loader,t_loader ,tv_loader, encoder  ,classifier,hpms, OUTPUT_DIR,device):
        
        self.encoder = encoder
        self.classifier = classifier
        self.t_loader = t_loader
        self.s_loader = s_loader
        self.tv_loader = tv_loader
        self.device = device
        self.hpms = hpms
        self.op = OUTPUT_DIR

    
    def train(self):
        T_LOSS , V_LOSS , V_AUC = fit(self.s_loader,self.t_loader ,self.tv_loader, self.encoder ,
                                            self.classifier,self.hpms,self.op,self.device)
        plot_train_metrics(T_LOSS , V_LOSS , V_AUC )