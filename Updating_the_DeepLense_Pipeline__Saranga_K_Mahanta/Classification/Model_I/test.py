import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc

from utils import device, calculate_accuracy, transforms
from model import EffNetB2_backbone_model
from dataloader import create_data_loaders
from config import  BATCH_SIZE, MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH


def test_epoch(model, dataloader,criterion):
    model.eval()
    test_loss = []
    test_accuracy = []
    
    y_pred_list = []
    y_pred_prob_list = []
    y_truth_list = []

    with torch.no_grad():
        loop=tqdm(enumerate(dataloader),total=len(dataloader))
        
        for batch_idx, (img_batch,labels) in loop:
            X = img_batch.to(device)
            y_truth = labels.to(device)
            y_truth_list.append(y_truth.detach().cpu().numpy())

            #forward prop
            y_pred = model(X)
            y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
            y_pred_prob_list.append(torch.softmax(y_pred, dim = 1).detach().cpu().numpy())
            _, y_pred_labels = torch.max(y_pred_softmax, dim = 1)
            y_pred_list.append(y_pred_labels.detach().cpu().numpy())

            #loss and accuracy calculation
            loss = criterion(y_pred, y_truth)
            accuracy = calculate_accuracy(y_pred, y_truth)

            #batch loss and accuracy
            # print(f'Partial train loss: {loss.data}')
            test_loss.append(loss.detach().cpu().numpy())
            test_accuracy.append(accuracy.detach().cpu().numpy())
            
    return y_pred_prob_list, y_pred_list, y_truth_list, np.mean(test_loss), np.mean(test_accuracy)


def flatten_list(x):
    flattened_list = []
    for i in x:
        for j in i:
            flattened_list.append(j)
            
    return flattened_list


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    model = EffNetB2_backbone_model().to(device)

    if device != 'cpu':
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location = torch.device('cpu')))
    
    _, _, test_loader, class_map = create_data_loaders(TRAIN_DATA_PATH, TEST_DATA_PATH, 
                                            val_split = 0.2, batch_size = BATCH_SIZE,
                                            transforms = transforms,
                                            class_map = True)
    idx2class = {v: k for k, v in class_map.items()}
    class_names = [i for i in class_map.keys()]

    y_pred_prob_list, y_pred_list, y_truth_list, test_loss, test_accuracy = test_epoch(model, test_loader, criterion)
    y_pred_list_flattened = flatten_list(y_pred_list)
    y_truth_list_flattened = flatten_list(y_truth_list)
    y_pred_prob_list_flattened = flatten_list(y_pred_prob_list)

    print(f"\nTest set loss: {test_loss} \nTest_accuracy:{test_accuracy}")
    print("Classification Report:\n")
    print(classification_report(y_truth_list_flattened, y_pred_list_flattened,target_names = class_names))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_pred_list_flattened, y_truth_list_flattened))
    print("\nMacro-average one-vs-rest ROC-AUC score: ", end = ' ')
    print(roc_auc_score(y_truth_list_flattened, y_pred_prob_list_flattened, average='macro', multi_class="ovr"))
    print("Macro-average one-vs-one ROC-AUC score: ", end = ' ')
    print(roc_auc_score(y_truth_list_flattened, y_pred_prob_list_flattened, average='macro', multi_class="ovo"))
    print("Weighted-average one-vs-rest ROC-AUC score: ", end = ' ')
    print(roc_auc_score(y_truth_list_flattened, y_pred_prob_list_flattened, average='weighted', multi_class="ovr"))
    print("Weighted-average one-vs-one ROC-AUC score: ", end = ' ')
    print(roc_auc_score(y_truth_list_flattened, y_pred_prob_list_flattened, average='weighted', multi_class="ovo"))
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    thresh ={}
    n_class = len(class_map)
    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(np.array(y_truth_list_flattened), np.array(y_pred_prob_list_flattened)[:,i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(12,10))
    plt.rcParams['font.size'] = '30'
    for i in range(0,n_class):
        plt.plot(fpr[i], tpr[i], linestyle='--', label=f'{idx2class[i]} AUC = {roc_auc[i]:.3f}')
        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
    # plt.savefig('ROC_curves.png',dpi = 352)
    plt.show()

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_truth_list_flattened, y_pred_list_flattened)).rename(columns=idx2class, index=idx2class)
    fig, ax = plt.subplots(figsize=(10,20))         
    sns.heatmap(confusion_matrix_df, fmt = ".0f", annot=True, ax=ax)
    # plt.savefig('Confusion_matrix.png', dpi = 352)
    plt.show()

    
