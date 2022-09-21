import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def calculate_accuracy(y_pred, y_truth):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_labels = torch.max(y_pred_softmax, dim = 1)
    
    correct_preds = (y_pred_labels == y_truth).float()
    acc = correct_preds.sum() / len(correct_preds)
    acc = torch.round(acc*100)
    
    return acc  

device ='cuda' if torch.cuda.is_available() else 'cpu'

#pre-processing transformation
transforms = A.Compose(
        [
            A.CenterCrop(height = 100, width = 100, p=1.0),
            ToTensorV2()
        ]
    )