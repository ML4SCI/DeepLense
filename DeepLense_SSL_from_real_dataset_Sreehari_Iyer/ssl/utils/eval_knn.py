# code adapted from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn_accuracy(
        model,
        train_dataloader, 
        test_dataloader, 
        classes, 
        knn_k, 
        knn_t = 0.1,
        device: str = "cuda",
        use_dense_prediction: bool = False,
        masked_loader: bool = True,
        return_all_tokens: bool = True,
    ):
    top1, top5 = 0., 0.
    count = 0
    train_X = []
    train_y = []
    
    with torch.no_grad():
        for data in train_dataloader:
            X, y = None, None
            if masked_loader:
                X, y, _ = data
            else:
                X, y = data
            feat = None
            if return_all_tokens:
                feat = model(X.to(device, non_blocking=True))[:, 0]
            else:
                feat = model(X.to(device, non_blocking=True))
            feat = F.normalize(feat, dim=1)
            train_X.append(feat)
            train_y.append(y.to(device, non_blocking=True))
        
        train_X = torch.cat(train_X, dim=0).t().contiguous()
        train_y = torch.cat(train_y, dim=0).contiguous()

        for data in test_dataloader:
            X, y = None, None
            if masked_loader:
                X, y, _ = data
            else:
                X, y = data
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            feat = None
            if return_all_tokens:
                feat = model(X)[:, 0]
            else:
                feat = model(X)
            feat = F.normalize(feat, dim=1)
            pred_labels = knn_predict(feat, train_X, train_y, classes, knn_k, knn_t)
            count += X.size(0)
            top1 += (pred_labels[:, 0] == y).float().sum().item()
            top5 += (pred_labels[:, :5] == y.unsqueeze(1)).any(dim=1).float().sum().item()
            
    return ((top1 / count) * 100), ((top5 / count) * 100) 
    
    
def knn_predict(
        feat, 
        train_X, 
        train_y, 
        num_classes, 
        knn_k, 
        knn_t):
    sim_matrix = torch.mm(feat, train_X)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(train_y.expand(feat.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feat.size(0) * knn_k, num_classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feat.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    
    return pred_labels
    




        