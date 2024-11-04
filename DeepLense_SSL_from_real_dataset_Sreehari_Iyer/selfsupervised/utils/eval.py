import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as Transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support
from typing import List
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
import copy
from typing import Optional, List

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def get_dataloader(
        data_path: str,
        eval_transforms: Transforms,
        indices: List[int],
        batch_size: int,
        shuffle: bool,
    ):
    dataset = datasets.DatasetFolder(
        root=data_path,
        loader=npy_loader,
        extensions=['.npy'],
        transform=eval_transforms
    )
    if indices is not None:
        dataset.samples = [dataset.samples[i] for i in indices.indices]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def test_(
        model,
        test_loader, 
        criterion,
    ):    
    val_preds = []
    val_y = []
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    test_loss = 0
    output = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        output.extend(outputs.detach().cpu())
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        val_y.extend(labels.tolist())
    y_true=np.array(val_y).astype(np.float16)
    y_pred=(np.argmax(output, axis=-1)).astype(np.float16)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Evaluation on held out test dataset")
    print(f"Confusion Matrix")
    t = PrettyTable(['', 'predicted lenses', 'predicted nonlenses'])
    t.add_row(['true lenses', cm[0][0], cm[0][1]])
    t.add_row(['true nonlenses', cm[1][0], cm[1][1]])
    print(t) 
    
    loss /= len(test_loader)
    acc = accuracy_score(y_true, y_pred)*100
    
    print("Test Metrics")
    t = PrettyTable(header=False)
    t.add_row(['accuracy', f'{acc:.4f}%'])
    t.add_row(['loss', f'{loss:.4f}'])
    auc = roc_auc_score(y_true, np.array(output).T[1], average = 'macro')
    t.add_row(['auc score', f'{auc:.4f}'])
    print(t)
    
    t = PrettyTable(['', 'precision', 'recall', 'f-score', 'support'])
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0.0)
    t.add_row(['lenses', f'{precision[0]:.4f}', f'{recall[0]:.4f}', f'{f_score[0]:.4f}', f'{support[0]}'])
    t.add_row(['nonlenses', f'{precision[1]:.4f}', f'{recall[1]:.4f}', f'{f_score[1]:.4f}', f'{support[1]}'])
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0.0)
    t.add_row(['macro averaged', f'{precision:.4f}', f'{recall:.4f}', f'{f_score:.4f}', ''])
    print(t)

    return output, y_true, acc, auc

def train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs: int = 100,
        scheduler: Optional = None,
        device: str = "cuda"
    ):
    model = model.to(device)
    best_loss, best_val_loss, best_acc, best_model = None, None, None, None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = np.argmax(outputs.cpu().numpy(), axis=-1)
                # val_preds.extend(predicted.tolist())
                # val_y.extend(labels.tolist())
                total += labels.size(0)
                correct += (np.count_nonzero(predicted == labels.cpu().numpy()))
        val_loss = val_loss/len(val_loader)
        acc = 100 * correct / total
        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
            best_loss = running_loss / len(train_loader)
            best_model = copy.deepcopy(model)
            best_acc = acc
        print(f'[{epoch+1}/{num_epochs}] Train Loss: {(running_loss / len(train_loader)):.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {acc:.2f}%\n')
    return model, best_model
