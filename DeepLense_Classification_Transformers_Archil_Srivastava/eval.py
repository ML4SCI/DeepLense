import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import auroc as auroc_fn, accuracy as accuracy_fn
import wandb
import numpy as np
import argparse
import os

from models import get_timm_model
from models.baseline import BaselineModel
from data import LensDataset, get_transforms
from constants import *
from utils import get_device

@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    loss, accuracy, class_auroc, micro_auroc, macro_auroc = [], [], [], [], []
    logits, y = [], []
    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device, dtype=torch.float), batch_y.type(torch.LongTensor)
        logits.append(model(batch_X).cpu())
        y.append(batch_y)
    
    logits, y = torch.cat(logits), torch.cat(y)
    loss.append(loss_fn(logits, y))
    accuracy.append(accuracy_fn(logits, y, num_classes=NUM_CLASSES))
    class_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average=None))
    #micro_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average='micro'))
    macro_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average='macro'))

    result = {
        'ground_truth': y,
        'logits': logits,
        'loss': np.mean(loss),
        'accuracy': np.mean(accuracy),
        'micro_auroc': np.mean(micro_auroc),
        'macro_auroc': np.mean(macro_auroc)
    }

    class_auroc = class_auroc[0]
    for i, label in enumerate(LABELS):
        result[f'{label}_auroc'] = class_auroc[i]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runid', type=str, help='ID of train run')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'best'], default='best')
    run_config = parser.parse_args()

    with wandb.init(entity='_archil', id=run_config.runid, resume='must'):
        complex = bool(wandb.config.complex)
        pretrained = bool(wandb.config.pretrained)
        tune = bool(wandb.config.tune)

        device = get_device(run_config.device)
        
        if wandb.config.dataset == 'Model_I':
            IMAGE_SIZE = 150
        elif wandb.config.dataset == 'Model_II' or wandb.config.dataset == 'Model_III':
            IMAGE_SIZE = 64
        else:
            IMAGE_SIZE = None

        INPUT_SIZE = IMAGE_SIZE
        if wandb.config.model_source == 'baseline':
            model = BaselineModel(image_size=INPUT_SIZE).to(device)
        elif wandb.config.model_source == 'timm':
            INPUT_SIZE = TIMM_IMAGE_SIZE[wandb.config.model_name]
            model = get_timm_model(wandb.config.model_name, complex=complex).to(device)
        else:
            model = None
        weights_file = wandb.restore('best_model.pt')
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'best_model.pt')))

        datapath = os.path.join('./data', wandb.config.dataset, 'memmap', 'test')
        dataset = LensDataset(image_size=IMAGE_SIZE, memmap_path=datapath,
                              mean=wandb.run.summary['norm_mean'], std=wandb.run.summary['norm_std'],
                              transform=get_transforms(wandb.config, final_size=INPUT_SIZE, mode='test'))
        data_loader = DataLoader(dataset, batch_size=wandb.config.batchsize, shuffle=False)

        if device == 'cuda' and torch.cuda.device_count() > 1:
            device = 'cuda:0'
            model = torch.nn.DataParallel(model)
            model = model.to(device)

        criterion = CrossEntropyLoss()

        metrics = evaluate(model, data_loader, criterion, device=device)

        wandb.run.summary['test_loss'] = metrics['loss']
        wandb.run.summary['test_accuracy'] = metrics['accuracy']
        wandb.run.summary['test_micro_auroc'] = metrics['micro_auroc']
        wandb.run.summary['test_macro_auroc'] = metrics['macro_auroc']
        for label in LABELS:
            wandb.run.summary[f'test_{label}_auroc'] = metrics[f'{label}_auroc']

        wandb.log({
            'test_roc': wandb.plot.roc_curve(metrics['ground_truth'],
                                        torch.nn.functional.softmax(metrics['logits'], dim=-1),
                                        labels=LABELS)
        })

