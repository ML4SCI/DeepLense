import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import argparse
import os
import wandb

from data import LensDataset, WrapperDataset, get_transforms
from constants import *
from models import get_timm_model
from models.baseline import BaselineModel
from utils import get_device, set_seed
from eval import evaluate

def train_step(model, images, labels, optimizer, scheduler, criterion, device='cpu'):
    images, labels = images.to(device, dtype=torch.float), labels.type(torch.LongTensor).to(device)
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step(loss)
    return loss


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, log_interval=100):
    wandb.watch(model, criterion, log='all', log_freq=log_interval)
    best_val_auroc, best_val_metrics = 0., dict()
    batch_num = 0
    for epoch in range(1, epochs + 1):
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch_num += 1
            model.train()
            images, labels = batch_data
            loss = train_step(model, images, labels, optimizer, scheduler, criterion, device=device)

            if batch_num % log_interval == 0:
                val_metrics = evaluate(model, val_loader, criterion, device=device)

                # Log in wandb
                log_dict = {
                    'epoch': epoch,
                    'batch_num': batch_num,
                    'train/loss': loss,
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/micro_auroc': val_metrics['micro_auroc'],
                    'val/macro_auroc': val_metrics['macro_auroc']
                }
                for label in LABELS:
                    log_dict[f'val/{label}_auroc'] = val_metrics[f'{label}_auroc']
                wandb.log(log_dict, step=batch_num)
                wandb.log({
                    'roc': wandb.plot.roc_curve(val_metrics['ground_truth'],
                                                torch.nn.functional.softmax(val_metrics['logits'], dim=-1),
                                                labels=LABELS)
                })

                # Track best val auroc
                if val_metrics['macro_auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['macro_auroc']
                    wandb.run.summary['best_val_micro_auroc'] = val_metrics['micro_auroc']
                    wandb.run.summary['best_val_macro_auroc'] = best_val_auroc
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_batch_num'] = batch_num
                    for label in LABELS:
                        wandb.run.summary[f'best_val_{label}_auroc'] = val_metrics[f'{label}_auroc']
                    best_val_metrics = val_metrics
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))

        # Sync best model at a lesser frequency (i.e. at the end of each epoch)
        wandb.save(os.path.join(wandb.run.dir, 'best_model.pt'))

    return best_val_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Fixed params
    parser.add_argument('--dataset', choices=['Model_I', 'Model_II', 'Model_III', 'Model_IV'], default='Model_I', help='which data model')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--model_source', choices=['baseline', 'timm'], default='timm')
    # Timm Specific
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--complex', type=int, choices=[0, 1], default=1)
    parser.add_argument('--pretrained', type=int, choices=[0, 1], default=1)
    parser.add_argument('--tune', type=int, choices=[0, 1], default=1, help='Whether to further tune (1) pretrained model (if any) or freeze the pretrained weights (0)')
    ###
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'tpu', 'best'], default='best')

    # Augmentations
    parser.add_argument('--random_zoom', type=float, default=0.9)
    parser.add_argument('--random_rotation', type=float, default=90)

    # Common hyperparameters
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--decay_lr', type=int, default=0)

    run_config = parser.parse_args()

    pretrained = bool(run_config.pretrained)
    tune = bool(run_config.tune) or not pretrained
    complex = bool(run_config.complex)

    # Group for wandb
    group = 'baseline'
    if run_config.model_source == 'timm':
        group = f'timm-{run_config.model_name}'
        if complex:
            group = f'{group}-complex'

    with wandb.init(entity='_archil', config=run_config, group=group, job_type=f'{run_config.dataset}'):
        if run_config.seed:
            set_seed(run_config.seed)
        
        device = get_device(run_config.device)
        
        if run_config.dataset == 'Model_I':
            IMAGE_SIZE = 150
        elif run_config.dataset == 'Model_II' or run_config.dataset == 'Model_III':
            IMAGE_SIZE = 64
        else:
            IMAGE_SIZE = None

        INPUT_SIZE = IMAGE_SIZE
        if run_config.model_source == 'baseline':
            model = BaselineModel(image_size=INPUT_SIZE, dropout_rate=run_config.dropout).to(device)
        elif run_config.model_source == 'timm':
            INPUT_SIZE = TIMM_IMAGE_SIZE[run_config.model_name]
            model = get_timm_model(run_config.model_name, complex=complex,
                                    dropout_rate=run_config.dropout, pretrained=pretrained, tune=tune).to(device)
        else:
            model = None
        
        if device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        
        datapath = os.path.join('./data', f'{run_config.dataset}', 'memmap', 'train')
        train_dataset = LensDataset(image_size=IMAGE_SIZE, memmap_path=datapath)
        wandb.run.summary['norm_mean'] = train_dataset.mean
        wandb.run.summary['norm_std'] = train_dataset.std
        # 90%-10% Train-validation split
        train_size = int(len(train_dataset) * 0.8)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_dataset = WrapperDataset(train_dataset, transform=get_transforms(run_config, final_size=INPUT_SIZE, mode='train'))
        val_dataset = WrapperDataset(val_dataset, transform=get_transforms(run_config, final_size=INPUT_SIZE, mode='test'))

        train_loader = DataLoader(train_dataset, batch_size=run_config.batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=run_config.batchsize, shuffle=False)

        if run_config.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=run_config.lr)
        else:
            optimizer = SGD(model.parameters(), lr=run_config.lr)
        
        criterion = CrossEntropyLoss()

        if run_config.decay_lr == 1:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6, verbose=True)
        else:
            scheduler = None

        best_val_metrics = train(model, train_loader, val_loader, criterion, optimizer, scheduler, run_config.epochs,
                                 device, run_config.log_interval)

