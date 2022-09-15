"""temporary trainer"""
from __future__ import print_function
import logging
import copy
import torch
from tqdm import tqdm
from typing import *
import wandb
import torch.nn as nn
from ray.tune.integration.wandb import wandb_mixin
import json
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup
import math
from torch.utils.data import DataLoader
import os
from ray import tune
from ray.air import session


@wandb_mixin
def train(
    config: dict,
    # epochs: int,
    model: nn.Module,
    device: Union[int, str],
    trainset: Any,
    testset: Any,
    criterion: nn.Module,
    path: str,
    log_dir: str,
    dataset_name: str,
    num_workers: int,
    log_freq=100,
    checkpoint_dir=None,
):
    """Supervised learning for image classification. Uses `wandb` for logging

    Args:
        epochs (int): # of epochs
        model (nn.Module): model for training
        device (Union[int, str]): number or name of device
        train_loader (Any): pytorch loader for trainset
        valid_loader (Any): pytorch loader for testset
        criterion (nn.Module): loss critirea
        optimizer (nn.Module): optimizer for model training
        use_lr_schedule (nn.Module): whether to use learning rate scheduler
        scheduler_step (nn.Module): type of learning rate scheduler
        path (str): path to save models
        config (dict): model hyperparameters as dict 
        dataset_name (str): type of dataset
        log_freq (int, optional): logging frequency. Defaults to 100.
   
   Example:
   >>>     train(
   >>>     epochs=25, 
   >>>     model=model,
   >>>     device=0,
   >>>     train_loader=train_loader,
   >>>     valid_loader=test_loader,
   >>>     criterion=criterion,
   >>>     optimizer=optimizer,
   >>>     use_lr_schedule=train_config["lr_schedule_config"]["use_lr_schedule"],
   >>>     scheduler_step=cosine_scheduler,
   >>>     path=PATH,
   >>>     log_freq=20,
   >>>     config=train_config,
   >>>     dataset_name=dataset_name)
    """
    wandb.init(config=config, group=dataset_name, job_type="train")  # ,mode="disabled"
    wandb.watch(model, criterion, log="all", log_freq=log_freq)

    optimizer_config = config["optimizer_config"]
    lr_schedule_config = config["lr_schedule_config"]

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        dataset=testset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )

    sample = next(iter(train_loader))
    print(sample[0].shape)

    epochs = config["num_epochs"]
    warmup_epochs = optimizer_config["warmup_epoch"]
    num_train_steps = math.ceil(len(train_loader))
    num_warmup_steps = num_train_steps * warmup_epochs
    num_training_steps = int(num_train_steps * epochs)

    # learning rate scheduler
    scheduler_step = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    use_lr_schedule = (config["lr_schedule_config"]["use_lr_schedule"],)
    config["dataset_name"] = dataset_name
    config["lr_schedule_config"]["cosine_scheduler"] = {}
    config["lr_schedule_config"]["cosine_scheduler"][
        "num_warmup_steps"
    ] = num_warmup_steps
    config["lr_schedule_config"]["cosine_scheduler"]["num_training_steps"] = int(
        num_training_steps
    )

    with open(
        f"{os.path.dirname(os.path.abspath(__file__))}/../{log_dir}/config.json", "w"
    ) as fp:
        json.dump(config, fp)

    path = f"{os.path.dirname(os.path.abspath(__file__))}/../{path}"
    steps = 0
    all_train_loss = []
    all_val_loss = []
    all_train_accuracy = []
    all_val_accuracy = []
    all_test_accuracy = []
    all_epoch_loss = []

    best_accuracy = 0
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        tr_loss_epoch = []
        running_loss = 0

        for data, label in tqdm(train_loader):
            # for step, (data, label) in loop:
            steps += 1
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if use_lr_schedule:
                # scheduler_plateau.step(epoch_val_loss)
                scheduler_step.step()

        epoch_loss = epoch_loss / len(train_loader)
        all_epoch_loss.append(epoch_loss)

        correct = 0

        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc
                epoch_val_loss += val_loss

            epoch_val_accuracy = epoch_val_accuracy / len(valid_loader)
            epoch_val_loss = epoch_val_loss / len(valid_loader)
            all_val_loss.append(epoch_val_loss)

        all_val_accuracy.append(epoch_val_accuracy.item() * 100)
        logging.debug(
            f"Epoch : {epoch+1} - LR {optimizer.param_groups[0]['lr']:.8f} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} \n"
        )

        # logging frequency = each epoch
        log_dict = {
            "epoch": epoch,
            "steps": steps,
            "train/loss": loss,
            "val/loss": epoch_val_loss,
            "val/accuracy": epoch_val_accuracy,
        }
        wandb.log(log_dict, step=steps)

        if epoch_val_accuracy > best_accuracy:
            best_accuracy = epoch_val_accuracy
            best_model = copy.deepcopy(model)
            wandb.run.summary["best_accuracy"] = epoch_val_accuracy
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_step"] = steps
            wandb.save(path)
            torch.save(best_model.state_dict(), path)

        session.report({"best_accuracy": best_accuracy})
        # tune.report(best_accuracy=best_accuracy)
    return best_accuracy

