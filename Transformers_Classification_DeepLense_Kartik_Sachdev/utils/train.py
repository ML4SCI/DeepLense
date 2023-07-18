from __future__ import print_function
import logging
import copy
import torch
from tqdm import tqdm
from typing import *
import wandb
import torch.nn as nn
from torch.nn.functional import normalize


def train(
    epochs: int,
    model: nn.Module,
    device: Union[int, str],
    train_loader: Any,
    valid_loader: Any,
    criterion: nn.Module,
    optimizer: nn.Module,
    use_lr_schedule: nn.Module,
    scheduler_step: nn.Module,
    path: str,
    config: dict,
    dataset_name: str,
    log_freq=100,
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


def train_contrastive_with_labels(
    epochs: int,
    model: nn.Module,
    device: Union[int, str],
    train_loader: Any,
    criterion: nn.Module,
    optimizer: nn.Module,
    saved_model_path: str,
):
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1 = model(img1)
            output2 = model(img2)
            loss = criterion(output1, output2, label)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}"
                )

        epoch_loss = epoch_loss / len(train_loader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), saved_model_path)


def train_contrastive(
    epochs: int,
    model: nn.Module,
    device: Union[int, str],
    train_loader: Any,
    criterion: nn.Module,
    optimizer: nn.Module,
    saved_model_path: str,
):
    best_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (img1, _, _) in enumerate(train_loader):
            img1 = img1.to(device)

            optimizer.zero_grad()
            embeddings = model(img1)

            # embeddings = model(img1)
            embeddings = normalize(embeddings, dim=1)

            loss = criterion(embeddings)
            loss.backward()
            optimizer.step()

            # TODO: for code testing
            if batch_idx == 30:
                break

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}"
                )

        epoch_loss = epoch_loss / len(train_loader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), saved_model_path)


def train_simplistic(
    epochs: int,
    model: nn.Module,
    device: Union[int, str],
    train_loader: Any,
    criterion: nn.Module,
    optimizer: nn.Module,
    saved_model_path: str,
):
    best_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (img1, _, label) in enumerate(train_loader):
            img1 = img1.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(img1)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}"
                )

        epoch_loss = epoch_loss / len(train_loader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), saved_model_path)
