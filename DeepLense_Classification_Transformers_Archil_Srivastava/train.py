import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, Adam, SGD
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
from models.transformers import get_transformer_model
from utils import get_device, set_seed
from eval import evaluate


def train_step(model, images, labels, optimizer, scheduler, criterion, device="cpu"):
    """
    Takes one train step (forward + backward pass) on a batch

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    images : torch.Tensor
        Batch of images of shape [None, H, W, C]
    labels : torch.Tensor
        Ground truth of shape [None]
    optimizer : torch.optim.Optimizer
        Optimizer algorithm
    scheduler : torch.optim.lr_scheduler._LRScheduler (or something similar)
        Train scheduler to use (If None, no scheduler is used)
    criterion : torch.nn.CrossEntropyLoss (or similar)
        Loss Function
    device : str
        Device to use for training

    Returns
    -------
    float
        Loss value from the forward pass
    """
    # Send to device
    images, labels = images.to(device, dtype=torch.float), labels.type(
        torch.LongTensor
    ).to(device)
    model.train()  # Set train mode
    optimizer.zero_grad()  # Reset gradients
    logits = model(images)  # Forward pass
    loss = criterion(logits, labels)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights step
    if scheduler is not None:
        scheduler.step(loss)  # Modify learning rate if scheduler is set
    return loss


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device,
    log_interval=100,
):
    """
    Train the model, log it on W&B, save best model in disk and return the validation metrics

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        Batched data loader for train data
    val_loader : torch.utils.data.DataLoader
        Batched data loader for validation data
    criterion : torch.nn.CrossEntropyLoss (or similar)
        Loss Function
    optimizer : torch.optim.Optimizer
        Optimizer algorithm
    scheduler : torch.optim.lr_scheduler._LRScheduler (or something similar)
        Train scheduler to use (If None, no scheduler is used)
    epochs : int
        Number of epochs to train
    device : str
        Device to use for training
    log_interval : int, optional
        Number of epochs after which to log imtermediate results, by default 100

    Returns
    -------
    dict
        Metrics on validation data from best model (based on val AUROC).
        It is a dict containing metrics like "loss", "accuracy", "micro_auroc", "macro_auroc".
    """
    _ = model(
        next(iter(train_loader))[0].to(device, dtype=torch.float)
    )  # Used to initialize parameters in lazy layers

    # Alert wandb to log this training
    wandb.watch(model, criterion, log="all", log_freq=log_interval)

    best_val_auroc, best_val_metrics = 0.0, dict()
    batch_num = 0
    for epoch in range(1, epochs + 1):
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch_num += 1
            model.train()  # Set train mode
            images, labels = batch_data  # Get data
            # Do forward + backward pass
            loss = train_step(
                model, images, labels, optimizer, scheduler, criterion, device=device
            )

            # Compute metrics for validation data after every few epochs
            if batch_num % log_interval == 0:
                # Get validation data metrics
                val_metrics = evaluate(model, val_loader, criterion, device=device)

                # Log in wandb
                log_dict = {
                    "epoch": epoch,
                    "batch_num": batch_num,
                    "train/loss": loss,
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/micro_auroc": val_metrics["micro_auroc"],
                    "val/macro_auroc": val_metrics["macro_auroc"],
                }
                # Classwise validation AUROC
                for label in LABELS:
                    log_dict[f"val/{label}_auroc"] = val_metrics[f"{label}_auroc"]

                # Log metrics in wandb
                wandb.log(log_dict, step=batch_num)

                # Log ROC curve for validation data
                wandb.log(
                    {
                        "roc": wandb.plot.roc_curve(
                            val_metrics["ground_truth"],
                            torch.nn.functional.softmax(val_metrics["logits"], dim=-1),
                            labels=LABELS,
                        )
                    }
                )

                # Update best val auroc
                if val_metrics["macro_auroc"] > best_val_auroc:
                    best_val_auroc = val_metrics["macro_auroc"]
                    wandb.run.summary["best_val_micro_auroc"] = val_metrics[
                        "micro_auroc"
                    ]
                    wandb.run.summary["best_val_macro_auroc"] = best_val_auroc
                    wandb.run.summary["best_epoch"] = epoch
                    wandb.run.summary["best_batch_num"] = batch_num
                    for label in LABELS:
                        wandb.run.summary[f"best_val_{label}_auroc"] = val_metrics[
                            f"{label}_auroc"
                        ]
                    best_val_metrics = val_metrics
                    # Save best model so far in disk
                    torch.save(
                        model.state_dict(), os.path.join(wandb.run.dir, "best_model.pt")
                    )

        # Sync best model at a lesser frequency (i.e. at the end of each epoch)
        wandb.save(os.path.join(wandb.run.dir, "best_model.pt"))

    return best_val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # W&B related parameters
    parser.add_argument(
        "--dataset",
        choices=["Model_I", "Model_II", "Model_III", "Model_IV"],
        default="Model_I",
        help="which data model",
    )
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--project", type=str, default="ml4sci_deeplense_final")

    # Timm-Specific parameters
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument(
        "--complex", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to further tune (1) pretrained model (if any) or freeze the pretrained weights (0)",
    )

    # Run-specific parameters
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda", "tpu", "best"], default="best"
    )

    # Augmentations
    parser.add_argument("--random_zoom", type=float, default=1)
    parser.add_argument("--random_rotation", type=float, default=0)

    # Common hyperparameters
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--optimizer", choices=["sgd", "adam", "adamw"], default="adamw"
    )
    parser.add_argument(
        "--decay_lr", action=argparse.BooleanOptionalAction, default=False
    )

    run_config = parser.parse_args()

    # Get  group for wandb
    group = f"timm-{run_config.model_name}"
    if run_config.complex:
        group = f"{group}-complex"

    # Start wandb run
    with wandb.init(
        entity="_archil",
        project=run_config.project,
        config=run_config,
        group=group,
        job_type=f"{run_config.dataset}",
    ):
        # Set random seed
        if run_config.seed:
            set_seed(run_config.seed)

        # Select image size based on dataset
        if run_config.dataset == "Model_I":
            IMAGE_SIZE = 150
        elif run_config.dataset in ["Model_II", "Model_III"]:
            IMAGE_SIZE = 64
        else:
            IMAGE_SIZE = None
            raise ValueError("Dataset not found")

        # Select best device on the machine
        device = get_device(run_config.device)

        # Get timm model
        INPUT_SIZE = TIMM_IMAGE_SIZE[run_config.model_name]
        model = get_timm_model(
            run_config.model_name,
            complex=run_config.complex,
            dropout_rate=run_config.dropout,
            pretrained=run_config.pretrained,
            tune=run_config.tune or not run_config.pretrained,
        ).to(device)

        # Parallelization across multiple GPUs, if available
        if device == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(
                model, device_ids=list(range(torch.cuda.device_count()))
            )

        # Initialize train dataset
        train_dataset = LensDataset(
            root_dir=os.path.join("./data", f"{run_config.dataset}", "train")
        )

        # 90%-10% Train-validation split
        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Initialize train and validation datasets
        train_dataset = WrapperDataset(
            train_dataset,
            transform=get_transforms(
                run_config, initial_size=IMAGE_SIZE, final_size=INPUT_SIZE, mode="train"
            ),
        )
        val_dataset = WrapperDataset(
            val_dataset,
            transform=get_transforms(
                run_config, initial_size=IMAGE_SIZE, final_size=INPUT_SIZE, mode="test"
            ),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=run_config.batchsize, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=run_config.batchsize, shuffle=False
        )

        # Initialize optimizer
        if run_config.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=run_config.lr, weight_decay=0.01)
        elif run_config.optimizer == "sgd":
            optimizer = SGD(model.parameters(), lr=run_config.lr, weight_decay=0.01)
        elif run_config.optimizer == "adamw":
            optimizer = AdamW(model.parameters(), lr=run_config.lr, weight_decay=0.01)
        else:
            optimizer = None

        # Loss function
        criterion = CrossEntropyLoss()

        # Scheduler
        if run_config.decay_lr:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=1, eta_min=1e-6, verbose=True
            )
        else:
            scheduler = None

        # Train the model and get the best validation metrics
        best_val_metrics = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            run_config.epochs,
            device,
            run_config.log_interval,
        )
