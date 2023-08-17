import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import auroc as auroc_fn, accuracy as accuracy_fn
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from models import get_timm_model
from models.transformers import get_transformer_model
from data import LensDataset, get_transforms
from constants import *
from utils import get_device


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    """
    Evaluate model on given dataset

    Parameters
    ----------
    model : torch.nn.Module
        The given trained model
    data_loader : torch.utils.data.DataLoader
        Batched data loader for the dataset
    loss_fn : torch.nn.CrossEntropyLoss (or similar)
        Loss Function
    device : str
        Device on which to run the inference

    Returns
    -------
    dict
        Metrics on the provided data by the given model.
        It is a dict containing metrics like "loss", "accuracy", "micro_auroc", "macro_auroc".
    """
    model.eval()  # Switch on evaluation model

    # Initialize lists for different metrics
    loss, accuracy, class_auroc, micro_auroc, macro_auroc = [], [], [], [], []
    logits, y = [], []

    # Iterate over batches and accumulate metrics
    for batch_X, batch_y in data_loader:
        # Send data to device
        batch_X, batch_y = batch_X.to(device, dtype=torch.float), batch_y.type(
            torch.LongTensor
        )
        logits.append(model(batch_X).cpu())  # Append the logits
        y.append(batch_y)  # Append the predictions

    # Concatenate all results
    logits, y = torch.cat(logits), torch.cat(y)
    loss.append(loss_fn(logits, y))
    accuracy.append(accuracy_fn(logits, y, num_classes=NUM_CLASSES))
    class_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average=None))
    macro_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average="macro"))

    result = {
        "ground_truth": y,
        "logits": logits,
        "loss": np.mean(loss),
        "accuracy": np.mean(accuracy),
        "micro_auroc": np.mean(micro_auroc),
        "macro_auroc": np.mean(macro_auroc),
    }

    # Class-wise AUROC
    class_auroc = class_auroc[0]
    for i, label in enumerate(LABELS):
        result[f"{label}_auroc"] = class_auroc[i]

    return result


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser()

    # Wandb-specific params
    parser.add_argument("--runid", type=str, help="ID of train run")
    parser.add_argument("--project", type=str, default="ml4sci_deeplense_final")

    # Device to run on
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda", "best"], default="best"
    )
    run_config = parser.parse_args()

    # Start wandb run
    with wandb.init(
        entity="_archil", project=run_config.project, id=run_config.runid, resume="must"
    ):
        # Get best device on machine
        device = get_device(run_config.device)

        # Set image size based on dataset
        if wandb.config.dataset == "Model_I":
            IMAGE_SIZE = 150
        elif wandb.config.dataset == "Model_II" or wandb.config.dataset == "Model_III":
            IMAGE_SIZE = 64
        else:
            IMAGE_SIZE = None

        # Get timm model object
        INPUT_SIZE = TIMM_IMAGE_SIZE[wandb.config.model_name]
        model = get_timm_model(
            wandb.config.model_name, complex=wandb.config.complex
        ).to(device)

        # Fetch weights from wandb train run
        weights_file = wandb.restore("best_model.pt")
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, "best_model.pt")))

        # Create the test dataset and batch loader
        dataset = LensDataset(
            root_dir=os.path.join("./data", wandb.config.dataset, "test"),
            transform=get_transforms(
                wandb.config,
                initial_size=IMAGE_SIZE,
                final_size=INPUT_SIZE,
                mode="test",
            ),
        )
        data_loader = DataLoader(
            dataset, batch_size=wandb.config.batchsize, shuffle=False
        )

        # Parallelization across multiple GPUs, if available
        if device == "cuda" and torch.cuda.device_count() > 1:
            device = "cuda:0"
            model = torch.nn.DataParallel(model)
            model = model.to(device)

        # Loss function
        criterion = CrossEntropyLoss()

        # Evaluate model on test data and get metrics
        metrics = evaluate(model, data_loader, criterion, device=device)

        # Log the summary into W&B
        wandb.run.summary["test_loss"] = metrics["loss"]
        wandb.run.summary["test_accuracy"] = metrics["accuracy"]
        wandb.run.summary["test_micro_auroc"] = metrics["micro_auroc"]
        wandb.run.summary["test_macro_auroc"] = metrics["macro_auroc"]
        for label in LABELS:
            wandb.run.summary[f"test_{label}_auroc"] = metrics[f"{label}_auroc"]

        # Log the ROC plot in W&B
        wandb.log(
            {
                "test_roc": wandb.plot.roc_curve(
                    metrics["ground_truth"],
                    torch.nn.functional.softmax(metrics["logits"], dim=-1),
                    labels=LABELS,
                )
            }
        )

        # Create confusion matric as a heatmap and log it into W&B and save in results folder on disk
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for idx, cls in enumerate(LABELS):
            class_truth = (metrics["ground_truth"].numpy() == idx).astype(int)
            class_pred = torch.nn.functional.softmax(metrics["logits"]).numpy()[
                ..., idx
            ]
            fpr[idx], tpr[idx], _ = roc_curve(class_truth, class_pred)
            _ = axes[0].plot(
                fpr[idx],
                tpr[idx],
                label="{} ({:.2f}%)".format(cls, metrics[f"{cls}_auroc"] * 100),
            )
        _ = axes[0].set_title(
            "Test AUROC: {:.2f}%".format(metrics["macro_auroc"] * 100)
        )
        _ = axes[0].legend()

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=metrics["ground_truth"].numpy(),
            y_pred=np.argmax(metrics["logits"], axis=-1),
            display_labels=LABELS,
            cmap=plt.cm.Blues,
            colorbar=False,
            ax=axes[1],
        )

        fig.tight_layout()

        fig.savefig(f"{wandb.config.model_name}__plots.jpg")

        wandb.log({"confusion_matrix": fig})
