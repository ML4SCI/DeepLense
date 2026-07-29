import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.functional.classification import multiclass_auroc as auroc_fn, multiclass_accuracy as accuracy_fn
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from models import get_timm_model
from data import LensDataset, get_transforms
from constants import *
from utils import get_device

# Helper to safely get config values without crashing
def get_cfg(attr, default):
    try:
        if wandb.run is not None:
            return getattr(wandb.config, attr)
        return default
    except:
        return default

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
    macro_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average="macro"))

    result = {
        "ground_truth": y,
        "logits": logits,
        "loss": np.mean(loss),
        "accuracy": np.mean(accuracy),
        "micro_auroc": np.mean(micro_auroc),
        "macro_auroc": np.mean(macro_auroc),
    }

    class_auroc = class_auroc[0]
    for i, label in enumerate(LABELS):
        result[f"{label}_auroc"] = class_auroc[i]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", type=str, help="ID of train run", default="test_run")
    parser.add_argument("--project", type=str, default="ml4sci_deeplense_final")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "best"], default="best")
    # Add local overrides so we don't have to rely on W&B
    parser.add_argument("--dataset", type=str, default="Model_I")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--complex", type=bool, default=False)
    parser.add_argument("--batchsize", type=int, default=2)
    run_config = parser.parse_args()

    with wandb.init(entity="_archil", project=run_config.project, id=run_config.runid, resume="must", mode="offline"):
        device = get_device(run_config.device)
        dataset_name = get_cfg('dataset', run_config.dataset)
        
        if dataset_name == "Model_I":
            IMAGE_SIZE = 150
        elif dataset_name in ["Model_II", "Model_III"]:
            IMAGE_SIZE = 64
        else:
            IMAGE_SIZE = None

        model_name = get_cfg('model_name', run_config.model_name)
        is_complex = get_cfg('complex', run_config.complex)
        batch_size = get_cfg('batchsize', run_config.batchsize)
        
        INPUT_SIZE = TIMM_IMAGE_SIZE.get(model_name, 224)
        model = get_timm_model(model_name, complex=is_complex).to(device)

        # Ensure we have weights
        weights_path = os.path.join(wandb.run.dir, "best_model.pt")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
        else:
            print("Warning: best_model.pt not found, evaluating with random weights.")

        dataset = LensDataset(
            root_dir=os.path.join("./data", dataset_name, "test"),
            transform=get_transforms(wandb.config, initial_size=IMAGE_SIZE, final_size=INPUT_SIZE, mode="test"),
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if device == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to("cuda:0")

        metrics = evaluate(model, data_loader, CrossEntropyLoss(), device=device)

        # Print metrics instead of forcing W&B sync if offline
        print(f"Evaluation Results: {metrics}")
        
        # ... (rest of plotting code remains the same)