"""Evaluation metrics for the three tasks (NumPy in, plain floats out)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, class_names: tuple[str, ...]) -> dict:
    """Accuracy, macro-F1, one-vs-rest AUROC per class and their mean."""
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }
    aucs = []
    for index, name in enumerate(class_names):
        positives = labels == index
        if positives.all() or not positives.any():
            metrics[f"auroc_{name}"] = float("nan")
            continue
        metrics[f"auroc_{name}"] = float(roc_auc_score(positives, probabilities[:, index]))
        aucs.append(metrics[f"auroc_{name}"])
    metrics["mean_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    return metrics


def regression_metrics(targets: np.ndarray, predictions: np.ndarray, deciles: int = 10) -> dict:
    """MAE/RMSE in dex, R^2, Pearson r, bias, and MAE per target decile."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    residual = predictions - targets
    pearson = float(np.corrcoef(predictions, targets)[0, 1]) if targets.std() > 0 and predictions.std() > 0 else float("nan")
    edges = np.quantile(targets, np.linspace(0, 1, deciles + 1))
    bins = np.clip(np.searchsorted(edges, targets, side="right") - 1, 0, deciles - 1)
    by_decile = [float(np.abs(residual[bins == k]).mean()) if np.any(bins == k) else float("nan") for k in range(deciles)]
    return {
        "mae_dex": float(np.abs(residual).mean()),
        "rmse_dex": float(np.sqrt(np.mean(residual**2))),
        "r2": float(r2_score(targets, predictions)),
        "pearson_r": pearson,
        "bias_dex": float(residual.mean()),
        "mae_by_mass_decile": by_decile,
    }


def ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 7, reduce: bool = True) -> torch.Tensor:
    """Structural similarity with a uniform ``window x window`` kernel.

    Images are expected in ``[0, 1]``. With ``reduce=False`` one value per image
    is returned. This is differentiable and also used inside the SR loss.
    """
    pad = window // 2
    mean_x = F.avg_pool2d(prediction, window, 1, pad)
    mean_y = F.avg_pool2d(target, window, 1, pad)
    var_x = F.avg_pool2d(prediction.square(), window, 1, pad) - mean_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, pad) - mean_y.square()
    cov = F.avg_pool2d(prediction * target, window, 1, pad) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mean_x * mean_y + c1) * (2 * cov + c2)) / ((mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2))
    per_image = ssim_map.flatten(1).mean(dim=1)
    return per_image.mean() if reduce else per_image


def super_resolution_metrics(predictions: np.ndarray, targets: np.ndarray, ssim_values: np.ndarray) -> dict:
    """Mean per-image PSNR (data range 1), SSIM, MAE and MSE."""
    axes = tuple(range(1, predictions.ndim))
    mse = np.mean((predictions - targets) ** 2, axis=axes)
    mae = np.mean(np.abs(predictions - targets), axis=axes)
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))
    return {
        "psnr": float(psnr.mean()),
        "ssim": float(np.mean(ssim_values)),
        "mae": float(mae.mean()),
        "mse": float(mse.mean()),
    }
