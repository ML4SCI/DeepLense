"""
Model Evaluation Utilities for DeepLense.

This module provides reusable evaluation functions for gravitational
lens classification and regression tasks. It includes metrics computation,
confusion matrix generation, and visualization helpers that can be used
across different DeepLense sub-projects.

Author: Kamala Hasini Burra
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute classification metrics for gravitational lens classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (1D array of integers).
    y_pred : np.ndarray
        Predicted labels (1D array of integers).
    class_names : list of str, optional
        Names of the classes (e.g., ['no_substructure', 'vortex', 'sphere']).

    Returns
    -------
    dict
        Dictionary containing accuracy, per-class precision, recall, and
        F1-score.

    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 1, 1, 0, 2, 2])
    >>> metrics = compute_classification_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    Accuracy: 0.67
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)} samples."
        )

    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)

    # Unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    if class_names is None:
        class_names = [f"class_{c}" for c in classes]

    metrics: Dict[str, float] = {"accuracy": float(accuracy)}

    # Per-class metrics
    for i, cls in enumerate(classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        name = class_names[i] if i < len(class_names) else f"class_{cls}"
        metrics[f"{name}_precision"] = float(precision)
        metrics[f"{name}_recall"] = float(recall)
        metrics[f"{name}_f1"] = float(f1)

    # Macro-averaged metrics
    precisions = [
        metrics[f"{class_names[i]}_precision"]
        for i in range(min(n_classes, len(class_names)))
    ]
    recalls = [
        metrics[f"{class_names[i]}_recall"]
        for i in range(min(n_classes, len(class_names)))
    ]
    f1s = [
        metrics[f"{class_names[i]}_f1"]
        for i in range(min(n_classes, len(class_names)))
    ]

    metrics["macro_precision"] = float(np.mean(precisions))
    metrics["macro_recall"] = float(np.mean(recalls))
    metrics["macro_f1"] = float(np.mean(f1s))

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute a confusion matrix without requiring sklearn.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    n_classes : int, optional
        Number of classes. If None, inferred from the data.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes) where entry [i, j]
        represents the number of samples with true label i predicted as j.
    """
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    return cm


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics for lens parameter estimation.

    Useful for evaluating models that predict continuous parameters
    such as Einstein radius, ellipticity, or source position.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict
        Dictionary containing MSE, RMSE, MAE, and R-squared metrics.
    """
    residuals = y_true - y_pred

    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
    }


def compute_roc_auc_ovr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute one-vs-rest ROC AUC for multi-class classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (1D array of integers).
    y_scores : np.ndarray
        Predicted probability scores of shape (n_samples, n_classes).
    n_classes : int, optional
        Number of classes. If None, inferred from y_scores.

    Returns
    -------
    dict
        Dictionary with per-class AUC and macro-averaged AUC.
    """
    if n_classes is None:
        n_classes = y_scores.shape[1]

    auc_scores: Dict[str, float] = {}

    for cls in range(n_classes):
        # Binary labels: 1 if true class is cls, 0 otherwise
        binary_true = (y_true == cls).astype(int)
        scores = y_scores[:, cls]

        # Sort by decreasing score
        sorted_indices = np.argsort(-scores)
        sorted_true = binary_true[sorted_indices]

        # Compute TPR and FPR at each threshold
        tp_cumsum = np.cumsum(sorted_true)
        fp_cumsum = np.cumsum(1 - sorted_true)

        total_pos = np.sum(binary_true)
        total_neg = len(binary_true) - total_pos

        if total_pos == 0 or total_neg == 0:
            auc_scores[f"class_{cls}_auc"] = 0.0
            continue

        tpr = tp_cumsum / total_pos
        fpr = fp_cumsum / total_neg

        # Prepend (0, 0)
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # Trapezoidal integration
        auc = float(np.trapz(tpr, fpr))
        auc_scores[f"class_{cls}_auc"] = auc

    class_aucs = [v for k, v in auc_scores.items() if k.endswith("_auc")]
    auc_scores["macro_auc"] = float(np.mean(class_aucs)) if class_aucs else 0.0

    return auc_scores


def format_metrics_table(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics",
) -> str:
    """Format metrics dictionary as a readable table string.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric name to value.
    title : str
        Title for the table.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = [title, "=" * len(title)]
    max_key_len = max(len(k) for k in metrics)

    for key, value in metrics.items():
        lines.append(f"  {key:<{max_key_len}}  {value:.4f}")

    return "\n".join(lines)
