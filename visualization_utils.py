"""
Visualization Utilities for DeepLense.

This module provides reusable plotting and visualization functions
for gravitational lensing image analysis. It includes functions for
displaying image grids, plotting training history, visualizing
confusion matrices, and generating classification reports.

Requires: matplotlib (standard in ML environments)

Author: Kamala Hasini Burra
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_image_grid(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Sample Images",
    save_path: Optional[str] = None,
    cmap: str = "viridis",
) -> Optional[object]:
    """Display a grid of images with optional labels.

    Parameters
    ----------
    images : np.ndarray
        Batch of images, shape (N, H, W) or (N, H, W, C).
    labels : np.ndarray, optional
        Labels for each image.
    class_names : list of str, optional
        Human-readable class names.
    n_cols : int
        Number of columns in the grid.
    figsize : tuple
        Figure size (width, height).
    title : str
        Main title for the figure.
    save_path : str, optional
        Path to save the figure. If None, returns the figure object.
    cmap : str
        Colormap for grayscale images.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if save_path is None, otherwise saves and returns None.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Skipping plot.")
        return None

    n_images = min(len(images), n_cols * 4)  # Max 4 rows
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else np.array([[axes]])

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n_images:
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)

            if labels is not None and i < len(labels):
                label = labels[i]
                if class_names and label < len(class_names):
                    ax.set_title(class_names[int(label)], fontsize=9)
                else:
                    ax.set_title(f"Class {label}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> Optional[object]:
    """Plot training and validation metrics over epochs.

    Parameters
    ----------
    history : dict
        Dictionary mapping metric names to lists of values per epoch.
        Expected keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    metrics : list of str, optional
        Which metrics to plot. If None, plots all available.
    figsize : tuple
        Figure size.
    title : str
        Main title.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Skipping plot.")
        return None

    if metrics is None:
        metrics = list(history.keys())

    # Group metrics by type (loss vs accuracy)
    loss_metrics = [m for m in metrics if "loss" in m.lower()]
    acc_metrics = [m for m in metrics if "acc" in m.lower() or "f1" in m.lower()]
    other_metrics = [m for m in metrics if m not in loss_metrics and m not in acc_metrics]

    n_plots = sum(1 for g in [loss_metrics, acc_metrics, other_metrics] if g)
    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if n_plots <= 1:
        axes = [axes]

    plot_idx = 0

    for group, ylabel in [
        (loss_metrics, "Loss"),
        (acc_metrics, "Score"),
        (other_metrics, "Value"),
    ]:
        if not group:
            continue
        ax = axes[plot_idx]
        for metric_name in group:
            if metric_name in history:
                epochs = range(1, len(history[metric_name]) + 1)
                style = "--" if "val" in metric_name else "-"
                ax.plot(epochs, history[metric_name], style, label=metric_name, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = False,
) -> Optional[object]:
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    class_names : list of str, optional
        Names for each class.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.
    normalize : bool
        Whether to normalize by row (show percentages).

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available. Skipping plot.")
        return None

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm.astype(float), row_sums,
            where=row_sums != 0, out=np.zeros_like(cm, dtype=float),
        )
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, cmap="Blues", aspect="auto")

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_display[i, j]
            color = "white" if value > cm_display.max() / 2 else "black"
            text = f"{value:{fmt}}" if isinstance(value, float) else str(value)
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def generate_classification_report(
    metrics: Dict[str, float],
    class_names: Optional[List[str]] = None,
) -> str:
    """Generate a formatted classification report string.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_classification_metrics().
    class_names : list of str, optional
        Class names for display.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CLASSIFICATION REPORT")
    lines.append("=" * 60)

    if "accuracy" in metrics:
        lines.append(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")

    # Find per-class metrics
    precision_keys = [k for k in metrics if k.endswith("_precision")]
    if precision_keys:
        lines.append(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        lines.append("-" * 52)

        for pk in precision_keys:
            cls_name = pk.replace("_precision", "")
            prec = metrics.get(f"{cls_name}_precision", 0.0)
            rec = metrics.get(f"{cls_name}_recall", 0.0)
            f1 = metrics.get(f"{cls_name}_f1", 0.0)
            lines.append(f"{cls_name:<20} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    # Macro averages
    lines.append("-" * 52)
    if "macro_precision" in metrics:
        lines.append(
            f"{'Macro Average':<20} "
            f"{metrics['macro_precision']:>10.4f} "
            f"{metrics['macro_recall']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
