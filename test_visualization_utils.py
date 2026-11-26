"""Tests for the visualization_utils module."""

import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization_utils import (
    plot_image_grid,
    plot_training_history,
    plot_confusion_matrix,
    generate_classification_report,
)


def test_plot_image_grid_saves():
    """Test that image grid saves to file."""
    images = np.random.rand(8, 32, 32)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    save_path = os.path.join(tempfile.gettempdir(), "test_grid.png")

    plot_image_grid(
        images, labels,
        class_names=["No Sub", "Vortex", "Sphere"],
        save_path=save_path,
    )
    assert os.path.exists(save_path), "Grid image should be saved"
    os.remove(save_path)
    print("[PASS] test_plot_image_grid_saves")


def test_plot_image_grid_returns_fig():
    """Test that image grid returns figure when no save path."""
    images = np.random.rand(4, 32, 32)
    fig = plot_image_grid(images, title="Test")
    assert fig is not None, "Should return figure object"
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("[PASS] test_plot_image_grid_returns_fig")


def test_plot_training_history_saves():
    """Test that training history plot saves."""
    history = {
        "train_loss": [0.9, 0.7, 0.5, 0.3, 0.2],
        "val_loss": [0.95, 0.8, 0.6, 0.45, 0.4],
        "train_acc": [0.6, 0.7, 0.8, 0.85, 0.9],
        "val_acc": [0.55, 0.65, 0.75, 0.8, 0.82],
    }
    save_path = os.path.join(tempfile.gettempdir(), "test_history.png")
    plot_training_history(history, save_path=save_path)
    assert os.path.exists(save_path), "History plot should be saved"
    os.remove(save_path)
    print("[PASS] test_plot_training_history_saves")


def test_plot_confusion_matrix_saves():
    """Test confusion matrix plot saves."""
    cm = np.array([[45, 3, 2], [5, 40, 5], [1, 4, 45]])
    save_path = os.path.join(tempfile.gettempdir(), "test_cm.png")
    plot_confusion_matrix(
        cm,
        class_names=["No Sub", "Vortex", "Sphere"],
        save_path=save_path,
    )
    assert os.path.exists(save_path), "Confusion matrix should be saved"
    os.remove(save_path)
    print("[PASS] test_plot_confusion_matrix_saves")


def test_plot_confusion_matrix_normalized():
    """Test normalized confusion matrix."""
    cm = np.array([[40, 10], [5, 45]])
    save_path = os.path.join(tempfile.gettempdir(), "test_cm_norm.png")
    plot_confusion_matrix(cm, normalize=True, save_path=save_path)
    assert os.path.exists(save_path)
    os.remove(save_path)
    print("[PASS] test_plot_confusion_matrix_normalized")


def test_generate_classification_report():
    """Test classification report generation."""
    metrics = {
        "accuracy": 0.85,
        "no_sub_precision": 0.90,
        "no_sub_recall": 0.88,
        "no_sub_f1": 0.89,
        "vortex_precision": 0.82,
        "vortex_recall": 0.80,
        "vortex_f1": 0.81,
        "macro_precision": 0.86,
        "macro_recall": 0.84,
        "macro_f1": 0.85,
    }
    report = generate_classification_report(metrics)
    assert "CLASSIFICATION REPORT" in report
    assert "0.8500" in report
    assert "no_sub" in report
    assert "vortex" in report
    assert "Macro Average" in report
    print("[PASS] test_generate_classification_report")


def test_plot_image_grid_rgb():
    """Test with RGB images."""
    images = np.random.rand(4, 32, 32, 3)
    fig = plot_image_grid(images)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("[PASS] test_plot_image_grid_rgb")


if __name__ == "__main__":
    test_plot_image_grid_saves()
    test_plot_image_grid_returns_fig()
    test_plot_training_history_saves()
    test_plot_confusion_matrix_saves()
    test_plot_confusion_matrix_normalized()
    test_generate_classification_report()
    test_plot_image_grid_rgb()
    print("\n=== All 7 tests passed! ===")
