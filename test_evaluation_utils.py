"""Tests for the evaluation_utils module."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_utils import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_regression_metrics,
    compute_roc_auc_ovr,
    format_metrics_table,
)


def test_classification_metrics_perfect():
    """Test classification metrics with perfect predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0, f"Expected 1.0, got {metrics['accuracy']}"
    assert metrics["macro_f1"] == 1.0, f"Expected 1.0, got {metrics['macro_f1']}"
    print("[PASS] test_classification_metrics_perfect")


def test_classification_metrics_partial():
    """Test classification metrics with some misclassifications."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    metrics = compute_classification_metrics(
        y_true, y_pred,
        class_names=["no_sub", "vortex", "sphere"],
    )

    assert 0.0 < metrics["accuracy"] < 1.0
    assert "no_sub_precision" in metrics
    assert "vortex_recall" in metrics
    print("[PASS] test_classification_metrics_partial")


def test_confusion_matrix():
    """Test confusion matrix computation."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = compute_confusion_matrix(y_true, y_pred)

    assert cm.shape == (3, 3), f"Expected shape (3, 3), got {cm.shape}"
    assert cm[0, 0] == 1  # True class 0, predicted 0
    assert cm[0, 1] == 1  # True class 0, predicted 1
    assert cm[1, 1] == 2  # True class 1, predicted 1
    assert cm[2, 2] == 1  # True class 2, predicted 2
    assert cm[2, 0] == 1  # True class 2, predicted 0
    print("[PASS] test_confusion_matrix")


def test_regression_metrics():
    """Test regression metrics for lens parameter estimation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mse"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert 0.0 < metrics["r_squared"] <= 1.0
    assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-10
    print("[PASS] test_regression_metrics")


def test_regression_metrics_perfect():
    """Test regression metrics with perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r_squared"] == 1.0
    print("[PASS] test_regression_metrics_perfect")


def test_roc_auc_ovr():
    """Test ROC AUC one-vs-rest computation."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    # Good predictions: high score for correct class
    y_scores = np.array([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.05, 0.9, 0.05],
        [0.1, 0.1, 0.8],
        [0.05, 0.05, 0.9],
    ])
    auc = compute_roc_auc_ovr(y_true, y_scores)

    assert auc["macro_auc"] > 0.5, "AUC should be > 0.5 for good predictions"
    print("[PASS] test_roc_auc_ovr")


def test_format_metrics_table():
    """Test metrics table formatting."""
    metrics = {"accuracy": 0.95, "f1": 0.93}
    table = format_metrics_table(metrics, "Test Results")

    assert "Test Results" in table
    assert "accuracy" in table
    assert "0.9500" in table
    print("[PASS] test_format_metrics_table")


def test_length_mismatch_raises_error():
    """Test that mismatched array lengths raise ValueError."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])
    try:
        compute_classification_metrics(y_true, y_pred)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] test_length_mismatch_raises_error")


if __name__ == "__main__":
    test_classification_metrics_perfect()
    test_classification_metrics_partial()
    test_confusion_matrix()
    test_regression_metrics()
    test_regression_metrics_perfect()
    test_roc_auc_ovr()
    test_format_metrics_table()
    test_length_mismatch_raises_error()
    print("\n=== All 8 tests passed! ===")
