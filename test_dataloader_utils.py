"""Tests for the dataloader_utils module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader_utils import (
    safe_to_tensor,
    robust_normalize,
    create_data_pipeline,
    validate_batch,
)


def test_safe_to_tensor_list():
    arr = safe_to_tensor([1, 2, 3])
    assert arr.dtype == np.float32
    assert arr.flags["C_CONTIGUOUS"]
    print("[PASS] test_safe_to_tensor_list")


def test_safe_to_tensor_scalar():
    arr = safe_to_tensor(42)
    assert arr.shape == (1,)
    print("[PASS] test_safe_to_tensor_scalar")


def test_safe_to_tensor_nan():
    arr = safe_to_tensor([1.0, float("nan"), 3.0])
    assert not np.any(np.isnan(arr))
    print("[PASS] test_safe_to_tensor_nan")


def test_safe_to_tensor_inf():
    arr = safe_to_tensor([1.0, float("inf"), -float("inf")])
    assert not np.any(np.isinf(arr))
    print("[PASS] test_safe_to_tensor_inf")


def test_safe_to_tensor_non_contiguous():
    orig = np.arange(16).reshape(4, 4).T  # Non-contiguous
    arr = safe_to_tensor(orig)
    assert arr.flags["C_CONTIGUOUS"]
    print("[PASS] test_safe_to_tensor_non_contiguous")


def test_safe_to_tensor_none():
    try:
        safe_to_tensor(None)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("[PASS] test_safe_to_tensor_none")


def test_normalize_standard():
    data = np.random.rand(10, 64, 64) * 255
    normed = robust_normalize(data, method="standard")
    assert abs(np.mean(normed)) < 0.01
    print("[PASS] test_normalize_standard")


def test_normalize_constant_image():
    """Edge case: constant image should not produce NaN."""
    data = np.ones((64, 64)) * 42.0
    normed = robust_normalize(data, method="standard")
    assert not np.any(np.isnan(normed))
    assert np.all(normed == 0.0)
    print("[PASS] test_normalize_constant_image")


def test_normalize_minmax():
    data = np.random.rand(64, 64) * 100
    normed = robust_normalize(data, method="minmax")
    assert normed.min() >= -0.01
    assert normed.max() <= 1.01
    print("[PASS] test_normalize_minmax")


def test_normalize_robust_scale():
    data = np.random.randn(64, 64) * 10
    normed = robust_normalize(data, method="robust_scale")
    assert not np.any(np.isnan(normed))
    print("[PASS] test_normalize_robust_scale")


def test_normalize_per_channel():
    data = np.random.rand(64, 64, 3) * 255
    normed = robust_normalize(data, method="per_channel")
    for c in range(3):
        assert abs(np.mean(normed[:, :, c])) < 0.01
    print("[PASS] test_normalize_per_channel")


def test_normalize_with_clip():
    data = np.random.randn(64, 64) * 10
    normed = robust_normalize(data, method="standard", clip_range=(-3, 3))
    assert normed.min() >= -3.0
    assert normed.max() <= 3.0
    print("[PASS] test_normalize_with_clip")


def test_normalize_nan_input():
    data = np.array([1.0, float("nan"), 3.0, float("nan")])
    normed = robust_normalize(data, method="minmax")
    assert not np.any(np.isnan(normed))
    print("[PASS] test_normalize_nan_input")


def test_pipeline():
    pipeline = create_data_pipeline([
        lambda x: x.astype(np.float64),
        lambda x: robust_normalize(x, method="minmax"),
    ])
    data = np.random.randint(0, 255, (64, 64)).astype(np.uint8)
    result = pipeline(data)
    assert result.dtype == np.float32
    assert result.min() >= -0.01
    print("[PASS] test_pipeline")


def test_validate_batch_valid():
    images = np.random.rand(10, 64, 64, 1).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    report = validate_batch(images, labels)
    assert report["valid"]
    assert report["stats"]["batch_size"] == 10
    assert report["stats"]["num_classes"] == 3
    print("[PASS] test_validate_batch_valid")


def test_validate_batch_nan():
    images = np.array([[[float("nan")]]]).astype(np.float32)
    report = validate_batch(images)
    assert not report["valid"]
    assert any("NaN" in w for w in report["warnings"])
    print("[PASS] test_validate_batch_nan")


def test_validate_batch_label_mismatch():
    images = np.random.rand(5, 32, 32).astype(np.float32)
    labels = np.array([0, 1, 2])  # Wrong count
    report = validate_batch(images, labels)
    assert not report["valid"]
    print("[PASS] test_validate_batch_label_mismatch")


def test_validate_batch_shape_mismatch():
    images = np.random.rand(5, 32, 32).astype(np.float32)
    report = validate_batch(images, expected_shape=(64, 64))
    assert not report["valid"]
    print("[PASS] test_validate_batch_shape_mismatch")


if __name__ == "__main__":
    test_safe_to_tensor_list()
    test_safe_to_tensor_scalar()
    test_safe_to_tensor_nan()
    test_safe_to_tensor_inf()
    test_safe_to_tensor_non_contiguous()
    test_safe_to_tensor_none()
    test_normalize_standard()
    test_normalize_constant_image()
    test_normalize_minmax()
    test_normalize_robust_scale()
    test_normalize_per_channel()
    test_normalize_with_clip()
    test_normalize_nan_input()
    test_pipeline()
    test_validate_batch_valid()
    test_validate_batch_nan()
    test_validate_batch_label_mismatch()
    test_validate_batch_shape_mismatch()
    print("\n=== All 18 tests passed! ===")
