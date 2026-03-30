"""Tests for the preprocessing_utils module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing_utils import (
    normalize_images,
    augment_image,
    stratified_split,
    compute_class_weights,
    create_image_patches,
)


def test_minmax_normalization():
    imgs = np.random.randint(0, 255, (5, 64, 64)).astype(float)
    normed = normalize_images(imgs, method="minmax")
    assert abs(normed.min()) < 1e-10, f"Min should be ~0, got {normed.min()}"
    assert abs(normed.max() - 1.0) < 1e-10, f"Max should be ~1, got {normed.max()}"
    print("[PASS] test_minmax_normalization")


def test_zscore_normalization():
    imgs = np.random.randn(5, 64, 64) * 50 + 100
    normed = normalize_images(imgs, method="zscore")
    assert abs(normed.mean()) < 1e-10, f"Mean should be ~0, got {normed.mean()}"
    assert abs(normed.std() - 1.0) < 1e-10, f"Std should be ~1, got {normed.std()}"
    print("[PASS] test_zscore_normalization")


def test_robust_normalization():
    imgs = np.random.randn(5, 64, 64) * 10
    normed = normalize_images(imgs, method="robust")
    assert normed.shape == imgs.shape
    print("[PASS] test_robust_normalization")


def test_invalid_normalization():
    imgs = np.random.randn(5, 64, 64)
    try:
        normalize_images(imgs, method="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] test_invalid_normalization")


def test_augment_flip():
    img = np.arange(16).reshape(4, 4).astype(float)
    flipped_h = augment_image(img, flip_horizontal=True)
    assert flipped_h[0, 0] == img[0, 3]
    flipped_v = augment_image(img, flip_vertical=True)
    assert flipped_v[0, 0] == img[3, 0]
    print("[PASS] test_augment_flip")


def test_augment_rotate():
    img = np.arange(16).reshape(4, 4).astype(float)
    rotated = augment_image(img, rotate_90=1)
    assert rotated.shape == (4, 4)
    print("[PASS] test_augment_rotate")


def test_augment_noise():
    img = np.ones((64, 64))
    noisy = augment_image(img, add_noise=True, noise_std=0.1)
    assert not np.allclose(img, noisy), "Noisy image should differ"
    print("[PASS] test_augment_noise")


def test_stratified_split():
    labels = np.array([0]*100 + [1]*100 + [2]*100)
    train, val, test = stratified_split(labels, 0.7, 0.15, 0.15)
    assert len(train) + len(val) + len(test) == 300
    assert len(train) == 210  # 70% of 300
    print("[PASS] test_stratified_split")


def test_stratified_split_invalid_ratios():
    labels = np.array([0, 1, 2])
    try:
        stratified_split(labels, 0.5, 0.5, 0.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] test_stratified_split_invalid_ratios")


def test_class_weights_balanced():
    labels = np.array([0]*10 + [1]*90)
    weights = compute_class_weights(labels, method="balanced")
    assert weights[0] > weights[1], "Minority class should have higher weight"
    print("[PASS] test_class_weights_balanced")


def test_class_weights_sqrt():
    labels = np.array([0]*10 + [1]*90)
    weights = compute_class_weights(labels, method="sqrt")
    assert weights[0] > weights[1]
    balanced = compute_class_weights(labels, method="balanced")
    assert weights[0] < balanced[0], "sqrt weights should be softer"
    print("[PASS] test_class_weights_sqrt")


def test_create_patches():
    img = np.random.randn(64, 64)
    patches = create_image_patches(img, patch_size=32, stride=16)
    assert patches.shape[1] == 32
    assert patches.shape[2] == 32
    expected_n = ((64 - 32) // 16 + 1) ** 2  # 3*3 = 9
    assert patches.shape[0] == expected_n, f"Expected {expected_n}, got {patches.shape[0]}"
    print("[PASS] test_create_patches")


if __name__ == "__main__":
    test_minmax_normalization()
    test_zscore_normalization()
    test_robust_normalization()
    test_invalid_normalization()
    test_augment_flip()
    test_augment_rotate()
    test_augment_noise()
    test_stratified_split()
    test_stratified_split_invalid_ratios()
    test_class_weights_balanced()
    test_class_weights_sqrt()
    test_create_patches()
    print("\n=== All 12 tests passed! ===")
