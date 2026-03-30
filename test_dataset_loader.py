"""Tests for the dataset_loader module."""

import numpy as np
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_loader import DatasetLoader


class TestDatasetLoader:
    """Test suite for DatasetLoader."""

    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def test_load_npy_safe_valid():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    loader = DatasetLoader(d)
    arr = np.random.rand(64, 64).astype(np.float32)
    path = os.path.join(d, "test.npy")
    np.save(path, arr)
    loaded = loader.load_npy_safe(path)
    assert loaded is not None
    assert loaded.shape == (64, 64)
    t.cleanup()
    print("[PASS] test_load_npy_safe_valid")


def test_load_npy_safe_invalid():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    loader = DatasetLoader(d)
    path = os.path.join(d, "bad.npy")
    with open(path, "wb") as f:
        f.write(b"\x00\x01\x02\x03invalid")
    loaded = loader.load_npy_safe(path)
    assert loaded is None
    t.cleanup()
    print("[PASS] test_load_npy_safe_invalid")


def test_validate_grayscale():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    loader = DatasetLoader(d, channel_format="channels_last")
    img = np.random.rand(64, 64).astype(np.float32)
    result = loader.validate_image_shape(img)
    assert result is not None
    assert result.shape == (64, 64, 1)
    t.cleanup()
    print("[PASS] test_validate_grayscale")


def test_validate_channels_first_to_last():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    loader = DatasetLoader(d, channel_format="channels_last")
    img = np.random.rand(3, 64, 64).astype(np.float32)
    result = loader.validate_image_shape(img)
    assert result is not None
    assert result.shape == (64, 64, 3)
    t.cleanup()
    print("[PASS] test_validate_channels_first_to_last")


def test_detect_uint8():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    loader = DatasetLoader(d)
    imgs = np.random.randint(0, 255, (10, 64, 64)).astype(np.float32)
    assert loader.detect_normalization(imgs) == "uint8"
    t.cleanup()
    print("[PASS] test_detect_uint8")


def test_detect_normalized():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    loader = DatasetLoader(d)
    imgs = np.random.rand(10, 64, 64).astype(np.float32)
    assert loader.detect_normalization(imgs) == "normalized"
    t.cleanup()
    print("[PASS] test_detect_normalized")


def test_detect_centered():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    loader = DatasetLoader(d)
    imgs = np.random.rand(10, 64, 64).astype(np.float32) * 2 - 1
    assert loader.detect_normalization(imgs) == "centered"
    t.cleanup()
    print("[PASS] test_detect_centered")


def test_normalize_uint8_to_01():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    loader = DatasetLoader(d)
    imgs = np.array([0, 128, 255], dtype=np.float32)
    normed = loader.normalize_to_range(imgs, "normalized")
    assert abs(normed[0]) < 1e-5
    assert abs(normed[-1] - 1.0) < 1e-5
    t.cleanup()
    print("[PASS] test_normalize_uint8_to_01")


def test_load_dataset():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    for i in range(5):
        np.save(os.path.join(d, f"img_{i}.npy"),
                np.random.rand(64, 64).astype(np.float32))
    loader = DatasetLoader(d)
    images, labels = loader.load_dataset()
    assert images.shape[0] == 5
    assert labels is None
    t.cleanup()
    print("[PASS] test_load_dataset")


def test_load_dataset_with_labels():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    for i in range(5):
        np.save(os.path.join(d, f"img_{i}.npy"),
                np.random.rand(64, 64).astype(np.float32))
    np.save(os.path.join(d, "labels.npy"), np.array([0, 1, 2, 0, 1]))
    loader = DatasetLoader(d)
    images, labels = loader.load_dataset(label_file="labels.npy")
    assert images.shape[0] == 5
    assert labels is not None
    assert len(labels) == 5
    t.cleanup()
    print("[PASS] test_load_dataset_with_labels")


def test_get_dataset_info():
    t = TestDatasetLoader()
    d = t.setup_temp_dir()
    np.save(os.path.join(d, "test.npy"), np.random.rand(32, 32))
    loader = DatasetLoader(d)
    info = loader.get_dataset_info()
    assert info["num_npy_files"] == 1
    assert "sample_shape" in info
    t.cleanup()
    print("[PASS] test_get_dataset_info")


def test_invalid_channel_format():
    try:
        DatasetLoader(".", channel_format="invalid")
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("[PASS] test_invalid_channel_format")


if __name__ == "__main__":
    test_load_npy_safe_valid()
    test_load_npy_safe_invalid()
    test_validate_grayscale()
    test_validate_channels_first_to_last()
    test_detect_uint8()
    test_detect_normalized()
    test_detect_centered()
    test_normalize_uint8_to_01()
    test_load_dataset()
    test_load_dataset_with_labels()
    test_get_dataset_info()
    test_invalid_channel_format()
    print("\n=== All 12 tests passed! ===")
