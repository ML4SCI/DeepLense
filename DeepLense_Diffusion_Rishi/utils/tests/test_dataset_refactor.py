
import unittest
import torch
import numpy as np
import os
import shutil
from dataset.dataset import LensDataset, WrapperDataset
from dataset.transforms import get_transforms

class TestDatasetRefactor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = "test_data_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy .npy files
        self.data_shape = (64, 64)
        
        # Create 'class_a' folder
        self.class_a_dir = os.path.join(self.test_dir, "class_a")
        os.makedirs(self.class_a_dir, exist_ok=True)
        self.file_a = os.path.join(self.class_a_dir, "file_a.npy")
        np.save(self.file_a, np.random.rand(*self.data_shape).astype(np.float32))
        
        # Create 'class_b' folder
        self.class_b_dir = os.path.join(self.test_dir, "class_b")
        os.makedirs(self.class_b_dir, exist_ok=True)
        self.file_b = os.path.join(self.class_b_dir, "file_b.npy")
        np.save(self.file_b, np.random.rand(*self.data_shape).astype(np.float32))

        # Create a file in root for LensDataset test
        self.file_root = os.path.join(self.test_dir, "root_file.npy")
        np.save(self.file_root, np.random.rand(*self.data_shape).astype(np.float32))

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_transforms(self):
        """Test if transforms return expected shape and values."""
        transforms = get_transforms()
        data = np.random.rand(*self.data_shape).astype(np.float32)
        out = transforms(data)
        
        # Check type
        self.assertTrue(isinstance(out, torch.Tensor))
        # Check MinMax normalization (0 to 1)
        self.assertTrue(out.min() >= 0)
        self.assertTrue(out.max() <= 1)
        # Check AddChannel (64, 64) -> (1, 64, 64)
        self.assertEqual(out.shape, (1, 64, 64))

    def test_lens_dataset(self):
        """Test basic LensDataset loading."""
        dataset = LensDataset(root_dir=self.test_dir, transform=None)
        # Should only find root_file.npy in the root dir, ignoring subdirs
        self.assertEqual(len(dataset), 1)
        
        data = dataset[0]
        self.assertEqual(data.shape, self.data_shape)

    def test_wrapper_dataset(self):
        """Test WrapperDataset for conditional loading."""
        dataset = WrapperDataset(root_dir=self.test_dir, transform=None)
        # Should find 2 files (one in class_a, one in class_b)
        self.assertEqual(len(dataset), 2)
        
        data, label = dataset[0]
        self.assertEqual(data.shape, self.data_shape)
        self.assertTrue(isinstance(label, (int, np.integer)))

    def test_lens_dataset_with_transforms(self):
        """Test LensDataset with transforms."""
        transforms = get_transforms()
        dataset = LensDataset(root_dir=self.test_dir, transform=transforms)
        data = dataset[0]
        self.assertEqual(data.shape, (1, 64, 64))
        self.assertTrue(isinstance(data, torch.Tensor))

if __name__ == '__main__':
    unittest.main()
