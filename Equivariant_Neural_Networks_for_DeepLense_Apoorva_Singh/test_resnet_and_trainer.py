import os
import tempfile
import unittest
import torch
import numpy as np

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from resnet_model import resnet18, conv1x1
from trainer import ECNN


class TestEquivariantResNetAndTrainer(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    def test_conv1x1_instantiation(self):
        layer = conv1x1(64, 128, stride=2)
        self.assertIsInstance(layer, torch.nn.Conv2d)
        self.assertEqual(layer.kernel_size, (1, 1))
        self.assertEqual(layer.stride, (2, 2))
        self.assertFalse(layer.bias)

    def test_resnet18_forward_and_backward(self):
        num_classes = 3
        model = resnet18(num_classes=num_classes).to(self.device)
        # DeepLense gravitational lensing images are single-channel grayscale
        x = torch.randn(2, 1, 224, 224, device=self.device, requires_grad=True)
        out = model(x)

        self.assertEqual(out.shape, (2, num_classes))

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.norm().item(), 0.0)

    def test_resnet18_arbitrary_classes(self):
        for n_cls in [2, 4, 10]:
            model = resnet18(num_classes=n_cls).to(self.device)
            x = torch.randn(1, 1, 64, 64, device=self.device)
            out = model(x)
            self.assertEqual(out.shape, (1, n_cls))

    def test_trainer_checkpoint_deserialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = resnet18(num_classes=3).to(self.device)
            ts = "test_run"
            save_dir = os.path.join(tmpdir, ts)
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, "best-model-parameters.pt")
            torch.save(model.state_dict(), checkpoint_path)

            import logging
            logger = logging.getLogger("TestLogger")

            # ECNN in Test mode must load checkpoint using map_location=device
            trainer = ECNN(
                logger=logger,
                log_dir=tmpdir,
                current_time=ts,
                n_classes=3,
                device=self.device,
                use_CNN=True,
                mode='Test'
            )
            self.assertIsNotNone(trainer.model)

    def test_trainer_dynamic_classes_one_hot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import logging
            logger = logging.getLogger("TestLogger")
            trainer = ECNN(
                logger=logger,
                log_dir=tmpdir,
                current_time="test_ts",
                n_classes=5,
                device=self.device,
                use_CNN=True,
                mode='Train'
            )
            labels = np.array([0, 2, 4])
            one_hot = trainer._ECNN__to_one_hot_vector(trainer.n_classes, labels)
            self.assertEqual(one_hot.shape, (3, 5))
            self.assertEqual(one_hot[0, 0], 1)
            self.assertEqual(one_hot[1, 2], 1)
            self.assertEqual(one_hot[2, 4], 1)


if __name__ == '__main__':
    unittest.main()
