import os
import sys
import unittest

import numpy as np
import torch

_current_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_current_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from components.ChangedPhysicsInformedEncoder import ChangedPhysicsInformedEncoder
from components.FeedForwardBlock import FeedForwardBlock
from components.Lensiformer import Lensiformer
from components.PhysicsInformedEncoder import PhysicsInformedEncoder


class TestPhysicsInformedEncoder(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.image_size = 64
        self.patch_size = 16
        self.embed_dim = 64
        self.encoder = PhysicsInformedEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embedding_dim=self.embed_dim,
            num_patches=16,
            num_heads=4,
            hidden_dim=128,
            transformer_block_activation_function=torch.nn.GELU,
            num_transformer_block=2,
            device=self.device,
            k_max=1.2,
            k_min=0.8,
        )

    def test_tensor_gradient_autograd(self):
        """Verifies that torch.Tensor gradients do not detach autograd or crash."""
        img = torch.randn(self.image_size, self.image_size, requires_grad=True)
        grad_tensor = torch.randn(self.image_size, self.image_size, requires_grad=True)

        source = self.encoder.image_to_source(img, gradient=grad_tensor)
        self.assertEqual(source.shape, (self.image_size, self.image_size))

        loss = source.sum()
        loss.backward()
        self.assertIsNotNone(img.grad)
        self.assertGreater(img.grad.norm().item(), 0.0)

    def test_numpy_gradient_compatibility(self):
        """Verifies backward compatibility when passing a NumPy gradient array."""
        img = torch.randn(self.image_size, self.image_size)
        grad_np = np.random.randn(self.image_size, self.image_size).astype(np.float32)

        source = self.encoder.image_to_source(img, gradient=grad_np)
        self.assertEqual(source.shape, (self.image_size, self.image_size))

    def test_device_buffer_transfer(self):
        """Verifies that registered buffers move with model across devices."""
        self.assertEqual(self.encoder.imones.device.type, self.device.type)
        self.assertEqual(self.encoder.imones_1d.device.type, self.device.type)

        if torch.backends.mps.is_available():
            mps_dev = torch.device("mps")
            encoder_mps = self.encoder.to(mps_dev)
            self.assertEqual(encoder_mps.imones.device.type, "mps")
            self.assertEqual(encoder_mps.imones_1d.device.type, "mps")

            img_mps = torch.randn(self.image_size, self.image_size, device=mps_dev)
            grad_mps = torch.randn(self.image_size, self.image_size, device=mps_dev)
            source_mps = encoder_mps.image_to_source(img_mps, gradient=grad_mps)
            self.assertEqual(source_mps.device.type, "mps")

    def test_feedforward_activation_flexibility(self):
        """Verifies FeedForwardBlock supports both activation classes and instances."""
        ff_class = FeedForwardBlock(64, 64, 128, 1, torch.nn.GELU, "regression")
        ff_inst = FeedForwardBlock(64, 64, 128, 1, torch.nn.GELU(), "regression")

        x = torch.randn(2, 64)
        out_class = ff_class(x)
        out_inst = ff_inst(x)

        self.assertEqual(out_class.shape, (2, 64))
        self.assertEqual(out_inst.shape, (2, 64))

    def test_lensiformer_invert_lens(self):
        """Verifies Lensiformer.invert_lens executes cleanly without missing argument errors."""
        model = Lensiformer(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            in_channels=1,
            num_classes=3,
            num_heads=4,
            num_hidden_neurons=128,
            num_hidden_layers=1,
            transformer_activation=torch.nn.GELU,
            feedforward_activation=torch.nn.GELU,
            num_transformer_blocks=2,
            device=self.device,
        )
        images = torch.randn(2, self.image_size, self.image_size)
        inverted = model.invert_lens(images)
        self.assertEqual(inverted.shape, (2, self.image_size, self.image_size))

    def test_changed_encoder_autograd(self):
        """Verifies ChangedPhysicsInformedEncoder autograd and device handling."""
        changed = ChangedPhysicsInformedEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embedding_dim=self.embed_dim,
            num_patches=16,
            num_heads=4,
            hidden_dim=128,
            transformer_block_activation_function=torch.nn.GELU,
            num_transformer_block=2,
            device=self.device,
            k_max=1.2,
            k_min=0.8,
        )
        img = torch.randn(self.image_size, self.image_size, requires_grad=True)
        grad = torch.randn(self.image_size, self.image_size, requires_grad=True)
        source = changed.image_to_source(img, gradient=grad)
        loss = source.sum()
        loss.backward()
        self.assertIsNotNone(img.grad)


if __name__ == "__main__":
    unittest.main()
