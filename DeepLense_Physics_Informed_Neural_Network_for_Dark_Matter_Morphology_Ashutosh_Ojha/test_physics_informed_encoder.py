import os
import sys
import unittest

import numpy as np
import torch
from torch import nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPONENTS_DIR = os.path.join(CURRENT_DIR, "components")
if COMPONENTS_DIR not in sys.path:
    sys.path.insert(0, COMPONENTS_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from components.ChangedPhysicsInformedEncoder import ChangedPhysicsInformedEncoder
from components.Physics import Physics
from components.PhysicsInformedEncoder import PhysicsInformedEncoder
from components.TransformerLSABlock import TransformerLSABlock


class TestPhysicsInformedEncoder(unittest.TestCase):
    def setUp(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.image_size = 64
        self.image = torch.ones((self.image_size, self.image_size), device=self.device)

    def test_component_instantiation(self):
        """TransformerLSABlock, PhysicsInformedEncoder, and ChangedPhysicsInformedEncoder must instantiate cleanly."""
        block = TransformerLSABlock(
            embedding_dim=64,
            num_head=4,
            num_patches=16,
            num_hidden_neurons=128,
            num_hidden_layers=1,
            activation_function=nn.GELU,
            device=self.device,
            dropout=0.1,
        )
        self.assertIsInstance(block, nn.Module)

        encoder = PhysicsInformedEncoder(
            image_size=self.image_size,
            patch_size=4,
            embedding_dim=64,
            num_patches=256,
            num_heads=4,
            hidden_dim=128,
            transformer_block_activation_function=nn.GELU,
            num_transformer_block=1,
            device=self.device,
            k_max=1.0,
            k_min=0.0,
        )
        self.assertIsInstance(encoder, nn.Module)

        changed_encoder = ChangedPhysicsInformedEncoder(
            image_size=self.image_size,
            patch_size=4,
            embedding_dim=64,
            num_patches=256,
            num_heads=4,
            hidden_dim=128,
            transformer_block_activation_function=nn.GELU,
            num_transformer_block=1,
            device=self.device,
            k_max=1.0,
            k_min=0.0,
        )
        self.assertIsInstance(changed_encoder, nn.Module)

    def test_encoder_image_to_source_with_requires_grad_tensor(self):
        """image_to_source must accept a gradient tensor with requires_grad=True without autograd detachment."""
        encoder = PhysicsInformedEncoder(
            image_size=self.image_size,
            patch_size=4,
            embedding_dim=64,
            num_patches=256,
            num_heads=4,
            hidden_dim=128,
            transformer_block_activation_function=nn.GELU,
            num_transformer_block=1,
            device=self.device,
            k_max=1.0,
            k_min=0.0,
        )

        grad_tensor = torch.randn(
            self.image_size, self.image_size, device=self.device, requires_grad=True
        )

        # Must not raise RuntimeError: Can't call numpy() on Tensor that requires grad.
        source = encoder.image_to_source(self.image, gradient=grad_tensor)
        self.assertIsInstance(source, torch.Tensor)
        self.assertEqual(source.shape, (self.image_size, self.image_size))
        self.assertEqual(source.device.type, self.device.type)

    def test_encoder_image_to_source_gradient_numerical_parity(self):
        """torch.gradient in image_to_source must yield identical results to np.gradient."""
        np.random.seed(42)
        np_arr = np.random.randn(self.image_size, self.image_size).astype(np.float32)
        tensor = torch.from_numpy(np_arr).to(self.device)

        # Compute with numpy
        np_gx = np.gradient(np_arr, axis=0)
        np_gy = np.gradient(np_arr, axis=1)

        # Compute with torch native
        torch_gx, torch_gy = torch.gradient(tensor, dim=(0, 1))

        np.testing.assert_allclose(torch_gx.cpu().numpy(), np_gx, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(torch_gy.cpu().numpy(), np_gy, rtol=1e-5, atol=1e-6)

    def test_encoder_image_to_source_with_numpy_input(self):
        """image_to_source must gracefully handle numpy arrays without device mismatch errors."""
        encoder = PhysicsInformedEncoder(
            image_size=self.image_size,
            patch_size=4,
            embedding_dim=64,
            num_patches=256,
            num_heads=4,
            hidden_dim=128,
            transformer_block_activation_function=nn.GELU,
            num_transformer_block=1,
            device=self.device,
            k_max=1.0,
            k_min=0.0,
        )

        np_grad = np.random.randn(self.image_size, self.image_size).astype(np.float32)
        source = encoder.image_to_source(self.image, gradient=np_grad)
        self.assertIsInstance(source, torch.Tensor)
        self.assertEqual(source.shape, (self.image_size, self.image_size))
        self.assertEqual(source.device.type, self.device.type)

    def test_changed_encoder_image_to_source(self):
        """ChangedPhysicsInformedEncoder must also support torch.gradient with device preservation."""
        changed_encoder = ChangedPhysicsInformedEncoder(
            image_size=self.image_size,
            patch_size=4,
            embedding_dim=64,
            num_patches=256,
            num_heads=4,
            hidden_dim=128,
            transformer_block_activation_function=nn.GELU,
            num_transformer_block=1,
            device=self.device,
            k_max=1.0,
            k_min=0.0,
        )

        grad_tensor = torch.randn(
            self.image_size, self.image_size, device=self.device, requires_grad=True
        )
        source = changed_encoder.image_to_source(self.image, gradient=grad_tensor)
        self.assertIsInstance(source, torch.Tensor)
        self.assertEqual(source.shape, (self.image_size, self.image_size))

    def test_physics_class_accepts_tensor_and_numpy(self):
        """Physics.image_to_source must accept both numpy arrays and torch tensors without raising exceptions."""
        phys = Physics(mag=1)
        np_image = np.ones((self.image_size, self.image_size), dtype=float)

        # 1. NumPy input
        np_grad = np.random.randn(self.image_size, self.image_size)
        res_np = phys.image_to_source(np_image, gradient=np_grad)
        self.assertIsInstance(res_np, np.ndarray)

        # 2. PyTorch tensor with requires_grad
        t_grad = torch.randn(self.image_size, self.image_size, requires_grad=True)
        res_t = phys.image_to_source(np_image, gradient=t_grad)
        self.assertIsInstance(res_t, np.ndarray)
        self.assertEqual(res_t.shape, (self.image_size, self.image_size))

    def test_torch_gradient_autograd_backward(self):
        """torch.gradient must support loss.backward() and propagate valid gradients."""
        grad_tensor = torch.randn(
            self.image_size, self.image_size, device=self.device, requires_grad=True
        )
        gx, gy = torch.gradient(grad_tensor, dim=(0, 1))
        loss = gx.sum() + gy.sum()
        loss.backward()
        self.assertIsNotNone(grad_tensor.grad)
        self.assertEqual(grad_tensor.grad.shape, grad_tensor.shape)
        self.assertGreater(grad_tensor.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
