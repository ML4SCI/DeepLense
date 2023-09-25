import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Tuple, Union, List
import torchvision.transforms as transforms
from models import FeedForward, TransformerLSABlock, Flatten, ShiftedPatchTokenization, RelativisticPhysicalInformedEncoder

class Lensiformer(nn.Module):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 embed_dim: int,
                 in_channels: int,
                 num_classes: int,
                 num_heads: int,
                 num_hidden_neurons: int,
                 num_hidden_layers: int,
                 transformer_activation: nn.Module,
                 feedforward_activation: nn.Module,
                 num_transformer_blocks: int,
                 device: torch.device,
                 dropout_rate: float = 0.1):
        """
        Initializes Lensiformer, a Relativistic Physics-Informed Vision Transformer (PIViT) Architecture for Dark Matter Morphology.

        Args:
            image_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each image patch (assumed square).
            embed_dim (int): Dimension of the embedding space.
            in_channels (int): Number of input channels.
            num_classes (int): Number of target classes.
            num_heads (int): Number of attention heads.
            num_hidden_neurons (int): Number of neurons in hidden layers.
            num_hidden_layers (int): Number of hidden layers.
            transformer_activation (nn.Module): Activation function for transformer blocks.
            feedforward_activation (nn.Module): Activation function for feedforward layers.
            num_transformer_blocks (int): Number of transformer blocks.
            device (torch.device): Computational device (CPU/GPU).
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(Lensiformer, self).__init__()

        # Initialize parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_hidden_neurons = num_hidden_neurons
        self.num_hidden_layers = num_hidden_layers
        self.transformer_activation = transformer_activation
        self.feedforward_activation = feedforward_activation
        self.num_transformer_blocks = num_transformer_blocks
        self.device = device
        self.dropout_rate = dropout_rate

        # Initialize modules
        self.initial_tokenizer = ShiftedPatchTokenization(image_size, patch_size, embed_dim, in_channels, device)
        self.secondary_tokenizer = ShiftedPatchTokenization(image_size, patch_size, embed_dim, in_channels, device)
        self.encoder = RelativisticPhysicalInformedEncoder(image_size, patch_size, embed_dim, self.initial_tokenizer.get_num_patches(), num_heads, num_hidden_neurons, transformer_activation, num_transformer_blocks, device)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerLSABlock(embed_dim, num_heads, self.initial_tokenizer.get_num_patches(), num_hidden_neurons, transformer_activation, device, dropout_rate)
            for _ in range(num_transformer_blocks)
        ])

        # Flatten and FeedForward layers
        self.flatten_layer = Flatten((self.initial_tokenizer.get_num_patches() + 1) * embed_dim)
        self.feedforward_layer = FeedForward(self.flatten_layer.num_neurons_flatten,
                                             num_classes, feedforward_activation, num_hidden_neurons,
                                             num_hidden_layers, task_type='multi_classification', dropout=dropout_rate)

    def invert_lens(self, images: Tensor) -> Tensor:
        batch_size = images.size(0)

        # Tokenize input images into patches
        initial_patches = self.initial_tokenizer(images.reshape(batch_size, 1, self.image_size, self.image_size))

        # Encode images and patches
        lens_corrected_images = self.encoder(images, initial_patches)

        return lens_corrected_images


    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass through the Physics-Informed Vision Transformer.

        Args:
            images (Tensor): Input images with shape (batch_size, channels, height, width).

        Returns:
            Tensor: Model predictions with shape (batch_size, num_classes).
        """
        batch_size = images.size(0)

        # Tokenize input images into patches
        initial_patches = self.initial_tokenizer(images.reshape(batch_size, 1, self.image_size, self.image_size))

        # Encode images and patches
        lens_corrected_images = self.encoder(images, initial_patches)
        lens_corrected_patches = self.secondary_tokenizer(lens_corrected_images.reshape(batch_size, 1, self.image_size, self.image_size))

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            initial_patches = block(key=initial_patches, value=lens_corrected_patches)

        # Flatten the patches
        flattened_patches = self.flatten_layer(initial_patches)

        # Generate final predictions
        final_predictions = self.feedforward_layer(flattened_patches)

        return final_predictions
