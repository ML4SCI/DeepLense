import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from typing import Tuple, Union, List
import time
import timm

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
        self.initial_tokenizer = ShiftPatchTokenizer(image_size=image_size, embedding_size=embed_dim, patch_size=patch_size, num_channels=in_channels, translation_mode="diagonal", device=device)
        self.secondary_tokenizer = ShiftPatchTokenizer(image_size=image_size, embedding_size=embed_dim, patch_size=patch_size, num_channels=in_channels, translation_mode="diagonal", device=device)
        self.tertiary_tokenizer = ShiftPatchTokenizer(image_size=image_size, embedding_size=embed_dim, patch_size=patch_size, num_channels=in_channels, translation_mode="diagonal", device=device)
        self.encoder = PhysicsInformedEncoder(image_size=image_size, 
                                              patch_size=patch_size, 
                                              embedding_dim = embed_dim, 
                                              num_patches = self.initial_tokenizer.get_num_patches(), 
                                              num_heads=num_heads,
                                              hidden_dim=num_hidden_neurons, 
                                              transformer_block_activation_function=feedforward_activation, 
                                              num_transformer_block = num_transformer_blocks,
                                              device = device, 
                                              k_max = 1.2, 
                                              k_min = 0.8)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerLSABlock(embedding_dim = embed_dim,
                                num_head = num_heads,
                                num_patches = self.initial_tokenizer.get_num_patches(),
                                num_hidden_neurons=num_hidden_neurons,
                                num_hidden_layers=1,
                                activation_function=transformer_activation,
                                device=device)
            for _ in range(num_transformer_blocks)
        ])

        # Flatten and FeedForward layers
        self.flatten_layer = Flatten((self.initial_tokenizer.get_num_patches() + 1) * embed_dim)
        self.feedforward_layer = FeedForwardBlock(in_dim = self.flatten_layer.num_neurons_flatten,
                                                  out_dim = num_classes,
                                                  hidden_dim = num_hidden_neurons,
                                                  num_hidden_layers = num_hidden_layers,
                                                  activation_function = feedforward_activation,
                                                  task_type = 'multiclass',
                                                  dropout=dropout_rate)
        """self.siam = SiameseNet("resnet18",
                               num_classes,
                               num_hidden_neurons,
                               num_hidden_layers,
                               feedforward_activation,
                               dropout_rate,
                               mode = "difference"
                              )"""
        #base_network = BaseNetwork()
        #self.siam = SiameseNetwork(base_network, 2)
        
    def invert_lens(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        images_T = images.reshape(batch_size, 1, self.image_size, self.image_size)
        # Tokenize input images into patches
        initial_patches = self.initial_tokenizer(images_T)

        # Encode images and patches
        lens_corrected_images = self.encoder(images, initial_patches)

        return lens_corrected_images

    def forward(self, images: torch.Tensor,distortions) -> torch.Tensor:
        """
        Forward pass through the Physics-Informed Vision Transformer.

        Args:
            images (Tensor): Input images with shape (batch_size, channels, height, width).

        Returns:
            Tensor: Model predictions with shape (batch_size, num_classes).
        """
        images = images.to(self.device)
        distortions = distortions.to(self.device)
        batch_size = images.size(0)

        # Tokenize input images into patches
        dis_patches = self.tertiary_tokenizer(distortions.reshape(batch_size,1,self.image_size,self.image_size))
        initial_patches = self.initial_tokenizer(images.reshape(batch_size, 1, self.image_size, self.image_size))
        #print(initial_patches.shape)
        # Encode images and patches
        #print("images",images.shape)
        
        info,lens_corrected_images = self.encoder(images, initial_patches,distortions)
        source = lens_corrected_images.view(batch_size,1,64,64)
        source = source.to(self.device)
        
        source_patches = self.secondary_tokenizer(source)
        concatenated = torch.concat((dis_patches,source_patches-initial_patches),dim=2)
        # Pass through transformer blocks
        verbose = 2
        if verbose ==0:
            print("source_shape = ",source_patches.shape)
            print("observed_shape = ",initial_patches.shape)
            print("distortion_shape = ",dis_patches.shape)
            print("concat feats",concatenated.shape)

        for block in self.transformer_blocks:
            concatenated = block(key = dis_patches,value = initial_patches,query = source_patches)
        # Flatten the patches
        flattened_patches = self.flatten_layer(concatenated)
        # Generate final predictions
        final_predictions = self.feedforward_layer(flattened_patches)
        return info,source,final_predictions
