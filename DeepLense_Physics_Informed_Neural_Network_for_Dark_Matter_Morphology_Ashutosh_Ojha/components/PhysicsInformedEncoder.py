import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from typing import Tuple, Union, List
import time

# --- ADDED IMPORTS TO FIX UNDEFINED VARIABLES ---
from Physics import Physics
from TransformerLSABlock import TransformerLSABlock

class PhysicsInformedEncoder(nn.Module):
    def __init__(self, image_size, patch_size, embedding_dim, num_patches, num_heads,
                 hidden_dim, transformer_block_activation_function, num_transformer_block,
                 device: torch.device, k_max, k_min, num_hidden_layer=1, dropout=0.1, pixel_scale=0.101,
                 min_angle=-3.323, max_angle=3.232, mag=1):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.k_max = k_max
        self.k_min = k_min
        self.transformer_block_activation_function = transformer_block_activation_function
        self.num_transformer_block = num_transformer_block
        self.device = device
        self.dropout = dropout
        self.pixel_scale = pixel_scale
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.source_mag = mag

        self.transformer = nn.ModuleList()

        for _ in range(self.num_transformer_block):
            self.transformer.append(
                TransformerLSABlock(embedding_dim=self.embedding_dim,
                                    num_head=self.num_heads,
                                    num_patches=self.num_patches,
                                    num_hidden_neurons=self.hidden_dim,
                                    num_hidden_layers=num_hidden_layer,
                                    activation_function=self.transformer_block_activation_function,
                                    device=self.device,
                                    dropout=dropout)
            )

        # Adding a compressor to give us the deflection angle for each pixel
        self.transformer.append(nn.Flatten())
        self.total_features = (self.num_patches + 1) * self.embedding_dim
        self.transformer.append(nn.Linear(self.total_features, self.image_size*self.image_size))
        self.physics = Physics(mag=1)
        self.imones = torch.ones((self.image_size,self.image_size)).to(self.device)
        self.imones_1d = torch.ones((self.image_size*self.image_size)).to(self.device)
        # Create a grid for image coordinates
        self.profile_size = self.image_size
        self.half_profile_size = self.profile_size // 2

        # --- OPTIMIZATION: Precompute static coordinate grids to save memory/time in forward pass ---
        self.cached_pixel_width = (self.max_angle - self.min_angle) / self.image_size
        self.cached_centre_x, self.cached_centre_y = self.image_size // 2, self.image_size // 2
        range_x = torch.arange(-(self.cached_centre_x - 1), self.image_size - (self.cached_centre_x - 1), device=self.device)
        range_y = torch.arange(-(self.cached_centre_y - 1), self.image_size - (self.cached_centre_y - 1), device=self.device)
        grid_x, grid_y = torch.meshgrid(range_x, range_y, indexing='ij')
        self.cached_x = grid_x * self.cached_pixel_width
        self.cached_y = grid_y * self.cached_pixel_width
        r_grid = torch.sqrt(self.cached_x**2 + self.cached_y**2)
        r_grid[r_grid == 0] = 1
        self.cached_r = r_grid
        # -----------------------------------------------------------------------------------------

    def image_to_source(self, image, centre=None, E_r=None, deflection=None, gradient=None):
        length, width = image.shape
        
        # OPTIMIZATION: Use cached grids if using default image size to avoid redundant GPU allocations
        if centre is None and length == self.image_size and width == self.image_size:
            pixel_width = self.cached_pixel_width
            centre_x, centre_y = self.cached_centre_x, self.cached_centre_y
            x, y, r = self.cached_x, self.cached_y, self.cached_r
        else:
            pixel_width = (self.max_angle - self.min_angle) / length
            if centre is None:
                centre = (length // 2, width // 2)
            centre_x, centre_y = centre
            if False:
                centre_x = centre_x.item()
                centre_y = centre_y.item()
            
            range_indices_x = torch.arange(-(centre_x - 1), length - (centre_x - 1), device=self.device)
            range_indices_y = torch.arange(-(centre_y - 1), width - (centre_y - 1), device=self.device)

            x, y = torch.meshgrid(range_indices_x, range_indices_y, indexing='ij')
            x, y = x * pixel_width, y * pixel_width

            r = torch.sqrt(x**2 + y**2)
            mask = (r == 0)
            r[mask] = 1

        if deflection is not None:
            if deflection.shape != (length, width):
                raise ValueError(f"The deflection should be of shape (2, {length}, {width}) but got {deflection.shape}")
            xdef = (deflection * x) / r
            ydef = (deflection * y) / r
        elif gradient is not None:
            if gradient.shape != image.shape:
                raise ValueError("The gradient and image should be of the same shape")
            xdef = np.gradient(gradient, axis=0)
            ydef = np.gradient(gradient, axis=1)
        elif E_r is not None:
            k = torch.ones((length, width), device=self.device) * E_r
            xdef = (k * x) / r
            ydef = (k * y) / r
        else:
            raise ValueError("Both deflection and gradient cannot be None")

        bx = x - xdef
        by = y - ydef

        bx, by = bx / pixel_width, by / pixel_width

        bx = torch.clamp(bx + centre_x * self.source_mag, 0, length * self.source_mag - 1).long()
        by = torch.clamp(by + centre_y * self.source_mag, 0, width * self.source_mag - 1).long()

        # Initialize the output image tensor
        sourceimage = torch.zeros((length * self.source_mag, width * self.source_mag), dtype=torch.float, device=self.device)
        
        # Flatten the image and coordinates
        flat_image = image.view(-1)
        flat_bx = bx.view(-1)
        flat_by = by.view(-1)

        # Calculate 1D indices
        one_d_indices = flat_bx * (width * self.source_mag) + flat_by

        # Scatter add the values to sourceimage
        sourceimage_flat = sourceimage.view(-1)
        count  = torch.zeros_like(sourceimage_flat)
        sourceimage_flat = sourceimage_flat.scatter_add(0, one_d_indices, flat_image)
        count = count.scatter_add(0, one_d_indices, self.imones_1d)
        mask = count!=0
        sourceimage_flat[mask]/=count[mask]
        sourceimage = sourceimage_flat.view(sourceimage.shape)

        return sourceimage


    def forward(self, input_images, patches,distortion):
        # Move tensors to the appropriate device
        input_images = input_images.to(self.device)
        distortion = distortion.to(self.device)
        patches = patches.to(self.device)

        batch_size = input_images.shape[0]

        # Process patches through transformer layers
        k_sigmoid = patches
        for layer in self.transformer:
            k_sigmoid = layer(k_sigmoid)

        # Extract output components from k_sigmoid
        k_sigmoid = k_sigmoid.view(batch_size,64,64)
        distortion = distortion.view(batch_size,64,64)
        einstein_angle = k_sigmoid*distortion

        # Reshape input images
        input_images = input_images.view(batch_size, self.image_size, self.image_size)

        # Initialize source_image tensor
        source_image = torch.zeros_like(input_images)

        # Transform each image in the batch
        for i in range(batch_size):
            source_image_np = self.image_to_source(image=input_images[i], E_r=einstein_angle[i])
            source_image[i] = source_image_np

        return k_sigmoid,source_image