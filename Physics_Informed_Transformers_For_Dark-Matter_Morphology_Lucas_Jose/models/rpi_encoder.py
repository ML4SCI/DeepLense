import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Tuple, Union, List
import torchvision.transforms as transforms
from models import FeedForward, TransformerLSABlock

class RelativisticPhysicalInformedEncoder(nn.Module):
    """
    A PyTorch module to perform inverse gravitational lensing using the Singular Isothermal Sphere (SIS) model.

    Attributes:
        pixel_scale (float): The scale of each pixel in the image, often in arcseconds per pixel.
        profile_size (int): The size of the image profile.
        half_profile_size (int): Half of the profile size.
        num_patches (int): Number of patches.
        embed_dim (int): Dimension of the embedding.
        num_heads (int): Number of heads in the transformer.
        num_hidden_neurons (int): Number of hidden neurons.
        eps (float): Float number used to avoid division by zero.
        transformer_activation_function (nn.Module): Activation function used in the transformer.
        num_transformer_blocks (int): Number of transformer blocks.
        device (torch.device): Device to which tensors will be moved.
        transformer (nn.ModuleList): List of transformer blocks.
        num_neurons_flatten (int): Number of neurons in the Flatten layer.
        grid_x (Tensor): Grid of x coordinates.
        grid_y (Tensor): Grid of y coordinates.
        flat_grid_x (Tensor): Flattened grid of x coordinates.
        flat_grid_y (Tensor): Flattened grid of y coordinates.
    """
    def __init__( self,
                  image_size: int,
                  patch_size: int ,
                  embed_dim: int,
                  num_patches: int,
                  num_heads: int,
                  num_hidden_neurons: int,
                  transformer_activation_function: nn.Module,
                  num_transformer_blocks: int,
                  device: torch.device,
                  dropout: float = 0.1,
                  pixel_scale:float =0.101,
                  k_min: float = 0.8,
                  k_max: float = 1.2,
                  eps: float = 1e-8
                  ):


        """
        Initialize the module.

        Args:
            image_size (int): The size of the image.
            patch_size (int): The size of each patch.
            embed_dim (int): The embedding dimension.
            num_patches (int): The number of patches.
            num_heads (int): The number of heads in the transformer.
            num_hidden_neurons (int): The number of hidden neurons.
            transformer_activation_function (nn.Module): The activation function used in the transformer.
            num_transformer_blocks (int): The number of transformer blocks.
            device (torch.device): The device to which tensors will be moved.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            pixel_scale (float, optional): The scale of each pixel in the image, often in arcseconds per pixel. Defaults to 0.101.
            k_min (float, optional): Minimum value of the potential correction parameter.  Defaults to 0.8.
            k_max (float, optional): Maximum value of the potencial correction parameter.  Defaults to 1.2.
            eps (float): Float number used to avoid division by zero. Defaluts to 1e-8.
        """

        super(RelativisticPhysicalInformedEncoder, self).__init__()

        # Initialize variables
        self.pixel_scale = pixel_scale
        self.profile_size = image_size
        self.half_profile_size = self.profile_size // 2
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_hidden_neurons = num_hidden_neurons
        self.transformer_activation_function = transformer_activation_function
        self.num_transformer_blocks = num_transformer_blocks
        self.device = device
        self.k_min = k_min
        self.k_max = k_max
        self.eps = eps

        # Create an empty list for the transformer blocks
        self.transformer = nn.ModuleList()

        # Calculate the number of neurons for the Flatten layer
        self.num_neurons_flatten = (self.num_patches+1)*embed_dim

        # Create an empty list for the transformer blocks
        self.transformer = nn.ModuleList()

        # Create a for loop that iterates over the number of transformer blocks

        for _ in range(num_transformer_blocks):
          # Add a TransformerLSABlock to the transformer list
          self.transformer.append(

              TransformerLSABlock(embed_dim,
                                  num_heads,
                                  self.num_patches,
                                  num_hidden_neurons,
                                  transformer_activation_function,
                                  device,
                                  dropout)
          )

        # Create a compressor (FeedFoward) compress the size of num_neurons_flatten
        self.transformer.append(nn.Flatten())
        self.transformer.append(nn.Linear(self.num_neurons_flatten,self.profile_size*self.profile_size))
        self.transformer.append(nn.Sigmoid())

        # Create a grid for image coordinates
        x_coordinates = torch.linspace(-self.half_profile_size, self.half_profile_size-1, self.profile_size) * self.pixel_scale
        y_coordinates = torch.linspace(-self.half_profile_size, self.half_profile_size-1, self.profile_size) * self.pixel_scale
        self.grid_x, self.grid_y = torch.meshgrid(x_coordinates, y_coordinates)
        self.flat_grid_x = self.grid_x.flatten().to(self.device)
        self.flat_grid_y = self.grid_y.flatten().to(self.device)


    def forward(self, input_images: Tensor, patches: Tensor)->Tensor:

        """
        Forward pass through the module.

        Args:
            input_images (torch.Tensor): The input images.
            patches (torch.Tensor): The patches extracted from the images.

        Returns:
            output_images (torch.Tensor): The output images after inverse gravitational lensing.
        """

        # Get the batch size from the input images
        batch_size = input_images.shape[0]

        # Generate k using the sequential model
        for i, layer in enumerate(self.transformer):
            if i == 0:
                k_sigmoid = layer(patches)
            else:
                k_sigmoid = layer(k_sigmoid)

        # Reshape k_sigmoid to have shape [batch_size, profile_size, profile_size]
        k_sigmoid = k_sigmoid.view(-1, self.profile_size, self.profile_size)

        # Flatten k_sigmoid to match the shape of non_zero_x and non_zero_radius
        k_sigmoid_flat = k_sigmoid.view(-1, self.profile_size*self.profile_size)

        # Bias and Scalling
        k_sigmoid_flat = self.k_min + (self.k_max-self.k_min)*k_sigmoid_flat

         # Flatten the input images for easier indexing
        flat_input_images = input_images.view(batch_size, -1)

        # Create a mask for non-zero coordinates in the grid
        non_zero_mask = (self.flat_grid_x != 0) | (self.flat_grid_y != 0)

        # Select only the non-zero indices to match with non_zero_x and non_zero_radius
        k_sigmoid_non_zero = k_sigmoid_flat[:, non_zero_mask]

        # Get the shape of k_sigmoid_non_zero
        shape_k_sigmoid_non_zero = k_sigmoid_non_zero.shape

        # Reshape k to have a batch dimension compatible for broadcasting
        k = k_sigmoid_non_zero.view(shape_k_sigmoid_non_zero[0], 1, 1, shape_k_sigmoid_non_zero[1])

        # Apply the mask to get non-zero coordinates
        non_zero_x = self.flat_grid_x[non_zero_mask]
        non_zero_y = self.flat_grid_y[non_zero_mask]

        # Calculate the radius for non-zero coordinates
        non_zero_radius = torch.sqrt(non_zero_x ** 2 + non_zero_y ** 2)

        # Expand dimensions for broadcasting
        non_zero_radius = non_zero_radius[None, None, None, :]

        # Compute shifted coordinates based on the Gravitational Lens Equation to SIS model
        shifted_x = (non_zero_x[None, None, None, :] - k * non_zero_x[None, None, None, :] / non_zero_radius)
        shifted_y = (non_zero_y[None, None, None, :] - k * non_zero_y[None, None, None, :] / non_zero_radius)

        # Convert shifted coordinates to indices in the image grid
        shifted_x_idx = torch.round(shifted_x / self.pixel_scale + self.half_profile_size).long()
        shifted_y_idx = torch.round(shifted_y / self.pixel_scale + self.half_profile_size).long()

        # Initialize the output image tensor and flatten it
        output_images = torch.zeros(batch_size, self.profile_size, self.profile_size).to(self.device)
        flat_output_images = output_images.view(batch_size, -1)

        # Calculate 1D indices from shifted_x_idx and shifted_y_idx
        one_d_indices = shifted_x_idx * self.profile_size + shifted_y_idx

        # Flatten the input images for easier indexing
        flat_input_images = input_images.view(batch_size, -1)

        # Get the current values at the shifted positions in the flat output images
        output_values_at_shifted_positions = flat_output_images.gather(1, one_d_indices.view(batch_size, -1))

        # Get the corresponding values from the original positions in the flat input images
        input_values_at_original_positions = flat_input_images[:, non_zero_mask]

        # Update the output image based on the algorithm
        updated_values = torch.where(output_values_at_shifted_positions == 0,
                                    input_values_at_original_positions,
                                    (output_values_at_shifted_positions + input_values_at_original_positions) / 2)

        # Assign the updated values back to the flat output images
        flat_output_images.scatter_(1, one_d_indices.view(batch_size, -1), updated_values)

        # Reshape the flat output images back to their original shape
        output_images = flat_output_images.view(batch_size, self.profile_size, self.profile_size)


        # Normalize the output images
        max_values, _ = output_images.max(dim=1, keepdim=True)
        max_values, _ = max_values.max(dim=2, keepdim=True)
        output_images = output_images / (max_values + self.eps)

        return output_images



