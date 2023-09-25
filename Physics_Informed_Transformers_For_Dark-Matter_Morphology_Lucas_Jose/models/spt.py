import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Tuple, Union, List
import torchvision.transforms as transforms



class ShiftedPatchTokenization(nn.Module):

    """

    This module performs shifted patch tokenization on input images,
    generating patches with positional encodings for use in transformer models.

    Attributes:
        embed_dim (int): The dimension of the output embedding.
        image_width (int): Width of the input image.
        image_height (int): Height of the input image.
        patch_width (int): Width of each patch.
        patch_height (int): Height of each patch.
        device (torch.device): The device to use for computation.
        num_patches (int): The total number of patches generated from an image.
        delta_x (int): The amount to shift along the x-axis.
        delta_y (int): The amount to shift along the y-axis.
        total_channels (int): The total number of channels in the output tensor.
        projection (nn.Conv2d): Conv2D layer for patch projection.
        layer_norm (nn.LayerNorm): Layer normalization.
        cls_token (nn.Parameter): Token for class.
        positional_encoding (nn.Parameter): Positional encoding for patches.

    """

    def __init__(self, image_size: Union[int,Tuple[int,int]],
                 patch_size: Union[int,Tuple[int,int]], embed_dim: int,
                 in_channels: int, device: torch.device):

        """
        Initializes the ShiftedPatchTokenization module.

        Args:
            image_size (Union[int, Tuple[int, int]]): Size of the input image. If an integer, it is assumed to be square-shaped.
            patch_size (Union[int, Tuple[int, int]]): Size of each patch. If an integer, it is assumed to be square-shaped.
            embed_dim (int): Dimension of the output embedding.
            in_channels (int): Number of channels in the input image.
            device (torch.device): Device to use for computation.

        Raises:
            ValueError: If the image dimensions are not divisible by the patch dimensions.
        """

        super(ShiftedPatchTokenization, self).__init__()

        # If image_size or patch_size is an integer, convert it to a tuple with equal width and height
        if type(image_size) == int:
            image_size = (image_size, image_size)
        if type(patch_size) == int:
            patch_size = (patch_size, patch_size)

        self.embed_dim = embed_dim
        self.image_width, self.image_height = image_size
        self.patch_width, self.patch_height = patch_size
        self.device = device

        # Check if the image dimensions are divisible by the patch dimensions
        if self.image_width % self.patch_width != 0 or self.image_height % self.patch_height != 0:
            raise ValueError("The image's width must be divisible by the patche's width and the image's height, by the patch's height")

        # Calculate the number of patches
        self.num_patches = int((self.image_width // self.patch_width) * (self.image_height // self.patch_height))

        # Set the translation amounts
        self.delta_x = self.patch_width // 2
        self.delta_y = self.patch_height // 2

        # Set the total number of channels for the output tensor
        self.num_transformations = 4
        self.total_channels = in_channels * (self.num_transformations + 1)

        # Define the projection layer
        self.projection = nn.Conv2d(in_channels=self.total_channels,
                                    out_channels=self.embed_dim,
                                    kernel_size=(self.patch_width, self.patch_height),
                                    stride=(self.patch_width, self.patch_height))

        # Define the layer normalization layer
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # Define the CLS token
        self.cls_token = nn.Parameter(torch.zeros((1, 1, self.embed_dim)))

        # Define the positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros((1, 1 + self.num_patches, self.embed_dim)))

    def translate_image(self, image: Tensor, delta_x: int, delta_y: int, device: torch.device):

        """
        Translates an image by specified amounts along the x and y axes.

        Args:
            image (Tensor): Input image tensor.
            delta_x (int): Shift amount along the x-axis.
            delta_y (int): Shift amount along the y-axis.
            device (torch.device): Device to use for computation.

        Returns:
            Tensor: Translated image.
        """

        translated_image = transforms.functional.affine(image, angle=0, translate=(delta_x, delta_y), fill=0, scale=1, shear=0)
        translated_image = translated_image.to(device)
        return translated_image

    def get_num_patches(self):

        """
        Gets the total number of patches that would be created from an image.

        Returns:
            int: The total number of patches.
        """

        return self.num_patches

    def forward(self, image: Tensor) -> Tensor:

        """
        Performs the forward pass, tokenizing the image into patches and adding positional encodings.

        Args:
            image (Tensor): A tensor representing an image, of shape (batch_size, in_channels, image_height, image_width).

        Returns:
            Tensor: A tensor of tokenized image patches, of shape (batch_size, num_patches+1, embed_dim).
        """

        batch_size = image.size(0)


        # generate shifted versions of the image
        shift_left_up = self.translate_image(image, -self.delta_x, -self.delta_y,self.device)
        shift_right_up = self.translate_image(image, self.delta_x, -self.delta_y,self.device)
        shift_left_down = self.translate_image(image, -self.delta_x, self.delta_y,self.device)
        shift_right_down = self.translate_image(image, self.delta_x, self.delta_y,self.device)

        # concatenate the original image with its shifted versions
        concatenated_images = torch.cat((image, shift_left_up, shift_right_up, shift_left_down, shift_right_down), dim=1)

        # project the concatenated image onto a lower-dimensional embedding
        projected_patches = self.projection(concatenated_images)
        patches = projected_patches.flatten(2)
        patches = patches.transpose(1, 2)

        patches.to(self.device)

        # apply layer normalization to the patches
        patches = self.layer_norm(patches)

        # append a learnable "class token" to the beginning of the patch sequence
        cls = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat((cls, patches), dim=1)

        patches.to(self.device)

        # add learnable positional encodings to the patches
        patches = patches + self.positional_encoding

        return patches
