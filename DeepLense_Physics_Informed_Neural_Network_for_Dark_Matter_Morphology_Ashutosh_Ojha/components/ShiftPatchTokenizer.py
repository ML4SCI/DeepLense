import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from typing import Tuple, Union, List
import time

class ShiftPatchTokenizer(nn.Module):
    def __init__(self, image_size, embedding_size, patch_size, num_channels, translation_mode, device='cpu'):
        super(ShiftPatchTokenizer, self).__init__()

        self.image_size = image_size
        self.embedding_size = embedding_size  # size of the features of each patch
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.translation_mode = translation_mode
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if self.image_size % self.patch_size != 0:
            raise ValueError("The image size must be divisible by the patch size")
        
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.num_translations = 4
        self.total_channels = self.num_channels * (1 + self.num_translations)

        self.tokenizer = nn.Conv2d(in_channels=self.total_channels,
                                   out_channels=self.embedding_size,
                                   kernel_size=(self.patch_size, self.patch_size),
                                   stride=(self.patch_size, self.patch_size))

        self.patch_width, self.patch_height = (patch_size, patch_size)
        self.shift_value_x, self.shift_value_y = (self.patch_width // 2, self.patch_height // 2)

        # This will convert the shape of image from I x I x C to self.num_patch(in square) x embedding_size
        # Now the class embeddings
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, self.embedding_size))

        # Now creating the learnable parameter for the positional encoding
        self.positional = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embedding_size))

    def translation(self, image, delta_x, delta_y):
        trans_image = F.affine(img=image,
                               angle=0,
                               translate=(delta_x, delta_y),
                               scale=1,
                               shear=0)
        trans_image = trans_image.to(self.device)
        return trans_image
    def get_num_patches(self):
        return self.num_patches
    
    def forward(self, image):
        image = image.to(self.device)
        batch = image.shape[0]

        if self.translation_mode == "diagonal":
            shift_left_up = self.translation(image, -self.shift_value_x, -self.shift_value_y)
            shift_right_up = self.translation(image, self.shift_value_x, -self.shift_value_y)
            shift_left_down = self.translation(image, -self.shift_value_x, self.shift_value_y)
            shift_right_down = self.translation(image, self.shift_value_x, self.shift_value_y)

            # Concatenate the image + all the shifts
            patched_images = torch.cat((image, shift_left_up, shift_right_up, shift_left_down, shift_right_down), dim=1)

        elif self.translation_mode == "rectangular":
            shift_down = self.translation(image, 0, self.shift_value_y)
            shift_up = self.translation(image, 0, -self.shift_value_y)
            shift_left = self.translation(image, -self.shift_value_x, 0)
            shift_right = self.translation(image, self.shift_value_x, 0)

            # Concatenate the image + all the shifts
            patched_images = torch.cat((image, shift_down, shift_up, shift_left, shift_right), dim=1)
        
        else:
            raise ValueError("Invalid translation_mode")

        # Projection of the images to get the embeddings of all the patches so we have the shape batch, (height/patch_size), (width/patch_size), embedding_size
        projected_embeddings = self.tokenizer(patched_images)
        #print(projected_embeddings.shape)
        patches = projected_embeddings.flatten(2).transpose(1, 2)

        # The new shape of patches is batch, num_patches, embedding_size
        patches = patches.to(self.device)
        
        # Concatenating the class embeddings
        #print(patches.shape)
        class_tokens = self.class_embedding.expand(batch, -1, -1)
        patches = torch.cat((class_tokens, patches), dim=1)

        # Adding the positional encodings
        positional_patches = patches + self.positional

        return positional_patches
