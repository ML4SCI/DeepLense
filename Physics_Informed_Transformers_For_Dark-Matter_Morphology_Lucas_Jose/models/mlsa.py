import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Tuple, Union, List
import torchvision.transforms as transforms

class MultiLocallySelfAttention(nn.Module):

    """
    Implements a MultiLocallySelfAttention layer, which is a specialized form of multi-head self-attention
    designed to attend over local patches of an image.

    Attributes:
        embed_dim (int): Embedding dimensionality of the input.
        num_heads (int): Number of attention heads.
        num_patches (int): Number of patches in the image.
        dropout (float): Dropout rate for regularization.
        device (torch.device): Device to run the computations on.
        attn_mask (torch.Tensor): Attention mask for self-attention operation.
        mha (nn.MultiheadAttention): Standard multi-head attention layer.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_patches: int, device: torch.device, dropout: float = 0.1):

        """
        Initializes the MultiLocallySelfAttention layer.

        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            num_patches (int): Number of patches in the image.
            device (torch.device): Device to run the computations on.
            dropout (float): Dropout rate for regularization. Default is 0.1.

        """

        super().__init__()

        # Store layer parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.dropout = dropout
        self.device = device

        # Initialize attention mask (a lower triangular matrix with True values)
        self.attn_mask = torch.eye(1 + self.num_patches, 1 + self.num_patches, dtype=torch.bool, requires_grad=False)
        self.attn_mask = self.attn_mask.to(device)

        # Initialize multi-head attention layer
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)

    def forward(self, key: Tensor, query: Tensor, value: Tensor) -> Tensor:

        """
        Performs the forward pass through the MultiLocallySelfAttention layer.

        Args:
            key (Tensor): The key tensor for attention mechanism.
            query (Tensor): The query tensor for attention mechanism.
            value (Tensor): The value tensor for attention mechanism.

        Returns:
            Tensor: The output tensor after applying multi-head attention.

        """

        # Apply multi-head attention
        x, _ = self.mha(query, key, value, attn_mask=self.attn_mask)

        return x
