import torch
import numpy as np
from torch import nn
from torch import Tensor
from typing import Tuple, Union, List
import torchvision.transforms as transforms
from models import FeedForward

class TransformerLSABlock(nn.Module):

    """
    Implements a TransformerLSABlock, a building block for transformer models designed
    to work on image patches. This block contains multi-locally self-attention,
    feedforward network, and layer normalization components.

    Attributes:
        mlsa (MultiLocallySelfAttention): MultiLocallySelfAttention layer.
        first_norm (nn.LayerNorm): First layer normalization.
        feedforward (FeedForward): FeedForward neural network layer.
        second_norm (nn.LayerNorm): Second layer normalization.
        dropout_layer (nn.Dropout): Dropout layer for regularization.
        device (torch.device): Device to run the computations on.
        temperature (nn.Parameter): Temperature parameter for attention mechanism.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_patches: int,
                 num_hidden_neurons: int,
                 activation_function: nn.Module,
                 device: torch.device,
                 dropout: float = 0.1):
        """
        Initializes the TransformerLSABlock layer.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            num_patches (int): Number of patches in the image.
            num_hidden_neurons (int): Number of neurons in the hidden layer of the feedforward network.
            activation_function (nn.Module): Activation function used in the feedforward network.
            device (torch.device): Device to run the computations on.
            dropout (float, optional): Dropout rate for regularization. Default is 0.1.
        """

        super().__init__()

        # initialize the multi-locally self-attention layer
        self.mlsa = MultiLocallySelfAttention(embed_dim, num_heads, num_patches, device, dropout)

        # initialize the first layer normalization
        self.first_norm = nn.LayerNorm(embed_dim)

        # initialize the feedforward network
        self.feedforward = FeedForward(embed_dim, embed_dim, activation_function, num_hidden_neurons, num_hidden_layers=1,
                                       task_type='regression', dropout=dropout)

        # initialize the second layer normalization
        self.second_norm = nn.LayerNorm(embed_dim)

        # initialize the dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # the device to store the tensors on
        self.device = device

        # initialize temperature parameter (a scalar used to divide the queries)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, key: torch.Tensor, query: Union[torch.Tensor, None] = None, value: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Performs the forward pass through the TransformerLSABlock layer.

        Args:
            key (Tensor): The key tensor for the attention mechanism.
            query (Union[Tensor, None], optional): The query tensor for the attention mechanism.
                                                  If None, it defaults to the key tensor divided by temperature.
                                                  Default is None.
            value (Union[Tensor, None], optional): The value tensor for the attention mechanism.
                                                   If None, it defaults to the key tensor.
                                                   Default is None.

        Returns:
            Tensor: The output tensor after applying multi-locally self-attention,
                    layer normalization, feedforward network, and dropout.
        """
        # Prepare key, query, and value tensors
        key = key.to(self.device)
        if query is None:
            query = (key / self.temperature).to(self.device)
        if value is None:
            value = key.to(self.device)

        # apply multi-locally self-attention
        value = value + self.mlsa(key, query, value)

        # apply first layer normalization
        value = self.first_norm(value)

        # apply feedforward network
        value = value + self.feedforward(value)

        # apply second layer normalization
        value = self.second_norm(value)

        # apply dropout
        value = self.dropout_layer(value)

        return value
