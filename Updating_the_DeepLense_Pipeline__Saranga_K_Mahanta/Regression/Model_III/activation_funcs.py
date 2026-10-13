import torch
import torch.nn.functional as F
from torch import nn


class Mish_layer(nn.Module):
    """
    The class represents Mish activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
