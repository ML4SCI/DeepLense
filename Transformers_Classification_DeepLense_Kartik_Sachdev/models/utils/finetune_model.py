from typing import Any
import torch.nn as nn
from utils.util import activate_requires_grad, deactivate_requires_grad
from models.byol import BYOLSingleChannel
import copy

class FinetuneModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()

        # Combine the pretrained model and the projection head
        self.model = nn.Sequential(*list(backbone.children())[:-1], head)

    def forward(self, x):
        x = self.model(x)
        return x



