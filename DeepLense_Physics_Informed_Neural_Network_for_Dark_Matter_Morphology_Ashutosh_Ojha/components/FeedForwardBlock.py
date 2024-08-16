import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from typing import Tuple, Union, List
import time

class FeedForwardBlock(nn.Module):
    """
    This is a FeedForward block with structure like:
    Linear -> Activation Function -> Dropout -> Linear... (repeated for num_hidden_layers)
    
    This block is designed for regression, binary classification, or multi-class classification tasks.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 num_hidden_layers: int,
                 activation_function: nn.Module,
                 task_type: str,
                 dropout: float = 0.1
                ):
        super(FeedForwardBlock, self).__init__()  # corrected super() call
        
        self.feed_list = nn.ModuleList()
        self.feed_list.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))  # add initial linear layer
        self.feed_list.append(activation_function())  # add activation function
        self.feed_list.append(nn.Dropout(dropout))  # add dropout layer
        
        # add hidden layers
        for _ in range(num_hidden_layers):
            self.feed_list.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            self.feed_list.append(activation_function())
            self.feed_list.append(nn.Dropout(dropout))
        
        # determine final activation layer based on task type
        if task_type == "regression":
            self.final_activation_layer = nn.Identity()
        elif task_type == "binary":
            if out_dim != 1:
                raise ValueError("For binary classification, out_dim should be 1.")
            self.final_activation_layer = nn.Sigmoid()
        elif task_type == "multiclass":
            self.final_activation_layer = nn.Softmax(dim=1)
        else:
            raise ValueError("Task type should be one of: 'regression', 'binary', 'multiclass'.")

        self.feed_list.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))
        self.feed_list.append(self.final_activation_layer)
        
    def forward(self, x):
        """
        Forward pass through the FeedForwardBlock.
        """
        for layer in self.feed_list:
            x = layer(x)
        return x
