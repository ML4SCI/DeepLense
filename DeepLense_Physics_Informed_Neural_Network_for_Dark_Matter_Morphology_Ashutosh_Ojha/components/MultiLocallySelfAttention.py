import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from typing import Tuple, Union, List
import time

class MultiLocallySelfAttention(nn.Module):
    """This is a special form of self attention where the intra self-attention 
    calculation is not calculated and the attention calculation is not done 
    from itself it is done by assigning a large weight for the inter-token
    attention calculation and zero for the attention calculation between 
    the tokens itself for this an attention mask is calculated with False 
    on the diagonal and True on the else where. 
    like this 
       [[ True, False, False, False],
        [False,  True, False, False],
        [False, False,  True, False],
        [False, False, False,  True]] 
    """
    def __init__(self, embedding_dim, num_heads, num_patches, device, dropout=0.1):
        super(MultiLocallySelfAttention, self).__init__()  # corrected super() call
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.device = device
        self.dropout = dropout
        
        # Initialize the attention mask
        #self.attention_mask = torch.ones(1 + self.num_patches, 1 + self.num_patches, dtype=torch.bool)
        #self.attention_mask = self.attention_mask.triu(1) + self.attention_mask.tril(-1)
        #self.attention_mask = self.attention_mask.to(self.device)
        self.attention_mask = torch.eye(1 + self.num_patches, 1 + self.num_patches, dtype=torch.bool, requires_grad=False)
        self.attention_mask = self.attention_mask.to(device)

        # Initializing the multi-head self-attention layer
        self.mha = nn.MultiheadAttention(self.embedding_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        
    def forward(self, key, query, value):
        x, _ = self.mha(query, key, value, attn_mask=self.attention_mask)
        return x
