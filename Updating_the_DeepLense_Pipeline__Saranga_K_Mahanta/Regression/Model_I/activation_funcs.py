import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish_layer(nn.Module):
    '''
    The class represents Mish activation function.
    '''
    def __init__(self):
        super(Mish_layer,self).__init__()
    def forward(self,x):
        return x*torch.tanh(F.softplus(x))