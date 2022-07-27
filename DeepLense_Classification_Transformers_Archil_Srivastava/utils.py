from logging import warn
import torch
from torch.nn import Softmax
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from constants import *

def get_device(device):
    if (device == 'tpu' or device == 'best') and 'COLAB_TPU_ADDR' in os.environ:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    if (device == 'cuda' or device == 'best') and torch.cuda.is_available():
        return 'cuda'
    if (device == 'mps' or device == 'best') and torch.has_mps:
        return 'mps'
    if (device == 'cpu' or device == 'best'):
        return 'cpu'
    warn(f'Requested device {device} not found, running on CPU')
    return 'cpu'


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


