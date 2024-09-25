import torch
import random
import numpy as np
import os

# utility function for reproducibility
# call this before running the train loop
SEED = 12
def set_seed(seed: int = SEED, device: str = "cuda") -> None:
    '''
    sets seed for reproducibility in training
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

set_seed()

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#torch.use_deterministic_algorithms(True)
