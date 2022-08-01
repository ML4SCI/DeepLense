import os
from typing import List
import random
import torch
import numpy as np
import logging


def make_directory(dirname: str) -> None:
    """makes a single directory

    Args:
        dirname (str): name of desired directory
    """
    os.makedirs(dirname, exist_ok=True)


def make_directories(dirnames: List[str]) -> None:
    """makes directories from a list of desired directories

    Args:
        dirnames (List[str]): list of desired directories
    """
    for dirname in dirnames:
        make_directory(dirname)


def seed_everything(seed):
    """Fixing various seeds

    Args:
        seed (int): any seed number

    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(use_cuda=True, cuda_idx=0):
    """Get the CUDA device

    Args:
        use_cuda (Bool): To used CUDA or not
        cuda_idx (int): index of CUDA device 
    
    Returns:
        device: CUDA device(s) being used 
    """

    if use_cuda:
        if torch.cuda.is_available():
            assert cuda_idx in range(
                0, torch.cuda.device_count()
            ), "GPU index out of range. index lies in [{}, {})".format(
                0, torch.cuda.device_count()
            )
            device = torch.device("cuda:" + str(cuda_idx))
        else:
            print("cuda not found, will switch to cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device = {str(device)}")
    return device


def init_logging_handler(log_dir, current_time, extra=""):
    """Initializes the handler for logger. Create the logger directory if it doest exists. 
        Define the format of logging
        DEBUG logging level being used

    Args:
        log_dir (str): Logger directory
        current_time (str): time from logging to begin  
        extra (str): Space for adding extra info in .txt file
    
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(os.path.join(log_dir, current_time)):
        os.makedirs(os.path.join(log_dir, current_time))

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        "{}/{}/log_{}.txt".format(log_dir, current_time, current_time + extra)
    )
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logging.getLogger("matplotlib.font_manager").disabled = True

    os.makedirs(f"{log_dir}/{current_time}/checkpoint", exist_ok=True)

