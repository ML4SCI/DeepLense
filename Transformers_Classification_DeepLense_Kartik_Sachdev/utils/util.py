import os
from typing import List
import random
import torch
import numpy as np
import logging
import torch.nn as nn
from typing import Optional


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


def init_logging_handler(log_dir, current_time, extra="", use_ray=False):
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

    if not use_ray:
        os.makedirs(f"{log_dir}/{current_time}/checkpoint", exist_ok=True)


def check_trainable_layers(model: nn.Module):
    # Iterate over the model parameters and check the trainability
    print("Trainable layers: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}")


def load_model_add_head(
    pretrain_model: nn.Module,
    saved_model_path,
    head: nn.Module,
    freeze_pretrain_layers: Optional[bool] = True,
) -> nn.Module:
    pretrain_model.load_state_dict(torch.load(saved_model_path))

    requires_grad = not (freeze_pretrain_layers)
    for param in pretrain_model.parameters():
        param.requires_grad = requires_grad

    check_trainable_layers(pretrain_model)

    # Combine the pretrained model and the projection head
    model = nn.Sequential(*list(pretrain_model.children())[:-1], head)

    return model


def load_dummy_model_with_head(
    backbone: nn.Module,
    head: nn.Module,
) -> nn.Module:
    # model = nn.Sequential(backbone, head)
    model = nn.Sequential(*list(backbone.children())[:-1], head)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_second_last_layer(model: nn.Module):
    # Get the second last layer
    layers = list(model.children())

    layer_names = list(model._modules.keys())
    second_last_layer_name = layer_names[-3]
    second_last_layer = model._modules[second_last_layer_name]
    return second_last_layer

    # second_last_layer = layers[-2]


def second_last_layer(model):
    random_input = torch.randn(1, 1, 224, 224)
    (model.children())[:-1]
    print(random_input)


def deactivate_requires_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def activate_requires_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Momentum encoders are a crucial component fo models such as MoCo or BYOL.

    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Momentum encoders are a crucial component fo models such as MoCo or BYOL.

    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


def get_last_layer_features(model: nn.Module, num_input=1, device="cuda"):
    random_input = []
    for num in range(num_input):
        random_input.append(torch.randn(1, 1, 224, 224).to(device))

    if num_input == 1:
        output = model(random_input[0])
    elif num_input == 2:
        output = model(random_input[0], random_input[1])

    num_last_features = output.size(1)
    return num_last_features
