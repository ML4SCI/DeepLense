import torch
from torchvision.models import resnet18
from utils.util import *

model = resnet18(pretrained=True)
second_last_layer(model)