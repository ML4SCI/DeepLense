from torch import nn
import numpy as np
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A CNN with residual connections

    Attributes
    ----------
    image_x: int
        dim 3 of the images to be processed
    image_y: int
        dim 2 of the images to be processed    
    image_c: int
        dim 1 of the images to be processed (number of channels)
    out_shape: int
        Shape of the output vector

    Methods
    -------
    forward(x)
        Feed-forward method
    """
    def __init__(self, image_x, image_y, image_c, out_shape):
        super(CNN, self).__init__()
        pow_2 = int(np.floor(np.log2(max(image_x, image_y))))
        self.module_list = nn.ModuleList()
        channels = image_c
        image_x_, image_y_ = image_x, image_y
        for _ in range(0,pow_2,2):
            self.module_list.append(
                nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=4, stride=2, padding=1)
            )
            self.module_list.append(
                nn.ReLU()
            )
            self.module_list.append(
                nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
            )
            channels *= 2
            image_x_ //= 4
            image_y_ //= 4
        self.fc1 = nn.Linear(in_features=channels*image_x_*image_y_, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=out_shape)
    
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
        