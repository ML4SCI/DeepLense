import os
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device ='cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    #Sets the seed for Reprocudibility
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

train_transforms = A.Compose(
            [
                A.HorizontalFlip(p = 0.5),
                A.VerticalFlip(p = 0.5),
                A.Rotate(limit = 360, p = 0.4),
                ToTensorV2()
            ]
        )


test_transforms = A.Compose(
            [
                ToTensorV2()
            ]
        )

