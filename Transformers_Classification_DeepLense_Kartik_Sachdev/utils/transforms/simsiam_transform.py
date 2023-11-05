from typing import Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SimSiamTransform:
    """Implements the augmentations for SimSiam.
    """

    def __init__(self):
        pass
    
    def get_transforms(self, 
                       global_crop_size: int = 224,
                       global_crop_scale: Tuple[float, float] = (0.4, 1.0),
                       local_crop_size: int = 96,
                       local_crop_scale: Tuple[float, float] = (0.05, 0.4),):

        # first global crop
        transform = A.Compose(
        [
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.Resize(global_crop_size, global_crop_size, p=1.0),
            A.RandomResizedCrop(height=global_crop_size, width=global_crop_size),
            A.Rotate(p=0.5), 
            ToTensorV2(),
        ])

        return [transform, transform]
    