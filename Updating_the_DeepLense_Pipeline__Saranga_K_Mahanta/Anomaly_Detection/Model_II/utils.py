import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device ='cuda' if torch.cuda.is_available() else 'cpu'

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

