from typing import Optional
from torchvision import transforms
from torchvision.transforms import (
    RandomRotation,
    RandomCrop,
    Pad,
    Resize,
    RandomAffine,
    ToTensor,
    Compose,
    RandomPerspective,
    Grayscale,
    RandomApply,
    RandomVerticalFlip,
    RandomHorizontalFlip,
)
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


def get_transform_train(
    upsample_size: int, final_size: int, channels: Optional[int] = 1
):
    """Trainset transformation

    Args:
        upsample_size (int): intermediate upsampling size
        final_size (int): final size of image to network
        channels (Optional[int], optional): number of channels of final image to network. Defaults to 1.

    Returns:
        Compose: transforms.Compose

    Example:
        >>> get_transform_test(387, 224, 1)
    """

    random_transform = []
    transform1 = transforms.Compose(
        [
            RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
            RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
            RandomPerspective(distortion_scale=0.3, p=0.1),
        ]
    )

    transform2 = transforms.Compose(
        [
            RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
            RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
        ]
    )

    transform_list = [transform1, transform2]

    transform_prob = 1.0 / len(transform_list)
    for transform in transform_list:
        random_transform.append(RandomApply([transform], transform_prob))

    transform_simple = Compose(
        [Resize(final_size), Grayscale(num_output_channels=1), ToTensor(),]
    )

    if channels == 3:
        transform_train = Compose(
            [
                RandomCrop(128),
                # Pad((0, 0, 1, 1), fill=0),
                Resize(upsample_size),
                RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
                RandomVerticalFlip(p=0.3),
                RandomHorizontalFlip(p=0.3),
                Resize(final_size),
                ToTensor(),
            ]
        )
        return transform_train

    transform_train = Compose(
        [
            RandomCrop(128),
            Resize(final_size),
            Grayscale(num_output_channels=1),
            ToTensor(),
        ]
    )

    transform_effective = A.Compose(
        [
            # A.RandomCrop(80, 80, p=0.5),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.Resize(final_size, final_size, p=1.0),
            ToTensorV2(),
        ]
    )
    return transform_effective


def get_transform_test(final_size: int, channels: Optional[int] = 1):
    """Testset transformation

    Args:
        final_size (int): final size of image to network
        channels (int, optional): number of channels of final image. Defaults to 1.

    Returns:
        Compose: transforms.Compose
    
    Example:
        >>> get_transform_test(387, 224, 1)
    """
    if channels == 3:
        transform_test = Compose([Resize(final_size), ToTensor(),])
        return transform_test

    transform_test = Compose(
        [Resize(final_size), Grayscale(num_output_channels=1), ToTensor(),]
    )

    transform_effective = A.Compose(
        [A.Resize(final_size, final_size, p=1.0), ToTensorV2()]
    )

    return transform_effective
