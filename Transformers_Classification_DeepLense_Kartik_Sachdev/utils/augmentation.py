from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFilter, ImageOps
from timm.data import create_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    Grayscale,
    Pad,
    RandomAffine,
    RandomApply,
    RandomCrop,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)


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

class GaussianBlur:
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class DefaultTransformations:
    def __init__(self) -> None:
        self.get_default_aug_cfg()

    def get_default_aug_cfg(self):
        self.default_aug_cfg = {}

        # images are padded to have shape 129x129.
        self.default_aug_cfg["pad"] = Pad((0, 0, 1, 1), fill=0)

        # to reduce interpolation artifacts
        # upsample an image by a factor of 3, rotate it and finally downsample it again
        self.default_aug_cfg["resize1"] = Resize(387)
        self.default_aug_cfg["resize2"] = Resize(224)  # 129
        self.default_aug_cfg["totensor"] = ToTensor()
        self.default_aug_cfg["togray"] = transforms.Grayscale(num_output_channels=1)

        return self.default_aug_cfg

    def get_test_transform(self):
        transform_test = Compose(
            [
                transforms.RandomCrop(128),
                self.default_aug_cfg["pad"],
                self.default_aug_cfg["resize2"],
                self.default_aug_cfg["togray"],
                self.default_aug_cfg["totensor"],
            ]
        )
        return transform_test

    def get_train_transform_eqv(self):
        transform_train_eqv = transforms.Compose(
            [
                transforms.RandomCrop(128),
                self.default_aug_cfg["pad"],
                self.default_aug_cfg["resize1"],
                RandomRotation(180, resample=Image.BILINEAR, expand=False),
                self.default_aug_cfg["resize2"],
                self.default_aug_cfg["togray"],
                self.default_aug_cfg["totensor"],
            ]
        )
        return transform_train_eqv

    def get_train_transform(self):
        transform_train = Compose(
            [
                transforms.RandomCrop(128),
                self.default_aug_cfg["pad"],
                self.default_aug_cfg["resize1"],
                RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
                RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
                RandomPerspective(distortion_scale=0.3, p=0.1),
                # transforms.RandomApply(random_transform, 0.9999),
                Resize((224, 224)),
                self.default_aug_cfg["togray"],
                self.default_aug_cfg["totensor"],
            ]
        )
        return transform_train

    def get_train_transforms_ssl(self):
        transform_train_1 = Compose(
            [
                transforms.RandomCrop(128),
                self.default_aug_cfg["pad"],
                self.default_aug_cfg["resize1"],
                RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
                RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
                RandomPerspective(distortion_scale=0.3, p=0.1),
                # transforms.RandomApply(random_transform, 0.9999),
                Resize((224, 224)),
                self.default_aug_cfg["togray"],
                self.default_aug_cfg["totensor"],
            ]
        )

        transform_train_2 = Compose(
            [
                transforms.RandomCrop(128),
                self.default_aug_cfg["pad"],
                self.default_aug_cfg["resize1"],
                RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
                # RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                # transforms.RandomApply(random_transform, 0.9999),
                Resize((224, 224)),
                self.default_aug_cfg["togray"],
                self.default_aug_cfg["totensor"],
            ]
        )

        return [transform_train_1, transform_train_2]

    def transform_factory(self):
        random_transform = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_1 = transforms.Compose(
            [
                RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
                RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
                RandomPerspective(distortion_scale=0.3, p=0.1),
            ]
        )

        transform_2 = transforms.Compose(
            [
                RandomRotation(degrees=(0, 180), resample=Image.BILINEAR, expand=False),
                RandomAffine(degrees=(20, 80), translate=(0.1, 0.2), scale=(0.4, 0.95)),
            ]
        )

        transform_list = [transform_1, transform_2]
        transform_prob = 1.0 / len(transform_list)
        for transform in transform_list:
            random_transform.append(transforms.RandomApply([transform], transform_prob))

        transform_3 = transforms.Compose(
            [
                transforms.RandomResizedCrop(150, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=1.0),
                transforms.ToTensor(),
                # normalize,
            ]
        )
        transform_4 = transforms.Compose(
            [
                transforms.RandomResizedCrop(150, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.1),
                transforms.RandomApply([ImageOps.solarize], p=0.2),
                transforms.ToTensor(),
                # normalize,
            ]
        )

        transform_random = create_transform(
            input_size=150,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            interpolation="bicubic",
            mean=(0.5, 0.5),
            std=(0.5, 0.5),
        )
