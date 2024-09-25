from PIL import Image
from PIL.ImageOps import solarize as Solarize
from PIL.ImageFilter import GaussianBlur 
import numpy as np
import random
import torch
import torchvision.transforms as Transforms
from typing import Tuple, Union, List

class gaussian_blur:
    def __init__(
            self,
            p: float,
            sigma: Tuple[float, float]
        ):
        self.p = p
        self.sigma = sigma 

    def __call__(
            self, 
            img):
        if random.random() > self.p:
            return img
        if isinstance(img, torch.Tensor):
            return Transforms.functional.gaussian_blur(img, 7, (float(self.sigma[0]), float(self.sigma[1])))
        else:
            return img.filter(
                GaussianBlur(
                    radius=random.uniform(float(self.sigma[0]), float(self.sigma[1]))
                )
            )

class randomrotation:
    def __init__(
            self,
            p: float,
            rotation_degree: Tuple[float, float]
        ):
        self.p = p
        self.rotation_degree = rotation_degree 

    def __call__(
            self, 
            img):
        if random.random() > self.p:
            return img
        angle = random.choice(np.arange(self.rotation_degree[0], self.rotation_degree[1]))
        if isinstance(img, torch.Tensor):
            return Transforms.functional.rotate(img, angle=angle)
        else:
            return img.rotate(angle=angle)

class solarize:
    def __init__(
            self, \
            p: float):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if isinstance(img, torch.Tensor):
            return Transforms.functional.solarize(img, img.median())
        else:
            return Solarize(img)


    
def random_color_jitter(
            brightness_jitter: float,
            contrast_jitter: float,
            saturation_jitter: float,
            hue_jitter: float,
            color_jitter_probability: float,
        ):
        return Transforms.RandomApply([
                    Transforms.ColorJitter(\
                        brightness=brightness_jitter,
                        contrast=contrast_jitter,
                        saturation=saturation_jitter,
                        hue=hue_jitter,)
                    ],
                    p=color_jitter_probability
                )

def build_transforms(
            np_input: bool = True,
            crop_size: int = 64,
            scale_range = [0.0, 1.0],
            horizontal_flip_probability: float = 0.5,
            center_crop: int = 64,
            color_jitter: bool = True,
            brightness_jitter: float = 0.,
            contrast_jitter: float = 0.,
            saturation_jitter: float = 0.,
            hue_jitter: float = 0.,
            color_jitter_probability: float = 0.,
            random_grayscale: bool = True,
            grayscale_probability: float = 0.,
            random_gaussian_blur: bool = True,
            gaussian_blur_sigma: Tuple[float, float] = (0., 0.),
            gaussian_blur_probability: float = 0.,
            random_solarize: bool = True,
            solarize_probability: float = 0.,
            random_rotation: bool = True,
            rotation_degree: Union[float, Tuple[float, float]] = (0., 0.),
            rotation_probability: float = 0.,
            normalize: bool = True,
            mean: Union[float, Tuple[float, ]] = 0.,
            std: Union[float, Tuple[float, ]] = 0.,
            to_tensor: bool = False # npy loader returns torch.Tensor
        ):
    transforms = []
    if np_input:
        transforms.append(Transforms.ToPILImage())
    if to_tensor:
        transforms.append(Transforms.ToTensor())
    
    if center_crop > 0:
        transforms.append(Transforms.CenterCrop(center_crop))
    transforms.extend([
            Transforms.RandomResizedCrop(crop_size,
                                 scale=scale_range,
                                 interpolation=Image.BICUBIC,
                                 antialias=True),
            Transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
            Transforms.RandomVerticalFlip(p=horizontal_flip_probability)
    ])
    if color_jitter:
        transforms.append(
            random_color_jitter(
                brightness_jitter = brightness_jitter,
                contrast_jitter = contrast_jitter,
                saturation_jitter = saturation_jitter,
                hue_jitter = hue_jitter,
                color_jitter_probability = color_jitter_probability,
            ))
    if random_grayscale:
        transforms.append(
            Transforms.RandomGrayscale(p=grayscale_probability)
        )
    if random_gaussian_blur:
        transforms.append(
            gaussian_blur(
                sigma = gaussian_blur_sigma,
                p=gaussian_blur_probability
            )
        )
    if random_solarize:
        transforms.append(
            solarize(p=solarize_probability)
        )
    if random_rotation:
        transforms.append(
            randomrotation(p=rotation_probability, rotation_degree=rotation_degree)
        )
    if normalize:
        transforms.append(Transforms.Normalize(mean, std))
    
                        
    return Transforms.Compose(transforms)
