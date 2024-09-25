from PIL import Image
import numpy as np
import torchvision.transforms as Transforms
from .utils import build_transforms
import random
from typing import Tuple, Union, List

class BaseAugmentationDINO:
    def __init__(self):
        self.global_1 = None
        self.global_2 = None
        self.local = None
        self.num_local_crops = None

    def __call__(self, img):
        
        if None in {self.global_1, self.global_2, self.local, self.num_local_crops}:
            nonelist = [var for var, val in {'global_1':self.global_1,\
                                             'global_2':self.global_2,\
                                             'local':self.local,\
                                             'num_local_crops':self.num_local_crops}.items() \
                        if val is None]
            print(f'{nonelist} not initialized')
            return
            
        crops = []
        crops.append(self.global_1(img))
        crops.append(self.global_2(img))
        crops.extend([self.local(img) \
                      for _ in range(self.num_local_crops)])
        return crops
# ---------------------------------------------------------------

def get_dino_augmentations(
        **kwargs
    ):
    augmentation = kwargs.get("augmentation", "AugmentationDINO").lower()
    if augmentation == "AugmentationDINO".lower():
        return AugmentationDINO(**kwargs)
    if augmentation == "AugmentationDINOSingleChannel".lower():
        return AugmentationDINOSingleChannel(**kwargs)
    elif augmentation == "AugmentationDINOexpt1".lower():
        return AugmentationDINOexpt1(**kwargs)
    elif augmentation == "AugmentationDINOexpt2".lower():
        return AugmentationDINOexpt2(**kwargs)
    else:
        raise NotImplementedError


# ---------------------------------------------------------------
class AugmentationDINO(BaseAugmentationDINO):
    '''   
    implements the standard DINO augmentations
    contains augmentations that doesn't affect
    channel information
    adapted from https://github.com/facebookresearch/dino/blob/main/main_dino.py
    '''
    def __init__(self, \
            center_crop: int = 64,
            global_crop_scale_range = [0.4, 1.0],
            global_crop_size: int = 64, 
            local_crop_scale_range = [0.05, 0.4],
            local_crop_size: int = 28,
            num_local_crops: int = 8,
            dataset_mean: Tuple[float, ] = None,
            dataset_std: Tuple[float, ] = None,
            **kwargs):
        super().__init__()
        assert dataset_mean is not None
        assert dataset_std is not None
        self.global_1 = build_transforms(
                    np_input = False,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.0),
                    hue_jitter = kwargs.get('hue_jitter', 0.0),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    # grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_1', 0.08),
                    random_solarize = False,
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )
        
        self.global_2 = build_transforms(
                    np_input = False,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.0),
                    hue_jitter  = kwargs.get('hue_jitter', 0.0),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    # grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_2', 0.08),
                    random_solarize = False,
                    # solarize_probability = kwargs.get('solarize_probability', 0.2),
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.local = build_transforms(
                    np_input = False,
                    crop_size = local_crop_size,
                    center_crop = center_crop,
                    scale_range = local_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.4),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.4),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.0),
                    hue_jitter = kwargs.get('hue_jitter', 0.0),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    # grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_local', 0.5),
                    random_solarize = False,
                    random_rotation = False,
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.num_local_crops = num_local_crops
# ---------------------------------------------------------------

# ---------------------------------------------------------------
class AugmentationDINOSingleChannel(BaseAugmentationDINO):
    '''   
    implements the augmentations that only affects individual channels
    adapted from https://github.com/facebookresearch/dino/blob/main/main_dino.py
    '''
    def __init__(self, 
            center_crop: int = 64,
            global_crop_scale_range: List[float] = [0.4, 1],
            global_crop_size: int = 64, 
            local_crop_scale_range: List[float] = [0.05, 0.4],
            local_crop_size: int = 28,
            num_local_crops: int = 8,
            dataset_mean: Tuple[float, ] = None,
            dataset_std: Tuple[float, ] = None,
            **kwargs):
        super().__init__()
        assert dataset_mean is not None
        assert dataset_std is not None

        self.global_1 = build_transforms(
                    np_input = False,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.8),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.8),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.0),
                    hue_jitter = kwargs.get('hue_jitter', 0.0),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    # grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.01, 4.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_1', 1.0),
                    random_solarize = False,
                    random_rotation = False,
                    # rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    # rotation_probability = (kwargs.get('rotation_probability', (0.08))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )
        
        self.global_2 = build_transforms(
                    np_input = False,
                    center_crop = center_crop,
                    crop_size = global_crop_size,
                    scale_range = global_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.8),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.8),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.0),
                    hue_jitter  = kwargs.get('hue_jitter', 0.0),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    # grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.01, 4.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_2', 0.3),
                    random_solarize = False,
                    random_rotation = False,
                    # rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    # rotation_probability = (kwargs.get('rotation_probability', (0.08))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.local = build_transforms(
                    np_input = False,
                    crop_size = local_crop_size,
                    center_crop = center_crop,
                    scale_range = local_crop_scale_range,
                    horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                    color_jitter = True,
                    brightness_jitter = kwargs.get('brightness_jitter', 0.8),
                    contrast_jitter = kwargs.get('contrast_jitter', 0.8),
                    saturation_jitter = kwargs.get('saturation_jitter', 0.0),
                    hue_jitter = kwargs.get('hue_jitter', 0.0),
                    color_jitter_probability = kwargs.get('color_jitter_probability', 0.1),
                    random_grayscale = False,
                    # grayscale_probability = kwargs.get('grayscale_probability', 0.2),
                    random_gaussian_blur = True,
                    gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.01, 4.0)),
                    gaussian_blur_probability = kwargs.get('gaussian_blur_probability_local', 0.5),
                    random_solarize = False,
                    random_rotation = False,
                    # rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                    # rotation_probability = (kwargs.get('rotation_probability', (0.08))),
                    normalize = True,
                    mean = dataset_mean,
                    std = dataset_std,
                )

        self.num_local_crops = num_local_crops
# ---------------------------------------------------------------
