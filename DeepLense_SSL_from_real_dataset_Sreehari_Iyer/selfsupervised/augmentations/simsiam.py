from PIL import Image
import numpy as np
import torchvision.transforms as Transforms
from .utils import build_transforms
import random
from typing import Tuple, Union, List

class BaseAugmentationSIMSIAM:
    def __init__(self):
        self.transforms = []

    def __call__(self, img):
        crops = [self.transform(img), self.transform(img)]
        return crops
# ---------------------------------------------------------------

def get_simsiam_augmentations(
        **kwargs
    ):
    augmentation = kwargs.get("augmentation", "AugmentationSIMSIAM").lower()
    if augmentation == "AugmentationSIMSIAM".lower():
        return AugmentationSIMSIAM(**kwargs)
    else:
        raise NotImplementedError
        
# ---------------------------------------------------------------
class AugmentationSIMSIAM(BaseAugmentationSIMSIAM):
    '''   
    implements the standard SWAV augmentations
    contains augmentations that doesn't affect
    channel information
    '''
    def __init__(self, \
            center_crop: int = 64,
            crop_size: int = 64,
            crop_scale_range: Union[int, List[int]] = [0.12, 1],
            dataset_mean: Tuple[float, ] = None,
            dataset_std: Tuple[float, ] = None,
            **kwargs):
        super().__init__()

        self.transform = build_transforms(
                            np_input = False,
                            center_crop = center_crop,
                            crop_size = crop_size,
                            scale_range = crop_scale_range,
                            horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                            color_jitter = True,
                            brightness_jitter = kwargs.get('brightness_jitter', 0.8), # 0.8
                            contrast_jitter = kwargs.get('contrast_jitter', 0.8), # 0.8
                            saturation_jitter = kwargs.get('saturation_jitter', 0.0),
                            hue_jitter = kwargs.get('hue_jitter', 0.0),
                            color_jitter_probability = kwargs.get('color_jitter_probability', 0.1), # 0.8
                            random_grayscale = False,
                            random_gaussian_blur = True,
                            gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.01, 4.0)),
                            gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_1', 0.5),
                            random_solarize = False,
                            random_rotation = False,
                            # rotation_degree = (kwargs.get('rotation_degree', (-180., 180.))),
                            # rotation_probability = (kwargs.get('rotation_probability', (0.1))),
                            normalize = True,
                            mean = dataset_mean,
                            std = dataset_std,
                        )
# ---------------------------------------------------------------