from PIL import Image
import numpy as np
import torchvision.transforms as Transforms
from .utils import build_transforms
import random
from typing import Tuple, Union, List

class BaseAugmentationSWAV:
    def __init__(self):
        self.transforms = []

    def __call__(self, img):
        crops = [transform(img) for transform in self.transforms]
        return crops
# ---------------------------------------------------------------

def get_swav_augmentations(
        **kwargs
    ):
    augmentation = kwargs.get("augmentation", "AugmentationSWAV").lower()
    if augmentation == "AugmentationSWAV".lower():
        return AugmentationSWAV(**kwargs)
    else:
        raise NotImplementedError
        
# ---------------------------------------------------------------
class AugmentationSWAV(BaseAugmentationSWAV):
    '''   
    implements the standard SWAV augmentations
    contains augmentations that doesn't affect
    channel information
    '''
    def __init__(self, \
            center_crop: int = 64,
            size_crops: Union[int, List[int]] = [64],
            num_crops: Union[int, List[int]] = [2], 
            min_scale_crops: Union[float, List[float]] = [.14],
            max_scale_crops: Union[float, List[float]] = [1.],
            dataset_mean: Tuple[float, ] = None,
            dataset_std: Tuple[float, ] = None,
            **kwargs):
        super().__init__()
        
        if not isinstance(size_crops, List):
            size_crops = [size_crops]
        if not isinstance(num_crops, List):
            num_crops = [num_crops]
        assert len(num_crops) == len(size_crops), f"len num_crops = {len(num_crops)} and len size_crops = {len(size_crops)} are different"
        if not isinstance(min_scale_crops, List):
            min_scale_crops = [min_scale_crops]
        assert len(min_scale_crops) == len(size_crops), f"len min_scale_crops = {len(min_scale_crops)} and len size_crops = {len(size_crops)} are different"
        if not isinstance(max_scale_crops, List):
            max_scale_crops = [max_scale_crops]
        assert len(max_scale_crops) == len(size_crops), f"len max_scale_crops = {len(max_scale_crops)} and len size_crops = {len(size_crops)} are different"
        
        assert dataset_mean is not None, f"dataset_mean is None"
        assert dataset_std is not None, f"dataset_std is None"

        for i in range(len(size_crops)):
            for _ in range(num_crops[i]):
                self.transforms.append(
                    build_transforms(
                        np_input = False,
                        center_crop = center_crop,
                        crop_size = size_crops[i],
                        scale_range = [min_scale_crops[i], max_scale_crops[i]],
                        horizontal_flip_probability = kwargs.get('horizontal_flip_probability', 0.5),
                        color_jitter = True,
                        brightness_jitter = kwargs.get('brightness_jitter', 0.4), # 0.8
                        contrast_jitter = kwargs.get('contrast_jitter', 0.4), # 0.8
                        saturation_jitter = kwargs.get('saturation_jitter', 0.0),
                        hue_jitter = kwargs.get('hue_jitter', 0.0),
                        color_jitter_probability = kwargs.get('color_jitter_probability', 0.1), # 0.8
                        random_grayscale = False,
                        random_gaussian_blur = True,
                        gaussian_blur_sigma = kwargs.get('gaussian_blur_sigma', (0.1, 2.0)),
                        gaussian_blur_probability = kwargs.get('gaussian_blur_probability_global_1', 0.5),
                        random_solarize = False,
                        random_rotation = False,
                        normalize = True,
                        mean = dataset_mean,
                        std = dataset_std,
                    )
                )
# ---------------------------------------------------------------