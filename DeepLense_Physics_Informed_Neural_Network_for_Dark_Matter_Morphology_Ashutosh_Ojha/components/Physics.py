import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from typing import Tuple, Union, List
import time

class Physics():
    def __init__(self,mag = 1,min_angle = -3.232,max_angle = 3.232):
        self.source_mag = mag
        self.min_angle = min_angle
        self.max_angle = max_angle
        
    #def sersic_fit(self,reconstructed_source):
         
    def image_to_source(self,image,centre = None,E_r = None,deflection=None, gradient=None):
        length, width = image.shape
        pixel_width = (self.max_angle-self.min_angle)/length
        if centre is None:
            centre = (length//2,width//2)
        centre_x = centre[0]
        centre_y = centre[1]

        range_indices_x = np.arange(-(centre_x-1),length-(centre_x-1))
        range_indices_y = np.arange(-(centre_y-1),width-(centre_y-1))

        x, y = np.meshgrid(range_indices_x, range_indices_y, indexing='ij')
        x,y = x*pixel_width,y*pixel_width
        
        r = np.sqrt(x**2 + y**2)
        mask = (r==0)
        r[mask] = 1

        if deflection is not None:
            if deflection.shape != (length, width):
                raise ValueError(f"The deflection should be of shape (2, {length}, {width}) but got {deflection.shape}")
            xdef = (deflection * x) / r
            ydef = (deflection * y) / r
        elif gradient is not None:
            if gradient.shape != image.shape:
                raise ValueError("The gradient and image should be of the same shape")
            #gradient = gradient*r
            xdef = np.gradient(gradient, axis=0)
            ydef = np.gradient(gradient, axis=1)
        elif E_r is not None:
            k = np.ones((length,width))*E_r
            xdef = (k * x) / r
            ydef = (k * y) / r
        else:
            raise ValueError("Both deflection and gradient cannot be None")

        bx = x - xdef
        by = y - ydef
        
        bx,by = bx/pixel_width,by/pixel_width
        
        bx = np.clip(bx + centre_x*self.source_mag, 0, length*self.source_mag - 1).astype(int)
        by = np.clip(by + centre_y*self.source_mag, 0, width*self.source_mag - 1).astype(int)
        
        sourceimage = np.zeros((length*self.source_mag,width*self.source_mag), dtype=float)
        counts = np.zeros_like(sourceimage, dtype=int)

        np.add.at(sourceimage, (bx, by), image)
        np.add.at(counts, (bx, by), 1)

        average_mask = counts > 0
        sourceimage[average_mask] /= counts[average_mask]
        
        return sourceimage