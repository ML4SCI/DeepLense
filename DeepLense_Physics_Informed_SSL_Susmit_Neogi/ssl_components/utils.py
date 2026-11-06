import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

class Physics:
    def __init__(self):
        pass

    @staticmethod
    def potential_gradient(potential):
        sobel_x = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]],
        device=potential.device,
        dtype=potential.dtype
        ).view(1,1,3,3) / 8.

        sobel_y = torch.tensor(
            [[-1.,-2.,-1.],
            [ 0., 0., 0.],
            [ 1., 2., 1.]],
            device=potential.device,
            dtype=potential.dtype
        ).view(1,1,3,3) / 8.

        alpha_x = F.conv2d(potential, sobel_x, padding=1)
        alpha_y = F.conv2d(potential, sobel_y, padding=1)

        # alpha = torch.cat([alpha_x, alpha_y], dim = 1)
        # alpha = alpha.permute(0,2,3,1)

        return alpha_x, alpha_y
    def laplacian_2d(self, field):
        kernel = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]],
        device=field.device,
        dtype=field.dtype
        ).view(1,1,3,3) / 8.
        return F.conv2d(field, kernel, padding=1)
    @staticmethod
    def lens_prediction(source, alpha):
        B, C, H, W = source.shape

        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,H,device=source.device),
            torch.linspace(-1,1,W,device=source.device),
            indexing="ij"
        )
        grid = torch.stack((xx,yy), dim=-1)              
        grid = grid.unsqueeze(0).repeat(B,1,1,1)
        alpha_grid = alpha.clone()
        alpha_grid[..., 0] = alpha_grid[..., 0] * (2.0 / max(W - 1, 1))
        alpha_grid[..., 1] = alpha_grid[..., 1] * (2.0 / max(H - 1, 1))
        beta = grid - alpha_grid
        pred_lens = F.grid_sample(
            source,
            beta,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        return pred_lens
    
    def poisson_loss(self, psi_pred, kappa_pred):
        laplacian_psi = self.laplacian_2d(psi_pred)
        return F.mse_loss(laplacian_psi, 2.0 * kappa_pred)

    def deflection_consistency_loss(self, psi_pred):
        alpha_x, alpha_y = self.potential_gradient(psi_pred)
        d_alpha_x_dy = alpha_x[:, :, 1:, :] - alpha_x[:, :, :-1, :]
        d_alpha_y_dx = alpha_y[:, :, :, 1:] - alpha_y[:, :, :, :-1]
        min_h = min(d_alpha_x_dy.shape[2], d_alpha_y_dx.shape[2])
        min_w = min(d_alpha_x_dy.shape[3], d_alpha_y_dx.shape[3])
        curl = d_alpha_x_dy[:, :, :min_h, :min_w] - d_alpha_y_dx[:, :, :min_h, :min_w]
        return (curl**2).mean()
    
    def substructure_sparsity_loss(kappa_pred, kappa_smooth):
        residual = kappa_pred - kappa_smooth
        return torch.abs(residual).mean()
    
    