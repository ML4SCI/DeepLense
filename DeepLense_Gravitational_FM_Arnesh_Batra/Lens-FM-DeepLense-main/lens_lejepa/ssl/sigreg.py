"""SIGReg: sketched isotropic Gaussian regularisation (Balestriero & LeCun, 2025).

The embeddings are projected onto random unit directions ("slices"). For each
slice an Epps-Pulley statistic compares the empirical characteristic function
with that of a standard normal. Keeping every 1-D projection Gaussian keeps
the whole embedding distribution close to an isotropic Gaussian, which is what
stops the teacher-free objective from collapsing.

Follows the reference implementation at github.com/galilai-group/lejepa
(single-process version).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EppsPulley(nn.Module):
    """Univariate Epps-Pulley test statistic against N(0, 1)."""

    def __init__(self, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        if n_points % 2 != 1:
            raise ValueError("n_points must be odd (trapezoidal rule).")
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())  # characteristic function of N(0, 1)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[N, S]`` projected samples; returns one statistic per slice."""
        x_t = x.unsqueeze(-1) * self.t
        error = (torch.cos(x_t).mean(0) - self.phi).square() + torch.sin(x_t).mean(0).square()
        return (error @ self.weights) * x.shape[0]


class SIGReg(nn.Module):
    """Mean Epps-Pulley statistic over ``num_slices`` random directions.

    The directions are re-drawn every call from a generator seeded with an
    internal step counter, so training is reproducible and resumable.
    """

    def __init__(self, num_slices: int = 256, n_points: int = 17) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.test = EppsPulley(n_points=n_points)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        flat = embeddings.reshape(-1, embeddings.shape[-1]).float()
        with torch.no_grad():
            generator = torch.Generator(device=flat.device).manual_seed(int(self.step.item()))
            directions = torch.randn(flat.shape[-1], self.num_slices, device=flat.device, generator=generator)
            directions = F.normalize(directions, dim=0)
            self.step.add_(1)
        return self.test(flat @ directions).mean()
