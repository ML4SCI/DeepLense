"""Lens-safe augmentations.

A view must be an image the telescope could have returned for the *same*
system. We therefore only use:

* a Gaussian blur with sigma up to ~1 native pixel (PSF variation), and
* low-amplitude Gaussian detector noise.

Crop, zoom and rescale are excluded because they change the apparent Einstein
radius, and ``M(<theta_E)`` is proportional to ``theta_E**2``. Translation is
excluded because both lensing priors assume a centred lens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.io import standardise


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur ``[B, C, H, W]`` images with an isotropic Gaussian kernel."""
    radius = max(1, int(round(3 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
    kernel = torch.exp(-0.5 * (axis / sigma).square())
    kernel = kernel / kernel.sum()
    kernel_2d = torch.outer(kernel, kernel).expand(images.shape[1], 1, -1, -1)
    return F.conv2d(images, kernel_2d, padding=radius, groups=images.shape[1])


def lens_safe_view(
    images: torch.Tensor,
    blur_probability: float = 0.5,
    blur_sigma_max: float = 1.0,
    noise_std_max: float = 0.03,
) -> torch.Tensor:
    """Randomly blur and add noise, then standardise each image.

    One blur sigma and one noise level are drawn per call (i.e. per batch).
    Input images are expected in ``[0, 1]``.
    """
    view = images
    if blur_probability > 0 and torch.rand(()).item() < blur_probability:
        sigma = torch.rand(()).item() * blur_sigma_max
        if sigma > 1e-4:
            view = gaussian_blur(view, sigma)
    if noise_std_max > 0:
        view = view + torch.randn_like(view) * (torch.rand(()).item() * noise_std_max)
    return standardise(view)
