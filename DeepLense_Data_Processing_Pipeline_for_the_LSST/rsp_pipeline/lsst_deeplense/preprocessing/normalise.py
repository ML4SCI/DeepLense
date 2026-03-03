"""
normalise.py
------------
Normalisation strategies for LSST / Rubin calibrated images before
they are fed into DeepLense models.

Three strategies are supported:

``'minmax'``
    Rescale pixel values to [0, 1] per image.

``'zscore'``
    Zero-mean, unit-variance normalisation per image.

``'asinh'``
    Inverse hyperbolic-sine stretch commonly used in astronomy to
    compress the dynamic range of images with bright nuclei and faint
    arcs simultaneously.  Scale parameter ``a`` controls the linear
    regime (default 0.1 in units of the image's 3-sigma sky level).

Usage
-----
>>> norm = Normaliser(strategy="asinh", asinh_a=0.1)
>>> img_norm = norm(img)          # np.ndarray (H, W) or (C, H, W)
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


_VALID_STRATEGIES = ("minmax", "zscore", "asinh")


class Normaliser:
    """
    Apply a pixel-level normalisation to a single-channel or multi-channel
    lensing image.

    Parameters
    ----------
    strategy : {'minmax', 'zscore', 'asinh'}
        Normalisation method (default ``'asinh'`` — recommended for LSST
        data where lens arcs are faint relative to the lens galaxy).
    asinh_a : float
        Softening parameter for the asinh stretch.  Smaller values
        preserve more contrast in faint features.  Only used when
        ``strategy='asinh'``.
    clip_sigma : float, optional
        If given, pixel values beyond ``clip_sigma`` standard deviations
        from the median are clipped *before* normalisation.  Useful for
        cosmic-ray rejection.
    eps : float
        Small constant added to denominators to avoid division by zero.
    """

    def __init__(
        self,
        strategy: Literal["minmax", "zscore", "asinh"] = "asinh",
        asinh_a: float = 0.1,
        clip_sigma: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'"
            )
        self.strategy = strategy
        self.asinh_a = asinh_a
        self.clip_sigma = clip_sigma
        self.eps = eps

    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise ``image`` and return a float32 array of the same shape.

        Parameters
        ----------
        image : np.ndarray
            Shape (H, W) or (C, H, W).  Any finite float dtype is accepted.

        Returns
        -------
        np.ndarray  float32
        """
        img = image.astype(np.float32)

        if self.clip_sigma is not None:
            img = self._sigma_clip(img)

        if self.strategy == "minmax":
            return self._minmax(img)
        elif self.strategy == "zscore":
            return self._zscore(img)
        else:  # asinh
            return self._asinh(img)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _minmax(self, img: np.ndarray) -> np.ndarray:
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo + self.eps)

    def _zscore(self, img: np.ndarray) -> np.ndarray:
        mu = img.mean()
        sigma = img.std()
        return (img - mu) / (sigma + self.eps)

    def _asinh(self, img: np.ndarray) -> np.ndarray:
        # Estimate the background level as the median of the lower half of
        # pixel values (robust against bright lens galaxy contribution).
        flat = img.ravel()
        background = np.median(flat[flat < np.median(flat)])
        sky_sigma = 1.4826 * np.median(np.abs(flat - background))  # MAD estimator

        # Shift so background ~ 0
        shifted = img - background
        scale = self.asinh_a * sky_sigma + self.eps

        stretched = np.arcsinh(shifted / scale)
        # Rescale to [0, 1]
        return self._minmax(stretched)

    def _sigma_clip(self, img: np.ndarray) -> np.ndarray:
        mu, sigma = img.mean(), img.std()
        lo = mu - self.clip_sigma * sigma
        hi = mu + self.clip_sigma * sigma
        return np.clip(img, lo, hi)
