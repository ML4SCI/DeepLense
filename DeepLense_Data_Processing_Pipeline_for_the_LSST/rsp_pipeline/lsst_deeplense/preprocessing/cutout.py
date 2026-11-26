"""
cutout.py
---------
Extract fixed-size cutouts from LSST calibrated images and resize them
to the dimensions expected by DeepLense models (default: 64 × 64 px).

LSST native pixel scale is ~0.2 arcsec / px, so a 10 arcsec cutout
corresponds to 50 × 50 pixels.  This module handles the resize to the
target model resolution and optional multi-band stacking.

Usage
-----
>>> from lsst_deeplense.preprocessing import CutoutExtractor
>>> extractor = CutoutExtractor(output_size=64)
>>> # single-band (H, W) array → (1, 64, 64) tensor-ready array
>>> patch = extractor.extract(image_array, centre_row=256, centre_col=256)
>>> # multi-band dict → (C, 64, 64)
>>> stack = extractor.stack_bands({"g": arr_g, "r": arr_r, "i": arr_i})
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class CutoutExtractor:
    """
    Extract and resize image cutouts to a standard square size.

    Parameters
    ----------
    output_size : int
        Side length of the output square patch (default 64, matching
        DeepLense Model I/II/III convention).
    half_size : int, optional
        Half-side of the raw cutout extracted before resize.  If ``None``
        the entire input array is used.
    resize_method : {'bilinear', 'bicubic', 'nearest'}
        Interpolation method for resizing.
    """

    def __init__(
        self,
        output_size: int = 64,
        half_size: Optional[int] = None,
        resize_method: str = "bilinear",
    ) -> None:
        self.output_size = output_size
        self.half_size = half_size
        self.resize_method = resize_method

    # ------------------------------------------------------------------

    def extract(
        self,
        image: np.ndarray,
        centre_row: Optional[int] = None,
        centre_col: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract a cutout centred at ``(centre_row, centre_col)`` and resize
        to ``(1, output_size, output_size)``.

        Parameters
        ----------
        image : np.ndarray
            Input array (H, W) or (C, H, W).
        centre_row, centre_col : int, optional
            Pixel coordinates of the target object.  Defaults to the image
            centre.

        Returns
        -------
        np.ndarray  float32  shape (1, output_size, output_size)
        """
        if image.ndim == 3:
            # Average channels if multi-band input passed here
            img2d = image.mean(axis=0)
        else:
            img2d = image

        h, w = img2d.shape
        cr = centre_row if centre_row is not None else h // 2
        cc = centre_col if centre_col is not None else w // 2

        if self.half_size is not None:
            r0 = max(0, cr - self.half_size)
            r1 = min(h, cr + self.half_size)
            c0 = max(0, cc - self.half_size)
            c1 = min(w, cc + self.half_size)
            img2d = img2d[r0:r1, c0:c1]

        resized = self._resize(img2d)
        return resized[np.newaxis].astype(np.float32)  # (1, H, W)

    def stack_bands(
        self,
        band_arrays: Dict[str, np.ndarray],
        band_order: Optional[Tuple[str, ...]] = None,
    ) -> np.ndarray:
        """
        Stack multiple single-band arrays into a (C, H, W) array.

        Parameters
        ----------
        band_arrays : dict[str, np.ndarray]
            Mapping of band name → (H, W) numpy array.
        band_order : tuple[str, ...], optional
            Explicit ordering of bands in the output tensor.  If ``None``
            the dict insertion order is used.

        Returns
        -------
        np.ndarray  float32  shape (C, output_size, output_size)
        """
        order = band_order if band_order is not None else tuple(band_arrays)
        missing = [b for b in order if b not in band_arrays]
        if missing:
            raise KeyError(f"Requested bands not available: {missing}")

        patches = []
        for band in order:
            patch = self._resize(band_arrays[band])
            patches.append(patch)

        return np.stack(patches, axis=0).astype(np.float32)  # (C, H, W)

    # ------------------------------------------------------------------
    # Internal resize
    # ------------------------------------------------------------------

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize a (H, W) array to (output_size, output_size)."""
        size = self.output_size
        if img.shape == (size, size):
            return img.astype(np.float32)

        if _CV2_AVAILABLE:
            interp_map = {
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "nearest": cv2.INTER_NEAREST,
            }
            interp = interp_map.get(self.resize_method, cv2.INTER_LINEAR)
            return cv2.resize(
                img.astype(np.float32), (size, size), interpolation=interp
            )

        if _PIL_AVAILABLE:
            resample_map = {
                "bilinear": PILImage.BILINEAR,
                "bicubic": PILImage.BICUBIC,
                "nearest": PILImage.NEAREST,
            }
            resample = resample_map.get(self.resize_method, PILImage.BILINEAR)
            pil_img = PILImage.fromarray(img.astype(np.float32))
            pil_img = pil_img.resize((size, size), resample=resample)
            return np.array(pil_img, dtype=np.float32)

        # Fallback: nearest-neighbour via index arithmetic
        row_idx = (np.arange(size) * img.shape[0] / size).astype(int)
        col_idx = (np.arange(size) * img.shape[1] / size).astype(int)
        return img[np.ix_(row_idx, col_idx)].astype(np.float32)
