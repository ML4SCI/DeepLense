"""
Robust Dataset Loader for DeepLense.

Addresses Issue #178: Add robust dataset loader utility for inconsistent
.npy formats. This module provides a flexible dataset loader that handles
different .npy file formats, shapes, and dtypes commonly found across the
various DeepLense sub-projects.

Author: Kamala Hasini Burra
"""

import os
import glob
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


class DatasetLoader:
    """Robust loader for .npy datasets with format validation.

    Handles common inconsistencies found in DeepLense sub-projects:
    - Mixed dtypes (float32, float64, uint8, int)
    - Inconsistent shapes (H,W), (H,W,C), (C,H,W)
    - Missing or corrupted files
    - Different normalization ranges ([0,1], [0,255], [-1,1])

    Parameters
    ----------
    data_dir : str
        Root directory containing .npy files.
    target_shape : tuple, optional
        Target image shape (H, W). Set to None to keep original shape.
    target_dtype : np.dtype, optional
        Target dtype. Default is np.float32.
    channel_format : str, optional
        'channels_last' (H,W,C) or 'channels_first' (C,H,W).
        Default is 'channels_last'.

    Examples
    --------
    >>> loader = DatasetLoader("path/to/data", target_shape=(64, 64))
    >>> images, labels = loader.load_dataset()
    >>> print(images.shape, images.dtype)
    """

    def __init__(
        self,
        data_dir: str,
        target_shape: Optional[Tuple[int, int]] = None,
        target_dtype: np.dtype = np.float32,
        channel_format: str = "channels_last",
    ):
        self.data_dir = data_dir
        self.target_shape = target_shape
        self.target_dtype = target_dtype
        self.channel_format = channel_format

        if channel_format not in ("channels_last", "channels_first"):
            raise ValueError(
                f"channel_format must be 'channels_last' or 'channels_first', "
                f"got '{channel_format}'"
            )

    def load_npy_safe(self, filepath: str) -> Optional[np.ndarray]:
        """Safely load a .npy file, handling common errors.

        Parameters
        ----------
        filepath : str
            Path to the .npy file.

        Returns
        -------
        np.ndarray or None
            Loaded array, or None if loading failed.
        """
        try:
            data = np.load(filepath, allow_pickle=True)

            # Handle object arrays (sometimes created with allow_pickle)
            if data.dtype == object:
                try:
                    data = np.array(data.tolist(), dtype=self.target_dtype)
                except (ValueError, TypeError):
                    warnings.warn(
                        f"Could not convert object array in {filepath}. Skipping."
                    )
                    return None

            return data.astype(self.target_dtype)

        except Exception as e:
            warnings.warn(f"Failed to load {filepath}: {e}")
            return None

    def validate_image_shape(
        self, image: np.ndarray
    ) -> Optional[np.ndarray]:
        """Validate and standardize image shape.

        Parameters
        ----------
        image : np.ndarray
            Input image array.

        Returns
        -------
        np.ndarray or None
            Standardized image, or None if shape is invalid.
        """
        ndim = image.ndim

        if ndim == 2:
            # Grayscale (H, W) -> (H, W, 1)
            if self.channel_format == "channels_last":
                image = image[:, :, np.newaxis]
            else:
                image = image[np.newaxis, :, :]

        elif ndim == 3:
            # Check if channels_first: (C, H, W) where C is small
            if image.shape[0] <= 4 and image.shape[1] > 4 and image.shape[2] > 4:
                if self.channel_format == "channels_last":
                    image = np.transpose(image, (1, 2, 0))
            elif image.shape[2] <= 4 and image.shape[0] > 4 and image.shape[1] > 4:
                if self.channel_format == "channels_first":
                    image = np.transpose(image, (2, 0, 1))
        else:
            warnings.warn(f"Unexpected image ndim={ndim}, shape={image.shape}")
            return None

        # Resize if target shape specified
        if self.target_shape is not None:
            image = self._resize_image(image)

        return image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image using simple nearest-neighbor interpolation.

        Uses numpy only (no PIL/cv2 dependency required).

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Resized image.
        """
        target_h, target_w = self.target_shape

        if self.channel_format == "channels_last":
            h, w = image.shape[0], image.shape[1]
            channels = image.shape[2] if image.ndim == 3 else 1
        else:
            h, w = image.shape[1], image.shape[2]
            channels = image.shape[0] if image.ndim == 3 else 1

        if h == target_h and w == target_w:
            return image

        # Nearest-neighbor resize
        row_indices = (np.arange(target_h) * h / target_h).astype(int)
        col_indices = (np.arange(target_w) * w / target_w).astype(int)
        row_indices = np.clip(row_indices, 0, h - 1)
        col_indices = np.clip(col_indices, 0, w - 1)

        if self.channel_format == "channels_last":
            return image[np.ix_(row_indices, col_indices)]
        else:
            return image[:, np.ix_(row_indices, col_indices)].reshape(
                channels, target_h, target_w
            )

    def detect_normalization(
        self, images: np.ndarray
    ) -> str:
        """Detect the normalization range of a batch of images.

        Parameters
        ----------
        images : np.ndarray
            Batch of images.

        Returns
        -------
        str
            One of 'uint8' ([0,255]), 'normalized' ([0,1]),
            'centered' ([-1,1]), or 'unknown'.
        """
        vmin, vmax = float(images.min()), float(images.max())

        if vmin >= 0 and vmax > 1 and vmax <= 255:
            return "uint8"
        elif vmin >= 0 and vmax <= 1.0:
            return "normalized"
        elif vmin >= -1.0 and vmax <= 1.0:
            return "centered"
        else:
            return "unknown"

    def normalize_to_range(
        self,
        images: np.ndarray,
        target_range: str = "normalized",
    ) -> np.ndarray:
        """Normalize images to a consistent range.

        Parameters
        ----------
        images : np.ndarray
            Input images.
        target_range : str
            Target range: 'normalized' ([0,1]) or 'centered' ([-1,1]).

        Returns
        -------
        np.ndarray
            Normalized images.
        """
        detected = self.detect_normalization(images)

        if detected == target_range:
            return images

        # First normalize to [0, 1]
        if detected == "uint8":
            images = images / 255.0
        elif detected == "centered":
            images = (images + 1.0) / 2.0
        elif detected == "unknown":
            vmin, vmax = images.min(), images.max()
            if vmax - vmin > 0:
                images = (images - vmin) / (vmax - vmin)

        # Convert to target
        if target_range == "centered":
            images = images * 2.0 - 1.0

        return images.astype(self.target_dtype)

    def load_dataset(
        self,
        image_pattern: str = "*.npy",
        label_file: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load a complete dataset from directory.

        Parameters
        ----------
        image_pattern : str
            Glob pattern for image files.
        label_file : str, optional
            Path to labels .npy file.

        Returns
        -------
        tuple
            (images, labels) where labels may be None.
        """
        pattern = os.path.join(self.data_dir, image_pattern)
        files = sorted(glob.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matching '{image_pattern}' in {self.data_dir}"
            )

        images = []
        skipped = 0

        for f in files:
            img = self.load_npy_safe(f)
            if img is None:
                skipped += 1
                continue

            # Handle batch files (multiple images in one .npy)
            if img.ndim >= 3 and img.shape[0] > 4:
                # Likely a batch of images
                for j in range(img.shape[0]):
                    validated = self.validate_image_shape(img[j])
                    if validated is not None:
                        images.append(validated)
            else:
                validated = self.validate_image_shape(img)
                if validated is not None:
                    images.append(validated)

        if skipped > 0:
            warnings.warn(f"Skipped {skipped}/{len(files)} files due to errors.")

        if not images:
            raise ValueError("No valid images loaded from the dataset.")

        images_array = np.array(images, dtype=self.target_dtype)

        # Load labels if provided
        labels = None
        if label_file:
            label_path = os.path.join(self.data_dir, label_file)
            labels = self.load_npy_safe(label_path)

        return images_array, labels

    def get_dataset_info(self) -> Dict[str, Union[str, int]]:
        """Get summary information about the dataset directory.

        Returns
        -------
        dict
            Dictionary with dataset statistics.
        """
        npy_files = glob.glob(os.path.join(self.data_dir, "*.npy"))
        total_size = sum(os.path.getsize(f) for f in npy_files)

        info = {
            "data_dir": self.data_dir,
            "num_npy_files": len(npy_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

        # Sample first file for shape/dtype info
        if npy_files:
            sample = self.load_npy_safe(npy_files[0])
            if sample is not None:
                info["sample_shape"] = str(sample.shape)
                info["sample_dtype"] = str(sample.dtype)
                info["normalization"] = self.detect_normalization(sample)

        return info
