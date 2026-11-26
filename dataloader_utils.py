"""
Robust Dataloader Utilities for DeepLense.

Addresses Issue #145: Dataloader normalization edge case and tensor
conversion robustness. Provides safe tensor conversion, robust
normalization that handles edge cases (constant images, NaN/Inf values),
and a configurable data pipeline for PyTorch-compatible datasets.

Author: Kamala Hasini Burra
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def safe_to_tensor(
    data: Union[np.ndarray, list, float, int],
    dtype: str = "float32",
) -> np.ndarray:
    """Safely convert various input types to a numpy tensor.

    Handles common edge cases that cause PyTorch tensor conversion
    to fail, including mixed types, non-contiguous arrays, and
    object arrays.

    Parameters
    ----------
    data : array-like
        Input data to convert.
    dtype : str
        Target dtype string (e.g., 'float32', 'int64').

    Returns
    -------
    np.ndarray
        Contiguous array ready for tensor conversion.

    Examples
    --------
    >>> arr = safe_to_tensor([1, 2, 3])
    >>> print(arr.dtype, arr.flags['C_CONTIGUOUS'])
    float32 True
    """
    # Handle None
    if data is None:
        raise ValueError("Cannot convert None to tensor")

    # Convert to numpy array
    if isinstance(data, (int, float)):
        return np.array([data], dtype=dtype)

    if isinstance(data, list):
        try:
            arr = np.array(data, dtype=dtype)
        except (ValueError, TypeError):
            # Handle ragged lists by padding
            arr = _pad_ragged_list(data, dtype)
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.array(data, dtype=dtype)

    # Handle object arrays
    if arr.dtype == object:
        try:
            arr = np.array(arr.tolist(), dtype=dtype)
        except (ValueError, TypeError):
            arr = _pad_ragged_list(arr.tolist(), dtype)

    # Ensure correct dtype
    arr = arr.astype(dtype)

    # Ensure C-contiguous memory layout (required by PyTorch)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    # Replace NaN/Inf with safe values
    arr = _sanitize_values(arr)

    return arr


def _pad_ragged_list(
    data: list, dtype: str = "float32"
) -> np.ndarray:
    """Pad a ragged list to create a regular array.

    Parameters
    ----------
    data : list
        Potentially ragged nested list.
    dtype : str
        Target dtype.

    Returns
    -------
    np.ndarray
        Padded regular array.
    """
    if not data:
        return np.array([], dtype=dtype)

    # Find max dimensions
    if isinstance(data[0], (list, np.ndarray)):
        max_len = max(len(item) if hasattr(item, "__len__") else 1
                      for item in data)
        padded = []
        for item in data:
            if hasattr(item, "__len__"):
                row = list(item) + [0] * (max_len - len(item))
            else:
                row = [item] + [0] * (max_len - 1)
            padded.append(row)
        return np.array(padded, dtype=dtype)

    return np.array(data, dtype=dtype)


def _sanitize_values(arr: np.ndarray) -> np.ndarray:
    """Replace NaN and Inf values with safe defaults.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Sanitized array with NaN replaced by 0 and Inf by max finite.
    """
    if np.issubdtype(arr.dtype, np.floating):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))

        if nan_count > 0 or inf_count > 0:
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

    return arr


def robust_normalize(
    data: np.ndarray,
    method: str = "standard",
    epsilon: float = 1e-8,
    clip_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Normalize data with robust handling of edge cases.

    Unlike standard normalization, this handles:
    - Constant images (zero variance)
    - Single-pixel images
    - NaN/Inf values
    - Very large or very small value ranges

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize.
    method : str
        Normalization method:
        - 'standard': Zero mean, unit variance
        - 'minmax': Scale to [0, 1]
        - 'robust_scale': Median-centered, IQR-scaled
        - 'per_channel': Normalize each channel independently
    epsilon : float
        Small value to prevent division by zero.
    clip_range : tuple, optional
        (min, max) to clip values after normalization.

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    data = data.astype(np.float64)

    # Sanitize first
    data = _sanitize_values(data)

    if method == "standard":
        mean = np.mean(data)
        std = np.std(data)
        if std < epsilon:
            # Constant image: return zeros
            result = np.zeros_like(data)
        else:
            result = (data - mean) / (std + epsilon)

    elif method == "minmax":
        vmin = np.min(data)
        vmax = np.max(data)
        value_range = vmax - vmin
        if value_range < epsilon:
            result = np.zeros_like(data)
        else:
            result = (data - vmin) / (value_range + epsilon)

    elif method == "robust_scale":
        median = np.median(data)
        q75 = np.percentile(data, 75)
        q25 = np.percentile(data, 25)
        iqr = q75 - q25
        if iqr < epsilon:
            result = np.zeros_like(data)
        else:
            result = (data - median) / (iqr + epsilon)

    elif method == "per_channel":
        result = data.copy()
        if data.ndim == 3:
            # Assume last dim is channels
            for c in range(data.shape[-1]):
                channel = data[..., c]
                mean = np.mean(channel)
                std = np.std(channel)
                if std < epsilon:
                    result[..., c] = 0.0
                else:
                    result[..., c] = (channel - mean) / (std + epsilon)
        else:
            mean = np.mean(data)
            std = np.std(data)
            if std < epsilon:
                result = np.zeros_like(data)
            else:
                result = (data - mean) / (std + epsilon)
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    # Apply clipping if specified
    if clip_range is not None:
        result = np.clip(result, clip_range[0], clip_range[1])

    return result.astype(np.float32)


def create_data_pipeline(
    transforms: List[Callable],
) -> Callable:
    """Create a composable data preprocessing pipeline.

    Parameters
    ----------
    transforms : list of callable
        List of transform functions, each taking and returning np.ndarray.

    Returns
    -------
    callable
        A function that applies all transforms sequentially.

    Examples
    --------
    >>> pipeline = create_data_pipeline([
    ...     lambda x: x.astype(np.float32),
    ...     lambda x: robust_normalize(x, method='minmax'),
    ... ])
    >>> processed = pipeline(raw_image)
    """
    def apply_pipeline(data: np.ndarray) -> np.ndarray:
        result = data
        for transform in transforms:
            result = transform(result)
        return result

    return apply_pipeline


def validate_batch(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Validate a batch of images and labels for common issues.

    Parameters
    ----------
    images : np.ndarray
        Batch of images.
    labels : np.ndarray, optional
        Corresponding labels.
    expected_shape : tuple, optional
        Expected shape for each image (excluding batch dim).

    Returns
    -------
    dict
        Validation report with warnings and statistics.
    """
    report: Dict[str, Any] = {
        "valid": True,
        "warnings": [],
        "stats": {},
    }

    # Check for empty batch
    if images.size == 0:
        report["valid"] = False
        report["warnings"].append("Empty image batch")
        return report

    report["stats"]["batch_size"] = images.shape[0]
    report["stats"]["dtype"] = str(images.dtype)
    report["stats"]["shape"] = str(images.shape)

    # Check for NaN/Inf
    if np.issubdtype(images.dtype, np.floating):
        nan_count = int(np.sum(np.isnan(images)))
        inf_count = int(np.sum(np.isinf(images)))
        if nan_count > 0:
            report["warnings"].append(f"Found {nan_count} NaN values")
            report["valid"] = False
        if inf_count > 0:
            report["warnings"].append(f"Found {inf_count} Inf values")
            report["valid"] = False

    # Check value range
    report["stats"]["min"] = float(np.min(images))
    report["stats"]["max"] = float(np.max(images))
    report["stats"]["mean"] = float(np.mean(images))
    report["stats"]["std"] = float(np.std(images))

    # Check for constant images
    if report["stats"]["std"] < 1e-10:
        report["warnings"].append("Images have zero variance (constant)")

    # Check expected shape
    if expected_shape is not None:
        actual_shape = images.shape[1:]
        if actual_shape != expected_shape:
            report["warnings"].append(
                f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
            report["valid"] = False

    # Validate labels
    if labels is not None:
        if len(labels) != images.shape[0]:
            report["warnings"].append(
                f"Label count ({len(labels)}) != image count ({images.shape[0]})"
            )
            report["valid"] = False
        report["stats"]["num_classes"] = len(np.unique(labels))

    return report
