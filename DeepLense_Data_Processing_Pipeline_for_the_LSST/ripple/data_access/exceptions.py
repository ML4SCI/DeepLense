"""
Custom exceptions for RIPPLe data access layer.

This module defines the exception hierarchy for data access operations,
providing specific error types for different failure scenarios with
appropriate recovery strategies.
"""

from typing import Any


class RippleError(Exception):
    """Base exception for all RIPPLe-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataAccessError(RippleError):
    """Base class for data access related errors."""


class ButlerConnectionError(DataAccessError):
    """Raised when Butler connection fails."""

    def __init__(
        self,
        message: str,
        repo_path: str | None = None,
        server_url: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.repo_path = repo_path
        self.server_url = server_url


class DataIdValidationError(DataAccessError):
    """Raised when DataId validation fails."""

    def __init__(self, message: str, data_id: dict[str, Any] | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.data_id = data_id


class CutoutExtractionError(DataAccessError):
    """Raised when cutout extraction fails."""

    def __init__(
        self,
        message: str,
        ra: float | None = None,
        dec: float | None = None,
        size: int | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.ra = ra
        self.dec = dec
        self.size = size


class CollectionError(DataAccessError):
    """Raised when collection operations fail."""

    def __init__(self, message: str, collection_name: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.collection_name = collection_name


class CoordinateConversionError(DataAccessError):
    """Raised when coordinate conversion fails."""

    def __init__(
        self,
        message: str,
        from_coords: tuple | None = None,
        to_coords: tuple | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.from_coords = from_coords
        self.to_coords = to_coords


class CacheError(DataAccessError):
    """Raised when cache operations fail."""


class PerformanceError(DataAccessError):
    """Raised when performance thresholds are exceeded."""

    def __init__(
        self,
        message: str,
        metric: str | None = None,
        threshold: float | None = None,
        measured: float | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.metric = metric
        self.threshold = threshold
        self.measured = measured
