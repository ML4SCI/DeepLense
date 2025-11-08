"""
Custom exceptions for RIPPLe data access layer.

This module defines the exception hierarchy for data access operations,
providing specific error types for different failure scenarios with
appropriate recovery strategies.
"""

from typing import Optional, Dict, Any


class RippleError(Exception):
    """Base exception for all RIPPLe-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataAccessError(RippleError):
    """Base class for data access related errors."""
    pass


class ButlerConnectionError(DataAccessError):
    """Raised when Butler connection fails."""
    
    def __init__(self, message: str, repo_path: Optional[str] = None, 
                 server_url: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.repo_path = repo_path
        self.server_url = server_url


class DataIdValidationError(DataAccessError):
    """Raised when DataId validation fails."""
    
    def __init__(self, message: str, data_id: Optional[Dict[str, Any]] = None, 
                 **kwargs):
        super().__init__(message, **kwargs)
        self.data_id = data_id


class CutoutExtractionError(DataAccessError):
    """Raised when cutout extraction fails."""
    
    def __init__(self, message: str, ra: Optional[float] = None, 
                 dec: Optional[float] = None, size: Optional[int] = None, 
                 **kwargs):
        super().__init__(message, **kwargs)
        self.ra = ra
        self.dec = dec
        self.size = size


class CollectionError(DataAccessError):
    """Raised when collection operations fail."""
    
    def __init__(self, message: str, collection_name: Optional[str] = None, 
                 **kwargs):
        super().__init__(message, **kwargs)
        self.collection_name = collection_name


class CoordinateConversionError(DataAccessError):
    """Raised when coordinate conversion fails."""
    
    def __init__(self, message: str, from_coords: Optional[tuple] = None, 
                 to_coords: Optional[tuple] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.from_coords = from_coords
        self.to_coords = to_coords


class CacheError(DataAccessError):
    """Raised when cache operations fail."""
    pass


class PerformanceError(DataAccessError):
    """Raised when performance thresholds are exceeded."""
    
    def __init__(self, message: str, metric: Optional[str] = None, 
                 threshold: Optional[float] = None, 
                 measured: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.metric = metric
        self.threshold = threshold
        self.measured = measured