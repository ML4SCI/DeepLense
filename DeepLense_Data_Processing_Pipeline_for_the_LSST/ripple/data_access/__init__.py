"""
Data Access Layer for RIPPLe

This module provides the core data access functionality for LSST data retrieval
using Butler Gen3 architecture with optimized performance and comprehensive 
error handling.
"""

from .data_fetcher import LsstDataFetcher, ButlerConfig
from .butler_client import ButlerClient
from .coordinate_utils import CoordinateConverter
from .exceptions import (
    DataAccessError,
    ButlerConnectionError,
    DataIdValidationError,
    CutoutExtractionError,
)

__all__ = [
    "LsstDataFetcher",
    "ButlerConfig",
    "ButlerClient",
    "CoordinateConverter",
    "DataAccessError",
    "ButlerConnectionError", 
    "DataIdValidationError",
    "CutoutExtractionError",
]