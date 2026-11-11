"""
Data Access Layer for RIPPLe

This module provides the core data access functionality for LSST data retrieval
using Butler Gen3 architecture with optimized performance and comprehensive
error handling.
"""

from .butler_client import ButlerClient
from .coordinate_utils import CoordinateConverter
from .data_fetcher import ButlerConfig, LsstDataFetcher
from .exceptions import (
    ButlerConnectionError,
    CutoutExtractionError,
    DataAccessError,
    DataIdValidationError,
)

__all__ = [
    "ButlerClient",
    "ButlerConfig",
    "ButlerConnectionError",
    "CoordinateConverter",
    "CutoutExtractionError",
    "DataAccessError",
    "DataIdValidationError",
    "LsstDataFetcher",
]
