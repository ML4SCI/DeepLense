"""
Butler Repository Management

This package provides functionality for validating, discovering, creating, and accessing
LSST Butler repositories for the RIPPLe pipeline.
"""

from .creator import (
    ButlerRepoCreator,
    DataDiscoveryResult,
    RepositoryCreationResult,
    create_butler_repo_from_data,
    create_butler_repo_from_dc2_data,
    create_butler_repo_from_directory_tree,
)
from .validator import (
    ButlerRepoValidator,
    CoverageReport,
    DataProductInfo,
    ValidationResult,
)

__all__ = [
    # Creation functionality
    "ButlerRepoCreator",
    # Validation functionality
    "ButlerRepoValidator",
    "CoverageReport",
    "DataDiscoveryResult",
    "DataProductInfo",
    "RepositoryCreationResult",
    "ValidationResult",
    "create_butler_repo_from_data",
    "create_butler_repo_from_dc2_data",
    "create_butler_repo_from_directory_tree",
]
