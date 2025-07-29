"""
Butler Repository Management

This package provides functionality for validating, discovering, creating, and accessing
LSST Butler repositories for the RIPPLe pipeline.
"""

from .validator import (
    ButlerRepoValidator,
    ValidationResult,
    DataProductInfo,
    CoverageReport
)

from .creator import (
    ButlerRepoCreator,
    DataDiscoveryResult,
    RepositoryCreationResult,
    create_butler_repo_from_data,
    create_butler_repo_from_directory_tree,
    create_butler_repo_from_dc2_data
)

__all__ = [
    # Validation functionality
    'ButlerRepoValidator',
    'ValidationResult', 
    'DataProductInfo',
    'CoverageReport',
    
    # Creation functionality
    'ButlerRepoCreator',
    'DataDiscoveryResult',
    'RepositoryCreationResult',
    'create_butler_repo_from_data',
    'create_butler_repo_from_directory_tree',
    'create_butler_repo_from_dc2_data'
]