"""
Configuration examples and validation for RIPPLe data access.

This module provides example configurations and validation utilities
for different deployment scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def get_default_config() -> 'ButlerConfig':
    """Get default configuration for local development."""
    from .data_fetcher import ButlerConfig
    
    return ButlerConfig(
        repo_path="/path/to/local/repo",
        collections=["2.2i/runs/DP0.2"],
        instrument="LSSTCam-imSim",
        cache_size=1000,
        enable_performance_monitoring=True,
        timeout=30.0,
        retry_attempts=3
    )


def get_production_config() -> 'ButlerConfig':
    """Get configuration optimized for production deployment."""
    from .data_fetcher import ButlerConfig
    
    return ButlerConfig(
        server_url="https://butler.lsst.org",
        collections=["2.2i/runs/DP0.2", "HSC/runs/RC2"],
        instrument="LSSTCam-imSim",
        cache_size=5000,
        enable_performance_monitoring=True,
        max_connections=20,
        timeout=60.0,
        retry_attempts=5,
        retry_delay=2.0,
        batch_size=64,
        max_workers=8
    )


def get_testing_config() -> 'ButlerConfig':
    """Get configuration for testing with MockButler."""
    from .data_fetcher import ButlerConfig
    
    return ButlerConfig(
        repo_path="/tmp/test_repo",
        collections=["test/mock_collection"],
        instrument="TestCam",
        cache_size=100,
        enable_performance_monitoring=False,
        timeout=10.0,
        retry_attempts=1
    )


def validate_config(config: 'ButlerConfig') -> Dict[str, Any]:
    """
    Validate configuration settings.
    
    Parameters
    ----------
    config : ButlerConfig
        Configuration to validate
        
    Returns
    -------
    Dict[str, Any]
        Validation results with errors and warnings
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate connection settings
    if not config.repo_path and not config.server_url:
        results['errors'].append("Must specify either repo_path or server_url")
        results['valid'] = False
    
    if config.repo_path and config.server_url:
        results['warnings'].append("Both repo_path and server_url specified, will use server_url")
    
    # Validate collections
    if not config.collections:
        results['errors'].append("Must specify at least one collection")
        results['valid'] = False
    
    # Validate performance settings
    if config.cache_size < 0:
        results['errors'].append("Cache size must be non-negative")
        results['valid'] = False
    
    if config.cache_size > 10000:
        results['warnings'].append("Large cache size may consume significant memory")
    
    if config.timeout <= 0:
        results['errors'].append("Timeout must be positive")
        results['valid'] = False
    
    if config.retry_attempts < 1:
        results['errors'].append("Retry attempts must be at least 1")
        results['valid'] = False
    
    if config.max_workers < 1:
        results['errors'].append("Max workers must be at least 1")
        results['valid'] = False
    
    if config.batch_size < 1:
        results['errors'].append("Batch size must be at least 1")
        results['valid'] = False
    
    # Performance warnings
    if config.batch_size > 128:
        results['warnings'].append("Large batch size may cause memory issues")
    
    if config.max_workers > 16:
        results['warnings'].append("High worker count may cause resource contention")
    
    return results


# Example usage patterns
EXAMPLE_USAGE = """
# Basic usage with local repository
from ripple.data_access import LsstDataFetcher, ButlerConfig

config = ButlerConfig(
    repo_path="/path/to/repo",
    collections=["2.2i/runs/DP0.2"],
    cache_size=1000
)

with LsstDataFetcher(config) as fetcher:
    # Single cutout
    cutout = fetcher.fetch_cutout(ra=150.0, dec=2.5, size=64)
    
    # Multi-band cutout
    multiband = fetcher.fetch_cutout(
        ra=150.0, dec=2.5, size=64, 
        filters=['g', 'r', 'i']
    )
    
    # Batch processing
    requests = [
        CutoutRequest(ra=150.0, dec=2.5, size=64),
        CutoutRequest(ra=151.0, dec=2.6, size=64),
        CutoutRequest(ra=152.0, dec=2.7, size=64)
    ]
    
    results = fetcher.fetch_batch_cutouts(requests)
    
    # Check available data
    availability = fetcher.get_available_data(ra=150.0, dec=2.5)
    
    # Performance monitoring
    metrics = fetcher.get_performance_metrics()
    cache_stats = fetcher.get_cache_statistics()

# Client/server usage
config = ButlerConfig(
    server_url="https://butler.lsst.org",
    collections=["2.2i/runs/DP0.2"],
    timeout=60.0,
    retry_attempts=5
)

with LsstDataFetcher(config) as fetcher:
    # Same API, different backend
    cutout = fetcher.fetch_cutout(ra=150.0, dec=2.5, size=64)
"""