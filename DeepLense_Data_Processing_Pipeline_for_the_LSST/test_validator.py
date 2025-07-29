#!/usr/bin/env python3
"""
Test script for ButlerRepoValidator using the existing demo_data repository.
"""

import logging
from ripple.butler import ButlerRepoValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_validator_for_repo(repo_path: str, repo_name: str):
    """Test the ButlerRepoValidator with a specific repository."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {repo_name}")
    logger.info(f"Repository path: {repo_path}")
    logger.info(f"{'='*60}")
    
    # Initialize validator
    validator = ButlerRepoValidator(repo_path)
    
    # Test basic validation
    logger.info("\n=== Repository Validation ===")
    result = validator.validate_repository()
    
    if result.is_valid:
        logger.info(f"✓ Repository is valid: {result.repo_path}")
        logger.info(f"✓ Found {result.total_collections} collections")
        logger.info(f"✓ Found {result.total_dataset_types} dataset types")
        logger.info(f"✓ Instruments: {result.instruments}")
        
        logger.info(f"\nCollections: {result.collections}")
        logger.info(f"Dataset types (first 10): {result.dataset_types[:10]}")
        
        # Test data products discovery
        logger.info("\n=== Data Products Analysis ===")
        for dataset_type, info in result.data_products.items():
            logger.info(f"\n{dataset_type}:")
            logger.info(f"  Available: {info.available_count}")
            logger.info(f"  Coverage: {info.coverage_percentage:.1f}%")
            logger.info(f"  Collections: {info.collections}")
            logger.info(f"  Dimensions: {info.dimensions}")
            if info.sample_data_ids:
                logger.info(f"  Sample data ID: {info.sample_data_ids[0]}")
        
        # Test coverage report
        logger.info("\n=== Coverage Report ===")
        coverage = validator.get_data_coverage(
            dataset_types=["calexp", "src"],
            filters=["r"]
        )
        
        logger.info(f"Total data IDs: {coverage.total_data_ids}")
        logger.info(f"Available: {coverage.available_data_ids}")
        logger.info(f"Missing: {coverage.missing_data_ids}")
        logger.info(f"Coverage: {coverage.coverage_percentage:.1f}%")
        logger.info(f"Instruments: {coverage.instruments}")
        logger.info(f"Filters: {coverage.filters}")
        
        if coverage.failed_data_ids:
            logger.info(f"Failed data IDs (first 3): {coverage.failed_data_ids[:3]}")
        
    else:
        logger.error(f"✗ Repository validation failed: {result.error_message}")

def test_validator():
    """Test the ButlerRepoValidator with multiple repositories."""
    
    repositories = [
        ("demo_data/pipelines_check-29.1.1/DATA_REPO", "Demo Data Repository"),
        ("rc2_subset/SMALL_HSC", "RC2 Subset Repository")
    ]
    
    logger.info("Testing ButlerRepoValidator with multiple repositories...")
    
    for repo_path, repo_name in repositories:
        try:
            test_validator_for_repo(repo_path, repo_name)
        except Exception as e:
            logger.error(f"Failed to test {repo_name}: {e}")
            continue

if __name__ == "__main__":
    test_validator()