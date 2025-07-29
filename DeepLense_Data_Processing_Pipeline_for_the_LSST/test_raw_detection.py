#!/usr/bin/env python3
"""
Quick test to verify raw data detection in RC2 repository.
"""

import logging
from ripple.butler import ButlerRepoValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_raw_detection():
    """Test if we can detect raw data in RC2 repository."""
    
    repo_path = "rc2_subset/SMALL_HSC"
    
    logger.info("Testing raw data detection...")
    logger.info(f"Repository: {repo_path}")
    
    # Initialize validator
    validator = ButlerRepoValidator(repo_path)
    result = validator.validate_repository()
    
    if not result.is_valid:
        logger.error(f"Repository validation failed: {result.error_message}")
        return
    
    logger.info(f"Repository validation successful")
    logger.info(f"Found dataset types: {len(result.data_products)}")
    
    # Check specifically for raw data
    if 'raw' in result.data_products:
        raw_info = result.data_products['raw']
        logger.info(f"SUCCESS: Found {raw_info.available_count} raw exposures!")
        logger.info(f"Raw data coverage: {raw_info.coverage_percentage:.1f}%")
        logger.info(f"Raw data collections: {raw_info.collections}")
        if raw_info.sample_data_ids:
            logger.info(f"Sample raw data ID: {raw_info.sample_data_ids[0]}")
    else:
        logger.error("RAW data not found in discovered data products")
        logger.info(f"Available data products: {list(result.data_products.keys())}")

if __name__ == "__main__":
    test_raw_detection()