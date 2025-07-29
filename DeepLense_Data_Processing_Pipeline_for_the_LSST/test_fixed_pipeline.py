#!/usr/bin/env python3
"""
Test script to verify the fixed pipeline functionality.
"""

import logging
from ripple.butler import ButlerRepoValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_coverage_report():
    """Test the fixed coverage report functionality."""
    
    repo_path = "demo_data/pipelines_check-29.1.1/DATA_REPO"
    
    logger.info("Testing fixed coverage report functionality...")
    logger.info(f"Repository: {repo_path}")
    
    # Initialize validator
    validator = ButlerRepoValidator(repo_path)
    result = validator.validate_repository()
    
    if not result.is_valid:
        logger.error(f"Repository validation failed: {result.error_message}")
        return
    
    logger.info(f"Repository validation successful")
    logger.info(f"Instruments found: {result.instruments}")
    
    # Test coverage report with instruments parameter
    logger.info("\nTesting coverage report with governor dimensions fix...")
    
    try:
        coverage = validator.get_data_coverage(
            dataset_types=["calexp", "src"],
            filters=["r"],
            visit_ranges=[[903342, 903342]],
            instruments=list(result.instruments)  # Pass discovered instruments
        )
        
        logger.info("Coverage report generated successfully!")
        logger.info(f"Total data IDs: {coverage.total_data_ids}")
        logger.info(f"Available: {coverage.available_data_ids}")
        logger.info(f"Missing: {coverage.missing_data_ids}")
        logger.info(f"Coverage: {coverage.coverage_percentage:.1f}%")
        logger.info(f"Instruments: {coverage.instruments}")
        logger.info(f"Filters: {coverage.filters}")
        
        if coverage.total_data_ids > 0:
            logger.info("SUCCESS: Coverage report is working with governor dimensions fix!")
        else:
            logger.warning("No data found matching criteria")
            
    except Exception as e:
        logger.error(f"Coverage report still failed: {e}")

if __name__ == "__main__":
    test_fixed_coverage_report()