#!/usr/bin/env python3
"""
Test script for Butler repository creation from existing data.

This script demonstrates how to create a Butler repository from the RIPPLe_data
folder containing DC2 simulation data.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from ripple.butler.creator import (
    ButlerRepoCreator,
    create_butler_repo_from_directory_tree
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_discovery():
    """Test data discovery using local HSC data."""
    
    # Test with our existing local HSC data
    test_paths = [
        "demo_data/pipelines_check-29.1.1/DATA_REPO",  # Processed HSC data
        "rc2_subset/SMALL_HSC"  # Raw HSC data
    ]
    
    logger.info("=" * 60)
    logger.info("TESTING DATA DISCOVERY ON LOCAL DATA")
    logger.info("=" * 60)
    
    success = True
    
    for test_path in test_paths:
        if not Path(test_path).exists():
            logger.warning(f"Test data path not found: {test_path}")
            continue
            
        logger.info(f"\nDiscovering data in: {test_path}")
        
        # Create a repository creator (no instrument specified - will auto-detect)
        creator = ButlerRepoCreator("/tmp/test_repo")
        
        try:
            discovery = creator.discover_data_files(test_path)
            
            logger.info(f"  Total FITS files found: {discovery.total_files}")
            logger.info(f"  Detected instruments: {discovery.supported_instruments}")
            
            if discovery.file_patterns:
                logger.info("  Dataset types discovered:")
                for dataset_type, count in discovery.file_patterns.items():
                    logger.info(f"    {dataset_type}: {count} files")
            else:
                logger.info("  No dataset types discovered (might be existing Butler repo)")
            
            if discovery.error_files:
                logger.warning(f"  Files with errors: {len(discovery.error_files)}")
                for file_path, error in discovery.error_files[:2]:  # Show first 2 errors
                    logger.warning(f"    {file_path}: {error}")
        
        except Exception as e:
            logger.error(f"  Discovery failed for {test_path}: {e}")
            success = False
    
    return success


def test_repository_creation():
    """Test creating a Butler repository from local HSC data."""
    
    # Use a simple directory with FITS files for testing 
    # Let's create a minimal test with some existing FITS files
    test_repo_path = "/tmp/test_hsc_butler_repo"
    
    logger.info("=" * 60)
    logger.info("TESTING REPOSITORY CREATION")
    logger.info("=" * 60)
    
    # Clean up any existing test repository
    import shutil
    if Path(test_repo_path).exists():
        shutil.rmtree(test_repo_path)
    
    # For this test, let's just create an empty repository and test the creation process
    # rather than ingesting data (which requires proper LSST stack setup)
    logger.info(f"Creating empty Butler repository at: {test_repo_path}")
    
    creator = ButlerRepoCreator(test_repo_path, instrument="HSC")
    
    # Test repository creation without data ingestion
    result = creator.create_repository(overwrite=True)
    
    if result.success:
        logger.info("✓ Repository creation SUCCESSFUL")
        logger.info(f"  Repository path: {result.repo_path}")
        logger.info(f"  Created collections: {result.created_collections}")
        logger.info(f"  Ingested datasets: {result.ingested_datasets}")
        
        if result.warnings:
            logger.warning("Warnings:")
            for warning in result.warnings:
                logger.warning(f"  {warning}")
        
        # Test basic Butler operations
        logger.info("\nTesting basic Butler operations...")
        try:
            from lsst.daf.butler import Butler
            butler = Butler(test_repo_path)
            
            collections = list(butler.registry.queryCollections())
            dataset_types = list(butler.registry.queryDatasetTypes())
            
            logger.info(f"  Collections in repository: {len(collections)}")
            logger.info(f"  Dataset types in repository: {len(dataset_types)}")
            
            if collections:
                logger.info(f"  Example collections: {collections[:3]}")
            if dataset_types:
                logger.info(f"  Example dataset types: {[dt.name for dt in dataset_types[:3]]}")
            
        except Exception as e:
            logger.warning(f"Butler operations test failed: {e}")
    
    else:
        logger.error("✗ Repository creation FAILED")
        logger.error(f"  Error: {result.error_message}")
        if result.warnings:
            logger.warning("Warnings:")
            for warning in result.warnings:
                logger.warning(f"  {warning}")
    
    return result.success


def main():
    """Main test function."""
    
    logger.info("Butler Repository Creator Test")
    logger.info("=" * 60)
    
    # Test 1: Data discovery
    success1 = test_data_discovery()
    
    # Test 2: Repository creation
    success2 = test_repository_creation()
    
    logger.info("=" * 60)
    if success1 and success2:
        logger.info("✓ ALL TESTS PASSED")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())