#!/usr/bin/env python3
"""
Simple test script for Butler repository creation functionality.

This script tests the ButlerRepoCreator module using local test data
to avoid slow network operations.
"""

import logging
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from ripple.butler.creator import (
    ButlerRepoCreator,
    DataDiscoveryResult,
    RepositoryCreationResult
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_fits_files(test_dir: Path) -> None:
    """Create some dummy FITS files for testing data discovery."""
    
    # Create subdirectories
    (test_dir / "raw").mkdir(parents=True, exist_ok=True)
    (test_dir / "calexp").mkdir(parents=True, exist_ok=True)
    (test_dir / "src").mkdir(parents=True, exist_ok=True)
    
    # Create dummy files with realistic LSST naming patterns
    sample_files = [
        "raw/raw_HSC-r_00123456_10.fits",
        "raw/raw_HSC-g_00123457_10.fits",
        "calexp/calexp_HSC-r_00123456_10.fits",
        "calexp/calexp_HSC-g_00123457_10.fits", 
        "src/src_HSC-r_00123456_10.fits",
        "src/src_HSC-g_00123457_10.fits",
    ]
    
    for file_path in sample_files:
        full_path = test_dir / file_path
        full_path.touch()  # Create empty file
        
    logger.info(f"Created {len(sample_files)} sample FITS files in {test_dir}")


def test_data_discovery():
    """Test data discovery with sample files."""
    
    logger.info("=" * 60)
    logger.info("TESTING DATA DISCOVERY")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_dir = Path(temp_dir) / "test_data"
        test_data_dir.mkdir()
        
        # Create sample FITS files
        create_sample_fits_files(test_data_dir)
        
        # Test data discovery
        creator = ButlerRepoCreator("/tmp/test_repo")
        
        logger.info(f"Discovering data in: {test_data_dir}")
        discovery = creator.discover_data_files(str(test_data_dir))
        
        logger.info(f"✓ Total FITS files found: {discovery.total_files}")
        logger.info(f"✓ Detected instruments: {discovery.supported_instruments}")
        
        logger.info("✓ Dataset types discovered:")
        for dataset_type, count in discovery.file_patterns.items():
            logger.info(f"    {dataset_type}: {count} files")
        
        if discovery.error_files:
            logger.warning(f"Files with errors: {len(discovery.error_files)}")
        
        # Verify expected results
        expected_types = {'raw', 'calexp', 'src'}
        found_types = set(discovery.file_patterns.keys())
        
        if expected_types.issubset(found_types):
            logger.info("✓ All expected dataset types discovered")
            return True
        else:
            logger.error(f"✗ Missing dataset types: {expected_types - found_types}")
            return False


def test_repository_creation():
    """Test creating an empty Butler repository."""
    
    logger.info("=" * 60)
    logger.info("TESTING REPOSITORY CREATION")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_repo_path = Path(temp_dir) / "test_butler_repo"
        
        # Test repository creation
        logger.info(f"Creating Butler repository at: {test_repo_path}")
        
        creator = ButlerRepoCreator(str(test_repo_path), instrument="HSC")
        result = creator.create_repository(overwrite=True)
        
        if result.success:
            logger.info("✓ Repository creation SUCCESSFUL")
            logger.info(f"  Repository path: {result.repo_path}")
            
            # Verify repository structure
            repo_path = Path(result.repo_path)
            
            # Check for butler.yaml
            if (repo_path / "butler.yaml").exists():
                logger.info("✓ butler.yaml file created")
            else:
                logger.warning("✗ butler.yaml file missing")
                return False
            
            # Check for SQLite database
            if (repo_path / "gen3.sqlite3").exists():
                logger.info("✓ SQLite registry database created")
            else:
                logger.warning("✗ SQLite database missing")
                return False
            
            # Test basic Butler operations
            try:
                from lsst.daf.butler import Butler
                butler = Butler(str(repo_path))
                
                # Try to query collections (should work even if empty)
                collections = list(butler.registry.queryCollections())
                logger.info(f"✓ Butler instance created successfully")
                logger.info(f"  Collections available: {len(collections)}")
                
                return True
                
            except Exception as e:
                logger.error(f"✗ Failed to create Butler instance: {e}")
                return False
        
        else:
            logger.error("✗ Repository creation FAILED")
            logger.error(f"  Error: {result.error_message}")
            return False


def test_instrument_detection():
    """Test instrument auto-detection from file paths."""
    
    logger.info("=" * 60)
    logger.info("TESTING INSTRUMENT DETECTION")
    logger.info("=" * 60)
    
    creator = ButlerRepoCreator("/tmp/test_repo")
    
    # Test different file path patterns
    test_cases = [
        ("/data/hsc/raw/HSC-r_12345.fits", "HSC"),
        ("/data/lsst/dc2/calexp_12345.fits", "LSSTCam"),
        ("/data/decam/raw/c4d_12345.fits", "DECam"),
        ("/data/cfht/mega_12345.fits", "CFHT"),
        ("/data/unknown/file_12345.fits", None),
    ]
    
    all_passed = True
    
    for file_path, expected_instrument in test_cases:
        detected = creator._detect_instrument(Path(file_path))
        
        if detected == expected_instrument:
            logger.info(f"✓ {file_path} -> {detected}")
        else:
            logger.error(f"✗ {file_path} -> {detected} (expected {expected_instrument})")
            all_passed = False
    
    return all_passed


def main():
    """Main test function."""
    
    logger.info("Butler Repository Creator - Simple Test Suite")
    logger.info("=" * 60)
    
    # Run tests
    test_results = []
    
    try:
        test_results.append(("Data Discovery", test_data_discovery()))
    except Exception as e:
        logger.error(f"Data Discovery test failed with exception: {e}")
        test_results.append(("Data Discovery", False))
    
    try:
        test_results.append(("Repository Creation", test_repository_creation()))
    except Exception as e:
        logger.error(f"Repository Creation test failed with exception: {e}")
        test_results.append(("Repository Creation", False))
    
    try:
        test_results.append(("Instrument Detection", test_instrument_detection()))
    except Exception as e:
        logger.error(f"Instrument Detection test failed with exception: {e}")
        test_results.append(("Instrument Detection", False))
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())