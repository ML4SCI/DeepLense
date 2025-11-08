#!/usr/bin/env python3
"""
Mock test script for Butler repository creation functionality.

This script tests the ButlerRepoCreator module logic without requiring
the full LSST stack to be loaded, using mocks for the LSST imports.
"""

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Mock the LSST imports globally before any imports
sys.modules['lsst'] = Mock()
sys.modules['lsst.daf'] = Mock()
sys.modules['lsst.daf.butler'] = Mock()
sys.modules['lsst.obs'] = Mock()
sys.modules['lsst.obs.base'] = Mock()
sys.modules['lsst.pipe'] = Mock()
sys.modules['lsst.pipe.base'] = Mock()
sys.modules['lsst.pex'] = Mock()
sys.modules['lsst.pex.config'] = Mock()

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


def test_data_discovery_mock():
    """Test data discovery with mocked LSST imports."""
    
    logger.info("=" * 60)
    logger.info("TESTING DATA DISCOVERY (MOCKED)")
    logger.info("=" * 60)
    
    # Mock the LSST imports to avoid library loading issues
    with patch.dict('sys.modules', {
        'lsst.daf.butler': Mock(),
        'lsst.obs.base': Mock(),
        'lsst.pipe.base': Mock(),
        'lsst.pex.config': Mock(),
    }):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data_dir = Path(temp_dir) / "test_data"
            test_data_dir.mkdir()
            
            # Create sample FITS files
            create_sample_fits_files(test_data_dir)
            
            # Import the creator after mocking
            from ripple.butler.creator import ButlerRepoCreator
            
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
                logger.error(f"Found types: {found_types}")
                return False


def test_instrument_detection_mock():
    """Test instrument auto-detection without LSST stack."""
    
    logger.info("=" * 60)
    logger.info("TESTING INSTRUMENT DETECTION (MOCKED)")
    logger.info("=" * 60)
    
    # Mock the LSST imports
    with patch.dict('sys.modules', {
        'lsst.daf.butler': Mock(),
        'lsst.obs.base': Mock(),
        'lsst.pipe.base': Mock(),
        'lsst.pex.config': Mock(),
    }):
        
        # Import after mocking
        from ripple.butler.creator import ButlerRepoCreator
        
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


def test_file_classification_mock():
    """Test file classification without LSST stack."""
    
    logger.info("=" * 60)
    logger.info("TESTING FILE CLASSIFICATION (MOCKED)")
    logger.info("=" * 60)
    
    # Mock the LSST imports
    with patch.dict('sys.modules', {
        'lsst.daf.butler': Mock(),
        'lsst.obs.base': Mock(),
        'lsst.pipe.base': Mock(),
        'lsst.pex.config': Mock(),
    }):
        
        # Import after mocking
        from ripple.butler.creator import ButlerRepoCreator
        
        creator = ButlerRepoCreator("/tmp/test_repo")
        
        # Test different file naming patterns
        test_cases = [
            ("calexp_HSC-r_00123456_10.fits", ["calexp"]),
            ("src_HSC-r_00123456_10.fits", ["src"]),
            ("raw_HSC-r_00123456_10.fits", ["raw"]),
            ("bkgd_HSC-r_00123456_10.fits", ["bkgd"]),
            ("postISRCCD_HSC-r_00123456_10.fits", ["postISRCCD"]),
            ("unknown_file_12345.fits", ["raw"]),  # Default fallback
        ]
        
        all_passed = True
        
        for filename, expected_types in test_cases:
            detected_types = creator._classify_file(Path(filename))
            
            if detected_types == expected_types:
                logger.info(f"✓ {filename} -> {detected_types}")
            else:
                logger.error(f"✗ {filename} -> {detected_types} (expected {expected_types})")
                all_passed = False
        
        return all_passed


def main():
    """Main test function."""
    
    logger.info("Butler Repository Creator - Mock Test Suite")
    logger.info("=" * 60)
    
    # Run tests
    test_results = []
    
    try:
        test_results.append(("Data Discovery", test_data_discovery_mock()))
    except Exception as e:
        logger.error(f"Data Discovery test failed with exception: {e}")
        test_results.append(("Data Discovery", False))
    
    try:
        test_results.append(("Instrument Detection", test_instrument_detection_mock()))
    except Exception as e:
        logger.error(f"Instrument Detection test failed with exception: {e}")
        test_results.append(("Instrument Detection", False))
    
    try:
        test_results.append(("File Classification", test_file_classification_mock()))
    except Exception as e:
        logger.error(f"File Classification test failed with exception: {e}")
        test_results.append(("File Classification", False))
    
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