#!/usr/bin/env python3
"""
Test 3: Butler Connection and Component Initialization Tests
============================================================

This script tests the Butler connection, component initialization, and basic
functionality of the LsstDataFetcher class.

Prerequisites:
- Environment test (01_environment_setup.py) must pass
- Configuration test (02_configuration_tests.py) must pass
- LSST stack activated

Usage:
    python 03_butler_connection_tests.py
"""

import sys
import time
import traceback
from pathlib import Path

# Add RIPPLe to path
ripple_path = Path(__file__).parent.parent
sys.path.insert(0, str(ripple_path))

from ripple.data_access import ButlerConfig, LsstDataFetcher
from ripple.data_access.butler_client import ButlerClient
from ripple.data_access.cache_manager import CacheManager
from ripple.data_access.coordinate_utils import CoordinateConverter
from ripple.data_access.exceptions import ButlerConnectionError


def find_demo_repo():
    """Find the first available demo repository."""
    demo_repos = [
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-29.1.1/DATA_REPO",
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-29.1.0/DATA_REPO",
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-28.0.2/DATA_REPO",
    ]

    for repo_path in demo_repos:
        if Path(repo_path).exists():
            return repo_path

    return None


def test_butler_client_initialization():
    """Test ButlerClient initialization and connection."""
    print("🔧 TESTING BUTLER CLIENT INITIALIZATION")
    print("=" * 50)

    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping Butler client tests")
        return True

    # Test 1: Basic initialization
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC",
            cache_size=100,
        )

        butler_client = ButlerClient(config)
        print("✅ ButlerClient initialized")
        print(f"   └─ Repository: {demo_repo}")
        print(f"   └─ Collections: {config.collections}")
        print(f"   └─ Is remote: {butler_client.is_remote}")

    except Exception as e:
        print(f"❌ ButlerClient initialization failed: {e}")
        return False

    # Test 2: Connection test
    try:
        connection_ok = butler_client.test_connection()
        print(f"✅ Connection test: {'PASSED' if connection_ok else 'FAILED'}")

        if not connection_ok:
            print("⚠️  Butler connection test failed")
            return False

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

    # Test 3: Get skymap
    try:
        skymap = butler_client.get_skymap()
        print(f"✅ Skymap retrieved: {type(skymap).__name__}")
        print(
            f"   └─ Skymap type: {skymap.__class__.__module__}.{skymap.__class__.__name__}"
        )

    except Exception as e:
        print(f"❌ Skymap retrieval failed: {e}")
        return False

    # Test 4: Query available datasets
    try:
        # Use known tract/patch from HSC demo data
        available_data = butler_client.query_available_datasets(tract=0, patch="0,0")
        print("✅ Dataset query successful")
        print(
            f"   └─ Available datasets: {list(available_data.get('datasets', {}).keys())}"
        )
        print(f"   └─ Available filters: {available_data.get('filters', [])}")

    except Exception as e:
        print(f"❌ Dataset query failed: {e}")
        return False

    # Cleanup
    try:
        butler_client.cleanup()
        print("✅ Butler client cleanup successful")
    except Exception as e:
        print(f"⚠️  Butler client cleanup warning: {e}")

    return True


def test_coordinate_converter():
    """Test CoordinateConverter initialization and basic functionality."""
    print("\n🗺️  TESTING COORDINATE CONVERTER")
    print("=" * 50)

    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping coordinate converter tests")
        return True

    try:
        # Initialize Butler client to get skymap
        config = ButlerConfig(
            repo_path=demo_repo, collections=["demo_collection"], instrument="HSC"
        )

        butler_client = ButlerClient(config)
        skymap = butler_client.get_skymap()

        # Test 1: CoordinateConverter initialization
        converter = CoordinateConverter(skymap)
        print("✅ CoordinateConverter initialized")
        print(f"   └─ Skymap: {type(skymap).__name__}")

        # Test 2: Coordinate validation
        valid_coords = [
            (150.0, 2.5),  # Valid RA/Dec
            (0.0, 0.0),  # Valid boundary
            (359.9, 89.9),  # Valid boundary
        ]

        invalid_coords = [
            (400.0, 2.5),  # Invalid RA
            (150.0, 100.0),  # Invalid Dec
            (-10.0, 2.5),  # Invalid RA
        ]

        for ra, dec in valid_coords:
            is_valid = converter.validate_coordinates(ra, dec)
            print(
                f"✅ Coordinate validation ({ra}, {dec}): {'VALID' if is_valid else 'INVALID'}"
            )

        for ra, dec in invalid_coords:
            is_valid = converter.validate_coordinates(ra, dec)
            print(
                f"✅ Coordinate validation ({ra}, {dec}): {'VALID' if is_valid else 'INVALID'}"
            )
            assert not is_valid, f"Invalid coordinates ({ra}, {dec}) should be rejected"

        # Test 3: Angular separation calculation
        sep = converter.angular_separation(150.0, 2.5, 150.1, 2.6)
        print(f"✅ Angular separation: {sep:.6f} degrees")

        # Cleanup
        butler_client.cleanup()

    except Exception as e:
        print(f"❌ CoordinateConverter test failed: {e}")
        return False

    return True


def test_cache_manager():
    """Test CacheManager initialization and basic functionality."""
    print("\n💾 TESTING CACHE MANAGER")
    print("=" * 50)

    try:
        # Test 1: Initialization
        cache = CacheManager(max_size=10)
        print("✅ CacheManager initialized")
        print(f"   └─ Max size: {cache.max_size}")

        # Test 2: Basic operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        value1 = cache.get("key1")
        value2 = cache.get("key2")
        value3 = cache.get("key3")  # Should be None

        print("✅ Cache operations")
        print(f"   └─ Get key1: {value1}")
        print(f"   └─ Get key2: {value2}")
        print(f"   └─ Get key3: {value3}")

        assert value1 == "value1", "Cache should return stored value"
        assert value2 == "value2", "Cache should return stored value"
        assert value3 is None, "Cache should return None for missing key"

        # Test 3: Statistics
        stats = cache.get_statistics()
        print("✅ Cache statistics")
        print(f"   └─ Size: {stats['size']}")
        print(f"   └─ Hit count: {stats['hit_count']}")
        print(f"   └─ Miss count: {stats['miss_count']}")
        print(f"   └─ Hit rate: {stats['hit_rate']:.2%}")

        # Test 4: Cleanup
        cache.cleanup()
        print("✅ Cache cleanup successful")

    except Exception as e:
        print(f"❌ CacheManager test failed: {e}")
        return False

    return True


def test_lsst_data_fetcher_initialization():
    """Test LsstDataFetcher initialization and component integration."""
    print("\n🌊 TESTING LSST DATA FETCHER INITIALIZATION")
    print("=" * 50)

    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping LsstDataFetcher tests")
        return True

    try:
        # Test 1: Basic initialization
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC",
            cache_size=100,
            enable_performance_monitoring=True,
        )

        fetcher = LsstDataFetcher(config)
        print("✅ LsstDataFetcher initialized")
        print(f"   └─ Config: {type(config).__name__}")
        print(f"   └─ Butler client: {type(fetcher.butler_client).__name__}")
        print(
            f"   └─ Coordinate converter: {type(fetcher.coordinate_converter).__name__}"
        )
        print(f"   └─ Cache manager: {type(fetcher.cache_manager).__name__}")

        # Test 2: Configuration validation
        validation = fetcher.validate_configuration()
        print("✅ Configuration validation")
        print(
            f"   └─ Butler connection: {'✅' if validation['butler_connection'] else '❌'}"
        )
        print(
            f"   └─ Coordinate converter: {'✅' if validation['coordinate_converter'] else '❌'}"
        )
        print(f"   └─ Cache manager: {'✅' if validation['cache_manager'] else '❌'}")
        print(
            f"   └─ Performance monitoring: {'✅' if validation['performance_monitoring'] else '❌'}"
        )

        if validation["errors"]:
            print(f"   └─ Errors: {validation['errors']}")

        # Test 3: Performance metrics
        metrics = fetcher.get_performance_metrics()
        print(f"✅ Performance metrics: {len(metrics)} entries")

        # Test 4: Cache statistics
        cache_stats = fetcher.get_cache_statistics()
        print(f"✅ Cache statistics: {cache_stats}")

        # Test 5: Context manager
        with LsstDataFetcher(config) as context_fetcher:
            print("✅ Context manager works")
            print(f"   └─ Type: {type(context_fetcher).__name__}")

        print("✅ Context manager cleanup successful")

    except Exception as e:
        print(f"❌ LsstDataFetcher initialization failed: {e}")
        return False

    return True


def test_error_handling():
    """Test error handling for various failure scenarios."""
    print("\n⚠️  TESTING ERROR HANDLING")
    print("=" * 50)

    # Test 1: Invalid repository path
    try:
        config = ButlerConfig(
            repo_path="/nonexistent/path",
            collections=["demo_collection"],
            instrument="HSC",
        )

        try:
            LsstDataFetcher(config)
            print("❌ Should have failed with invalid repo path")
            return False
        except ButlerConnectionError as e:
            print(f"✅ Correctly caught ButlerConnectionError: {e}")
        except Exception as e:
            print(f"✅ Caught expected error: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"❌ Unexpected error in invalid repo test: {e}")
        return False

    # Test 2: Empty collections
    try:
        config = ButlerConfig(
            repo_path="/tmp/test",
            collections=[],  # Empty collections
            instrument="HSC",
        )

        # This should be caught by config validation
        from ripple.data_access.config_examples import validate_config

        validation = validate_config(config)
        assert not validation["valid"], "Empty collections should be invalid"
        print("✅ Empty collections correctly rejected")

    except Exception as e:
        print(f"❌ Empty collections test failed: {e}")
        return False

    # Test 3: Invalid timeout values
    try:
        config = ButlerConfig(
            repo_path="/tmp/test",
            collections=["test"],
            timeout=-1.0,  # Invalid timeout
        )

        from ripple.data_access.config_examples import validate_config

        validation = validate_config(config)
        assert not validation["valid"], "Negative timeout should be invalid"
        print("✅ Negative timeout correctly rejected")

    except Exception as e:
        print(f"❌ Invalid timeout test failed: {e}")
        return False

    return True


def test_component_integration():
    """Test integration between components."""
    print("\n🔗 TESTING COMPONENT INTEGRATION")
    print("=" * 50)

    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping integration tests")
        return True

    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC",
            cache_size=50,
        )

        with LsstDataFetcher(config) as fetcher:
            # Test 1: Butler client and coordinate converter integration
            print("✅ Components integrated successfully")
            print(
                f"   └─ Butler client connected: {fetcher.butler_client.test_connection()}"
            )
            print(
                f"   └─ Coordinate converter ready: {fetcher.coordinate_converter is not None}"
            )
            print(f"   └─ Cache manager ready: {fetcher.cache_manager is not None}")

            # Test 2: Cache operations
            fetcher.cache_manager.put("test_key", "test_value")
            cached_value = fetcher.cache_manager.get("test_key")
            assert cached_value == "test_value", "Cache integration failed"
            print("✅ Cache integration working")

            # Test 3: Performance monitoring
            time.time()
            fetcher._record_performance_metric("test_operation", 0.1, 100, True)
            metrics = fetcher.get_performance_metrics()
            print(f"✅ Performance monitoring working: {len(metrics)} metrics")

            # Test 4: Clear cache
            fetcher.clear_cache()
            cached_value = fetcher.cache_manager.get("test_key")
            assert cached_value is None, "Cache clear failed"
            print("✅ Cache clear working")

    except Exception as e:
        print(f"❌ Component integration test failed: {e}")
        return False

    return True


def main():
    """Run all Butler connection and component tests."""
    print("🚀 RIPPLE DATA FETCHER - BUTLER CONNECTION TESTS")
    print("=" * 60)

    try:
        # Run all tests
        test_results = [
            ("Butler Client Init", test_butler_client_initialization()),
            ("Coordinate Converter", test_coordinate_converter()),
            ("Cache Manager", test_cache_manager()),
            ("LsstDataFetcher Init", test_lsst_data_fetcher_initialization()),
            ("Error Handling", test_error_handling()),
            ("Component Integration", test_component_integration()),
        ]

        # Summary
        print("\n📋 SUMMARY")
        print("=" * 50)

        all_passed = True
        for test_name, result in test_results:
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n🎉 ALL BUTLER CONNECTION TESTS PASSED!")
            print("Ready to test data availability and queries.")
            return True
        else:
            print("\n⚠️  Some Butler connection tests failed.")
            return False

    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
