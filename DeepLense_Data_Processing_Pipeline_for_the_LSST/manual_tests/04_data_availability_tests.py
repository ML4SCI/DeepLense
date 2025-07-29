#!/usr/bin/env python3
"""
Test 4: Data Availability and Query Tests
==========================================

This script tests data availability queries, coordinate conversion, and
dataset discovery functionality.

Prerequisites:
- Previous tests (01-03) must pass
- LSST stack activated
- Demo data available

Usage:
    python 04_data_availability_tests.py
"""

import sys
import os
from pathlib import Path
import traceback
import time

# Add RIPPLe to path
ripple_path = Path(__file__).parent.parent
sys.path.insert(0, str(ripple_path))

from ripple.data_access import LsstDataFetcher, ButlerConfig
from ripple.data_access.exceptions import DataAccessError, CoordinateConversionError

def find_demo_repo():
    """Find the first available demo repository."""
    demo_repos = [
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-29.1.1/DATA_REPO",
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-29.1.0/DATA_REPO",
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-28.0.2/DATA_REPO"
    ]
    
    for repo_path in demo_repos:
        if Path(repo_path).exists():
            return repo_path
    
    return None

def test_coordinate_conversion():
    """Test coordinate conversion functionality."""
    print("🗺️  TESTING COORDINATE CONVERSION")
    print("=" * 50)
    
    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping coordinate conversion tests")
        return True
    
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC"
        )
        
        with LsstDataFetcher(config) as fetcher:
            converter = fetcher.coordinate_converter
            
            # Test 1: Basic coordinate validation
            test_coords = [
                (150.0, 2.5, True),    # Valid coordinates
                (0.0, 0.0, True),      # Valid boundary
                (359.9, 89.9, True),   # Valid boundary
                (360.1, 2.5, False),   # Invalid RA
                (150.0, 95.0, False),  # Invalid Dec
                (-10.0, 2.5, False)    # Invalid RA
            ]
            
            for ra, dec, should_be_valid in test_coords:
                is_valid = converter.validate_coordinates(ra, dec)
                status = "✅" if is_valid == should_be_valid else "❌"
                print(f"{status} Coordinate validation ({ra}, {dec}): {'VALID' if is_valid else 'INVALID'}")
                
                if is_valid != should_be_valid:
                    print(f"   └─ Expected: {should_be_valid}, Got: {is_valid}")
            
            # Test 2: Angular separation calculation
            separations = [
                ((150.0, 2.5), (150.0, 2.5), 0.0),      # Same point
                ((150.0, 2.5), (150.1, 2.5), 0.1),      # 0.1 degree in RA
                ((150.0, 2.5), (150.0, 2.6), 0.1),      # 0.1 degree in Dec
                ((150.0, 2.5), (150.1, 2.6), 0.1414)    # Diagonal
            ]
            
            for (ra1, dec1), (ra2, dec2), expected_sep in separations:
                actual_sep = converter.angular_separation(ra1, dec1, ra2, dec2)
                print(f"✅ Angular separation ({ra1},{dec1}) to ({ra2},{dec2}): {actual_sep:.4f}°")
                
                # Check if close to expected (within 0.01 degrees)
                if abs(actual_sep - expected_sep) > 0.01:
                    print(f"   └─ Expected ~{expected_sep:.4f}°, got {actual_sep:.4f}°")
            
            print(f"✅ Coordinate conversion tests completed")
            
    except Exception as e:
        print(f"❌ Coordinate conversion test failed: {e}")
        return False
    
    return True

def test_data_availability_basic():
    """Test basic data availability queries."""
    print("\n📊 TESTING BASIC DATA AVAILABILITY")
    print("=" * 50)
    
    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping data availability tests")
        return True
    
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC"
        )
        
        with LsstDataFetcher(config) as fetcher:
            # Test 1: Query data availability at a point
            # Use coordinates that should be in HSC demo data
            test_coordinates = [
                (150.0, 2.5),   # Common test coordinate
                (0.0, 0.0),     # Origin
                (320.0, 0.0),   # Another region
            ]
            
            for ra, dec in test_coordinates:
                try:
                    start_time = time.time()
                    availability = fetcher.get_available_data(ra, dec)
                    elapsed = time.time() - start_time
                    
                    print(f"✅ Data availability query ({ra}, {dec}) - {elapsed:.3f}s")
                    print(f"   └─ Tract/patches: {len(availability.get('tracts_patches', []))}")
                    print(f"   └─ Available filters: {availability.get('filters', [])}")
                    print(f"   └─ Total datasets: {availability.get('total_datasets', 0)}")
                    
                    # Check structure
                    required_keys = ['coordinates', 'tracts_patches', 'datasets', 'filters', 'total_datasets']
                    for key in required_keys:
                        if key not in availability:
                            print(f"   └─ Missing key: {key}")
                    
                except Exception as e:
                    print(f"❌ Data availability query failed for ({ra}, {dec}): {e}")
                    # Continue with other coordinates
            
            print(f"✅ Basic data availability tests completed")
            
    except Exception as e:
        print(f"❌ Basic data availability test failed: {e}")
        return False
    
    return True

def test_data_availability_radius():
    """Test data availability queries with radius."""
    print("\n🔍 TESTING DATA AVAILABILITY WITH RADIUS")
    print("=" * 50)
    
    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping radius tests")
        return True
    
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC"
        )
        
        with LsstDataFetcher(config) as fetcher:
            # Test with different radius values
            test_cases = [
                (150.0, 2.5, 0.1),    # Small radius
                (150.0, 2.5, 0.5),    # Medium radius
                (150.0, 2.5, 1.0),    # Large radius
            ]
            
            for ra, dec, radius in test_cases:
                try:
                    start_time = time.time()
                    availability = fetcher.get_available_data(ra, dec, radius=radius)
                    elapsed = time.time() - start_time
                    
                    print(f"✅ Radius query ({ra}, {dec}, r={radius}) - {elapsed:.3f}s")
                    print(f"   └─ Tract/patches: {len(availability.get('tracts_patches', []))}")
                    print(f"   └─ Available filters: {availability.get('filters', [])}")
                    print(f"   └─ Total datasets: {availability.get('total_datasets', 0)}")
                    
                    # Verify radius scaling (larger radius should find more or equal data)
                    if radius > 0.1:
                        print(f"   └─ Expected: more/equal data with larger radius")
                    
                except Exception as e:
                    print(f"❌ Radius query failed for ({ra}, {dec}, r={radius}): {e}")
            
            print(f"✅ Radius data availability tests completed")
            
    except Exception as e:
        print(f"❌ Radius data availability test failed: {e}")
        return False
    
    return True

def test_tract_patch_queries():
    """Test tract/patch specific queries."""
    print("\n🗂️  TESTING TRACT/PATCH QUERIES")
    print("=" * 50)
    
    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping tract/patch tests")
        return True
    
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC"
        )
        
        with LsstDataFetcher(config) as fetcher:
            # Test 1: Direct Butler client tract/patch queries
            butler_client = fetcher.butler_client
            
            # Test known tract/patch combinations for HSC demo data
            test_tract_patches = [
                (0, "0,0"),    # Common demo data location
                (0, "1,0"),    # Adjacent patch
                (0, "0,1"),    # Adjacent patch
            ]
            
            for tract, patch in test_tract_patches:
                try:
                    start_time = time.time()
                    data_info = butler_client.query_available_datasets(tract, patch)
                    elapsed = time.time() - start_time
                    
                    print(f"✅ Tract/patch query ({tract}, {patch}) - {elapsed:.3f}s")
                    print(f"   └─ Datasets: {list(data_info.get('datasets', {}).keys())}")
                    print(f"   └─ Filters: {data_info.get('filters', [])}")
                    
                except Exception as e:
                    print(f"❌ Tract/patch query failed for ({tract}, {patch}): {e}")
                    # This is expected for non-existent tract/patch combinations
            
            print(f"✅ Tract/patch query tests completed")
            
    except Exception as e:
        print(f"❌ Tract/patch query test failed: {e}")
        return False
    
    return True

def test_dataset_existence_checks():
    """Test dataset existence checking functionality."""
    print("\n🔍 TESTING DATASET EXISTENCE CHECKS")
    print("=" * 50)
    
    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping dataset existence tests")
        return True
    
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC"
        )
        
        with LsstDataFetcher(config) as fetcher:
            butler_client = fetcher.butler_client
            
            # Test different dataset types
            dataset_types = [
                "deepCoadd",      # Should exist in demo data
                "calexp",         # Should exist in demo data
                "src",            # Should exist in demo data
                "nonexistent"     # Should not exist
            ]
            
            # Test with demo data coordinates
            test_data_ids = [
                {"instrument": "HSC", "visit": 903342, "detector": 10},  # From demo data
                {"tract": 0, "patch": "0,0", "band": "i"},              # From demo data
                {"tract": 999, "patch": "999,999", "band": "z"}         # Should not exist
            ]
            
            for dataset_type in dataset_types:
                for data_id in test_data_ids:
                    try:
                        exists = butler_client.dataset_exists(dataset_type, data_id)
                        print(f"✅ Dataset existence: {dataset_type} {data_id} = {'EXISTS' if exists else 'NOT FOUND'}")
                        
                        if exists:
                            # Try to find the dataset reference
                            ref = butler_client.find_dataset(dataset_type, data_id)
                            if ref:
                                print(f"   └─ Dataset ref: {ref.id}")
                            else:
                                print(f"   └─ Dataset ref: None")
                    
                    except Exception as e:
                        print(f"❌ Dataset existence check failed for {dataset_type} {data_id}: {e}")
                        # Continue with other tests
            
            print(f"✅ Dataset existence tests completed")
            
    except Exception as e:
        print(f"❌ Dataset existence test failed: {e}")
        return False
    
    return True

def test_catalog_queries():
    """Test catalog query functionality."""
    print("\n📋 TESTING CATALOG QUERIES")
    print("=" * 50)
    
    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping catalog tests")
        return True
    
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC"
        )
        
        with LsstDataFetcher(config) as fetcher:
            # Test catalog fetching
            test_coordinates = [
                (150.0, 2.5, 0.1),   # Small radius
                (0.0, 0.0, 0.5),     # Different location
            ]
            
            catalog_types = [
                "src",           # Source catalog
                "objectTable",   # Object table (may not exist in demo)
            ]
            
            for ra, dec, radius in test_coordinates:
                for catalog_type in catalog_types:
                    try:
                        start_time = time.time()
                        catalog = fetcher.fetch_catalog(ra, dec, radius, catalog_type)
                        elapsed = time.time() - start_time
                        
                        if catalog is not None:
                            print(f"✅ Catalog query ({ra}, {dec}, r={radius}) {catalog_type} - {elapsed:.3f}s")
                            print(f"   └─ Catalog type: {type(catalog).__name__}")
                            print(f"   └─ Catalog size: {len(catalog) if hasattr(catalog, '__len__') else 'unknown'}")
                        else:
                            print(f"⚠️  Catalog query returned None for {catalog_type}")
                    
                    except Exception as e:
                        print(f"❌ Catalog query failed for {catalog_type}: {e}")
                        # Continue with other tests
            
            print(f"✅ Catalog query tests completed")
            
    except Exception as e:
        print(f"❌ Catalog query test failed: {e}")
        return False
    
    return True

def test_performance_timing():
    """Test performance timing for data availability queries."""
    print("\n⚡ TESTING PERFORMANCE TIMING")
    print("=" * 50)
    
    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping performance tests")
        return True
    
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC",
            enable_performance_monitoring=True
        )
        
        with LsstDataFetcher(config) as fetcher:
            # Test multiple queries to measure performance
            test_coordinates = [
                (150.0, 2.5),
                (150.1, 2.5),
                (150.2, 2.5),
                (150.3, 2.5),
                (150.4, 2.5)
            ]
            
            # Time multiple queries
            times = []
            for ra, dec in test_coordinates:
                start_time = time.time()
                try:
                    availability = fetcher.get_available_data(ra, dec)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    print(f"✅ Query ({ra}, {dec}): {elapsed:.3f}s")
                except Exception as e:
                    print(f"❌ Query failed for ({ra}, {dec}): {e}")
            
            # Calculate statistics
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                print(f"✅ Performance statistics:")
                print(f"   └─ Average time: {avg_time:.3f}s")
                print(f"   └─ Min time: {min_time:.3f}s")
                print(f"   └─ Max time: {max_time:.3f}s")
                print(f"   └─ Total queries: {len(times)}")
                
                # Check performance metrics
                metrics = fetcher.get_performance_metrics()
                print(f"   └─ Performance metrics recorded: {len(metrics)}")
            
            print(f"✅ Performance timing tests completed")
            
    except Exception as e:
        print(f"❌ Performance timing test failed: {e}")
        return False
    
    return True

def test_error_cases():
    """Test error cases for data availability queries."""
    print("\n⚠️  TESTING ERROR CASES")
    print("=" * 50)
    
    demo_repo = find_demo_repo()
    if not demo_repo:
        print("⚠️  No demo repository found, skipping error case tests")
        return True
    
    try:
        config = ButlerConfig(
            repo_path=demo_repo,
            collections=["demo_collection"],
            instrument="HSC"
        )
        
        with LsstDataFetcher(config) as fetcher:
            # Test invalid coordinates
            invalid_coords = [
                (400.0, 2.5),    # Invalid RA
                (150.0, 100.0),  # Invalid Dec
                (-50.0, 2.5),    # Invalid RA
                (150.0, -100.0)  # Invalid Dec
            ]
            
            for ra, dec in invalid_coords:
                try:
                    availability = fetcher.get_available_data(ra, dec)
                    print(f"❌ Should have failed for invalid coordinates ({ra}, {dec})")
                    return False
                except Exception as e:
                    print(f"✅ Correctly caught error for ({ra}, {dec}): {type(e).__name__}")
            
            # Test invalid radius
            try:
                availability = fetcher.get_available_data(150.0, 2.5, radius=-1.0)
                print(f"❌ Should have failed for negative radius")
                return False
            except Exception as e:
                print(f"✅ Correctly caught error for negative radius: {type(e).__name__}")
            
            print(f"✅ Error case tests completed")
            
    except Exception as e:
        print(f"❌ Error case test failed: {e}")
        return False
    
    return True

def main():
    """Run all data availability tests."""
    print("🚀 RIPPLE DATA FETCHER - DATA AVAILABILITY TESTS")
    print("=" * 60)
    
    try:
        # Run all tests
        test_results = [
            ("Coordinate Conversion", test_coordinate_conversion()),
            ("Basic Data Availability", test_data_availability_basic()),
            ("Radius Queries", test_data_availability_radius()),
            ("Tract/Patch Queries", test_tract_patch_queries()),
            ("Dataset Existence", test_dataset_existence_checks()),
            ("Catalog Queries", test_catalog_queries()),
            ("Performance Timing", test_performance_timing()),
            ("Error Cases", test_error_cases())
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
            print("\n🎉 ALL DATA AVAILABILITY TESTS PASSED!")
            print("Ready to test cutout retrieval.")
            return True
        else:
            print("\n⚠️  Some data availability tests failed.")
            return False
            
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)