#!/usr/bin/env python3
"""
Test 2: Configuration Validation Tests
=======================================

This script tests the ButlerConfig class and configuration validation functionality.
Tests different configuration scenarios and validates config parameters.

Prerequisites:
- Environment test (01_environment_setup.py) must pass
- LSST stack activated

Usage:
    python 02_configuration_tests.py
"""

import sys
import os
from pathlib import Path
import traceback

# Add RIPPLe to path
ripple_path = Path(__file__).parent.parent
sys.path.insert(0, str(ripple_path))

from ripple.data_access.data_fetcher import ButlerConfig
from ripple.data_access.config_examples import (
    get_default_config, 
    get_production_config, 
    get_testing_config,
    validate_config
)

def test_butler_config_creation():
    """Test ButlerConfig dataclass creation and defaults."""
    print("🔧 TESTING BUTLER CONFIG CREATION")
    print("=" * 50)
    
    # Test 1: Default config
    try:
        config = ButlerConfig()
        print(f"✅ Default config created")
        print(f"   └─ repo_path: {config.repo_path}")
        print(f"   └─ server_url: {config.server_url}")
        print(f"   └─ collections: {config.collections}")
        print(f"   └─ instrument: {config.instrument}")
        print(f"   └─ cache_size: {config.cache_size}")
        print(f"   └─ timeout: {config.timeout}")
    except Exception as e:
        print(f"❌ Default config failed: {e}")
        return False
    
    # Test 2: Local repository config
    try:
        config = ButlerConfig(
            repo_path="/tmp/test_repo",
            collections=["test_collection"],
            instrument="HSC",
            cache_size=500,
            timeout=60.0
        )
        print(f"✅ Local repo config created")
        print(f"   └─ repo_path: {config.repo_path}")
        print(f"   └─ collections: {config.collections}")
        print(f"   └─ instrument: {config.instrument}")
    except Exception as e:
        print(f"❌ Local repo config failed: {e}")
        return False
    
    # Test 3: Remote server config
    try:
        config = ButlerConfig(
            server_url="https://butler.lsst.org",
            collections=["HSC/runs/RC2"],
            timeout=120.0,
            retry_attempts=5
        )
        print(f"✅ Remote server config created")
        print(f"   └─ server_url: {config.server_url}")
        print(f"   └─ collections: {config.collections}")
        print(f"   └─ retry_attempts: {config.retry_attempts}")
    except Exception as e:
        print(f"❌ Remote server config failed: {e}")
        return False
        
    return True

def test_config_examples():
    """Test pre-defined configuration examples."""
    print("\n📋 TESTING CONFIG EXAMPLES")
    print("=" * 50)
    
    # Test 1: Default config example
    try:
        config = get_default_config()
        print(f"✅ Default config example: {type(config).__name__}")
        print(f"   └─ repo_path: {config.repo_path}")
        print(f"   └─ collections: {config.collections}")
    except Exception as e:
        print(f"❌ Default config example failed: {e}")
        return False
    
    # Test 2: Production config example
    try:
        config = get_production_config()
        print(f"✅ Production config example: {type(config).__name__}")
        print(f"   └─ server_url: {config.server_url}")
        print(f"   └─ max_connections: {config.max_connections}")
        print(f"   └─ cache_size: {config.cache_size}")
    except Exception as e:
        print(f"❌ Production config example failed: {e}")
        return False
    
    # Test 3: Testing config example
    try:
        config = get_testing_config()
        print(f"✅ Testing config example: {type(config).__name__}")
        print(f"   └─ repo_path: {config.repo_path}")
        print(f"   └─ collections: {config.collections}")
        print(f"   └─ cache_size: {config.cache_size}")
    except Exception as e:
        print(f"❌ Testing config example failed: {e}")
        return False
    
    return True

def test_config_validation():
    """Test configuration validation function."""
    print("\n🔍 TESTING CONFIG VALIDATION")
    print("=" * 50)
    
    # Test 1: Valid configuration
    try:
        config = ButlerConfig(
            repo_path="/tmp/test_repo",
            collections=["test_collection"],
            cache_size=1000,
            timeout=30.0,
            retry_attempts=3
        )
        
        validation = validate_config(config)
        print(f"✅ Valid config validation")
        print(f"   └─ Valid: {validation['valid']}")
        print(f"   └─ Errors: {validation['errors']}")
        print(f"   └─ Warnings: {validation['warnings']}")
        
        assert validation['valid'] == True, "Valid config should pass validation"
        
    except Exception as e:
        print(f"❌ Valid config validation failed: {e}")
        return False
    
    # Test 2: Invalid configuration (no repo_path or server_url)
    try:
        config = ButlerConfig(
            collections=["test_collection"],
            cache_size=1000
        )
        
        validation = validate_config(config)
        print(f"✅ Invalid config validation (no repo/server)")
        print(f"   └─ Valid: {validation['valid']}")
        print(f"   └─ Errors: {validation['errors']}")
        
        assert validation['valid'] == False, "Invalid config should fail validation"
        assert len(validation['errors']) > 0, "Should have validation errors"
        
    except Exception as e:
        print(f"❌ Invalid config validation failed: {e}")
        return False
    
    # Test 3: Configuration with warnings
    try:
        config = ButlerConfig(
            repo_path="/tmp/test_repo",
            server_url="https://butler.lsst.org",  # Both specified
            collections=["test_collection"],
            cache_size=15000,  # Large cache size
            max_workers=20     # High worker count
        )
        
        validation = validate_config(config)
        print(f"✅ Config with warnings")
        print(f"   └─ Valid: {validation['valid']}")
        print(f"   └─ Errors: {validation['errors']}")
        print(f"   └─ Warnings: {validation['warnings']}")
        
        assert len(validation['warnings']) > 0, "Should have warnings"
        
    except Exception as e:
        print(f"❌ Config with warnings failed: {e}")
        return False
    
    # Test 4: Configuration with errors
    try:
        config = ButlerConfig(
            repo_path="/tmp/test_repo",
            collections=[],  # Empty collections
            cache_size=-100,  # Negative cache size
            timeout=-5.0,     # Negative timeout
            retry_attempts=0  # Zero retry attempts
        )
        
        validation = validate_config(config)
        print(f"✅ Config with errors")
        print(f"   └─ Valid: {validation['valid']}")
        print(f"   └─ Errors: {validation['errors']}")
        
        assert validation['valid'] == False, "Should be invalid"
        assert len(validation['errors']) > 0, "Should have multiple errors"
        
    except Exception as e:
        print(f"❌ Config with errors failed: {e}")
        return False
        
    return True

def test_demo_data_configs():
    """Test configurations for available demo data."""
    print("\n📊 TESTING DEMO DATA CONFIGS")
    print("=" * 50)
    
    # Find available demo repositories
    demo_repos = [
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-29.1.1/DATA_REPO",
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-29.1.0/DATA_REPO",
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-28.0.2/DATA_REPO"
    ]
    
    available_repos = [repo for repo in demo_repos if Path(repo).exists()]
    
    if not available_repos:
        print("⚠️  No demo data repositories found")
        return True
    
    for repo_path in available_repos:
        try:
            # Create config for this demo repo
            config = ButlerConfig(
                repo_path=repo_path,
                collections=["demo_collection"],
                instrument="HSC",
                cache_size=100,
                enable_performance_monitoring=True
            )
            
            # Validate config
            validation = validate_config(config)
            
            print(f"✅ Demo repo config: {Path(repo_path).name}")
            print(f"   └─ Path: {repo_path}")
            print(f"   └─ Valid: {validation['valid']}")
            
            if validation['errors']:
                print(f"   └─ Errors: {validation['errors']}")
            if validation['warnings']:
                print(f"   └─ Warnings: {validation['warnings']}")
                
        except Exception as e:
            print(f"❌ Demo repo config failed for {repo_path}: {e}")
            return False
    
    return True

def test_config_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n⚡ TESTING CONFIG EDGE CASES")
    print("=" * 50)
    
    edge_cases = [
        # (description, config_params, should_be_valid)
        ("Empty collections list", {"repo_path": "/tmp", "collections": []}, False),
        ("Very large cache", {"repo_path": "/tmp", "collections": ["test"], "cache_size": 100000}, True),
        ("Zero cache size", {"repo_path": "/tmp", "collections": ["test"], "cache_size": 0}, True),
        ("Very long timeout", {"repo_path": "/tmp", "collections": ["test"], "timeout": 3600.0}, True),
        ("Very short timeout", {"repo_path": "/tmp", "collections": ["test"], "timeout": 0.1}, True),
        ("Max retry attempts", {"repo_path": "/tmp", "collections": ["test"], "retry_attempts": 10}, True),
        ("Min retry attempts", {"repo_path": "/tmp", "collections": ["test"], "retry_attempts": 1}, True),
        ("Large batch size", {"repo_path": "/tmp", "collections": ["test"], "batch_size": 1000}, True),
        ("Small batch size", {"repo_path": "/tmp", "collections": ["test"], "batch_size": 1}, True),
    ]
    
    for description, params, should_be_valid in edge_cases:
        try:
            config = ButlerConfig(**params)
            validation = validate_config(config)
            
            actual_valid = validation['valid']
            status = "✅" if actual_valid == should_be_valid else "❌"
            
            print(f"{status} {description}")
            print(f"   └─ Expected valid: {should_be_valid}, Actual: {actual_valid}")
            
            if validation['errors']:
                print(f"   └─ Errors: {validation['errors']}")
                
        except Exception as e:
            print(f"❌ {description} - Exception: {e}")
            return False
    
    return True

def main():
    """Run all configuration tests."""
    print("🚀 RIPPLE DATA FETCHER - CONFIGURATION TESTS")
    print("=" * 60)
    
    try:
        # Run all tests
        test_results = [
            ("Config Creation", test_butler_config_creation()),
            ("Config Examples", test_config_examples()),
            ("Config Validation", test_config_validation()),
            ("Demo Data Configs", test_demo_data_configs()),
            ("Edge Cases", test_config_edge_cases())
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
            print("\n🎉 ALL CONFIGURATION TESTS PASSED!")
            print("Ready to test Butler connections.")
            return True
        else:
            print("\n⚠️  Some configuration tests failed.")
            return False
            
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)