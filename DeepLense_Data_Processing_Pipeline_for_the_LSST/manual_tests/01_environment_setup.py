#!/usr/bin/env python3
"""
Test 1: Environment Setup and LSST Stack Verification
=======================================================

This script verifies that the LSST stack is properly installed and accessible.
Run this FIRST before any other tests.

Prerequisites:
- LSST stack should be activated in your shell
- Run: source ~/RIPPLe/lsst_stack/loadLSST.sh && setup lsst_distrib

Usage:
    python 01_environment_setup.py
"""

import importlib
import os
import sys
import traceback
from pathlib import Path


def test_environment_setup():
    """Test basic environment setup and LSST imports."""
    print("🔧 TESTING ENVIRONMENT SETUP")
    print("=" * 50)

    # Test 1: Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Test 2: Current working directory
    print(f"Current directory: {os.getcwd()}")

    # Test 3: LSST environment variables
    print("\n📋 LSST Environment Variables:")
    lsst_vars = ["LSST_STACK_VERSION", "LSST_LIBRARY_PATH", "LSST_CONDA_ENV_NAME"]
    for var in lsst_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"  {var}: {value}")

    # Test 4: Python path
    print(f"\n🐍 Python path contains {len(sys.path)} entries")
    lsst_paths = [p for p in sys.path if "lsst" in p.lower()]
    print(f"LSST-related paths: {len(lsst_paths)}")
    for path in lsst_paths[:3]:  # Show first 3
        print(f"  {path}")

    return True


def test_lsst_imports():
    """Test critical LSST imports."""
    print("\n🔍 TESTING LSST IMPORTS")
    print("=" * 50)

    # Core LSST modules required by data_fetcher.py
    required_modules = [
        "lsst.daf.butler",
        "lsst.afw.image",
        "lsst.geom",
        "lsst.skymap",
        "lsst.afw.table",
    ]

    results = {}

    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} - SUCCESS")
            results[module_name] = True

            # Test key classes
            if module_name == "lsst.daf.butler":
                from lsst.daf.butler import Butler, DataIdValueError, DatasetRef

                print("   └─ Butler class: ✅")
            elif module_name == "lsst.afw.image":
                from lsst.afw.image import Exposure, ExposureF

                print("   └─ Exposure class: ✅")
            elif module_name == "lsst.geom":
                from lsst.geom import Box2I, Point2I, SpherePoint, degrees

                print("   └─ Geometry classes: ✅")
            elif module_name == "lsst.skymap":
                from lsst.skymap import BaseSkyMap

                print("   └─ BaseSkyMap class: ✅")

        except ImportError as e:
            print(f"❌ {module_name} - FAILED: {e}")
            results[module_name] = False
        except Exception as e:
            print(f"⚠️  {module_name} - ERROR: {e}")
            results[module_name] = False

    return results


def test_ripple_imports():
    """Test RIPPLe module imports."""
    print("\n🌊 TESTING RIPPLE IMPORTS")
    print("=" * 50)

    # Add RIPPLe to path
    ripple_path = Path(__file__).parent.parent
    sys.path.insert(0, str(ripple_path))

    ripple_modules = [
        "ripple.data_access",
        "ripple.data_access.data_fetcher",
        "ripple.data_access.butler_client",
        "ripple.data_access.coordinate_utils",
        "ripple.data_access.cache_manager",
        "ripple.data_access.exceptions",
    ]

    results = {}

    for module_name in ripple_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} - SUCCESS")
            results[module_name] = True
        except ImportError as e:
            print(f"❌ {module_name} - FAILED: {e}")
            results[module_name] = False
        except Exception as e:
            print(f"⚠️  {module_name} - ERROR: {e}")
            results[module_name] = False

    return results


def test_demo_data_availability():
    """Test demo data availability."""
    print("\n📊 TESTING DEMO DATA AVAILABILITY")
    print("=" * 50)

    # Check for demo data repositories
    demo_repos = [
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-29.1.1/DATA_REPO",
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-29.1.0/DATA_REPO",
        "/home/kartikmandar/RIPPLe/demo_data/pipelines_check-28.0.2/DATA_REPO",
    ]

    available_repos = []

    for repo_path in demo_repos:
        repo = Path(repo_path)
        if repo.exists():
            print(f"✅ {repo_path} - EXISTS")

            # Check for butler.yaml
            butler_yaml = repo / "butler.yaml"
            if butler_yaml.exists():
                print("   └─ butler.yaml: ✅")
            else:
                print("   └─ butler.yaml: ❌")

            # Check for collections
            collections = list(repo.glob("demo_collection*"))
            print(f"   └─ Collections found: {len(collections)}")

            available_repos.append(repo_path)
        else:
            print(f"❌ {repo_path} - NOT FOUND")

    return available_repos


def main():
    """Run all environment tests."""
    print("🚀 RIPPLE DATA FETCHER - ENVIRONMENT TESTS")
    print("=" * 60)

    # Run tests
    try:
        # Test 1: Environment setup
        env_ok = test_environment_setup()

        # Test 2: LSST imports
        lsst_results = test_lsst_imports()

        # Test 3: RIPPLe imports
        ripple_results = test_ripple_imports()

        # Test 4: Demo data
        demo_repos = test_demo_data_availability()

        # Summary
        print("\n📋 SUMMARY")
        print("=" * 50)

        lsst_success = all(lsst_results.values())
        ripple_success = all(ripple_results.values())
        demo_available = len(demo_repos) > 0

        print(f"Environment setup: {'✅' if env_ok else '❌'}")
        print(f"LSST imports: {'✅' if lsst_success else '❌'}")
        print(f"RIPPLe imports: {'✅' if ripple_success else '❌'}")
        print(f"Demo data: {'✅' if demo_available else '❌'}")

        if lsst_success and ripple_success and demo_available:
            print("\n🎉 ALL TESTS PASSED! Ready for data fetcher testing.")
            return True
        else:
            print("\n⚠️  Some tests failed. Please fix issues before proceeding.")
            return False

    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
