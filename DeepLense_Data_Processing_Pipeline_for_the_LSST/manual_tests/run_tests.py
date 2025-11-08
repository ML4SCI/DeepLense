#!/usr/bin/env python3
"""
RIPPLe Data Fetcher Manual Test Runner
======================================

This script runs all manual tests in sequence and provides a summary.

Usage:
    python run_tests.py [test_number]

Examples:
    python run_tests.py         # Run all tests
    python run_tests.py 1       # Run only test 1
    python run_tests.py 1-3     # Run tests 1 through 3
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_test(test_number, test_name, script_path):
    """Run a single test and return result."""
    print(f"\n{'='*60}")
    print(f"RUNNING TEST {test_number}: {test_name}")
    print(f"{'='*60}")
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Return success/failure
        success = result.returncode == 0
        return success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def main():
    """Run manual tests."""
    parser = argparse.ArgumentParser(description='Run RIPPLe manual tests')
    parser.add_argument('tests', nargs='?', default='all', 
                       help='Test number(s) to run (e.g., 1, 1-3, or all)')
    args = parser.parse_args()
    
    # Define all tests
    all_tests = [
        (1, "Environment Setup", "01_environment_setup.py"),
        (2, "Configuration Tests", "02_configuration_tests.py"),
        (3, "Butler Connection Tests", "03_butler_connection_tests.py"),
        (4, "Data Availability Tests", "04_data_availability_tests.py"),
        # Add more tests as they're created
    ]
    
    # Determine which tests to run
    if args.tests == 'all':
        tests_to_run = all_tests
    elif '-' in args.tests:
        # Range of tests
        start, end = map(int, args.tests.split('-'))
        tests_to_run = [t for t in all_tests if start <= t[0] <= end]
    else:
        # Single test
        test_num = int(args.tests)
        tests_to_run = [t for t in all_tests if t[0] == test_num]
    
    if not tests_to_run:
        print(f"❌ No tests found for: {args.tests}")
        return False
    
    # Run tests
    print("🚀 RIPPLE DATA FETCHER - MANUAL TEST RUNNER")
    print("=" * 60)
    print(f"Running {len(tests_to_run)} test(s)...")
    
    results = []
    for test_num, test_name, script_path in tests_to_run:
        script_full_path = Path(__file__).parent / script_path
        
        if not script_full_path.exists():
            print(f"❌ Test script not found: {script_path}")
            results.append((test_num, test_name, False))
            continue
        
        success = run_test(test_num, test_name, script_full_path)
        results.append((test_num, test_name, success))
        
        if not success:
            print(f"\n❌ Test {test_num} failed. Consider fixing before continuing.")
            # Ask if user wants to continue
            if len(tests_to_run) > 1:
                continue_response = input("\nContinue with remaining tests? (y/n): ")
                if continue_response.lower() != 'y':
                    break
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_num, test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Test {test_num}: {test_name:<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)