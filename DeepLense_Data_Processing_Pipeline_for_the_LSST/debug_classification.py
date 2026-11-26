#!/usr/bin/env python3
"""Debug script to test file classification logic."""

import fnmatch
from pathlib import Path

# Test the logic directly
DATASET_PATTERNS = {
    'raw': ['raw_*.fits', '*.fits', '*_raw.fits'],
    'calexp': ['calexp_*.fits', '*_calexp.fits'],
    'src': ['src_*.fits', '*_src.fits', 'sources_*.fits'],
    'postISRCCD': ['postISRCCD_*.fits', '*_postISRCCD.fits'],
    'bkgd': ['bkgd_*.fits', '*_bkgd.fits', 'background_*.fits'],
    'deepCoadd': ['deepCoadd_*.fits', '*_deepCoadd.fits', 'coadd_*.fits'],
    'deepCoadd_src': ['deepCoadd_src_*.fits', '*_deepCoadd_src.fits'],
}

def debug_classify_file(file_path: Path):
    """Debug version of file classification."""
    filename = file_path.name.lower()
    print(f"\nClassifying: {filename}")
    matched_types = []
    
    # Check patterns in order of specificity (Most specific -> Least specific)
    # moved deepCoadd_src BEFORE src to prevent partial matches
    pattern_order = [
        'deepCoadd_src', 
        'deepCoadd', 
        'calexp', 
        'src', 
        'postISRCCD', 
        'bkgd', 
        'raw'
    ]
    
    for dataset_type in pattern_order:
        print(f"  Checking dataset_type: {dataset_type}")
        if dataset_type in DATASET_PATTERNS:
            patterns = DATASET_PATTERNS[dataset_type]
            for pattern in patterns:
                # Use fnmatch for proper glob pattern matching (e.g. *.fits)
                is_match = fnmatch.fnmatch(filename, pattern.lower())
                
                print(f"    Pattern: '{pattern}'")
                print(f"    Match result: {is_match}")
                
                if is_match:
                    matched_types.append(dataset_type)
                    print(f"    ✓ MATCH! Adding {dataset_type}")
                    break
        
            if matched_types:
                print(f"  Found match, stopping: {matched_types}")
                break

    if not matched_types:
        matched_types = ['raw']
        print(f"  No matches, defaulting to: {matched_types}")
    
    return matched_types

# Test the files from our test
test_files = [
    "calexp_HSC-r_00123456_10.fits",
    "src_HSC-r_00123456_10.fits", 
    "raw_HSC-r_00123456_10.fits",
    "deepCoadd_src_HSC-r.fits" # Added to test the specific case
]

print("DEBUGGING FILE CLASSIFICATION")
print("=" * 50)

for filename in test_files:
    result = debug_classify_file(Path(filename))
    print(f"RESULT: {filename} -> {result}")
    print()
