#!/usr/bin/env python3
"""Debug script to test file classification logic."""

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
    
    # Check patterns in order of specificity (most specific first)
    pattern_order = ['calexp', 'src', 'postISRCCD', 'bkgd', 'deepCoadd', 'deepCoadd_src', 'raw']
    
    for dataset_type in pattern_order:
        print(f"  Checking dataset_type: {dataset_type}")
        if dataset_type in DATASET_PATTERNS:
            patterns = DATASET_PATTERNS[dataset_type]
            for pattern in patterns:
                # Convert glob pattern to simple string matching
                # Remove * and .fits, just check for the prefix/suffix
                pattern_lower = pattern.lower().replace('*', '').replace('.fits', '')
                filename_no_fits = filename.replace('.fits', '')
                print(f"    Pattern: '{pattern}' → '{pattern_lower}'")
                print(f"    Check if '{pattern_lower}' in '{filename_no_fits}': {pattern_lower in filename_no_fits}")
                if pattern_lower in filename_no_fits:
                    matched_types.append(dataset_type)
                    print(f"    ✓ MATCH! Adding {dataset_type}")
                    break
            # Stop at first match to avoid multiple classifications
            if matched_types:
                print(f"  Found match, stopping: {matched_types}")
                break
    
    # Default to 'raw' if no specific type detected
    if not matched_types:
        matched_types = ['raw']
        print(f"  No matches, defaulting to: {matched_types}")
    
    return matched_types

# Test the files from our test
test_files = [
    "calexp_HSC-r_00123456_10.fits",
    "src_HSC-r_00123456_10.fits", 
    "raw_HSC-r_00123456_10.fits"
]

print("DEBUGGING FILE CLASSIFICATION")
print("=" * 50)

for filename in test_files:
    result = debug_classify_file(Path(filename))
    print(f"RESULT: {filename} → {result}")
    print()