#!/usr/bin/env python3
"""
DeepLense Unified Dataset Downloader

This script automates the downloading of DeepLense datasets from Google Drive.
It provides a simple command-line interface to download datasets for different
models and splits (train/test).

Usage:
    python download_datasets.py --model 1 --split train
    python download_datasets.py --model all --split both
    python download_datasets.py --help

Author: DeepLense Contributors
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Error: 'gdown' library not found.")
    print("Please install it using: pip install gdown")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: 'tqdm' library not found. Progress bars will not be displayed.")
    print("Install it using: pip install tqdm")
    tqdm = None


# Dataset metadata: Google Drive file IDs
DATASETS = {
    'Model_I': {
        'train': {
            'file_id': '1QMVLpqag6S9JWqzmGM_pK4C0F1eBVIfV',
            'filename': 'Model_I_train.h5',
            'description': 'Model I Training Data (150x150, Gaussian PSF, SNR~25)'
        },
        'test': {
            'file_id': '1rUAKLLS3p9jDaL9R9m84JVKvMcUuVsO1',
            'filename': 'Model_I_test.h5',
            'description': 'Model I Testing Data'
        }
    },
    'Model_II': {
        'train': {
            'file_id': '1HYPkdtVUj9xsoGzFDxT4rhl37KmqDCg4',
            'filename': 'Model_II_train.h5',
            'description': 'Model II Training Data (64x64, Euclid observation)'
        },
        'test': {
            'file_id': '1PFdpqk7XOAKtg0Cnav4HTzyJiudx9dZv',
            'filename': 'Model_II_test.h5',
            'description': 'Model II Testing Data'
        }
    },
    'Model_III': {
        'train': {
            'file_id': '1ynKMJoEeKKJqLfuKRR1Y7rQjeBMM0w94',
            'filename': 'Model_III_train.h5',
            'description': 'Model III Training Data (64x64, HST observation)'
        },
        'test': {
            'file_id': '18BuCv40t6qmiNnhjJF1y9rqSBhBOfDon',
            'filename': 'Model_III_test.h5',
            'description': 'Model III Testing Data'
        }
    }
}


def download_file(file_id, output_path, description, force=False):
    """
    Download a file from Google Drive using gdown.
    
    Args:
        file_id (str): Google Drive file ID
        output_path (str): Path where the file should be saved
        description (str): Description of the dataset
        force (bool): Force re-download even if file exists
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Check if file already exists
    if os.path.exists(output_path) and not force:
        print(f"✓ File already exists: {output_path}")
        print(f"  Use --force to re-download")
        return True
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Construct Google Drive URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"\n{'='*70}")
    print(f"Downloading: {description}")
    print(f"Destination: {output_path}")
    print(f"{'='*70}")
    
    try:
        # Download with gdown
        gdown.download(url, output_path, quiet=False)
        print(f"✓ Successfully downloaded: {output_path}\n")
        return True
    except Exception as e:
        print(f"✗ Error downloading {description}: {str(e)}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download DeepLense datasets from Google Drive',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Model I training data
  python download_datasets.py --model 1 --split train
  
  # Download all Model II data (train and test)
  python download_datasets.py --model 2 --split both
  
  # Download all datasets
  python download_datasets.py --model all --split both
  
  # Download to custom directory
  python download_datasets.py --model 1 --split train --output-dir ./my_data/
  
  # Force re-download existing files
  python download_datasets.py --model 1 --split train --force

Dataset Information:
  Model I:   150x150 pixels, Gaussian PSF, SNR~25
  Model II:  64x64 pixels, Euclid observation characteristics
  Model III: 64x64 pixels, HST observation characteristics
  
  All datasets contain 3 classes:
    - Axion (vortex substructure)
    - CDM (cold dark matter subhalos)
    - No substructure (baseline)
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['1', '2', '3', 'all'],
        required=True,
        help='Model dataset to download (1, 2, 3, or all)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'both'],
        required=True,
        help='Dataset split to download (train, test, or both)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for datasets (default: ./data)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file already exists'
    )
    
    args = parser.parse_args()
    
    # Determine which models to download
    if args.model == 'all':
        models = ['Model_I', 'Model_II', 'Model_III']
    else:
        model_map = {'1': 'Model_I', '2': 'Model_II', '3': 'Model_III'}
        models = [model_map[args.model]]
    
    # Determine which splits to download
    if args.split == 'both':
        splits = ['train', 'test']
    else:
        splits = [args.split]
    
    # Download datasets
    total_files = len(models) * len(splits)
    successful_downloads = 0
    failed_downloads = []
    
    print(f"\nDeepLense Dataset Downloader")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Total files to download: {total_files}")
    print(f"{'='*70}\n")
    
    for model in models:
        for split in splits:
            dataset_info = DATASETS[model][split]
            file_id = dataset_info['file_id']
            filename = dataset_info['filename']
            description = dataset_info['description']
            
            # Create output path
            output_path = os.path.join(args.output_dir, model, split, filename)
            
            # Download file
            success = download_file(file_id, output_path, description, args.force)
            
            if success:
                successful_downloads += 1
            else:
                failed_downloads.append(f"{model}/{split}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Download Summary")
    print(f"{'='*70}")
    print(f"Successful: {successful_downloads}/{total_files}")
    
    if failed_downloads:
        print(f"Failed: {len(failed_downloads)}")
        print(f"Failed downloads: {', '.join(failed_downloads)}")
    else:
        print(f"All downloads completed successfully! ✓")
    
    print(f"{'='*70}\n")
    
    # Exit with appropriate code
    sys.exit(0 if len(failed_downloads) == 0 else 1)


if __name__ == '__main__':
    main()
