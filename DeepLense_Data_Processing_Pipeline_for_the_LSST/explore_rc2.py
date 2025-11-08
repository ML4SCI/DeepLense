#!/usr/bin/env python3
"""
Explore the RC2 subset repository to find data products.
"""

import logging
from ripple.butler import ButlerRepoValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_rc2_repository():
    """Explore the RC2 repository in detail."""
    
    repo_path = "rc2_subset/SMALL_HSC"
    logger.info(f"Exploring RC2 repository: {repo_path}")
    
    validator = ButlerRepoValidator(repo_path)
    result = validator.validate_repository()
    
    if not result.is_valid:
        logger.error(f"Repository invalid: {result.error_message}")
        return
    
    logger.info(f"\n=== RC2 Repository Analysis ===")
    logger.info(f"Collections: {len(result.collections)}")
    logger.info(f"Dataset types: {len(result.dataset_types)}")
    
    # Check if calexp and src exist in the dataset types
    target_types = ['calexp', 'src', 'postISRCCD', 'raw', 'deepCoadd', 'objectTable']
    
    logger.info(f"\n=== Checking for Key Dataset Types ===")
    for dtype in target_types:
        if dtype in result.dataset_types:
            logger.info(f"✓ {dtype} - FOUND")
        else:
            logger.info(f"✗ {dtype} - NOT FOUND")
    
    # Let's manually check specific collections for calexp
    logger.info(f"\n=== Manual Collection Analysis ===")
    butler = validator.butler
    
    # Check some promising collections
    promising_collections = [
        'HSC/RC2/defaults',
        'HSC/RC2_subset/defaults', 
        'HSC/raw/RC2',
        'HSC/raw/RC2_subset',
        'HSC/raw/all'
    ]
    
    for collection in promising_collections:
        if collection in result.collections:
            logger.info(f"\nChecking collection: {collection}")
            try:
                # Try to find calexp in this collection
                if 'calexp' in result.dataset_types:
                    dataset_type_obj = butler.registry.getDatasetType('calexp')
                    data_ids = list(butler.registry.queryDataIds(
                        dataset_type_obj.dimensions,
                        datasets=dataset_type_obj,
                        collections=[collection]
                    ))
                    logger.info(f"  calexp data IDs found: {len(data_ids)}")
                    if data_ids:
                        logger.info(f"  Sample: {dict(data_ids[0].mapping)}")
                
                # Try raw data
                if 'raw' in result.dataset_types:
                    dataset_type_obj = butler.registry.getDatasetType('raw')
                    data_ids = list(butler.registry.queryDataIds(
                        dataset_type_obj.dimensions,
                        datasets=dataset_type_obj,
                        collections=[collection]
                    ))
                    logger.info(f"  raw data IDs found: {len(data_ids)}")
                    if data_ids:
                        logger.info(f"  Sample: {dict(data_ids[0].mapping)}")
                        
            except Exception as e:
                logger.info(f"  Error querying {collection}: {e}")
    
    # Check what dataset types actually have data across all collections
    logger.info(f"\n=== Dataset Types with Data ===")
    types_with_data = []
    
    for dtype in ['raw', 'calexp', 'src', 'postISRCCD', 'deepCoadd']:
        if dtype in result.dataset_types:
            try:
                dataset_type_obj = butler.registry.getDatasetType(dtype)
                data_ids = list(butler.registry.queryDataIds(
                    dataset_type_obj.dimensions,
                    datasets=dataset_type_obj,
                    collections=result.collections
                ))
                if data_ids:
                    types_with_data.append((dtype, len(data_ids)))
                    logger.info(f"✓ {dtype}: {len(data_ids)} data IDs")
                    
                    # Show sample data ID
                    sample = dict(data_ids[0].mapping)
                    logger.info(f"  Sample data ID: {sample}")
                    
                    # Show which collections have this data
                    collections_with_data = []
                    for collection in result.collections[:10]:  # Check first 10 collections
                        try:
                            coll_data_ids = list(butler.registry.queryDataIds(
                                dataset_type_obj.dimensions,
                                datasets=dataset_type_obj,
                                collections=[collection]
                            ))
                            if coll_data_ids:
                                collections_with_data.append(collection)
                        except:
                            continue
                    
                    logger.info(f"  Found in collections: {collections_with_data[:5]}")
                else:
                    logger.info(f"✗ {dtype}: 0 data IDs")
                    
            except Exception as e:
                logger.info(f"✗ {dtype}: Error - {e}")

if __name__ == "__main__":
    explore_rc2_repository()