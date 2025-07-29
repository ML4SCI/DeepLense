#!/usr/bin/env python3
"""
Check DC2 directories for Butler repositories.
"""

import os
import logging
from pathlib import Path
from ripple.butler import ButlerRepoValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dc2_repositories():
    """Check DC2 directories for Butler repositories."""
    
    base_path = Path("/Volumes/ExternalSSD/RIPPLe_data/lsstdesc-public/dc2/run2.2i-dr6-v4")
    
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        return
    
    # Check potential repository directories
    potential_repos = [
        "calexp-t3828-t3829-small",
        "coadd-t3828-t3829-small", 
        "coadd-t3828-t3829",
        "raw-t3828-t3829-small"
    ]
    
    logger.info(f"Checking DC2 directories for Butler repositories...")
    
    for repo_name in potential_repos:
        repo_path = base_path / repo_name
        
        if not repo_path.exists():
            logger.info(f"❌ {repo_name}: Directory does not exist")
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Checking: {repo_name}")
        logger.info(f"Path: {repo_path}")
        
        # Check for Butler files
        butler_yaml = repo_path / "butler.yaml"
        gen3_db = repo_path / "gen3.sqlite3"
        
        if butler_yaml.exists():
            logger.info(f"✓ Found butler.yaml")
        else:
            logger.info(f"❌ No butler.yaml found")
            
        if gen3_db.exists():
            logger.info(f"✓ Found gen3.sqlite3")
        else:
            logger.info(f"❌ No gen3.sqlite3 found")
        
        # If it looks like a Butler repo, try to validate it
        if butler_yaml.exists() and gen3_db.exists():
            try:
                logger.info(f"🔍 Attempting Butler validation...")
                validator = ButlerRepoValidator(str(repo_path))
                result = validator.validate_repository()
                
                if result.is_valid:
                    logger.info(f"✅ VALID Butler repository!")
                    logger.info(f"   Collections: {len(result.collections)}")
                    logger.info(f"   Dataset types: {len(result.dataset_types)}")
                    logger.info(f"   Instruments: {result.instruments}")
                    
                    # Check for key data products
                    key_products = ['calexp', 'src', 'deepCoadd', 'objectTable']
                    found_products = []
                    
                    for product in key_products:
                        if product in result.dataset_types:
                            # Try to count data IDs
                            try:
                                butler = validator.butler
                                dataset_type_obj = butler.registry.getDatasetType(product)
                                data_ids = list(butler.registry.queryDataIds(
                                    dataset_type_obj.dimensions,
                                    datasets=dataset_type_obj,
                                    collections=result.collections
                                ))
                                if data_ids:
                                    found_products.append(f"{product}({len(data_ids)})")
                            except:
                                found_products.append(f"{product}(?)")
                    
                    if found_products:
                        logger.info(f"   Data products: {', '.join(found_products)}")
                    else:
                        logger.info(f"   Data products: None found")
                        
                else:
                    logger.info(f"❌ Invalid Butler repository: {result.error_message}")
                    
            except Exception as e:
                logger.info(f"❌ Butler validation failed: {e}")
        else:
            logger.info(f"❌ Not a Butler repository (missing required files)")
        
        # Show directory structure sample
        try:
            subdirs = [d.name for d in repo_path.iterdir() if d.is_dir()][:10]
            if subdirs:
                logger.info(f"   Sample subdirectories: {', '.join(subdirs)}")
        except:
            pass

if __name__ == "__main__":
    check_dc2_repositories()