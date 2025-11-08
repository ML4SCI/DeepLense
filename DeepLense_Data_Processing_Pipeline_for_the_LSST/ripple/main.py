#!/usr/bin/env python
"""
RIPPLe Main Pipeline Entry Point.

This is the main entry point for the RIPPLe (Rubin Image Preparation and 
Processing Lensing engine) pipeline. It handles repository setup, data access,
and pipeline execution based on user configuration.

Usage:
    python -m ripple.main config.yaml
    python -m ripple.main --help
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add ripple to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from ripple.butler_repo import ButlerRepoManager, load_config, get_default_config, save_config
from ripple.butler_repo.utils import check_lsst_environment, validate_butler_command
from ripple.data_access import LsstDataFetcher, ButlerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RipplePipeline:
    """Main RIPPLe pipeline orchestrator."""
    
    def __init__(self, config_path: str):
        """
        Initialize RIPPLe pipeline.
        
        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = None
        self.repo_manager = None
        self.data_fetcher = None
        self.repo_path = None
        
    def run(self) -> int:
        """
        Run the complete pipeline.
        
        Returns
        -------
        int
            Exit code (0 for success, non-zero for failure)
        """
        logger.info("=" * 60)
        logger.info("RIPPLe Pipeline - Rubin Image Preparation and Processing")
        logger.info("=" * 60)
        
        try:
            # Step 1: Check environment
            if not self._check_environment():
                return 1
            
            # Step 2: Load configuration
            if not self._load_configuration():
                return 1
            
            # Step 3: Setup Butler repository
            if not self._setup_repository():
                return 1
            
            # For config-based runs, we only setup the butler repository
            logger.info("\n" + "=" * 60)
            logger.info("Butler repository setup completed successfully!")
            logger.info(f"Repository location: {self.repo_path}")
            logger.info("=" * 60)
            
            return 0
            
        except KeyboardInterrupt:
            logger.warning("\nPipeline interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            return 1
    
    def _check_environment(self) -> bool:
        """Check if LSST environment is properly configured."""
        logger.info("\nStep 1: Checking environment...")
        
        # Check LSST stack
        if not check_lsst_environment():
            logger.error("\nLSST Science Pipelines not found!")
            logger.error("Please activate the LSST environment:")
            logger.error("  source /path/to/lsst_stack/loadLSST.sh")
            logger.error("  setup lsst_distrib")
            return False
        
        # Check butler command
        if not validate_butler_command():
            logger.error("Butler command not found. Please ensure LSST stack is properly set up.")
            return False
        
        logger.info("✓ Environment check passed")
        return True
    
    def _load_configuration(self) -> bool:
        """Load and validate configuration."""
        logger.info(f"\nStep 2: Loading configuration from {self.config_path}")
        
        try:
            self.config = load_config(self.config_path)
            logger.info("✓ Configuration loaded successfully")
            
            # Log key configuration details
            logger.info(f"  Data source: {self.config.data_source.type}")
            logger.info(f"  Instrument: {self.config.instrument.name}")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            logger.error("Use --generate-config to create a template configuration")
            return False
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _setup_repository(self) -> bool:
        """Set up Butler repository based on configuration."""
        logger.info("\nStep 3: Setting up Butler repository...")
        
        self.repo_manager = ButlerRepoManager(self.config)
        success, result = self.repo_manager.setup_repository()
        
        if success:
            self.repo_path = result
            logger.info(f"✓ Repository ready at: {result}")
            return True
        else:
            logger.error(f"Failed to setup repository: {result}")
            return False
    
    def _initialize_data_access(self) -> bool:
        """Initialize data access layer."""
        logger.info("\nStep 4: Initializing data access...")
        
        # Skip for server-based repositories
        if self.config.data_source.type == 'butler_server':
            logger.info("Using remote Butler server - skipping local data access initialization")
            return True
        
        try:
            # Get data fetcher from repo manager if available
            self.data_fetcher = self.repo_manager.get_data_fetcher()
            
            if self.data_fetcher:
                logger.info("✓ Data access initialized from repository manager")
            else:
                # Create new data fetcher
                butler_config = ButlerConfig(
                    repo_path=str(self.repo_path),
                    collections=self.config.data_source.collections or [
                        f"{self.config.instrument.name}/defaults",
                        f"{self.config.instrument.name}/raw/all",
                        f"{self.config.instrument.name}/calib",
                        "refcats"
                    ],
                    instrument=self.config.instrument.name,
                    cache_size=self.config.processing.cache_size,
                    enable_performance_monitoring=self.config.processing.enable_performance_monitoring
                )
                
                self.data_fetcher = LsstDataFetcher(butler_config)
                logger.info("✓ Data access initialized")
            
            # Test data access
            validation = self.data_fetcher.validate_configuration()
            if validation['butler_connection']:
                logger.info("✓ Butler connection verified")
            else:
                logger.warning("Butler connection validation failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize data access: {e}")
            return False
    
    def _run_pipeline(self) -> bool:
        """Run the main pipeline operations."""
        logger.info("\nStep 5: Running pipeline operations...")
        
        # This is where you would add actual pipeline operations
        # For now, we'll just demonstrate data access
        
        if self.data_fetcher:
            try:
                # Example: Query available data
                logger.info("\nQuerying available data...")
                
                # You can add specific coordinates from config or use defaults
                test_ra = 320.37  # Example coordinate
                test_dec = -0.33
                
                availability = self.data_fetcher.get_available_data(
                    ra=test_ra,
                    dec=test_dec
                )
                
                logger.info(f"Available data at (RA={test_ra}, Dec={test_dec}):")
                logger.info(f"  Filters: {availability.get('filters', [])}")
                logger.info(f"  Tract/Patch: {availability.get('tracts_patches', [])}")
                
                # Example: Fetch a cutout if requested in config
                if hasattr(self.config.processing, 'test_cutout') and self.config.processing.test_cutout:
                    logger.info("\nFetching test cutout...")
                    cutout = self.data_fetcher.fetch_cutout(
                        ra=test_ra,
                        dec=test_dec,
                        size=self.config.processing.cutout_size,
                        filters=['r']
                    )
                    logger.info(f"✓ Successfully fetched cutout: {cutout.getImage().array.shape}")
                
            except Exception as e:
                logger.warning(f"Pipeline operation warning: {e}")
        
        logger.info("\n✓ Pipeline operations completed")
        return True


def main():
    """Main entry point for RIPPLe pipeline."""
    parser = argparse.ArgumentParser(
        description="RIPPLe - Rubin Image Preparation and Processing Lensing engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline with configuration file
  python -m ripple.main config.yaml
  
  # Generate default configuration
  python -m ripple.main --generate-config my_config.yaml
  
  # Run with verbose output
  python -m ripple.main config.yaml --verbose
  
  # Show version information
  python -m ripple.main --version
        """
    )
    
    parser.add_argument(
        "config",
        nargs="?",
        help="Configuration YAML file"
    )
    
    parser.add_argument(
        "--generate-config",
        metavar="FILE",
        help="Generate default configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information"
    )
    
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment setup only"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Show version
    if args.version:
        print("RIPPLe (Rubin Image Preparation and Processing Lensing engine)")
        print("Version: 0.1.0")
        print("Part of Google Summer of Code 2025 - ML4SCI")
        return 0
    
    # Check environment only
    if args.check_env:
        if check_lsst_environment() and validate_butler_command():
            print("✓ Environment check passed")
            print("  LSST Science Pipelines: Available")
            print("  Butler command: Available")
            return 0
        else:
            print("✗ Environment check failed")
            return 1
    
    # Generate configuration
    if args.generate_config:
        config = get_default_config()
        save_config(config, args.generate_config)
        print(f"Generated configuration file: {args.generate_config}")
        print("\nPlease edit the configuration file to:")
        print("  1. Set the correct data source path")
        print("  2. Choose the appropriate instrument")
        print("  3. Configure processing parameters")
        return 0
    
    # Check if config file is provided
    if not args.config:
        parser.error("Configuration file is required. Use --generate-config to create one.")
    
    # Run pipeline
    pipeline = RipplePipeline(args.config)
    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())