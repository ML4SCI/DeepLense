"""
Butler Repository Manager - Main orchestrator for repository operations.

This module provides the main interface for managing Butler repositories,
including creation, data ingestion, and configuration.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .config_handler import RepoConfig, load_config, validate_config, get_default_config, save_config
from .create_repo import initialize_repository, verify_repository
from .ingest_data import DataIngestor

# Import RIPPLe data access for later use
try:
    from ..data_access import LsstDataFetcher, ButlerConfig as DataAccessButlerConfig
except ImportError:
    LsstDataFetcher = None
    DataAccessButlerConfig = None

logger = logging.getLogger(__name__)


class ButlerRepoManager:
    """
    Main orchestrator for Butler repository operations.
    
    This class manages the complete workflow of creating and setting up
    Butler repositories based on configuration.
    """
    
    def __init__(self, config: RepoConfig):
        """
        Initialize Butler Repository Manager.
        
        Parameters
        ----------
        config : RepoConfig
            Repository configuration
        """
        self.config = config
        self.repo_path = None
        self.data_fetcher = None
        
    def setup_repository(self) -> Tuple[bool, str]:
        """
        Set up Butler repository based on configuration.
        
        Returns
        -------
        Tuple[bool, str]
            Success status and repository path or error message
        """
        logger.info("Starting Butler repository setup...")
        
        try:
            # Determine repository path and setup method
            repo_path, needs_creation = self._determine_repository_path()
            self.repo_path = repo_path
            
            if needs_creation:
                logger.info(f"Creating new Butler repository at {repo_path}")
                success = self._create_and_setup_repository(repo_path)
                if not success:
                    return False, "Failed to create repository"
            else:
                logger.info(f"Using existing Butler repository at {repo_path}")
                # Verify existing repository
                verification = verify_repository(str(repo_path))
                if verification["errors"]:
                    logger.warning(f"Repository verification warnings: {verification['errors']}")
            
            # Initialize data fetcher if repository is ready
            if self.config.data_source.type in ['butler_repo', 'data_folder']:
                self._initialize_data_fetcher()
            
            return True, str(repo_path)
            
        except Exception as e:
            logger.error(f"Repository setup failed: {e}")
            return False, str(e)
    
    def _determine_repository_path(self) -> Tuple[Path, bool]:
        """
        Determine repository path and whether it needs to be created.
        
        Returns
        -------
        Tuple[Path, bool]
            Repository path and whether creation is needed
        """
        if self.config.data_source.type == 'butler_repo':
            # Use existing repository
            repo_path = Path(self.config.data_source.path)
            needs_creation = False
            
            # Check if it exists
            if not (repo_path / "butler.yaml").exists():
                if self.config.data_source.create_if_missing:
                    needs_creation = True
                else:
                    raise ValueError(f"Butler repository not found at {repo_path}")
                    
        elif self.config.data_source.type == 'data_folder':
            # Create repository for data folder
            data_path = Path(self.config.data_source.path)
            if not data_path.exists():
                raise ValueError(f"Data folder not found: {data_path}")
            
            # Create repository in data folder or alongside it
            repo_path = data_path / "butler_repo"
            # Check if repository already exists
            if (repo_path / "butler.yaml").exists():
                logger.info(f"Found existing Butler repository at {repo_path}")
                needs_creation = False
            else:
                needs_creation = True
            
        elif self.config.data_source.type == 'butler_server':
            # Remote butler server - no local repository needed
            return None, False
        else:
            raise ValueError(f"Unknown data source type: {self.config.data_source.type}")
        
        return repo_path, needs_creation
    
    def _check_data_exists(self, repo_path: Path) -> bool:
        """
        Check if the repository already contains data.
        
        Parameters
        ----------
        repo_path : Path
            Repository path
            
        Returns
        -------
        bool
            True if data exists
        """
        try:
            # Try to query collections to see if data exists
            import subprocess
            cmd = ["butler", "query-collections", str(repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                # Check for data collections (not just instrument collections)
                collections = result.stdout.strip().split('\n')
                data_collections = [c for c in collections if any(x in c for x in ['/raw/', '/calib/', 'refcats'])]
                if data_collections:
                    logger.info(f"Found existing data collections: {data_collections}")
                    return True
                    
        except Exception as e:
            logger.debug(f"Error checking for existing data: {e}")
            
        return False
    
    def _create_and_setup_repository(self, repo_path: Path) -> bool:
        """
        Create and set up a new Butler repository.
        
        Parameters
        ----------
        repo_path : Path
            Path where repository will be created
            
        Returns
        -------
        bool
            True if successful
        """
        # Initialize repository (this will handle existing repos gracefully)
        if not initialize_repository(self.config, str(repo_path)):
            return False
        
        # Check for export file in data folder
        if self.config.data_source.type == 'data_folder':
            data_path = Path(self.config.data_source.path)
            export_file = None
            
            # Look for export.yaml
            for possible_export in ['export.yaml', 'exports.yaml', '*/export.yaml']:
                export_candidates = list(data_path.glob(possible_export))
                if export_candidates:
                    export_file = export_candidates[0]
                    break
            
            if export_file:
                # Check if data already exists in the repository
                if self._check_data_exists(repo_path):
                    logger.info("Repository already contains data, skipping import")
                    return True
                
                # Use butler import
                logger.info(f"Found export file: {export_file}")
                ingestor = DataIngestor(str(repo_path), self.config)
                
                # Determine data directory (parent of export file or data path)
                if export_file.parent != data_path:
                    import_dir = export_file.parent
                else:
                    import_dir = data_path
                
                success = ingestor.import_from_export(str(export_file), str(import_dir))
                
                if success:
                    logger.info("Data import completed successfully")
                else:
                    logger.warning("Data import failed, trying manual ingestion")
                    return self._manual_data_ingestion(repo_path)
            else:
                # Manual ingestion
                return self._manual_data_ingestion(repo_path)
        
        return True
    
    def _manual_data_ingestion(self, repo_path: Path) -> bool:
        """
        Manually ingest data when no export file is available.
        
        Parameters
        ----------
        repo_path : Path
            Repository path
            
        Returns
        -------
        bool
            True if successful
        """
        logger.info("Starting manual data ingestion...")
        
        # Update config to point to data location
        if self.config.data_source.type == 'data_folder':
            # Set ingestion paths based on data folder structure
            data_path = Path(self.config.data_source.path)
            
            # Look for common data structures
            if (data_path / "raw").exists():
                self.config.ingestion.raw_data_pattern = "raw/**/*.fits"
            elif (data_path / "HSC" / "raw").exists():
                self.config.ingestion.raw_data_pattern = "HSC/raw/**/*.fits"
            
            if (data_path / "calib").exists():
                self.config.ingestion.calibration_path = str(data_path / "calib")
            elif (data_path / "HSC" / "calib").exists():
                self.config.ingestion.calibration_path = str(data_path / "HSC" / "calib")
            
            if (data_path / "refcats").exists():
                self.config.ingestion.reference_catalog_path = str(data_path / "refcats")
        
        # Run ingestion
        ingestor = DataIngestor(str(repo_path), self.config)
        results = ingestor.ingest_all()
        
        # Check results
        total_success = (
            results["raw_data"]["success"] or
            results["calibrations"]["success"] or
            results["reference_catalogs"]["success"]
        )
        
        if total_success:
            logger.info("Data ingestion completed with some successes")
            self._log_ingestion_summary(results)
        else:
            logger.error("All data ingestion operations failed")
            
        return total_success
    
    def _initialize_data_fetcher(self) -> None:
        """Initialize RIPPLe data fetcher for the repository."""
        if not LsstDataFetcher or not self.repo_path:
            return
        
        try:
            # Create data access configuration
            da_config = DataAccessButlerConfig(
                repo_path=str(self.repo_path),
                collections=self.config.data_source.collections or [
                    f"{self.config.instrument.name}/defaults",
                    f"{self.config.instrument.name}/raw/all",
                    f"{self.config.instrument.name}/calib",
                    "refcats"
                ],
                instrument=self.config.instrument.name,
                cache_size=self.config.processing.cache_size,
                enable_performance_monitoring=self.config.processing.enable_performance_monitoring,
                timeout=30.0,
                retry_attempts=3
            )
            
            self.data_fetcher = LsstDataFetcher(da_config)
            logger.info("Data fetcher initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize data fetcher: {e}")
            self.data_fetcher = None
    
    def get_data_fetcher(self) -> Optional[LsstDataFetcher]:
        """
        Get configured data fetcher instance.
        
        Returns
        -------
        Optional[LsstDataFetcher]
            Data fetcher instance if available
        """
        return self.data_fetcher
    
    def _log_ingestion_summary(self, results: Dict[str, Any]) -> None:
        """Log summary of ingestion results."""
        logger.info("\nIngestion Summary:")
        logger.info("-" * 40)
        
        if results["raw_data"]["count"] > 0:
            logger.info(f"Raw data: {results['raw_data']['count']} files ingested")
        
        if results["calibrations"]["count"] > 0:
            logger.info(f"Calibrations: {results['calibrations']['count']} files ingested")
        
        if results["reference_catalogs"]["count"] > 0:
            logger.info(f"Reference catalogs: {results['reference_catalogs']['count']} files ingested")
        
        if results["visits_defined"]:
            logger.info("Visits: Successfully defined")
        
        # Log any errors
        all_errors = []
        for category in ["raw_data", "calibrations", "reference_catalogs"]:
            if results[category]["errors"]:
                all_errors.extend(results[category]["errors"])
        
        if all_errors:
            logger.warning("\nErrors encountered:")
            for error in all_errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
            if len(all_errors) > 5:
                logger.warning(f"  ... and {len(all_errors) - 5} more errors")


def main():
    """Command-line interface for Butler repository management."""
    parser = argparse.ArgumentParser(
        description="RIPPLe Butler Repository Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create repository from config file
  python -m ripple.butler_repo.repo_manager config.yaml
  
  # Create repository with command-line options
  python -m ripple.butler_repo.repo_manager --data-path /path/to/data --instrument HSC
  
  # Generate default config
  python -m ripple.butler_repo.repo_manager --generate-config my_config.yaml
        """
    )
    
    parser.add_argument(
        "config",
        nargs="?",
        help="Configuration YAML file"
    )
    
    parser.add_argument(
        "--data-path",
        help="Path to data folder or Butler repository"
    )
    
    parser.add_argument(
        "--data-type",
        choices=["butler_repo", "data_folder"],
        default="data_folder",
        help="Type of data source"
    )
    
    parser.add_argument(
        "--instrument",
        choices=["HSC", "LSSTCam", "DECam"],
        help="Instrument name"
    )
    
    parser.add_argument(
        "--transfer",
        choices=["symlink", "copy", "move", "direct"],
        default="symlink",
        help="File transfer mode"
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
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate config if requested
    if args.generate_config:
        config = get_default_config()
        save_config(config, args.generate_config)
        print(f"Generated configuration file: {args.generate_config}")
        return 0
    
    # Load or create configuration
    if args.config:
        # Load from file
        try:
            config = load_config(args.config)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
    else:
        # Create from command-line arguments
        if not args.data_path:
            parser.error("Either config file or --data-path is required")
        
        config = get_default_config()
        config.data_source.type = args.data_type
        config.data_source.path = args.data_path
        
        if args.instrument:
            config.instrument.name = args.instrument
            # Set instrument class
            instrument_classes = {
                "HSC": "lsst.obs.subaru.HyperSuprimeCam",
                "LSSTCam": "lsst.obs.lsst.LsstCam",
                "DECam": "lsst.obs.decam.DarkEnergyCamera"
            }
            config.instrument.class_name = instrument_classes[args.instrument]
        
        if args.transfer:
            config.ingestion.transfer_mode = args.transfer
    
    # Create and run manager
    try:
        manager = ButlerRepoManager(config)
        success, result = manager.setup_repository()
        
        if success:
            logger.info(f"\nRepository successfully set up at: {result}")
            logger.info("\nYou can now use this repository with RIPPLe data access:")
            logger.info(f"  repo_path: {result}")
            return 0
        else:
            logger.error(f"\nRepository setup failed: {result}")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())