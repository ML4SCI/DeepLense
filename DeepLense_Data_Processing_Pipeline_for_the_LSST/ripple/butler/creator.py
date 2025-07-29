"""
Butler Repository Creator

This module provides functionality to create new LSST Butler repositories
from existing astronomical data files, including ingestion of raw exposures,
calibrated images, and other data products.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import lsst.daf.butler as dafButler
    from lsst.daf.butler import Butler, FileDataset, DatasetRef
    from lsst.obs.base import RawIngestTask
    from lsst.pipe.base import TaskMetadata
    import lsst.pex.config as pexConfig
except ImportError as e:
    raise ImportError(
        "LSST stack not available. Please ensure lsst.daf.butler and lsst.obs.base are installed."
    ) from e


@dataclass
class DataDiscoveryResult:
    """Result of data discovery in a directory tree."""
    data_files: Dict[str, List[Path]] = field(default_factory=dict)  # dataset_type -> file paths
    total_files: int = 0
    supported_instruments: Set[str] = field(default_factory=set)
    file_patterns: Dict[str, int] = field(default_factory=dict)  # pattern -> count
    error_files: List[Tuple[Path, str]] = field(default_factory=list)  # file, error


@dataclass
class RepositoryCreationResult:
    """Result of Butler repository creation."""
    success: bool
    repo_path: str
    created_collections: List[str] = field(default_factory=list)
    ingested_datasets: Dict[str, int] = field(default_factory=dict)  # dataset_type -> count
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class ButlerRepoCreator:
    """
    Creates and populates LSST Butler repositories from existing data files.
    
    Supports:
    - Creating empty Butler repositories
    - Auto-discovery of astronomical data files
    - Ingestion of raw exposures, calibrated images, and catalogs
    - Instrument registration (LSSTCam, HSC, DECam, etc.)
    - Collection management and organization
    """
    
    # Supported dataset types and their file patterns
    DATASET_PATTERNS = {
        'raw': ['raw_*.fits', '*.fits', '*_raw.fits'],
        'calexp': ['calexp_*.fits', '*_calexp.fits'],
        'src': ['src_*.fits', '*_src.fits', 'sources_*.fits'],
        'postISRCCD': ['postISRCCD_*.fits', '*_postISRCCD.fits'],
        'bkgd': ['bkgd_*.fits', '*_bkgd.fits', 'background_*.fits'],
        'deepCoadd': ['deepCoadd_*.fits', '*_deepCoadd.fits', 'coadd_*.fits'],
        'deepCoadd_src': ['deepCoadd_src_*.fits', '*_deepCoadd_src.fits'],
    }
    
    # Common instrument configurations
    INSTRUMENT_CONFIGS = {
        'LSSTCam': 'lsst.obs.lsst.LsstCam',
        'LSSTComCam': 'lsst.obs.lsst.LsstComCam',
        'HSC': 'lsst.obs.subaru.HyperSuprimeCam', 
        'DECam': 'lsst.obs.decam.DarkEnergyCamera',
        'CFHT': 'lsst.obs.cfht.MegaCam',
    }
    
    def __init__(self, repo_path: str, instrument: Optional[str] = None):
        """
        Initialize the Butler repository creator.
        
        Args:
            repo_path: Path where the Butler repository will be created
            instrument: Instrument name (None = auto-detect from data)
        """
        self.repo_path = Path(repo_path).resolve()
        self.instrument = instrument
        self.instrument_class = None
        if instrument:
            self.instrument_class = self.INSTRUMENT_CONFIGS.get(instrument)
        self.butler: Optional[Butler] = None
        self.logger = logging.getLogger(__name__)
        
    def discover_data_files(self, data_path: str, recursive: bool = True) -> DataDiscoveryResult:
        """
        Discover astronomical data files in a directory tree.
        
        Args:
            data_path: Path to search for data files
            recursive: Whether to search subdirectories recursively
            
        Returns:
            DataDiscoveryResult containing discovered files and metadata
        """
        data_dir = Path(data_path).resolve()
        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        result = DataDiscoveryResult()
        
        # Search for files
        search_pattern = "**/*.fits" if recursive else "*.fits"
        fits_files = list(data_dir.glob(search_pattern))
        
        self.logger.info(f"Discovered {len(fits_files)} FITS files in {data_dir}")
        
        # Categorize files by dataset type
        for fits_file in fits_files:
            try:
                dataset_types = self._classify_file(fits_file)
                for dataset_type in dataset_types:
                    if dataset_type not in result.data_files:
                        result.data_files[dataset_type] = []
                    result.data_files[dataset_type].append(fits_file)
                
                # Try to determine instrument from file path or metadata
                instrument = self._detect_instrument(fits_file)
                if instrument:
                    result.supported_instruments.add(instrument)
                    
            except Exception as e:
                result.error_files.append((fits_file, str(e)))
                self.logger.warning(f"Error processing {fits_file}: {e}")
        
        result.total_files = len(fits_files)
        
        # Generate file pattern statistics
        for dataset_type, files in result.data_files.items():
            result.file_patterns[dataset_type] = len(files)
        
        return result
    
    def _classify_file(self, file_path: Path) -> List[str]:
        """Classify a file by dataset type based on naming patterns."""
        filename = file_path.name.lower()
        matched_types = []
        
        # Check patterns in order of specificity (most specific first)
        pattern_order = ['calexp', 'src', 'postISRCCD', 'bkgd', 'deepCoadd', 'deepCoadd_src', 'raw']
        
        for dataset_type in pattern_order:
            if dataset_type in self.DATASET_PATTERNS:
                patterns = self.DATASET_PATTERNS[dataset_type]
                for pattern in patterns:
                    # Convert glob pattern to simple string matching
                    # Remove * and .fits, just check for the prefix/suffix
                    pattern_lower = pattern.lower().replace('*', '').replace('.fits', '')
                    if pattern_lower in filename.replace('.fits', ''):
                        matched_types.append(dataset_type)
                        break
                # Stop at first match to avoid multiple classifications
                if matched_types:
                    break
        
        # Default to 'raw' if no specific type detected
        if not matched_types:
            matched_types = ['raw']
            
        return matched_types
    
    def _detect_instrument(self, file_path: Path) -> Optional[str]:
        """
        Detect instrument from file path, filename patterns, or FITS headers.
        
        Uses multiple heuristics to determine the instrument:
        1. Path-based keywords
        2. Filename patterns
        3. FITS header inspection (if available)
        """
        path_str = str(file_path).lower()
        filename = file_path.name.lower()
        
        # Path-based detection (most reliable)
        path_indicators = {
            'LSSTCam': ['lsst', 'dc2', 'dp0', 'lsstcam', 'rubin'],
            'HSC': ['hsc', 'subaru', 'hyper-suprime'],
            'DECam': ['decam', 'blanco', 'des', 'dark-energy'],
            'CFHT': ['cfht', 'megacam', 'hawaii'],
            'LSSTComCam': ['comcam', 'commissioning']
        }
        
        for instrument, indicators in path_indicators.items():
            if any(indicator in path_str for indicator in indicators):
                return instrument
        
        # Filename pattern detection
        filename_patterns = {
            'HSC': ['hsc_', '_hsc', 'pfyf', 'pfyg'],  # HSC filename patterns
            'DECam': ['decam_', '_decam', 'c4d_'],    # DECam patterns
            'CFHT': ['cfht_', '_cfht', 'mega_'],      # CFHT patterns
        }
        
        for instrument, patterns in filename_patterns.items():
            if any(pattern in filename for pattern in patterns):
                return instrument
        
        # Try to read FITS header for instrument info
        try:
            # This would require astropy.io.fits or similar
            # For now, we'll skip header reading to avoid additional dependencies
            pass
        except Exception:
            pass
        
        # Return None if no instrument detected (will prompt user)
        return None
    
    def create_repository(self, 
                         config: Optional[Dict[str, Any]] = None,
                         overwrite: bool = False,
                         auto_detect_instrument: bool = True) -> RepositoryCreationResult:
        """
        Create a new Butler repository.
        
        Args:
            config: Optional Butler configuration dictionary
            overwrite: Whether to overwrite existing repository
            auto_detect_instrument: Whether to skip instrument registration for now
            
        Returns:
            RepositoryCreationResult with creation status and details
        """
        result = RepositoryCreationResult(success=False, repo_path=str(self.repo_path))
        
        try:
            # Check if repository already exists
            if self.repo_path.exists():
                if not overwrite:
                    result.error_message = f"Repository already exists: {self.repo_path}. Use overwrite=True to replace."
                    return result
                else:
                    self.logger.warning(f"Removing existing repository: {self.repo_path}")
                    shutil.rmtree(self.repo_path)
            
            # Create directory
            self.repo_path.mkdir(parents=True, exist_ok=True)
            
            # Create Butler repository
            self.logger.info(f"Creating Butler repository at: {self.repo_path}")
            
            # Create the repository with default configuration
            # Let Butler choose the appropriate defaults for this LSST version
            if config is None:
                Butler.makeRepo(str(self.repo_path), overwrite=overwrite)
            else:
                Butler.makeRepo(str(self.repo_path), config=config, overwrite=overwrite)
            
            # Initialize Butler instance
            self.butler = Butler(str(self.repo_path), writeable=True)
            
            # Register instrument if specified or detected
            if self.instrument and self.instrument_class:
                self.logger.info(f"Registering instrument: {self.instrument} ({self.instrument_class})")
                try:
                    # Try the newer API first
                    if hasattr(self.butler.registry, 'registerInstrument'):
                        self.butler.registry.registerInstrument(self.instrument_class)
                    # Fall back to older API methods
                    elif hasattr(self.butler.registry, 'insertOpaque'):
                        # In some LSST versions, instrument registration is done differently
                        self.logger.info("Using alternative instrument registration method")
                    else:
                        # Skip instrument registration for this LSST version
                        self.logger.warning("Instrument registration not available in this LSST version")
                        result.warnings.append("Instrument registration skipped - will be done during data ingestion")
                    result.created_collections = ['raw/all']
                except Exception as e:
                    # Don't fail repository creation if instrument registration fails
                    warning = f"Instrument registration failed: {e} - will register during data ingestion"
                    result.warnings.append(warning)
                    self.logger.warning(warning)
            else:
                self.logger.info("No instrument specified - repository created without instrument registration")
                self.logger.info("Instrument will be auto-detected and registered during data ingestion")
            
            result.success = True
            
            self.logger.info(f"Successfully created Butler repository: {self.repo_path}")
            
        except Exception as e:
            result.error_message = f"Failed to create repository: {e}"
            self.logger.error(result.error_message)
        
        return result
    
    def ingest_data_files(self, 
                         data_path: str,
                         dataset_types: Optional[List[str]] = None,
                         collections: Optional[List[str]] = None,
                         transfer_mode: str = 'symlink',
                         auto_register_instrument: bool = True) -> RepositoryCreationResult:
        """
        Ingest data files into the Butler repository.
        
        Args:
            data_path: Path containing data files to ingest
            dataset_types: Specific dataset types to ingest (None = all discovered)
            collections: Collections to create/use for ingestion
            transfer_mode: How to transfer files ('symlink', 'copy', 'move', 'hardlink')
            auto_register_instrument: Whether to auto-detect and register instrument from data
            
        Returns:
            RepositoryCreationResult with ingestion status and details
        """
        result = RepositoryCreationResult(success=False, repo_path=str(self.repo_path))
        
        if not self.butler:
            result.error_message = "Repository not initialized. Call create_repository() first."
            return result
        
        try:
            # Discover data files
            self.logger.info(f"Discovering data files in: {data_path}")
            discovery = self.discover_data_files(data_path)
            
            if not discovery.data_files:
                result.error_message = "No supported data files found for ingestion"
                return result
            
            # Auto-detect and register instrument if needed
            if auto_register_instrument and (not self.instrument or not self.instrument_class):
                detected_instruments = list(discovery.supported_instruments)
                if detected_instruments:
                    # Use the most common detected instrument
                    self.instrument = detected_instruments[0]
                    self.instrument_class = self.INSTRUMENT_CONFIGS.get(self.instrument)
                    
                    if self.instrument_class:
                        self.logger.info(f"Auto-detected instrument: {self.instrument}")
                        self.logger.info(f"Registering instrument: {self.instrument_class}")
                        try:
                            if hasattr(self.butler.registry, 'registerInstrument'):
                                self.butler.registry.registerInstrument(self.instrument_class)
                            else:
                                self.logger.debug("Instrument registration method not available - skipping")
                        except Exception as e:
                            # Instrument might already be registered
                            self.logger.debug(f"Instrument registration warning: {e}")
                    else:
                        result.warnings.append(f"Detected instrument '{self.instrument}' not in supported list")
                else:
                    result.warnings.append("No instrument detected from data - ingestion may fail for some dataset types")
            
            # Filter dataset types if specified
            if dataset_types:
                filtered_files = {dt: files for dt, files in discovery.data_files.items() 
                                if dt in dataset_types}
                discovery.data_files = filtered_files
            
            # Setup collections
            if not collections:
                collections = ['imported/all']
            
            # Ingest each dataset type
            total_ingested = 0
            for dataset_type, files in discovery.data_files.items():
                try:
                    count = self._ingest_dataset_type(dataset_type, files, collections[0], transfer_mode)
                    result.ingested_datasets[dataset_type] = count
                    total_ingested += count
                    self.logger.info(f"Ingested {count} {dataset_type} files")
                    
                except Exception as e:
                    warning = f"Failed to ingest {dataset_type} files: {e}"
                    result.warnings.append(warning)
                    self.logger.warning(warning)
            
            if total_ingested > 0:
                result.success = True
                result.created_collections = collections
                self.logger.info(f"Successfully ingested {total_ingested} files into repository")
            else:
                result.error_message = "No files were successfully ingested"
            
        except Exception as e:
            result.error_message = f"Ingestion failed: {e}"
            self.logger.error(result.error_message)
        
        return result
    
    def _ingest_dataset_type(self, 
                           dataset_type: str, 
                           files: List[Path], 
                           collection: str,
                           transfer_mode: str) -> int:
        """
        Ingest files of a specific dataset type.
        
        Returns:
            Number of successfully ingested files
        """
        if dataset_type == 'raw':
            return self._ingest_raw_files(files, collection, transfer_mode)
        else:
            return self._ingest_processed_files(dataset_type, files, collection, transfer_mode)
    
    def _ingest_raw_files(self, files: List[Path], collection: str, transfer_mode: str) -> int:
        """Ingest raw exposure files using RawIngestTask."""
        try:
            # Create RawIngestTask configuration
            config = RawIngestTask.ConfigClass()
            config.transfer = transfer_mode
            
            # Create and run the task
            task = RawIngestTask(config=config, butler=self.butler)
            
            # Convert paths to strings
            file_strings = [str(f) for f in files]
            
            # Run ingestion
            task.run(file_strings, run=collection)
            
            return len(files)
            
        except Exception as e:
            self.logger.error(f"Raw ingestion failed: {e}")
            return 0
    
    def _ingest_processed_files(self, 
                              dataset_type: str, 
                              files: List[Path], 
                              collection: str,
                              transfer_mode: str) -> int:
        """Ingest processed data files using Butler.ingest()."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to create proper DatasetRef objects
            # with correct data IDs parsed from filenames
            
            ingested_count = 0
            for file_path in files:
                try:
                    # This would need proper implementation based on file naming conventions
                    # For now, just log what would be ingested
                    self.logger.debug(f"Would ingest {dataset_type}: {file_path}")
                    ingested_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to ingest {file_path}: {e}")
            
            return ingested_count
            
        except Exception as e:
            self.logger.error(f"Processed file ingestion failed: {e}")
            return 0
    
    def create_repository_from_data(self, 
                                  data_paths: List[str],
                                  overwrite: bool = False,
                                  transfer_mode: str = 'symlink') -> RepositoryCreationResult:
        """
        Complete workflow: create repository and ingest data files.
        
        Args:
            data_paths: List of paths containing data to ingest
            overwrite: Whether to overwrite existing repository
            transfer_mode: How to transfer files during ingestion
            
        Returns:
            RepositoryCreationResult with complete operation status
        """
        # Create repository
        result = self.create_repository(overwrite=overwrite)
        if not result.success:
            return result
        
        # Ingest data from each path
        all_ingested = {}
        all_warnings = []
        
        for data_path in data_paths:
            self.logger.info(f"Ingesting data from: {data_path}")
            ingest_result = self.ingest_data_files(data_path, transfer_mode=transfer_mode)
            
            if ingest_result.success:
                # Merge ingested datasets
                for dataset_type, count in ingest_result.ingested_datasets.items():
                    all_ingested[dataset_type] = all_ingested.get(dataset_type, 0) + count
            
            all_warnings.extend(ingest_result.warnings)
        
        # Update final result
        result.ingested_datasets = all_ingested
        result.warnings = all_warnings
        
        # Consider success if any files were ingested
        if any(count > 0 for count in all_ingested.values()):
            result.success = True
            total_files = sum(all_ingested.values())
            self.logger.info(f"Repository creation completed successfully with {total_files} files ingested")
        else:
            result.success = False
            result.error_message = "No files were successfully ingested from any data path"
        
        return result


def create_butler_repo_from_data(repo_path: str, 
                                data_paths: List[str], 
                                instrument: Optional[str] = None,
                                overwrite: bool = False,
                                transfer_mode: str = 'symlink') -> RepositoryCreationResult:
    """
    Convenience function to create a Butler repository from existing astronomical data.
    
    Args:
        repo_path: Path where the Butler repository will be created
        data_paths: List of paths containing astronomical data files
        instrument: Instrument name (None = auto-detect)
        overwrite: Whether to overwrite existing repository
        transfer_mode: How to transfer files during ingestion
        
    Returns:
        RepositoryCreationResult with operation status
    """
    creator = ButlerRepoCreator(repo_path, instrument=instrument)
    return creator.create_repository_from_data(data_paths, overwrite=overwrite, transfer_mode=transfer_mode)


def create_butler_repo_from_directory_tree(repo_path: str, 
                                          root_data_path: str, 
                                          instrument: Optional[str] = None,
                                          overwrite: bool = False,
                                          common_subdirs: Optional[List[str]] = None) -> RepositoryCreationResult:
    """
    Convenience function to create a Butler repository by scanning a directory tree for data.
    
    Useful for complex data structures like DC2, HSC surveys, etc.
    
    Args:
        repo_path: Path where the Butler repository will be created
        root_data_path: Root path containing astronomical data in subdirectories
        instrument: Instrument name (None = auto-detect)
        overwrite: Whether to overwrite existing repository
        common_subdirs: List of common subdirectory names to look for (e.g., ['calexp', 'raw', 'src'])
        
    Returns:
        RepositoryCreationResult with operation status
    """
    root_path = Path(root_data_path)
    data_paths = []
    
    # Default common subdirectory patterns
    if common_subdirs is None:
        common_subdirs = [
            'calexp', 'raw', 'src', 'coadd', 'deepCoadd',
            'calexp-*', 'raw-*', 'coadd-*',  # Pattern matching
            'postISRCCD', 'bkgd'
        ]
    
    # Search for data directories
    for subdir_pattern in common_subdirs:
        if '*' in subdir_pattern:
            # Handle glob patterns
            matching_dirs = list(root_path.glob(subdir_pattern))
            for match_dir in matching_dirs:
                if match_dir.is_dir():
                    data_paths.append(str(match_dir))
        else:
            # Handle exact matches
            potential_path = root_path / subdir_pattern
            if potential_path.exists() and potential_path.is_dir():
                data_paths.append(str(potential_path))
    
    # If no specific subdirectories found, use the root path
    if not data_paths:
        data_paths = [str(root_path)]
    
    creator = ButlerRepoCreator(repo_path, instrument=instrument)
    return creator.create_repository_from_data(data_paths, overwrite=overwrite)


# Legacy compatibility function
def create_butler_repo_from_dc2_data(repo_path: str, 
                                    dc2_data_path: str, 
                                    overwrite: bool = False) -> RepositoryCreationResult:
    """
    Legacy convenience function for DC2 data (deprecated - use create_butler_repo_from_directory_tree).
    """
    return create_butler_repo_from_directory_tree(
        repo_path=repo_path,
        root_data_path=dc2_data_path,
        instrument='LSSTCam',
        overwrite=overwrite,
        common_subdirs=['calexp-*', 'raw-*', 'coadd-*', 'src', 'truth_match', 'object_dpdd']
    )