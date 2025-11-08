"""
Butler Repository Validator

This module provides functionality to validate and discover LSST Butler repositories,
including auto-discovery of collections, dataset types, and data coverage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    import lsst.daf.butler as dafButler
    from lsst.daf.butler import Butler, CollectionType
except ImportError as e:
    raise ImportError(
        "LSST stack not available. Please ensure lsst.daf.butler is installed."
    ) from e


@dataclass
class DataProductInfo:
    """Information about a specific data product type."""
    dataset_type: str
    available_count: int
    total_possible: int
    coverage_percentage: float
    collections: List[str]
    dimensions: List[str]
    sample_data_ids: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Report of data coverage for specific selection criteria."""
    total_data_ids: int
    available_data_ids: int
    missing_data_ids: int
    coverage_percentage: float
    instruments: Set[str]
    filters: Set[str]
    visit_ranges: List[Tuple[int, int]]
    detector_ranges: List[Tuple[int, int]]
    failed_data_ids: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of Butler repository validation."""
    is_valid: bool
    repo_path: str
    error_message: Optional[str] = None
    collections: List[str] = field(default_factory=list)
    dataset_types: List[str] = field(default_factory=list)
    instruments: Set[str] = field(default_factory=set)
    data_products: Dict[str, DataProductInfo] = field(default_factory=dict)
    total_collections: int = 0
    total_dataset_types: int = 0


class ButlerRepoValidator:
    """
    Validates and discovers information about LSST Butler repositories.
    
    Provides functionality for:
    - Basic repository validation
    - Auto-discovery of collections and dataset types
    - Data product availability mapping
    - Coverage analysis for different selection criteria
    """
    
    def __init__(self, repo_path: str):
        """
        Initialize the validator.
        
        Args:
            repo_path: Path to the Butler repository
        """
        self.repo_path = Path(repo_path).resolve()
        self.butler: Optional[Butler] = None
        self.logger = logging.getLogger(__name__)
        
    def validate_repository(self) -> ValidationResult:
        """
        Perform comprehensive repository validation.
        
        Returns:
            ValidationResult containing validation status and discovered information
        """
        try:
            # Step 1: Basic validation
            if not self._validate_basic_structure():
                return ValidationResult(
                    is_valid=False,
                    repo_path=str(self.repo_path),
                    error_message="Invalid Butler repository structure"
                )
            
            # Step 2: Initialize Butler
            if not self._initialize_butler():
                return ValidationResult(
                    is_valid=False,
                    repo_path=str(self.repo_path),
                    error_message="Failed to initialize Butler"
                )
            
            # Step 3: Discover repository contents
            collections = self.discover_collections()
            dataset_types = self.discover_dataset_types()
            instruments = self.discover_instruments()
            
            # Step 4: Analyze data products
            data_products = self.discover_data_products()
            
            return ValidationResult(
                is_valid=True,
                repo_path=str(self.repo_path),
                collections=collections,
                dataset_types=dataset_types,
                instruments=instruments,
                data_products=data_products,
                total_collections=len(collections),
                total_dataset_types=len(dataset_types)
            )
            
        except Exception as e:
            self.logger.error(f"Repository validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                repo_path=str(self.repo_path),
                error_message=str(e)
            )
    
    def _validate_basic_structure(self) -> bool:
        """Validate basic Butler repository structure."""
        if not self.repo_path.exists():
            self.logger.error(f"Repository path does not exist: {self.repo_path}")
            return False
        
        butler_yaml = self.repo_path / "butler.yaml"
        if not butler_yaml.exists():
            self.logger.error(f"butler.yaml not found in {self.repo_path}")
            return False
        
        return True
    
    def _initialize_butler(self) -> bool:
        """Initialize Butler connection."""
        try:
            self.butler = Butler(str(self.repo_path))
            self.logger.info(f"Successfully initialized Butler for {self.repo_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Butler: {e}")
            return False
    
    def discover_collections(self) -> List[str]:
        """
        Auto-discover all available collections in the repository.
        
        Returns:
            List of collection names
        """
        if not self.butler:
            return []
        
        try:
            collections = []
            for collection_type in [CollectionType.RUN, CollectionType.TAGGED, CollectionType.CHAINED]:
                collection_names = list(self.butler.registry.queryCollections(
                    collectionTypes={collection_type}
                ))
                collections.extend(collection_names)
            
            self.logger.info(f"Discovered {len(collections)} collections")
            return sorted(collections)
            
        except Exception as e:
            self.logger.error(f"Failed to discover collections: {e}")
            return []
    
    def discover_dataset_types(self) -> List[str]:
        """
        Discover all available dataset types in the repository.
        
        Returns:
            List of dataset type names
        """
        if not self.butler:
            return []
        
        try:
            dataset_types = list(self.butler.registry.queryDatasetTypes())
            dataset_type_names = [dt.name for dt in dataset_types]
            
            self.logger.info(f"Discovered {len(dataset_type_names)} dataset types")
            return sorted(dataset_type_names)
            
        except Exception as e:
            self.logger.error(f"Failed to discover dataset types: {e}")
            return []
    
    def discover_instruments(self) -> Set[str]:
        """
        Discover all instruments present in the repository.
        
        Returns:
            Set of instrument names
        """
        if not self.butler:
            return set()
        
        try:
            instruments = set()
            for record in self.butler.registry.queryDimensionRecords("instrument"):
                instruments.add(record.name)
            
            self.logger.info(f"Discovered instruments: {instruments}")
            return instruments
            
        except Exception as e:
            self.logger.error(f"Failed to discover instruments: {e}")
            return set()
    
    def discover_data_products(self, collections: Optional[List[str]] = None) -> Dict[str, DataProductInfo]:
        """
        Discover and analyze availability of data products.
        
        Args:
            collections: Specific collections to analyze. If None, uses all discovered collections.
            
        Returns:
            Dictionary mapping dataset types to their availability information
        """
        if not self.butler:
            return {}
        
        if collections is None:
            collections = self.discover_collections()
        
        data_products = {}
        
        # Focus on key dataset types for LSST-DeepLense pipeline
        key_dataset_types = [
            "raw",              # Raw exposures (starting point)
            "calexp",           # Calibrated exposures
            "src",              # Source catalogs
            "postISRCCD",       # Post-ISR images
            "deepCoadd",        # Deep coadded images
            "deepCoadd_src",    # Deep coadd source catalogs
            "objectTable",      # Object tables
            "visitTable"        # Visit tables
        ]
        
        for dataset_type in key_dataset_types:
            try:
                info = self._analyze_dataset_type(dataset_type, collections)
                if info.available_count > 0:
                    data_products[dataset_type] = info
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze dataset type {dataset_type}: {e}")
        
        return data_products
    
    def _analyze_dataset_type(self, dataset_type: str, collections: List[str]) -> DataProductInfo:
        """Analyze availability of a specific dataset type."""
        try:
            # First check if this dataset type exists in the registry
            try:
                dataset_type_obj = self.butler.registry.getDatasetType(dataset_type)
                dimensions = list(dataset_type_obj.dimensions.names)
            except Exception as e:
                self.logger.warning(f"Dataset type '{dataset_type}' not found in registry: {e}")
                return DataProductInfo(
                    dataset_type=dataset_type,
                    available_count=0,
                    total_possible=0,
                    coverage_percentage=0.0,
                    collections=[],
                    dimensions=[]
                )
            
            # Query available data IDs for this dataset type
            data_ids = list(self.butler.registry.queryDataIds(
                dataset_type_obj.dimensions,
                datasets=dataset_type_obj,
                collections=collections
            ))
            
            # Sample a few data IDs for reference
            sample_size = min(5, len(data_ids))
            sample_data_ids = [dict(data_id.mapping) for data_id in data_ids[:sample_size]]
            
            # Determine which collections have this dataset type
            available_collections = []
            for collection in collections:
                try:
                    test_query = list(self.butler.registry.queryDataIds(
                        dataset_type_obj.dimensions,
                        datasets=dataset_type_obj,
                        collections=[collection]
                    ))
                    if test_query:
                        available_collections.append(collection)
                except Exception as collection_error:
                    self.logger.debug(f"Collection {collection} doesn't have {dataset_type}: {collection_error}")
                    continue
            
            return DataProductInfo(
                dataset_type=dataset_type,
                available_count=len(data_ids),
                total_possible=len(data_ids),  # We don't know the theoretical maximum
                coverage_percentage=100.0 if len(data_ids) > 0 else 0.0,
                collections=available_collections,
                dimensions=dimensions,
                sample_data_ids=sample_data_ids
            )
            
        except Exception as e:
            self.logger.warning(f"Error analyzing {dataset_type}: {e}")
            return DataProductInfo(
                dataset_type=dataset_type,
                available_count=0,
                total_possible=0,
                coverage_percentage=0.0,
                collections=[],
                dimensions=[]
            )
    
    def get_data_coverage(
        self,
        dataset_types: List[str],
        collections: Optional[List[str]] = None,
        filters: Optional[List[str]] = None,
        visit_ranges: Optional[List[Tuple[int, int]]] = None,
        detector_ranges: Optional[List[Tuple[int, int]]] = None,
        instruments: Optional[List[str]] = None
    ) -> CoverageReport:
        """
        Generate a coverage report for specific selection criteria.
        
        Args:
            dataset_types: List of dataset types to check
            collections: Collections to search in
            filters: Photometric filters to include
            visit_ranges: List of (min_visit, max_visit) tuples
            detector_ranges: List of (min_detector, max_detector) tuples
            instruments: List of instrument names (required for governor dimensions)
            
        Returns:
            CoverageReport with availability statistics
        """
        if not self.butler:
            return CoverageReport(0, 0, 0, 0.0, set(), set(), [], [])
        
        if collections is None:
            collections = self.discover_collections()
        
        try:
            # Build query constraints
            where_clauses = []
            bind_params = {}
            
            # Add instrument constraint (required for governor dimensions)
            if instruments:
                # Use the first instrument as governor dimension constraint
                where_clauses.append(f"instrument = '{instruments[0]}'")
            
            if filters:
                where_clauses.append("band IN (band_list)")
                bind_params["band_list"] = filters
            
            if visit_ranges:
                visit_constraints = []
                for i, (min_visit, max_visit) in enumerate(visit_ranges):
                    visit_constraints.append(f"(visit >= min_visit_{i} AND visit <= max_visit_{i})")
                    bind_params[f"min_visit_{i}"] = min_visit
                    bind_params[f"max_visit_{i}"] = max_visit
                if visit_constraints:
                    where_clauses.append(f"({' OR '.join(visit_constraints)})")
            
            where_clause = " AND ".join(where_clauses) if where_clauses else ""
            
            # Query data IDs for the primary dataset type
            primary_dataset = dataset_types[0]
            try:
                primary_dataset_obj = self.butler.registry.getDatasetType(primary_dataset)
            except Exception as e:
                self.logger.error(f"Primary dataset type '{primary_dataset}' not found: {e}")
                return CoverageReport(0, 0, 0, 0.0, set(), set(), [], [])
            
            all_data_ids = list(self.butler.registry.queryDataIds(
                primary_dataset_obj.dimensions,
                datasets=primary_dataset_obj,
                collections=collections,
                where=where_clause,
                bind=bind_params
            ))
            
            # Check availability for each dataset type
            available_count = 0
            failed_data_ids = []
            instruments_found = set()
            filters_found = set()
            
            for data_id in all_data_ids:
                data_id_dict = dict(data_id.mapping)
                
                # Collect metadata
                if 'instrument' in data_id_dict:
                    instruments_found.add(data_id_dict['instrument'])
                if 'band' in data_id_dict:
                    filters_found.add(data_id_dict['band'])
                
                # Check if all required dataset types exist for this data ID
                all_exist = True
                for dataset_type in dataset_types:
                    try:
                        dataset_type_obj = self.butler.registry.getDatasetType(dataset_type)
                        refs = list(self.butler.registry.queryDatasets(
                            dataset_type_obj,
                            collections=collections,
                            dataId=data_id
                        ))
                        if not refs:
                            all_exist = False
                            break
                    except Exception as e:
                        self.logger.debug(f"Dataset type {dataset_type} check failed for {data_id_dict}: {e}")
                        all_exist = False
                        break
                
                if all_exist:
                    available_count += 1
                else:
                    failed_data_ids.append(data_id_dict)
            
            total_data_ids = len(all_data_ids)
            missing_count = total_data_ids - available_count
            coverage_pct = (available_count / total_data_ids * 100.0) if total_data_ids > 0 else 0.0
            
            return CoverageReport(
                total_data_ids=total_data_ids,
                available_data_ids=available_count,
                missing_data_ids=missing_count,
                coverage_percentage=coverage_pct,
                instruments=instruments_found,
                filters=filters_found,
                visit_ranges=visit_ranges or [],
                detector_ranges=detector_ranges or [],
                failed_data_ids=failed_data_ids
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate coverage report: {e}")
            return CoverageReport(0, 0, 0, 0.0, set(), set(), [], [])