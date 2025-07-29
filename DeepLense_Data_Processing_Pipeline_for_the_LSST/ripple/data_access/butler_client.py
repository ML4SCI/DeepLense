"""
Butler Client wrapper for RIPPLe data access.

This module provides a wrapper around the LSST Butler with enhanced error
handling, retry logic, and support for both direct and client/server modes.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import time
from pathlib import Path

# LSST imports
from lsst.daf.butler import Butler, DatasetRef, DataIdValueError
from lsst.afw.image import Exposure
from lsst.skymap import BaseSkyMap
from lsst.geom import Box2I

# RIPPLe imports
from .exceptions import ButlerConnectionError, DataAccessError
from .collection_manager import CollectionManager

logger = logging.getLogger(__name__)


class ButlerClient:
    """
    Wrapper around LSST Butler with enhanced functionality.
    
    This class provides a high-level interface to the Butler with:
    - Automatic retry logic for failed operations
    - Support for both direct and client/server modes
    - Connection management and health checking
    - Performance monitoring and optimization
    """
    
    def __init__(self, config):
        """Initialize Butler client with configuration."""
        self.config = config
        self.butler: Optional[Butler] = None
        self.skymap: Optional[BaseSkyMap] = None
        self.is_remote = False
        self.collection_manager: Optional[CollectionManager] = None
        
        self._initialize_butler()
    
    def _initialize_butler(self) -> None:
        """Initialize Butler with appropriate configuration."""
        max_attempts = self.config.retry_attempts
        delay = self.config.retry_delay
        
        for attempt in range(max_attempts):
            try:
                # Check if we're using client/server mode
                if self.config.server_url:
                    # Client/server mode (for 2025+ deployment)
                    self._initialize_remote_butler()
                else:
                    # Direct Butler mode
                    self._initialize_direct_butler()
                
                # Test connection
                if self.test_connection():
                    logger.info(f"Butler initialized successfully on attempt {attempt + 1}")
                    return
                else:
                    raise ButlerConnectionError("Butler connection test failed")
                    
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Butler initialization attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise ButlerConnectionError(
                        f"Failed to initialize Butler after {max_attempts} attempts: {e}",
                        repo_path=self.config.repo_path,
                        server_url=self.config.server_url
                    )
    
    def _initialize_direct_butler(self) -> None:
        """Initialize direct Butler connection."""
        if not self.config.repo_path:
            raise ButlerConnectionError("repo_path is required for direct Butler")
        
        # Validate repository path
        repo_path = Path(self.config.repo_path)
        if not repo_path.exists():
            raise ButlerConnectionError(f"Repository path does not exist: {repo_path}")
        
        # Initialize Butler with collections
        self.butler = Butler(
            self.config.repo_path,
            collections=self.config.collections,
            instrument=self.config.instrument if self.config.instrument else None
        )
        
        # Cache skymap for coordinate conversion
        try:
            self.skymap = self.butler.get("skyMap")
        except Exception as e:
            logger.warning(f"Failed to load skymap: {e}")
            self.skymap = None
        
        self.is_remote = False
        
        # Initialize collection manager
        try:
            self.collection_manager = CollectionManager(self.butler)
        except Exception as e:
            logger.warning(f"Failed to initialize collection manager: {e}")
            self.collection_manager = None
        
        logger.info(f"Direct Butler initialized with repo: {self.config.repo_path}")
    
    def _initialize_remote_butler(self) -> None:
        """Initialize remote Butler connection (client/server mode)."""
        if not self.config.server_url:
            raise ButlerConnectionError("server_url is required for remote Butler")
        
        # TODO: Implement remote Butler initialization for 2025+ deployment
        # This will use the client/server architecture when available
        # For now, we'll use a placeholder that can be updated when the
        # LSST client/server Butler is fully deployed
        
        try:
            # Attempt to create remote Butler
            # This is a placeholder for future client/server implementation
            self.butler = Butler(
                server=self.config.server_url,
                collections=self.config.collections,
                instrument=self.config.instrument if self.config.instrument else None
            )
            
            # Cache skymap for coordinate conversion
            try:
                self.skymap = self.butler.get("skyMap")
            except Exception as e:
                logger.warning(f"Failed to load skymap from remote: {e}")
                self.skymap = None
            
            self.is_remote = True
            
            # Initialize collection manager
            try:
                self.collection_manager = CollectionManager(self.butler)
            except Exception as e:
                logger.warning(f"Failed to initialize collection manager: {e}")
                self.collection_manager = None
            
            logger.info(f"Remote Butler initialized with server: {self.config.server_url}")
            
        except Exception as e:
            # Fallback to direct Butler if remote fails
            logger.warning(f"Remote Butler initialization failed: {e}")
            if self.config.repo_path:
                logger.info("Falling back to direct Butler")
                self._initialize_direct_butler()
            else:
                raise ButlerConnectionError(
                    f"Remote Butler failed and no repo_path provided for fallback: {e}",
                    server_url=self.config.server_url
                )
    
    def test_connection(self) -> bool:
        """Test Butler connection health."""
        if not self.butler:
            return False
        
        try:
            # Test basic registry operations
            _ = list(self.butler.registry.queryCollections())
            _ = list(self.butler.registry.queryDatasetTypes())
            
            # Test datastore access if possible
            if hasattr(self.butler, 'datastore'):
                _ = self.butler.datastore.get_opaque_table_definitions()
            
            return True
            
        except Exception as e:
            logger.error(f"Butler connection test failed: {e}")
            return False
    
    def get_skymap(self) -> BaseSkyMap:
        """Get skymap for coordinate conversion."""
        if self.skymap is None:
            try:
                self.skymap = self.butler.get("skyMap")
            except Exception as e:
                raise DataAccessError(f"Failed to retrieve skymap: {e}")
        return self.skymap
    
    def query_available_datasets(self, tract: int, patch: str) -> Dict[str, Any]:
        """Query available datasets for tract/patch."""
        try:
            # Query available dataset types
            dataset_types = ['deepCoadd', 'calexp', 'objectTable', 'sourceCatalog']
            
            available_data = {
                'tract': tract,
                'patch': patch,
                'datasets': {},
                'filters': set()
            }
            
            for dataset_type in dataset_types:
                try:
                    # Query datasets for this tract/patch
                    # Use optimized collections if available
                    if self.collection_manager:
                        optimal_collections = self.collection_manager.get_optimal_collections_for_dataset(
                            dataset_type, self.config.collections
                        )
                        collections_to_use = optimal_collections
                    else:
                        collections_to_use = self.config.collections
                    
                    refs = list(self.butler.registry.queryDatasets(
                        dataset_type,
                        where=f"tract = {tract} AND patch = '{patch}'",
                        collections=collections_to_use
                    ))
                    
                    if refs:
                        available_data['datasets'][dataset_type] = {
                            'count': len(refs),
                            'refs': refs
                        }
                        
                        # Extract available filters
                        for ref in refs:
                            if 'band' in ref.dataId:
                                available_data['filters'].add(ref.dataId['band'])
                            elif 'filter' in ref.dataId:
                                available_data['filters'].add(ref.dataId['filter'])
                                
                except Exception as e:
                    logger.warning(f"Failed to query {dataset_type}: {e}")
            
            available_data['filters'] = list(available_data['filters'])
            return available_data
            
        except Exception as e:
            raise DataAccessError(f"Failed to query available datasets: {e}")
    
    def get_catalog(self, tract: int, patch: str, catalog_type: str, 
                   filters: Optional[List[str]] = None) -> Any:
        """Get catalog for specified tract/patch."""
        try:
            # Construct dataId for catalog
            dataId = {
                'tract': tract,
                'patch': patch
            }
            
            # Add filter if specified and supported
            if filters and len(filters) == 1:
                dataId['band'] = filters[0]
            
            # Retrieve catalog
            catalog = self.butler.get(catalog_type, dataId)
            return catalog
            
        except Exception as e:
            logger.warning(f"Failed to retrieve catalog {catalog_type}: {e}")
            return None
    
    def get_exposure(self, dataId: Dict[str, Any], dataset_type: str = "deepCoadd", 
                   bbox: Optional[Any] = None) -> Optional[Exposure]:
        """Get exposure with optional bbox parameter."""
        try:
            parameters = {}
            if bbox is not None:
                parameters['bbox'] = bbox
                parameters['origin'] = 'PARENT'
            
            if parameters:
                return self.butler.get(dataset_type, dataId, parameters=parameters)
            else:
                return self.butler.get(dataset_type, dataId)
                
        except Exception as e:
            logger.error(f"Failed to retrieve exposure: {e}")
            return None
    
    def get_wcs(self, dataId: Dict[str, Any], dataset_type: str = "deepCoadd") -> Optional[Any]:
        """Get WCS for specified dataset."""
        try:
            return self.butler.get(f"{dataset_type}.wcs", dataId)
        except Exception as e:
            logger.error(f"Failed to retrieve WCS: {e}")
            return None
    
    def get_psf(self, dataId: Dict[str, Any], dataset_type: str = "deepCoadd") -> Optional[Any]:
        """Get PSF for specified dataset."""
        try:
            return self.butler.get(f"{dataset_type}.psf", dataId)
        except Exception as e:
            logger.error(f"Failed to retrieve PSF: {e}")
            return None
    
    def get_mask(self, dataId: Dict[str, Any], dataset_type: str = "deepCoadd") -> Optional[Any]:
        """Get mask for specified dataset."""
        try:
            return self.butler.get(f"{dataset_type}.mask", dataId)
        except Exception as e:
            logger.error(f"Failed to retrieve mask: {e}")
            return None
    
    def dataset_exists(self, dataset_type: str, dataId: Dict[str, Any]) -> bool:
        """Check if dataset exists."""
        try:
            return self.butler.datasetExists(dataset_type, dataId, collections=self.config.collections)
        except Exception as e:
            logger.error(f"Failed to check dataset existence: {e}")
            return False
    
    def find_dataset(self, dataset_type: str, dataId: Dict[str, Any]) -> Optional[DatasetRef]:
        """Find dataset reference."""
        try:
            return self.butler.find_dataset(dataset_type, dataId, collections=self.config.collections)
        except Exception as e:
            logger.error(f"Failed to find dataset: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear cached skymap
            self.skymap = None
            
            # Close Butler connections if possible
            if hasattr(self.butler, 'close'):
                self.butler.close()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_batch_datasets(self, dataset_type: str, data_ids: List[Dict[str, Any]], 
                          max_workers: int = 4) -> List[Tuple[Dict[str, Any], Optional[Any]]]:
        """Retrieve multiple datasets in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        def fetch_single_dataset(data_id):
            try:
                data = self.butler.get(dataset_type, data_id)
                return (data_id, data)
            except Exception as e:
                logger.error(f"Failed to fetch dataset {data_id}: {e}")
                return (data_id, None)
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_data_id = {
                    executor.submit(fetch_single_dataset, data_id): data_id
                    for data_id in data_ids
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_data_id):
                    data_id, result = future.result()
                    results.append((data_id, result))
                    
        except Exception as e:
            logger.error(f"Batch dataset retrieval failed: {e}")
            
        return results
    
    def get_batch_cutouts(self, dataset_type: str, data_ids: List[Dict[str, Any]], 
                         bboxes: List[Box2I], max_workers: int = 4) -> List[Tuple[Dict[str, Any], Optional[Any]]]:
        """Retrieve multiple cutouts in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if len(data_ids) != len(bboxes):
            raise ValueError("Number of data_ids must match number of bboxes")
        
        results = []
        
        def fetch_single_cutout(data_id, bbox):
            try:
                parameters = {"bbox": bbox, "origin": "PARENT"}
                data = self.butler.get(dataset_type, data_id, parameters=parameters)
                return (data_id, data)
            except Exception as e:
                logger.error(f"Failed to fetch cutout {data_id}: {e}")
                return (data_id, None)
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_data_id = {
                    executor.submit(fetch_single_cutout, data_id, bbox): data_id
                    for data_id, bbox in zip(data_ids, bboxes)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_data_id):
                    data_id, result = future.result()
                    results.append((data_id, result))
                    
        except Exception as e:
            logger.error(f"Batch cutout retrieval failed: {e}")
            
        return results
    
    def stream_datasets(self, dataset_type: str, data_ids: List[Dict[str, Any]], 
                       batch_size: int = 10):
        """Stream datasets in batches to manage memory usage."""
        for i in range(0, len(data_ids), batch_size):
            batch = data_ids[i:i + batch_size]
            
            try:
                # Process batch
                for data_id in batch:
                    try:
                        data = self.butler.get(dataset_type, data_id)
                        yield (data_id, data)
                    except Exception as e:
                        logger.error(f"Failed to stream dataset {data_id}: {e}")
                        yield (data_id, None)
                        
            except Exception as e:
                logger.error(f"Streaming batch failed: {e}")
                break
    
    def get_deferred_datasets(self, dataset_type: str, data_ids: List[Dict[str, Any]]) -> List[Any]:
        """Get deferred dataset references for lazy loading."""
        try:
            refs = []
            for data_id in data_ids:
                ref = self.butler.find_dataset(dataset_type, data_id, collections=self.config.collections)
                if ref:
                    refs.append(ref)
            return refs
        except Exception as e:
            logger.error(f"Failed to get deferred datasets: {e}")
            return []