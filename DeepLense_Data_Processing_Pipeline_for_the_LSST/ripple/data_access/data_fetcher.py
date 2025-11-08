"""
LsstDataFetcher: Main interface for LSST data retrieval

This module provides the primary interface for retrieving LSST astronomical
data using Butler Gen3 architecture with optimized performance, comprehensive
error handling, and support for both direct and client/server configurations.
"""

from typing import (
    Dict, List, Optional, Union, Tuple, Any, 
    AsyncGenerator, Generator, Callable
)
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import numpy as np

# LSST imports
from lsst.daf.butler import Butler, DatasetRef, DataIdValueError
from lsst.afw.image import Exposure, ExposureF
from lsst.geom import Box2I, Point2I, Extent2I, SpherePoint, degrees, Point2D
from lsst.afw.table import SourceCatalog

# RIPPLe imports
from .exceptions import (
    DataAccessError, ButlerConnectionError, DataIdValidationError,
    CutoutExtractionError, CollectionError, CoordinateConversionError,
    PerformanceError
)
from .butler_client import ButlerClient
from .coordinate_utils import CoordinateConverter
from .cache_manager import CacheManager
from .retry_utils import RetryManager, RetryConfig, CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


@dataclass
class ButlerConfig:
    """Configuration for Butler initialization."""
    repo_path: Optional[str] = None
    server_url: Optional[str] = None
    collections: List[str] = field(default_factory=list)
    instrument: str = "LSSTCam-imSim"
    max_connections: int = 10
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    cache_size: int = 1000
    enable_performance_monitoring: bool = True


@dataclass
class CutoutRequest:
    """Request parameters for cutout extraction."""
    ra: float
    dec: float
    size: int
    filters: List[str] = field(default_factory=lambda: ['r'])
    dataset_type: str = "deepCoadd"
    skymap: str = "DC2"
    quality_threshold: float = 0.5


@dataclass
class PerformanceMetrics:
    """Performance metrics for data access operations."""
    operation: str
    duration: float
    memory_usage: int
    success: bool
    error_message: Optional[str] = None
    cache_hit: bool = False


class LsstDataFetcher:
    """
    Main interface for LSST data retrieval using Butler Gen3.
    
    This class provides high-level methods for retrieving LSST images, catalogs,
    and metadata with optimized performance, comprehensive error handling, and
    support for both direct Butler and client/server configurations.
    
    Key Features:
    - Efficient cutout extraction with bbox parameters
    - Multi-band synchronized data retrieval
    - Comprehensive error handling and retry logic
    - Performance monitoring and optimization
    - Flexible configuration for different deployment scenarios
    - Memory-efficient batch processing
    - Automatic coordinate conversion (RA/Dec to tract/patch)
    """
    
    def __init__(self, config: ButlerConfig):
        """
        Initialize the LsstDataFetcher with configuration.
        
        Parameters
        ----------
        config : ButlerConfig
            Configuration object containing Butler settings and parameters
            
        Raises
        ------
        ButlerConnectionError
            If Butler initialization fails
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize core components
        self.butler_client: Optional[ButlerClient] = None
        self.coordinate_converter: Optional[CoordinateConverter] = None
        self.cache_manager: Optional[CacheManager] = None
        
        # Performance monitoring
        self.performance_metrics: List[PerformanceMetrics] = []
        self.start_time = time.time()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all core components."""
        try:
            # Initialize Butler client
            self.butler_client = ButlerClient(self.config)
            
            # Initialize coordinate converter
            try:
                skymap = self.butler_client.get_skymap()
                self.coordinate_converter = CoordinateConverter(skymap)
            except Exception as e:
                logger.warning(f"Could not initialize coordinate converter with skymap: {e}")
                self.coordinate_converter = None
            
            # Initialize cache manager
            if self.config.cache_size > 0:
                self.cache_manager = CacheManager(self.config.cache_size)
            
            self.logger.info(f"LsstDataFetcher initialized successfully")
            
        except Exception as e:
            raise ButlerConnectionError(
                f"Failed to initialize LsstDataFetcher: {e}",
                repo_path=self.config.repo_path,
                server_url=self.config.server_url
            )
    
    # Primary Public API Methods
    
    def fetch_cutout(self, 
                    ra: float, 
                    dec: float, 
                    size: int, 
                    filters: List[str] = None,
                    dataset_type: str = "deepCoadd",
                    quality_check: bool = True) -> Union[Exposure, Dict[str, Exposure]]:
        """
        Fetch image cutout(s) at specified coordinates.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        size : int
            Cutout size in pixels
        filters : List[str], optional
            List of filters to retrieve (default: ['r'])
        dataset_type : str, optional
            Dataset type to retrieve (default: 'deepCoadd')
        quality_check : bool, optional
            Whether to perform quality validation (default: True)
            
        Returns
        -------
        Union[Exposure, Dict[str, Exposure]]
            Single Exposure if one filter, dict of Exposures if multiple
            
        Raises
        ------
        CutoutExtractionError
            If cutout extraction fails
        DataIdValidationError
            If coordinates are invalid
        """
        if filters is None:
            filters = ['r']
        
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_coordinates(ra, dec)
            self._validate_cutout_size(size)
            
            # Convert coordinates to tract/patch
            tract, patch = self.coordinate_converter.radec_to_tract_patch(ra, dec)
            
            # Single filter case
            if len(filters) == 1:
                return self._fetch_single_cutout(
                    ra, dec, size, filters[0], dataset_type, tract, patch, quality_check
                )
            
            # Multi-filter case
            return self._fetch_multiband_cutout(
                ra, dec, size, filters, dataset_type, tract, patch, quality_check
            )
            
        except Exception as e:
            self._record_performance_metric(
                "fetch_cutout", time.time() - start_time, 0, False, str(e)
            )
            raise CutoutExtractionError(
                f"Failed to fetch cutout at ({ra}, {dec}): {e}",
                ra=ra, dec=dec, size=size
            )
    
    def fetch_batch_cutouts(self,
                           requests: List[CutoutRequest],
                           batch_size: int = 32,
                           max_workers: int = 4,
                           progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Fetch multiple cutouts in batches for optimal performance.
        
        Parameters
        ----------
        requests : List[CutoutRequest]
            List of cutout requests
        batch_size : int, optional
            Number of cutouts to process simultaneously (default: 32)
        max_workers : int, optional
            Maximum number of worker threads (default: 4)
        progress_callback : Optional[Callable], optional
            Callback function for progress updates
            
        Returns
        -------
        List[Dict[str, Any]]
            List of results with metadata
        """
        start_time = time.time()
        results = []
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Process in batches
                for i in range(0, len(requests), batch_size):
                    batch = requests[i:i+batch_size]
                    
                    # Submit batch for processing
                    futures = {
                        executor.submit(self._process_single_request, req): req
                        for req in batch
                    }
                    
                    # Collect results
                    for future in as_completed(futures):
                        req = futures[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if progress_callback:
                                progress_callback(len(results), len(requests))
                                
                        except Exception as e:
                            self.logger.error(f"Failed to process request {req}: {e}")
                            results.append({
                                'request': req,
                                'success': False,
                                'error': str(e)
                            })
            
            self._record_performance_metric(
                "fetch_batch_cutouts", time.time() - start_time, 0, True
            )
            
            return results
            
        except Exception as e:
            self._record_performance_metric(
                "fetch_batch_cutouts", time.time() - start_time, 0, False, str(e)
            )
            raise DataAccessError(f"Batch processing failed: {e}")
    
    def get_available_data(self,
                          ra: float,
                          dec: float,
                          radius: Optional[float] = None) -> Dict[str, Any]:
        """
        Query available data products for given coordinates.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        radius : float, optional
            Search radius in degrees (default: None for single tract/patch)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing available datasets, filters, and metadata
        """
        try:
            # Convert coordinates to tract/patch
            if radius is None:
                tract, patch = self.coordinate_converter.radec_to_tract_patch(ra, dec)
                tracts_patches = [(tract, patch)]
            else:
                tracts_patches = self.coordinate_converter.radec_to_tract_patch_radius(
                    ra, dec, radius
                )
            
            # Query available datasets
            available_data = {
                'coordinates': {'ra': ra, 'dec': dec},
                'tracts_patches': tracts_patches,
                'datasets': {},
                'filters': set(),
                'total_datasets': 0
            }
            
            for tract, patch in tracts_patches:
                data_info = self.butler_client.query_available_datasets(tract, patch)
                available_data['datasets'][f"{tract}_{patch}"] = data_info
                available_data['filters'].update(data_info.get('filters', []))
                available_data['total_datasets'] += len(data_info.get('datasets', []))
            
            available_data['filters'] = sorted(list(available_data['filters']))
            
            return available_data
            
        except Exception as e:
            raise DataAccessError(f"Failed to query available data: {e}")
    
    def fetch_catalog(self,
                     ra: float,
                     dec: float,
                     radius: float,
                     catalog_type: str = "objectTable",
                     filters: Optional[List[str]] = None) -> SourceCatalog:
        """
        Fetch source catalog for specified region.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        radius : float
            Search radius in degrees
        catalog_type : str, optional
            Type of catalog to retrieve (default: 'objectTable')
        filters : List[str], optional
            Filters to include in catalog
            
        Returns
        -------
        SourceCatalog
            LSST source catalog
        """
        try:
            # Find all tract/patch combinations in radius
            tracts_patches = self.coordinate_converter.radec_to_tract_patch_radius(
                ra, dec, radius
            )
            
            # Fetch catalogs from all relevant tract/patch combinations
            catalogs = []
            for tract, patch in tracts_patches:
                catalog = self.butler_client.get_catalog(
                    tract, patch, catalog_type, filters
                )
                if catalog is not None:
                    catalogs.append(catalog)
            
            # Merge catalogs if multiple
            if len(catalogs) == 0:
                raise DataAccessError("No catalogs found for specified region")
            elif len(catalogs) == 1:
                return catalogs[0]
            else:
                return self._merge_catalogs(catalogs)
                
        except Exception as e:
            raise DataAccessError(f"Failed to fetch catalog: {e}")
    
    # Configuration and Management Methods
    
    def get_performance_metrics(self) -> List[PerformanceMetrics]:
        """Get recorded performance metrics."""
        return self.performance_metrics.copy()
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache_manager:
            self.cache_manager.clear()
            self.logger.info("Cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if self.cache_manager:
            return self.cache_manager.get_statistics()
        return {}
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return status."""
        validation_results = {
            'butler_connection': False,
            'coordinate_converter': False,
            'cache_manager': False,
            'performance_monitoring': False,
            'errors': []
        }
        
        try:
            # Test Butler connection
            if self.butler_client and self.butler_client.test_connection():
                validation_results['butler_connection'] = True
            else:
                validation_results['errors'].append("Butler connection failed")
            
            # Test coordinate converter
            if self.coordinate_converter:
                validation_results['coordinate_converter'] = True
            else:
                validation_results['errors'].append("Coordinate converter not initialized")
            
            # Test cache manager
            if self.cache_manager:
                validation_results['cache_manager'] = True
            else:
                validation_results['errors'].append("Cache manager not initialized")
            
            # Test performance monitoring
            if self.config.enable_performance_monitoring:
                validation_results['performance_monitoring'] = True
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {e}")
        
        return validation_results
    
    # Private Helper Methods
    
    def _fetch_single_cutout(self, ra: float, dec: float, size: int, 
                           filter_name: str, dataset_type: str, 
                           tract: int, patch: str, quality_check: bool) -> Exposure:
        """Fetch a single cutout from specified tract/patch."""
        try:
            # Construct dataId
            dataId = {
                'tract': tract,
                'patch': patch,
                'band': filter_name
            }
            
            # Check if dataset exists
            if not self.butler_client.dataset_exists(dataset_type, dataId):
                raise DataAccessError(f"Dataset {dataset_type} not found for {dataId}")
            
            # Get WCS for coordinate conversion
            wcs = self.butler_client.get_wcs(dataId, dataset_type)
            if wcs is None:
                raise DataAccessError(f"WCS not available for {dataId}")
            
            # Calculate bbox using coordinate converter
            bbox = self.coordinate_converter.calculate_bbox_from_radec(ra, dec, size, wcs)
            
            # Fetch cutout with bbox
            cutout = self.butler_client.get_exposure(dataId, dataset_type, bbox)
            if cutout is None:
                raise DataAccessError(f"Failed to retrieve cutout for {dataId}")
            
            # Perform quality check if requested
            if quality_check:
                if not self._validate_cutout_quality(cutout, size):
                    raise DataAccessError(f"Cutout quality check failed for {dataId}")
            
            return cutout
            
        except Exception as e:
            raise CutoutExtractionError(
                f"Failed to fetch single cutout: {e}",
                ra=ra, dec=dec, size=size
            )
    
    def _fetch_multiband_cutout(self, ra: float, dec: float, size: int,
                              filters: List[str], dataset_type: str,
                              tract: int, patch: str, quality_check: bool) -> Dict[str, Exposure]:
        """Fetch synchronized multi-band cutouts."""
        try:
            cutouts = {}
            failed_filters = []
            
            # Get reference WCS (from first available filter)
            ref_wcs = None
            for filter_name in filters:
                try:
                    dataId = {'tract': tract, 'patch': patch, 'band': filter_name}
                    if self.butler_client.dataset_exists(dataset_type, dataId):
                        ref_wcs = self.butler_client.get_wcs(dataId, dataset_type)
                        break
                except Exception:
                    continue
            
            if ref_wcs is None:
                raise DataAccessError(f"No WCS available for any filter in {filters}")
            
            # Calculate bbox once using reference WCS
            bbox = self.coordinate_converter.calculate_bbox_from_radec(ra, dec, size, ref_wcs)
            
            # Fetch cutouts for each filter
            for filter_name in filters:
                try:
                    dataId = {
                        'tract': tract,
                        'patch': patch,
                        'band': filter_name
                    }
                    
                    # Check if dataset exists
                    if not self.butler_client.dataset_exists(dataset_type, dataId):
                        logger.warning(f"Dataset {dataset_type} not found for filter {filter_name}")
                        failed_filters.append(filter_name)
                        continue
                    
                    # Fetch cutout with bbox
                    cutout = self.butler_client.get_exposure(dataId, dataset_type, bbox)
                    if cutout is None:
                        logger.warning(f"Failed to retrieve cutout for filter {filter_name}")
                        failed_filters.append(filter_name)
                        continue
                    
                    # Perform quality check if requested
                    if quality_check:
                        if not self._validate_cutout_quality(cutout, size):
                            logger.warning(f"Cutout quality check failed for filter {filter_name}")
                            failed_filters.append(filter_name)
                            continue
                    
                    cutouts[filter_name] = cutout
                    
                except Exception as e:
                    logger.error(f"Failed to fetch cutout for filter {filter_name}: {e}")
                    failed_filters.append(filter_name)
            
            # Check if we have any successful cutouts
            if not cutouts:
                raise DataAccessError(f"Failed to retrieve cutouts for all filters: {filters}")
            
            # Log any failed filters
            if failed_filters:
                logger.warning(f"Failed to retrieve cutouts for filters: {failed_filters}")
            
            return cutouts
            
        except Exception as e:
            raise CutoutExtractionError(
                f"Failed to fetch multi-band cutout: {e}",
                ra=ra, dec=dec, size=size
            )
    
    def _process_single_request(self, request: CutoutRequest) -> Dict[str, Any]:
        """Process a single cutout request."""
        try:
            cutout = self.fetch_cutout(
                request.ra, request.dec, request.size, 
                request.filters, request.dataset_type, 
                quality_check=request.quality_threshold > 0
            )
            
            return {
                'request': request,
                'cutout': cutout,
                'success': True,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'request': request,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _validate_coordinates(self, ra: float, dec: float) -> None:
        """Validate coordinate values."""
        if not (0 <= ra <= 360):
            raise DataIdValidationError(f"Invalid RA: {ra} (must be 0-360)")
        if not (-90 <= dec <= 90):
            raise DataIdValidationError(f"Invalid Dec: {dec} (must be -90 to 90)")
    
    def _validate_cutout_size(self, size: int) -> None:
        """Validate cutout size."""
        if size <= 0:
            raise DataIdValidationError(f"Invalid cutout size: {size} (must be > 0)")
        if size > 2048:
            raise DataIdValidationError(f"Cutout size too large: {size} (max: 2048)")
    
    def _record_performance_metric(self, operation: str, duration: float, 
                                 memory_usage: int, success: bool, 
                                 error_message: Optional[str] = None,
                                 cache_hit: bool = False) -> None:
        """Record performance metric."""
        if self.config.enable_performance_monitoring:
            metric = PerformanceMetrics(
                operation=operation,
                duration=duration,
                memory_usage=memory_usage,
                success=success,
                error_message=error_message,
                cache_hit=cache_hit
            )
            self.performance_metrics.append(metric)
            
            # Log performance warning if threshold exceeded
            if duration > 5.0:  # 5 second threshold
                self.logger.warning(
                    f"Performance warning: {operation} took {duration:.2f}s"
                )
    
    def _validate_cutout_quality(self, cutout: Exposure, expected_size: int) -> bool:
        """Validate cutout quality and completeness."""
        try:
            # Get masked image
            masked_image = cutout.getMaskedImage()
            image_array = masked_image.getImage().array
            mask_array = masked_image.getMask().array
            
            # Check if cutout has expected dimensions
            actual_height, actual_width = image_array.shape
            if actual_height < expected_size * 0.8 or actual_width < expected_size * 0.8:
                logger.warning(f"Cutout too small: {actual_width}x{actual_height}, expected ~{expected_size}x{expected_size}")
                return False
            
            # Check for excessive bad pixels
            bad_pixel_mask = mask_array > 0
            bad_pixel_fraction = bad_pixel_mask.sum() / bad_pixel_mask.size
            if bad_pixel_fraction > 0.5:  # More than 50% bad pixels
                logger.warning(f"Too many bad pixels: {bad_pixel_fraction:.2%}")
                return False
            
            # Check for reasonable signal
            valid_pixels = image_array[~bad_pixel_mask]
            if len(valid_pixels) == 0:
                logger.warning("No valid pixels in cutout")
                return False
            
            # Check if we have some signal variation (not all zeros or constant)
            signal_std = valid_pixels.std()
            if signal_std == 0:
                logger.warning("No signal variation in cutout")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Cutout quality validation failed: {e}")
            return False
    
    def _merge_catalogs(self, catalogs: List[SourceCatalog]) -> SourceCatalog:
        """Merge multiple source catalogs."""
        try:
            if len(catalogs) == 0:
                raise ValueError("Cannot merge empty catalog list")
            
            if len(catalogs) == 1:
                return catalogs[0]
            
            # Use the first catalog as base
            merged_catalog = catalogs[0].copy(deep=True)
            
            # Append records from other catalogs
            for catalog in catalogs[1:]:
                for record in catalog:
                    merged_catalog.addNew(record)
            
            return merged_catalog
            
        except Exception as e:
            logger.error(f"Catalog merging failed: {e}")
            raise DataAccessError(f"Failed to merge catalogs: {e}")
    
    def _handle_bbox_error(self, original_bbox, wcs, ra: float, dec: float, size: int):
        """Handle bbox errors by adjusting size and position."""
        try:
            # Try reducing the cutout size
            reduced_sizes = [int(size * 0.8), int(size * 0.6), int(size * 0.4)]
            
            for reduced_size in reduced_sizes:
                try:
                    adjusted_bbox = self.coordinate_converter.calculate_bbox_from_radec(
                        ra, dec, reduced_size, wcs
                    )
                    logger.info(f"Adjusted bbox size from {size} to {reduced_size}")
                    return adjusted_bbox, reduced_size
                except Exception:
                    continue
            
            # If size reduction doesn't work, try slight coordinate adjustments
            coordinate_offsets = [
                (0.0001, 0.0001),   # Small positive offset
                (-0.0001, -0.0001), # Small negative offset
                (0.0001, -0.0001),  # Mixed offsets
                (-0.0001, 0.0001)
            ]
            
            for ra_offset, dec_offset in coordinate_offsets:
                try:
                    adjusted_bbox = self.coordinate_converter.calculate_bbox_from_radec(
                        ra + ra_offset, dec + dec_offset, size, wcs
                    )
                    logger.info(f"Adjusted coordinates by ({ra_offset}, {dec_offset})")
                    return adjusted_bbox, size
                except Exception:
                    continue
            
            # Last resort: return minimal bbox
            center_x, center_y = self.coordinate_converter.sky_to_pixel(ra, dec, wcs)
            minimal_size = 10  # Minimal 10x10 pixel cutout
            minimal_bbox = Box2I(
                Point2I(int(center_x - minimal_size//2), int(center_y - minimal_size//2)),
                Extent2I(minimal_size, minimal_size)
            )
            logger.warning(f"Using minimal bbox of size {minimal_size}")
            return minimal_bbox, minimal_size
            
        except Exception as e:
            logger.error(f"Bbox error recovery failed: {e}")
            raise CutoutExtractionError(f"Cannot recover from bbox error: {e}")
    
    def _handle_collection_fallback(self, dataset_type: str, dataId: Dict[str, Any]) -> Any:
        """Handle collection fallback when primary collections fail."""
        try:
            original_collections = self.config.collections.copy()
            
            # Try individual collections
            for collection in original_collections:
                try:
                    # Temporarily use single collection
                    self.butler_client.config.collections = [collection]
                    
                    if self.butler_client.dataset_exists(dataset_type, dataId):
                        data = self.butler_client.get_exposure(dataId, dataset_type)
                        if data is not None:
                            logger.info(f"Successfully retrieved data from collection: {collection}")
                            return data
                            
                except Exception as e:
                    logger.warning(f"Collection {collection} failed: {e}")
                    continue
                finally:
                    # Restore original collections
                    self.butler_client.config.collections = original_collections
            
            # Try querying all available collections
            try:
                all_collections = list(self.butler_client.butler.registry.queryCollections())
                
                # Filter to relevant collections (avoid calibration collections)
                relevant_collections = [
                    c for c in all_collections 
                    if not any(skip in c.lower() for skip in ['calib', 'bias', 'dark', 'flat'])
                ]
                
                for collection in relevant_collections:
                    if collection not in original_collections:
                        try:
                            self.butler_client.config.collections = [collection]
                            
                            if self.butler_client.dataset_exists(dataset_type, dataId):
                                data = self.butler_client.get_exposure(dataId, dataset_type)
                                if data is not None:
                                    logger.info(f"Found data in alternative collection: {collection}")
                                    return data
                                    
                        except Exception as e:
                            logger.warning(f"Alternative collection {collection} failed: {e}")
                            continue
                        finally:
                            self.butler_client.config.collections = original_collections
                            
            except Exception as e:
                logger.error(f"Failed to query alternative collections: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Collection fallback failed: {e}")
            return None
    
    def _handle_partial_data_failure(self, results: List[Dict[str, Any]], 
                                   min_success_rate: float = 0.5) -> List[Dict[str, Any]]:
        """Handle partial failures in batch operations."""
        try:
            total_requests = len(results)
            successful_results = [r for r in results if r.get('success', False)]
            success_rate = len(successful_results) / total_requests if total_requests > 0 else 0
            
            if success_rate < min_success_rate:
                logger.warning(f"Low success rate: {success_rate:.2%} ({len(successful_results)}/{total_requests})")
                
                # Attempt to retry failed requests
                failed_results = [r for r in results if not r.get('success', False)]
                retry_results = []
                
                for failed_result in failed_results:
                    try:
                        request = failed_result.get('request')
                        if request:
                            # Retry with reduced requirements
                            retry_cutout = self.fetch_cutout(
                                request.ra, request.dec, request.size, 
                                request.filters, request.dataset_type, 
                                quality_check=False  # Disable quality check for retry
                            )
                            
                            retry_results.append({
                                'request': request,
                                'cutout': retry_cutout,
                                'success': True,
                                'timestamp': time.time(),
                                'retry': True
                            })
                            
                    except Exception as e:
                        logger.error(f"Retry failed for request: {e}")
                        retry_results.append(failed_result)  # Keep original failure
                
                # Combine successful and retry results
                final_results = successful_results + retry_results
                
                final_success_rate = len([r for r in final_results if r.get('success', False)]) / total_requests
                logger.info(f"Final success rate after retries: {final_success_rate:.2%}")
                
                return final_results
            
            return results
            
        except Exception as e:
            logger.error(f"Partial data failure handling failed: {e}")
            return results
    
    def _aggregate_errors(self, errors: List[Exception]) -> str:
        """Aggregate multiple errors into a comprehensive error message."""
        if not errors:
            return "No errors to aggregate"
        
        error_counts = {}
        error_messages = []
        
        for error in errors:
            error_type = type(error).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            # Keep unique error messages
            error_msg = str(error)
            if error_msg not in error_messages:
                error_messages.append(error_msg)
        
        # Build summary
        summary_parts = []
        for error_type, count in error_counts.items():
            summary_parts.append(f"{error_type}: {count}")
        
        summary = f"Error summary: {', '.join(summary_parts)}"
        
        if len(error_messages) <= 3:
            # Show all messages if few
            details = " | ".join(error_messages)
        else:
            # Show first 3 messages if many
            details = " | ".join(error_messages[:3]) + f" ... and {len(error_messages) - 3} more"
        
        return f"{summary}. Details: {details}"
    
    def _recover_from_cutout_error(self, ra: float, dec: float, size: int, 
                                 filter_name: str, dataset_type: str, 
                                 tract: int, patch: str) -> Optional[Exposure]:
        """Comprehensive error recovery for cutout extraction."""
        try:
            # First, try collection fallback
            dataId = {'tract': tract, 'patch': patch, 'band': filter_name}
            
            fallback_data = self._handle_collection_fallback(dataset_type, dataId)
            if fallback_data is not None:
                return fallback_data
            
            # Try with different dataset types
            alternative_dataset_types = ['calexp', 'deepCoadd_calexp', 'goodSeeingCoadd']
            
            for alt_dataset_type in alternative_dataset_types:
                if alt_dataset_type != dataset_type:
                    try:
                        if self.butler_client.dataset_exists(alt_dataset_type, dataId):
                            wcs = self.butler_client.get_wcs(dataId, alt_dataset_type)
                            if wcs:
                                bbox = self.coordinate_converter.calculate_bbox_from_radec(
                                    ra, dec, size, wcs
                                )
                                cutout = self.butler_client.get_exposure(dataId, alt_dataset_type, bbox)
                                if cutout is not None:
                                    logger.info(f"Successfully retrieved using alternative dataset type: {alt_dataset_type}")
                                    return cutout
                    except Exception as e:
                        logger.warning(f"Alternative dataset type {alt_dataset_type} failed: {e}")
                        continue
            
            # Try nearby patches
            adjacent_patches = self._get_adjacent_patches(patch)
            
            for adj_patch in adjacent_patches:
                try:
                    adj_dataId = {'tract': tract, 'patch': adj_patch, 'band': filter_name}
                    
                    if self.butler_client.dataset_exists(dataset_type, adj_dataId):
                        wcs = self.butler_client.get_wcs(adj_dataId, dataset_type)
                        if wcs:
                            bbox = self.coordinate_converter.calculate_bbox_from_radec(
                                ra, dec, size, wcs
                            )
                            cutout = self.butler_client.get_exposure(adj_dataId, dataset_type, bbox)
                            if cutout is not None:
                                logger.info(f"Successfully retrieved from adjacent patch: {adj_patch}")
                                return cutout
                                
                except Exception as e:
                    logger.warning(f"Adjacent patch {adj_patch} failed: {e}")
                    continue
            
            logger.error("All recovery attempts failed")
            return None
            
        except Exception as e:
            logger.error(f"Cutout error recovery failed: {e}")
            return None
    
    def _get_adjacent_patches(self, patch: str) -> List[str]:
        """Get list of adjacent patches for fallback."""
        try:
            if ',' in patch:
                x, y = map(int, patch.split(','))
                
                # Return adjacent patches in order of preference
                adjacent = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip current patch
                        adjacent.append(f"{x + dx},{y + dy}")
                
                return adjacent
            else:
                # Handle non-standard patch format
                return []
                
        except Exception as e:
            logger.error(f"Failed to get adjacent patches: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            if self.cache_manager:
                self.cache_manager.cleanup()
            if self.butler_client:
                self.butler_client.cleanup()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.__exit__(None, None, None)
        except Exception:
            pass  # Ignore cleanup errors in destructor