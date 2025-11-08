"""
Cutout optimization features for RIPPLe data access.

This module provides advanced cutout optimization including adaptive sizing,
prefetching, compression, and intelligent caching strategies.
"""

import logging
import time
import threading
import pickle
import zlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque, defaultdict
from enum import Enum
import numpy as np

# LSST imports
from lsst.afw.image import Exposure
from lsst.geom import Box2I, Point2I, Extent2I

# RIPPLe imports
from .cache_manager import CacheManager
from .exceptions import CutoutExtractionError, PerformanceError

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Compression methods for cutout data."""
    NONE = "none"
    ZLIB = "zlib"
    LZMA = "lzma"
    NUMPY = "numpy"  # NumPy array compression


class PrefetchStrategy(Enum):
    """Prefetching strategies."""
    NONE = "none"
    SPATIAL = "spatial"  # Prefetch spatially adjacent regions
    TEMPORAL = "temporal"  # Prefetch based on access patterns
    PREDICTIVE = "predictive"  # Machine learning-based prediction


@dataclass
class CutoutRequest:
    """Enhanced cutout request with optimization hints."""
    ra: float
    dec: float
    size: int
    filters: List[str]
    priority: int = 1
    prefetch_neighbors: bool = False
    compression: CompressionMethod = CompressionMethod.NONE
    quality_threshold: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CutoutMetrics:
    """Metrics for cutout performance analysis."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    cache_hit: bool
    fetch_time: float
    quality_score: float


class AdaptiveSizeManager:
    """Manages adaptive cutout sizing based on performance metrics."""
    
    def __init__(self, initial_size: int = 64, min_size: int = 32, max_size: int = 256):
        self.initial_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.size_history: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=1000)
        self.optimal_sizes: Dict[str, int] = {}  # Per-dataset optimal sizes
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_optimal_size(self, 
                        dataset_type: str, 
                        filters: List[str], 
                        target_performance: float = 2.0) -> int:
        """Get optimal cutout size for given parameters."""
        try:
            # Create key for this configuration
            key = f"{dataset_type}_{'-'.join(sorted(filters))}"
            
            # Return cached optimal size if available
            if key in self.optimal_sizes:
                return self.optimal_sizes[key]
            
            # Analyze performance history to find optimal size
            optimal_size = self._analyze_performance_history(key, target_performance)
            
            # Cache the result
            self.optimal_sizes[key] = optimal_size
            
            return optimal_size
            
        except Exception as e:
            self.logger.error(f"Failed to get optimal size: {e}")
            return self.initial_size
    
    def _analyze_performance_history(self, key: str, target_performance: float) -> int:
        """Analyze performance history to find optimal size."""
        # Filter performance data for this configuration
        relevant_data = [
            (entry['size'], entry['performance']) 
            for entry in self.performance_history 
            if entry.get('key') == key and entry.get('performance', 0) > 0
        ]
        
        if not relevant_data:
            return self.initial_size
        
        # Find size that best meets target performance
        best_size = self.initial_size
        best_score = float('inf')
        
        for size, performance in relevant_data:
            # Score based on how close to target performance
            score = abs(performance - target_performance)
            
            if score < best_score:
                best_score = score
                best_size = size
        
        # Ensure size is within bounds
        return max(self.min_size, min(self.max_size, best_size))
    
    def record_performance(self, 
                          dataset_type: str, 
                          filters: List[str], 
                          size: int, 
                          performance: float):
        """Record performance for a specific configuration."""
        key = f"{dataset_type}_{'-'.join(sorted(filters))}"
        
        entry = {
            'key': key,
            'size': size,
            'performance': performance,
            'timestamp': time.time()
        }
        
        self.performance_history.append(entry)
        
        # Invalidate cached optimal size to force recalculation
        if key in self.optimal_sizes:
            del self.optimal_sizes[key]
    
    def suggest_size_adjustment(self, 
                               current_size: int, 
                               current_performance: float, 
                               target_performance: float) -> int:
        """Suggest size adjustment based on current performance."""
        if current_performance > target_performance * 1.5:
            # Too slow, try smaller size
            return max(self.min_size, int(current_size * 0.8))
        elif current_performance < target_performance * 0.5:
            # Too fast, can try larger size
            return min(self.max_size, int(current_size * 1.2))
        else:
            # Performance is acceptable
            return current_size
    
    def get_size_statistics(self) -> Dict[str, Any]:
        """Get statistics about size optimization."""
        if not self.performance_history:
            return {}
        
        sizes = [entry['size'] for entry in self.performance_history]
        performances = [entry['performance'] for entry in self.performance_history]
        
        return {
            'total_requests': len(self.performance_history),
            'size_range': (min(sizes), max(sizes)),
            'avg_size': sum(sizes) / len(sizes),
            'performance_range': (min(performances), max(performances)),
            'avg_performance': sum(performances) / len(performances),
            'optimal_sizes_cached': len(self.optimal_sizes)
        }


class CutoutCompressor:
    """Handles compression and decompression of cutout data."""
    
    def __init__(self):
        self.compression_stats: Dict[str, List[float]] = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compress_cutout(self, 
                       cutout: Exposure, 
                       method: CompressionMethod = CompressionMethod.ZLIB) -> Tuple[bytes, CutoutMetrics]:
        """Compress cutout data using specified method."""
        try:
            start_time = time.time()
            
            # Serialize the cutout
            serialized = pickle.dumps(cutout, protocol=pickle.HIGHEST_PROTOCOL)
            original_size = len(serialized)
            
            # Apply compression
            if method == CompressionMethod.ZLIB:
                compressed = zlib.compress(serialized, level=6)
            elif method == CompressionMethod.LZMA:
                import lzma
                compressed = lzma.compress(serialized, preset=6)
            elif method == CompressionMethod.NUMPY:
                compressed = self._compress_numpy(cutout)
            else:  # NONE
                compressed = serialized
            
            compressed_size = len(compressed)
            compression_time = time.time() - start_time
            
            # Calculate metrics
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            metrics = CutoutMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                compression_time=compression_time,
                decompression_time=0.0,
                cache_hit=False,
                fetch_time=0.0,
                quality_score=0.0
            )
            
            # Record compression statistics
            self.compression_stats[method.value].append(compression_ratio)
            
            return compressed, metrics
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            raise CutoutExtractionError(f"Failed to compress cutout: {e}")
    
    def decompress_cutout(self, 
                         compressed_data: bytes, 
                         method: CompressionMethod = CompressionMethod.ZLIB) -> Tuple[Exposure, float]:
        """Decompress cutout data."""
        try:
            start_time = time.time()
            
            # Decompress based on method
            if method == CompressionMethod.ZLIB:
                decompressed = zlib.decompress(compressed_data)
            elif method == CompressionMethod.LZMA:
                import lzma
                decompressed = lzma.decompress(compressed_data)
            elif method == CompressionMethod.NUMPY:
                return self._decompress_numpy(compressed_data)
            else:  # NONE
                decompressed = compressed_data
            
            # Deserialize the cutout
            cutout = pickle.loads(decompressed)
            
            decompression_time = time.time() - start_time
            return cutout, decompression_time
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise CutoutExtractionError(f"Failed to decompress cutout: {e}")
    
    def _compress_numpy(self, cutout: Exposure) -> bytes:
        """Compress cutout using NumPy-specific methods."""
        try:
            # Extract image data
            masked_image = cutout.getMaskedImage()
            image_array = masked_image.getImage().array
            mask_array = masked_image.getMask().array
            variance_array = masked_image.getVariance().array
            
            # Compress arrays separately
            compressed_data = {
                'image': zlib.compress(image_array.tobytes()),
                'mask': zlib.compress(mask_array.tobytes()),
                'variance': zlib.compress(variance_array.tobytes()),
                'shape': image_array.shape,
                'dtype': str(image_array.dtype)
            }
            
            return pickle.dumps(compressed_data)
            
        except Exception as e:
            self.logger.error(f"NumPy compression failed: {e}")
            raise
    
    def _decompress_numpy(self, compressed_data: bytes) -> Tuple[Exposure, float]:
        """Decompress NumPy-compressed cutout."""
        try:
            start_time = time.time()
            
            # Deserialize compressed data
            data = pickle.loads(compressed_data)
            
            # Decompress arrays
            image_bytes = zlib.decompress(data['image'])
            mask_bytes = zlib.decompress(data['mask'])
            variance_bytes = zlib.decompress(data['variance'])
            
            # Reconstruct arrays
            shape = data['shape']
            dtype = np.dtype(data['dtype'])
            
            image_array = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
            mask_array = np.frombuffer(mask_bytes, dtype=np.uint16).reshape(shape)
            variance_array = np.frombuffer(variance_bytes, dtype=dtype).reshape(shape)
            
            # Reconstruct cutout (simplified - would need full Exposure reconstruction)
            # This is a placeholder - actual implementation would need to properly
            # reconstruct the LSST Exposure object
            decompression_time = time.time() - start_time
            
            # For now, return None and time (would need proper Exposure reconstruction)
            return None, decompression_time
            
        except Exception as e:
            self.logger.error(f"NumPy decompression failed: {e}")
            raise
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        stats = {}
        
        for method, ratios in self.compression_stats.items():
            if ratios:
                stats[method] = {
                    'count': len(ratios),
                    'avg_ratio': sum(ratios) / len(ratios),
                    'min_ratio': min(ratios),
                    'max_ratio': max(ratios),
                    'median_ratio': sorted(ratios)[len(ratios)//2]
                }
        
        return stats


class CutoutPrefetcher:
    """Handles intelligent prefetching of cutout data."""
    
    def __init__(self, 
                 cache_manager: CacheManager,
                 max_prefetch_workers: int = 2,
                 prefetch_strategy: PrefetchStrategy = PrefetchStrategy.SPATIAL):
        self.cache_manager = cache_manager
        self.max_prefetch_workers = max_prefetch_workers
        self.prefetch_strategy = prefetch_strategy
        self.access_history: deque = deque(maxlen=10000)
        self.prefetch_stats = {
            'requested': 0,
            'successful': 0,
            'cache_hits': 0,
            'time_saved': 0.0
        }
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def prefetch_cutouts(self, 
                        primary_request: CutoutRequest, 
                        fetch_function: Callable,
                        max_prefetch: int = 5) -> int:
        """Prefetch related cutouts based on strategy."""
        try:
            if self.prefetch_strategy == PrefetchStrategy.NONE:
                return 0
            
            # Generate prefetch candidates
            candidates = self._generate_prefetch_candidates(primary_request, max_prefetch)
            
            if not candidates:
                return 0
            
            # Prefetch in background
            prefetched = 0
            
            with ThreadPoolExecutor(max_workers=self.max_prefetch_workers) as executor:
                # Submit prefetch tasks
                futures = []
                for candidate in candidates:
                    cache_key = self._generate_cache_key(candidate)
                    
                    # Skip if already cached
                    if self.cache_manager.get(cache_key) is not None:
                        self.prefetch_stats['cache_hits'] += 1
                        continue
                    
                    future = executor.submit(self._prefetch_single, candidate, fetch_function, cache_key)
                    futures.append(future)
                
                # Wait for completion (with timeout)
                for future in as_completed(futures, timeout=30):
                    try:
                        if future.result():
                            prefetched += 1
                            self.prefetch_stats['successful'] += 1
                    except Exception as e:
                        self.logger.debug(f"Prefetch task failed: {e}")
            
            self.prefetch_stats['requested'] += len(candidates)
            return prefetched
            
        except Exception as e:
            self.logger.error(f"Prefetch operation failed: {e}")
            return 0
    
    def _generate_prefetch_candidates(self, 
                                    primary_request: CutoutRequest, 
                                    max_candidates: int) -> List[CutoutRequest]:
        """Generate prefetch candidates based on strategy."""
        candidates = []
        
        if self.prefetch_strategy == PrefetchStrategy.SPATIAL:
            candidates = self._generate_spatial_candidates(primary_request, max_candidates)
        elif self.prefetch_strategy == PrefetchStrategy.TEMPORAL:
            candidates = self._generate_temporal_candidates(primary_request, max_candidates)
        elif self.prefetch_strategy == PrefetchStrategy.PREDICTIVE:
            candidates = self._generate_predictive_candidates(primary_request, max_candidates)
        
        return candidates
    
    def _generate_spatial_candidates(self, 
                                   primary_request: CutoutRequest, 
                                   max_candidates: int) -> List[CutoutRequest]:
        """Generate spatially adjacent prefetch candidates."""
        candidates = []
        
        # Generate nearby coordinates
        offsets = [
            (-0.01, 0), (0.01, 0), (0, -0.01), (0, 0.01),  # Adjacent
            (-0.01, -0.01), (0.01, 0.01), (-0.01, 0.01), (0.01, -0.01)  # Diagonal
        ]
        
        for i, (ra_offset, dec_offset) in enumerate(offsets):
            if i >= max_candidates:
                break
            
            candidate = CutoutRequest(
                ra=primary_request.ra + ra_offset,
                dec=primary_request.dec + dec_offset,
                size=primary_request.size,
                filters=primary_request.filters,
                priority=primary_request.priority - 1,  # Lower priority
                compression=primary_request.compression,
                quality_threshold=primary_request.quality_threshold * 0.8  # Relaxed quality
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _generate_temporal_candidates(self, 
                                    primary_request: CutoutRequest, 
                                    max_candidates: int) -> List[CutoutRequest]:
        """Generate candidates based on access patterns."""
        candidates = []
        
        # Analyze recent access history
        recent_accesses = list(self.access_history)[-100:]  # Last 100 accesses
        
        # Find frequently accessed regions near this request
        nearby_accesses = []
        for access in recent_accesses:
            distance = ((access['ra'] - primary_request.ra) ** 2 + 
                       (access['dec'] - primary_request.dec) ** 2) ** 0.5
            
            if distance < 0.1:  # Within 0.1 degree
                nearby_accesses.append(access)
        
        # Sort by frequency and recency
        access_counts = defaultdict(int)
        for access in nearby_accesses:
            key = (access['ra'], access['dec'])
            access_counts[key] += 1
        
        # Generate candidates from most frequent accesses
        for i, ((ra, dec), count) in enumerate(sorted(access_counts.items(), 
                                                     key=lambda x: x[1], 
                                                     reverse=True)):
            if i >= max_candidates:
                break
            
            candidate = CutoutRequest(
                ra=ra,
                dec=dec,
                size=primary_request.size,
                filters=primary_request.filters,
                priority=primary_request.priority - 1,
                compression=primary_request.compression,
                quality_threshold=primary_request.quality_threshold * 0.9
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _generate_predictive_candidates(self, 
                                      primary_request: CutoutRequest, 
                                      max_candidates: int) -> List[CutoutRequest]:
        """Generate candidates using predictive modeling."""
        # Placeholder for machine learning-based prediction
        # This would analyze access patterns and predict likely next requests
        candidates = []
        
        # For now, fall back to spatial strategy
        return self._generate_spatial_candidates(primary_request, max_candidates)
    
    def _prefetch_single(self, 
                        request: CutoutRequest, 
                        fetch_function: Callable, 
                        cache_key: str) -> bool:
        """Prefetch a single cutout."""
        try:
            start_time = time.time()
            
            # Fetch the cutout
            cutout = fetch_function(request.ra, request.dec, request.size, 
                                  request.filters[0] if request.filters else 'r')
            
            # Cache the result
            self.cache_manager.put(cache_key, cutout)
            
            # Record time saved for statistics
            fetch_time = time.time() - start_time
            self.prefetch_stats['time_saved'] += fetch_time
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Prefetch failed for {cache_key}: {e}")
            return False
    
    def _generate_cache_key(self, request: CutoutRequest) -> str:
        """Generate cache key for a cutout request."""
        return f"cutout_{request.ra:.6f}_{request.dec:.6f}_{request.size}_{'_'.join(request.filters)}"
    
    def record_access(self, ra: float, dec: float, size: int, filters: List[str]):
        """Record an access for pattern analysis."""
        access = {
            'ra': ra,
            'dec': dec,
            'size': size,
            'filters': filters,
            'timestamp': time.time()
        }
        
        self.access_history.append(access)
    
    def get_prefetch_statistics(self) -> Dict[str, Any]:
        """Get prefetch performance statistics."""
        stats = self.prefetch_stats.copy()
        
        if stats['requested'] > 0:
            stats['success_rate'] = stats['successful'] / stats['requested']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['requested']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats


class CutoutOptimizer:
    """
    Main cutout optimization coordinator.
    
    This class coordinates all optimization features including adaptive sizing,
    compression, prefetching, and intelligent caching.
    """
    
    def __init__(self, 
                 cache_manager: CacheManager,
                 enable_compression: bool = True,
                 enable_prefetching: bool = True,
                 enable_adaptive_sizing: bool = True):
        """
        Initialize cutout optimizer.
        
        Parameters
        ----------
        cache_manager : CacheManager
            Cache manager for storing optimized cutouts
        enable_compression : bool
            Whether to enable cutout compression
        enable_prefetching : bool
            Whether to enable intelligent prefetching
        enable_adaptive_sizing : bool
            Whether to enable adaptive sizing
        """
        self.cache_manager = cache_manager
        self.enable_compression = enable_compression
        self.enable_prefetching = enable_prefetching
        self.enable_adaptive_sizing = enable_adaptive_sizing
        
        # Initialize optimization components
        self.size_manager = AdaptiveSizeManager() if enable_adaptive_sizing else None
        self.compressor = CutoutCompressor() if enable_compression else None
        self.prefetcher = CutoutPrefetcher(cache_manager) if enable_prefetching else None
        
        # Performance tracking
        self.optimization_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'compression_savings': 0,
            'prefetch_hits': 0,
            'size_optimizations': 0
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def optimize_cutout_request(self, 
                              request: CutoutRequest,
                              fetch_function: Callable) -> Tuple[Any, CutoutMetrics]:
        """
        Optimize a cutout request using all available optimization techniques.
        
        Parameters
        ----------
        request : CutoutRequest
            The cutout request to optimize
        fetch_function : Callable
            Function to fetch cutout data
            
        Returns
        -------
        Tuple[Any, CutoutMetrics]
            Optimized cutout data and performance metrics
        """
        try:
            start_time = time.time()
            self.optimization_stats['total_requests'] += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            
            # Check cache first
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.optimization_stats['cache_hits'] += 1
                
                # Decompress if necessary
                if self.enable_compression and request.compression != CompressionMethod.NONE:
                    cutout, decomp_time = self.compressor.decompress_cutout(
                        cached_result, request.compression
                    )
                    
                    metrics = CutoutMetrics(
                        original_size=0,
                        compressed_size=len(cached_result),
                        compression_ratio=0.0,
                        compression_time=0.0,
                        decompression_time=decomp_time,
                        cache_hit=True,
                        fetch_time=0.0,
                        quality_score=1.0
                    )
                    
                    return cutout, metrics
                else:
                    metrics = CutoutMetrics(
                        original_size=0,
                        compressed_size=0,
                        compression_ratio=1.0,
                        compression_time=0.0,
                        decompression_time=0.0,
                        cache_hit=True,
                        fetch_time=0.0,
                        quality_score=1.0
                    )
                    
                    return cached_result, metrics
            
            # Optimize request parameters
            optimized_request = self._optimize_request_parameters(request)
            
            # Fetch cutout
            fetch_start = time.time()
            cutout = fetch_function(
                optimized_request.ra,
                optimized_request.dec,
                optimized_request.size,
                optimized_request.filters[0] if optimized_request.filters else 'r'
            )
            fetch_time = time.time() - fetch_start
            
            # Compress if enabled
            metrics = CutoutMetrics(
                original_size=0,
                compressed_size=0,
                compression_ratio=1.0,
                compression_time=0.0,
                decompression_time=0.0,
                cache_hit=False,
                fetch_time=fetch_time,
                quality_score=1.0
            )
            
            cutout_to_cache = cutout
            
            if self.enable_compression and request.compression != CompressionMethod.NONE:
                compressed_data, compression_metrics = self.compressor.compress_cutout(
                    cutout, request.compression
                )
                cutout_to_cache = compressed_data
                metrics = compression_metrics
                metrics.fetch_time = fetch_time
                metrics.cache_hit = False
                
                self.optimization_stats['compression_savings'] += (
                    compression_metrics.original_size - compression_metrics.compressed_size
                )
            
            # Cache the result
            self.cache_manager.put(cache_key, cutout_to_cache)
            
            # Record performance for adaptive sizing
            if self.enable_adaptive_sizing and self.size_manager:
                total_time = time.time() - start_time
                self.size_manager.record_performance(
                    'deepCoadd',  # Default dataset type
                    optimized_request.filters,
                    optimized_request.size,
                    total_time
                )
            
            # Start prefetching if enabled
            if self.enable_prefetching and self.prefetcher:
                if optimized_request.prefetch_neighbors:
                    prefetched = self.prefetcher.prefetch_cutouts(
                        optimized_request, fetch_function
                    )
                    if prefetched > 0:
                        self.optimization_stats['prefetch_hits'] += prefetched
                
                # Record access for pattern analysis
                self.prefetcher.record_access(
                    optimized_request.ra,
                    optimized_request.dec,
                    optimized_request.size,
                    optimized_request.filters
                )
            
            return cutout, metrics
            
        except Exception as e:
            self.logger.error(f"Cutout optimization failed: {e}")
            raise CutoutExtractionError(f"Optimization failed: {e}")
    
    def _optimize_request_parameters(self, request: CutoutRequest) -> CutoutRequest:
        """Optimize request parameters using adaptive sizing."""
        optimized_request = request
        
        if self.enable_adaptive_sizing and self.size_manager:
            optimal_size = self.size_manager.get_optimal_size(
                'deepCoadd',  # Default dataset type
                request.filters
            )
            
            if optimal_size != request.size:
                optimized_request = CutoutRequest(
                    ra=request.ra,
                    dec=request.dec,
                    size=optimal_size,
                    filters=request.filters,
                    priority=request.priority,
                    prefetch_neighbors=request.prefetch_neighbors,
                    compression=request.compression,
                    quality_threshold=request.quality_threshold,
                    context=request.context
                )
                
                self.optimization_stats['size_optimizations'] += 1
        
        return optimized_request
    
    def _generate_cache_key(self, request: CutoutRequest) -> str:
        """Generate cache key for a cutout request."""
        return f"opt_cutout_{request.ra:.6f}_{request.dec:.6f}_{request.size}_{'_'.join(request.filters)}_{request.compression.value}"
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = self.optimization_stats.copy()
        
        # Calculate rates
        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
            stats['size_optimization_rate'] = stats['size_optimizations'] / stats['total_requests']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['size_optimization_rate'] = 0.0
        
        # Add component statistics
        if self.size_manager:
            stats['size_manager'] = self.size_manager.get_size_statistics()
        
        if self.compressor:
            stats['compression'] = self.compressor.get_compression_statistics()
        
        if self.prefetcher:
            stats['prefetch'] = self.prefetcher.get_prefetch_statistics()
        
        return stats
    
    def clear_optimization_data(self):
        """Clear all optimization data and statistics."""
        self.optimization_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'compression_savings': 0,
            'prefetch_hits': 0,
            'size_optimizations': 0
        }
        
        if self.size_manager:
            self.size_manager.optimal_sizes.clear()
            self.size_manager.performance_history.clear()
        
        if self.compressor:
            self.compressor.compression_stats.clear()
        
        if self.prefetcher:
            self.prefetcher.access_history.clear()
            self.prefetcher.prefetch_stats = {
                'requested': 0,
                'successful': 0,
                'cache_hits': 0,
                'time_saved': 0.0
            }
        
        self.logger.info("Cleared all optimization data")
    
    def tune_parameters(self, target_performance: float = 2.0):
        """Tune optimization parameters based on performance history."""
        try:
            if self.size_manager:
                # Analyze size optimization effectiveness
                size_stats = self.size_manager.get_size_statistics()
                
                if size_stats.get('total_requests', 0) > 100:
                    avg_performance = size_stats.get('avg_performance', 0)
                    
                    if avg_performance > target_performance * 1.5:
                        # Performance is too slow, be more aggressive with smaller sizes
                        self.size_manager.initial_size = max(32, self.size_manager.initial_size - 8)
                        self.logger.info(f"Reduced initial size to {self.size_manager.initial_size}")
                    elif avg_performance < target_performance * 0.5:
                        # Performance is very fast, can try larger sizes
                        self.size_manager.initial_size = min(128, self.size_manager.initial_size + 8)
                        self.logger.info(f"Increased initial size to {self.size_manager.initial_size}")
            
            self.logger.info("Optimization parameters tuned")
            
        except Exception as e:
            self.logger.error(f"Parameter tuning failed: {e}")
    
    def get_optimization_recommendations(self) -> Dict[str, str]:
        """Get recommendations for optimization improvements."""
        recommendations = []
        
        stats = self.get_optimization_statistics()
        
        # Cache hit rate recommendations
        cache_hit_rate = stats.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.3:
            recommendations.append("Consider increasing cache size or improving prefetching")
        
        # Compression recommendations
        if self.enable_compression and 'compression' in stats:
            compression_stats = stats['compression']
            if compression_stats:
                avg_ratios = [s.get('avg_ratio', 1.0) for s in compression_stats.values()]
                if avg_ratios and max(avg_ratios) < 2.0:
                    recommendations.append("Compression ratios are low, consider different compression methods")
        
        # Size optimization recommendations
        size_opt_rate = stats.get('size_optimization_rate', 0)
        if size_opt_rate > 0.5:
            recommendations.append("High size optimization rate suggests initial size estimates are poor")
        
        # Prefetch recommendations
        if self.enable_prefetching and 'prefetch' in stats:
            prefetch_stats = stats['prefetch']
            success_rate = prefetch_stats.get('success_rate', 0)
            if success_rate < 0.5:
                recommendations.append("Prefetch success rate is low, consider adjusting prefetch strategy")
        
        return {
            'recommendations': recommendations,
            'performance_score': self._calculate_performance_score(stats),
            'optimization_health': 'good' if len(recommendations) < 2 else 'needs_attention'
        }
    
    def _calculate_performance_score(self, stats: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        score = 0.0
        
        # Cache hit rate (40% of score)
        cache_hit_rate = stats.get('cache_hit_rate', 0)
        score += cache_hit_rate * 40
        
        # Compression effectiveness (30% of score)
        if 'compression' in stats:
            compression_stats = stats['compression']
            if compression_stats:
                avg_ratios = [s.get('avg_ratio', 1.0) for s in compression_stats.values()]
                if avg_ratios:
                    compression_score = min(max(avg_ratios) / 3.0, 1.0)  # Normalize to 0-1
                    score += compression_score * 30
        
        # Prefetch effectiveness (20% of score)
        if 'prefetch' in stats:
            prefetch_stats = stats['prefetch']
            success_rate = prefetch_stats.get('success_rate', 0)
            score += success_rate * 20
        
        # Size optimization stability (10% of score)
        size_opt_rate = stats.get('size_optimization_rate', 0)
        size_stability = 1.0 - min(size_opt_rate, 1.0)  # Lower optimization rate is better
        score += size_stability * 10
        
        return min(score, 100.0)