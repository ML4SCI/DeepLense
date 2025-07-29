"""
Collection management for RIPPLe data access.

This module provides optimized collection handling with precedence management,
CHAINED navigation, filtering, and caching for efficient Butler operations.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum

# LSST imports
from lsst.daf.butler import Butler, CollectionType

# RIPPLe imports
from .exceptions import CollectionError, DataAccessError

logger = logging.getLogger(__name__)


class CollectionPriority(Enum):
    """Priority levels for collection ordering."""
    HIGHEST = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    LOWEST = 5


@dataclass
class CollectionInfo:
    """Information about a collection."""
    name: str
    type: CollectionType
    priority: CollectionPriority = CollectionPriority.MEDIUM
    description: Optional[str] = None
    created_time: Optional[float] = None
    last_accessed: Optional[float] = None
    dataset_count: int = 0
    is_active: bool = True
    tags: Set[str] = field(default_factory=set)


class CollectionManager:
    """
    Manages Butler collections with optimization and caching.
    
    This class provides intelligent collection management with:
    - Priority-based ordering
    - CHAINED collection navigation
    - Collection filtering and caching
    - Usage statistics and optimization
    """
    
    def __init__(self, butler: Butler, cache_size: int = 500):
        """
        Initialize collection manager.
        
        Parameters
        ----------
        butler : Butler
            LSST Butler instance
        cache_size : int
            Maximum number of cached collection queries
        """
        self.butler = butler
        self.cache_size = cache_size
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Collection information cache
        self.collection_info_cache: Dict[str, CollectionInfo] = {}
        
        # Collection hierarchy cache
        self.hierarchy_cache: Dict[str, List[str]] = {}
        
        # Query result cache
        self.query_cache: OrderedDict = OrderedDict()
        
        # Usage statistics
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.performance_metrics: Dict[str, List[float]] = {
            'query_time': [],
            'cache_hit_rate': [],
            'resolution_time': []
        }
        
        # Initialize collection information
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize collection information from Butler registry."""
        try:
            start_time = time.time()
            
            # Query all collections
            all_collections = list(self.butler.registry.queryCollections())
            
            for collection_name in all_collections:
                try:
                    # Get collection type
                    collection_type = self.butler.registry.getCollectionType(collection_name)
                    
                    # Create collection info
                    info = CollectionInfo(
                        name=collection_name,
                        type=collection_type,
                        priority=self._determine_priority(collection_name, collection_type),
                        created_time=time.time(),
                        last_accessed=None,
                        tags=self._extract_tags(collection_name)
                    )
                    
                    self.collection_info_cache[collection_name] = info
                    
                    # Initialize usage stats
                    self.usage_stats[collection_name] = {
                        'access_count': 0,
                        'success_count': 0,
                        'failure_count': 0,
                        'avg_response_time': 0.0,
                        'last_success': None,
                        'last_failure': None
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to initialize collection {collection_name}: {e}")
                    continue
            
            elapsed_time = time.time() - start_time
            self.performance_metrics['query_time'].append(elapsed_time)
            
            self.logger.info(f"Initialized {len(self.collection_info_cache)} collections in {elapsed_time:.2f}s")
            
        except Exception as e:
            raise CollectionError(f"Failed to initialize collections: {e}")
    
    def _determine_priority(self, collection_name: str, collection_type: CollectionType) -> CollectionPriority:
        """Determine priority for a collection based on name and type."""
        # Convert to lowercase for pattern matching
        name_lower = collection_name.lower()
        
        # High priority patterns
        if any(pattern in name_lower for pattern in ['dp0', 'dr', 'data_release']):
            return CollectionPriority.HIGHEST
        
        if any(pattern in name_lower for pattern in ['run', 'processed', 'coadd']):
            return CollectionPriority.HIGH
        
        # Low priority patterns
        if any(pattern in name_lower for pattern in ['calib', 'bias', 'dark', 'flat']):
            return CollectionPriority.LOW
        
        if any(pattern in name_lower for pattern in ['test', 'debug', 'temp']):
            return CollectionPriority.LOWEST
        
        # Type-based priority
        if collection_type == CollectionType.RUN:
            return CollectionPriority.HIGH
        elif collection_type == CollectionType.CHAINED:
            return CollectionPriority.MEDIUM
        elif collection_type == CollectionType.TAGGED:
            return CollectionPriority.MEDIUM
        
        return CollectionPriority.MEDIUM
    
    def _extract_tags(self, collection_name: str) -> Set[str]:
        """Extract tags from collection name for categorization."""
        tags = set()
        
        # Extract common tags
        name_lower = collection_name.lower()
        
        if 'dp0' in name_lower:
            tags.add('data_preview')
        if 'dr' in name_lower or 'data_release' in name_lower:
            tags.add('data_release')
        if 'run' in name_lower:
            tags.add('processing_run')
        if 'calib' in name_lower:
            tags.add('calibration')
        if 'coadd' in name_lower:
            tags.add('coadd')
        if 'raw' in name_lower:
            tags.add('raw_data')
        if 'test' in name_lower:
            tags.add('test')
        
        return tags
    
    def get_ordered_collections(self, 
                              collections: Optional[List[str]] = None,
                              include_tags: Optional[Set[str]] = None,
                              exclude_tags: Optional[Set[str]] = None,
                              active_only: bool = True) -> List[str]:
        """
        Get collections ordered by priority and usage statistics.
        
        Parameters
        ----------
        collections : List[str], optional
            Specific collections to order. If None, uses all collections.
        include_tags : Set[str], optional
            Only include collections with these tags
        exclude_tags : Set[str], optional
            Exclude collections with these tags
        active_only : bool
            Only include active collections
        
        Returns
        -------
        List[str]
            Ordered list of collection names
        """
        try:
            # Use specified collections or all cached collections
            if collections is None:
                candidate_collections = list(self.collection_info_cache.keys())
            else:
                candidate_collections = collections
            
            # Filter collections
            filtered_collections = []
            
            for collection_name in candidate_collections:
                if collection_name not in self.collection_info_cache:
                    continue
                
                info = self.collection_info_cache[collection_name]
                
                # Check if active
                if active_only and not info.is_active:
                    continue
                
                # Check include tags
                if include_tags and not (info.tags & include_tags):
                    continue
                
                # Check exclude tags
                if exclude_tags and (info.tags & exclude_tags):
                    continue
                
                filtered_collections.append(collection_name)
            
            # Sort by priority and usage statistics
            def sort_key(collection_name: str) -> Tuple[int, float, float]:
                info = self.collection_info_cache[collection_name]
                stats = self.usage_stats[collection_name]
                
                # Primary: Priority (lower number = higher priority)
                priority_score = info.priority.value
                
                # Secondary: Success rate (higher = better)
                total_attempts = stats['access_count']
                success_rate = stats['success_count'] / total_attempts if total_attempts > 0 else 0
                
                # Tertiary: Average response time (lower = better)
                avg_response_time = stats['avg_response_time']
                
                return (priority_score, -success_rate, avg_response_time)
            
            ordered_collections = sorted(filtered_collections, key=sort_key)
            
            self.logger.debug(f"Ordered {len(ordered_collections)} collections")
            return ordered_collections
            
        except Exception as e:
            self.logger.error(f"Failed to order collections: {e}")
            return collections or []
    
    def resolve_chained_collections(self, collection_name: str) -> List[str]:
        """
        Resolve CHAINED collection to its constituent collections.
        
        Parameters
        ----------
        collection_name : str
            Name of the collection to resolve
        
        Returns
        -------
        List[str]
            List of constituent collection names
        """
        # Check cache first
        if collection_name in self.hierarchy_cache:
            return self.hierarchy_cache[collection_name]
        
        try:
            start_time = time.time()
            
            # Get collection info
            if collection_name not in self.collection_info_cache:
                self.logger.warning(f"Collection {collection_name} not found in cache")
                return [collection_name]
            
            info = self.collection_info_cache[collection_name]
            
            # If not a chained collection, return as-is
            if info.type != CollectionType.CHAINED:
                result = [collection_name]
                self.hierarchy_cache[collection_name] = result
                return result
            
            # Query the chain definition
            try:
                chain_definition = self.butler.registry.getCollectionChain(collection_name)
                resolved_collections = list(chain_definition)
                
                # Cache the result
                self.hierarchy_cache[collection_name] = resolved_collections
                
                elapsed_time = time.time() - start_time
                self.performance_metrics['resolution_time'].append(elapsed_time)
                
                self.logger.debug(f"Resolved {collection_name} to {len(resolved_collections)} collections in {elapsed_time:.3f}s")
                return resolved_collections
                
            except Exception as e:
                self.logger.error(f"Failed to resolve chained collection {collection_name}: {e}")
                return [collection_name]
                
        except Exception as e:
            self.logger.error(f"Error resolving collection {collection_name}: {e}")
            return [collection_name]
    
    def get_optimal_collections_for_dataset(self, 
                                          dataset_type: str,
                                          collections: Optional[List[str]] = None,
                                          max_collections: int = 5) -> List[str]:
        """
        Get optimal collections for a specific dataset type.
        
        Parameters
        ----------
        dataset_type : str
            Type of dataset to find
        collections : List[str], optional
            Collections to consider. If None, uses all collections.
        max_collections : int
            Maximum number of collections to return
        
        Returns
        -------
        List[str]
            Optimal ordered list of collections
        """
        try:
            # Create cache key
            cache_key = f"{dataset_type}_{hash(tuple(collections or []))}"
            
            # Check cache
            if cache_key in self.query_cache:
                # Move to end (LRU)
                self.query_cache.move_to_end(cache_key)
                cached_result = self.query_cache[cache_key]
                
                # Update cache hit rate
                if self.performance_metrics['cache_hit_rate']:
                    current_rate = self.performance_metrics['cache_hit_rate'][-1]
                    new_rate = (current_rate + 1.0) / 2.0  # Simple moving average
                    self.performance_metrics['cache_hit_rate'].append(new_rate)
                else:
                    self.performance_metrics['cache_hit_rate'].append(1.0)
                
                return cached_result
            
            start_time = time.time()
            
            # Get candidate collections
            candidate_collections = self.get_ordered_collections(collections)
            
            # Score collections based on dataset availability
            collection_scores = []
            
            for collection_name in candidate_collections:
                try:
                    # Query dataset availability
                    datasets = list(self.butler.registry.queryDatasets(
                        dataset_type,
                        collections=[collection_name]
                    ))
                    
                    dataset_count = len(datasets)
                    
                    # Update collection info
                    self.collection_info_cache[collection_name].dataset_count = dataset_count
                    
                    # Calculate score
                    info = self.collection_info_cache[collection_name]
                    stats = self.usage_stats[collection_name]
                    
                    # Score based on dataset count, priority, and success rate
                    priority_score = (6 - info.priority.value) / 5.0  # Normalize to 0-1
                    dataset_score = min(dataset_count / 1000.0, 1.0)  # Normalize to 0-1
                    
                    total_attempts = stats['access_count']
                    success_rate = stats['success_count'] / total_attempts if total_attempts > 0 else 0.5
                    
                    combined_score = (dataset_score * 0.5 + priority_score * 0.3 + success_rate * 0.2)
                    
                    if dataset_count > 0:
                        collection_scores.append((collection_name, combined_score))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to score collection {collection_name}: {e}")
                    continue
            
            # Sort by score and take top collections
            collection_scores.sort(key=lambda x: x[1], reverse=True)
            optimal_collections = [name for name, score in collection_scores[:max_collections]]
            
            # Cache the result
            self.query_cache[cache_key] = optimal_collections
            
            # Maintain cache size
            if len(self.query_cache) > self.cache_size:
                self.query_cache.popitem(last=False)
            
            elapsed_time = time.time() - start_time
            self.performance_metrics['query_time'].append(elapsed_time)
            
            # Update cache hit rate
            if self.performance_metrics['cache_hit_rate']:
                current_rate = self.performance_metrics['cache_hit_rate'][-1]
                new_rate = (current_rate + 0.0) / 2.0  # Cache miss
                self.performance_metrics['cache_hit_rate'].append(new_rate)
            else:
                self.performance_metrics['cache_hit_rate'].append(0.0)
            
            self.logger.debug(f"Found {len(optimal_collections)} optimal collections for {dataset_type} in {elapsed_time:.3f}s")
            return optimal_collections
            
        except Exception as e:
            self.logger.error(f"Failed to get optimal collections for {dataset_type}: {e}")
            return collections or []
    
    def update_collection_stats(self, collection_name: str, success: bool, response_time: float):
        """Update usage statistics for a collection."""
        try:
            if collection_name not in self.usage_stats:
                self.usage_stats[collection_name] = {
                    'access_count': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'avg_response_time': 0.0,
                    'last_success': None,
                    'last_failure': None
                }
            
            stats = self.usage_stats[collection_name]
            
            # Update counters
            stats['access_count'] += 1
            
            if success:
                stats['success_count'] += 1
                stats['last_success'] = time.time()
            else:
                stats['failure_count'] += 1
                stats['last_failure'] = time.time()
            
            # Update average response time
            current_avg = stats['avg_response_time']
            count = stats['access_count']
            stats['avg_response_time'] = (current_avg * (count - 1) + response_time) / count
            
            # Update last accessed time
            if collection_name in self.collection_info_cache:
                self.collection_info_cache[collection_name].last_accessed = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to update stats for collection {collection_name}: {e}")
    
    def refresh_collection_info(self, collection_name: Optional[str] = None):
        """Refresh collection information from Butler registry."""
        try:
            if collection_name:
                # Refresh specific collection
                collections_to_refresh = [collection_name]
            else:
                # Refresh all collections
                collections_to_refresh = list(self.collection_info_cache.keys())
            
            for collection in collections_to_refresh:
                try:
                    # Re-query collection type
                    collection_type = self.butler.registry.getCollectionType(collection)
                    
                    # Update cached info
                    if collection in self.collection_info_cache:
                        self.collection_info_cache[collection].type = collection_type
                        self.collection_info_cache[collection].priority = self._determine_priority(
                            collection, collection_type
                        )
                    
                    # Clear hierarchy cache for this collection
                    if collection in self.hierarchy_cache:
                        del self.hierarchy_cache[collection]
                    
                except Exception as e:
                    self.logger.warning(f"Failed to refresh collection {collection}: {e}")
            
            # Clear query cache to force re-evaluation
            self.query_cache.clear()
            
            self.logger.info(f"Refreshed {len(collections_to_refresh)} collections")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh collection info: {e}")
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        try:
            total_collections = len(self.collection_info_cache)
            active_collections = sum(1 for info in self.collection_info_cache.values() if info.is_active)
            
            # Collection type distribution
            type_distribution = {}
            for info in self.collection_info_cache.values():
                type_name = info.type.name
                type_distribution[type_name] = type_distribution.get(type_name, 0) + 1
            
            # Priority distribution
            priority_distribution = {}
            for info in self.collection_info_cache.values():
                priority_name = info.priority.name
                priority_distribution[priority_name] = priority_distribution.get(priority_name, 0) + 1
            
            # Performance metrics
            avg_query_time = sum(self.performance_metrics['query_time']) / len(self.performance_metrics['query_time']) if self.performance_metrics['query_time'] else 0
            avg_cache_hit_rate = sum(self.performance_metrics['cache_hit_rate']) / len(self.performance_metrics['cache_hit_rate']) if self.performance_metrics['cache_hit_rate'] else 0
            
            # Top performing collections
            top_collections = sorted(
                self.usage_stats.items(),
                key=lambda x: x[1]['success_count'],
                reverse=True
            )[:10]
            
            return {
                'total_collections': total_collections,
                'active_collections': active_collections,
                'type_distribution': type_distribution,
                'priority_distribution': priority_distribution,
                'cache_size': len(self.query_cache),
                'cache_hit_rate': avg_cache_hit_rate,
                'avg_query_time': avg_query_time,
                'top_collections': [(name, stats['success_count']) for name, stats in top_collections],
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection statistics: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all cached data."""
        self.query_cache.clear()
        self.hierarchy_cache.clear()
        self.logger.info("Cleared collection caches")
    
    def deactivate_collection(self, collection_name: str):
        """Deactivate a collection (mark as inactive)."""
        if collection_name in self.collection_info_cache:
            self.collection_info_cache[collection_name].is_active = False
            self.logger.info(f"Deactivated collection: {collection_name}")
    
    def activate_collection(self, collection_name: str):
        """Activate a collection (mark as active)."""
        if collection_name in self.collection_info_cache:
            self.collection_info_cache[collection_name].is_active = True
            self.logger.info(f"Activated collection: {collection_name}")