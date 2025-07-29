"""
Cache management for RIPPLe data access.

This module provides caching functionality for cutouts, metadata, and other
frequently accessed data to improve performance.
"""

from typing import Dict, Any, Optional, List
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CacheManager:
    """
    LRU cache manager for RIPPLe data access.
    
    This class provides efficient caching of cutouts, metadata, and other
    frequently accessed data with automatic eviction and statistics tracking.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache manager."""
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.start_time = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            # Update existing item
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new item
            self.cache[key] = value
            
            # Evict if necessary
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'uptime': time.time() - self.start_time
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear()
    
    def get_memory_usage(self) -> int:
        """Estimate memory usage of cached items."""
        import sys
        total_size = 0
        for key, value in self.cache.items():
            total_size += sys.getsizeof(key) + sys.getsizeof(value)
        return total_size
    
    def evict_oldest(self, count: int = 1) -> int:
        """Evict oldest items from cache."""
        evicted = 0
        for _ in range(min(count, len(self.cache))):
            if self.cache:
                self.cache.popitem(last=False)
                evicted += 1
        return evicted
    
    def get_item_age(self, key: str) -> Optional[float]:
        """Get age of cached item in seconds."""
        if key not in self.cache:
            return None
        
        # Simple age estimation based on position in OrderedDict
        keys = list(self.cache.keys())
        try:
            position = keys.index(key)
            # Newer items are at the end
            age_factor = (len(keys) - position) / len(keys)
            return age_factor * (time.time() - self.start_time)
        except ValueError:
            return None
    
    def prefetch_keys(self, keys: List[str], fetch_func: callable) -> int:
        """Prefetch multiple keys if not already cached."""
        prefetched = 0
        for key in keys:
            if key not in self.cache:
                try:
                    value = fetch_func(key)
                    self.put(key, value)
                    prefetched += 1
                except Exception as e:
                    logger.warning(f"Failed to prefetch {key}: {e}")
        return prefetched