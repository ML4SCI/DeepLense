"""
Spatial indexing for RIPPLe data access performance optimization.

This module provides spatial indexing capabilities for efficient tract/patch lookup,
region overlap detection, and spatial query optimization.
"""

import logging
import time
import math
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# LSST imports
from lsst.geom import SpherePoint, degrees
from lsst.skymap import BaseSkyMap

# RIPPLe imports
from .exceptions import CoordinateConversionError
from .coordinate_utils import CoordinateConverter

logger = logging.getLogger(__name__)


class SpatialIndexType(Enum):
    """Types of spatial indexing strategies."""
    GRID = "grid"
    QUADTREE = "quadtree"
    HASH = "hash"


@dataclass
class SpatialBounds:
    """Represents spatial bounds of a region."""
    min_ra: float
    max_ra: float
    min_dec: float
    max_dec: float
    
    def contains(self, ra: float, dec: float) -> bool:
        """Check if coordinates are within bounds."""
        return (self.min_ra <= ra <= self.max_ra and 
                self.min_dec <= dec <= self.max_dec)
    
    def overlaps(self, other: 'SpatialBounds') -> bool:
        """Check if this bounds overlaps with another."""
        return not (self.max_ra < other.min_ra or 
                   self.min_ra > other.max_ra or
                   self.max_dec < other.min_dec or
                   self.min_dec > other.max_dec)
    
    def area(self) -> float:
        """Calculate area of bounds in square degrees."""
        return (self.max_ra - self.min_ra) * (self.max_dec - self.min_dec)


@dataclass
class SpatialRegion:
    """Represents a spatial region with associated data."""
    bounds: SpatialBounds
    tract: int
    patch: str
    data: Optional[Dict[str, Any]] = None
    
    def center(self) -> Tuple[float, float]:
        """Get center coordinates of the region."""
        center_ra = (self.bounds.min_ra + self.bounds.max_ra) / 2
        center_dec = (self.bounds.min_dec + self.bounds.max_dec) / 2
        return (center_ra, center_dec)


class GridSpatialIndex:
    """Grid-based spatial index for fast lookup."""
    
    def __init__(self, grid_size: float = 1.0):
        """
        Initialize grid spatial index.
        
        Parameters
        ----------
        grid_size : float
            Size of each grid cell in degrees
        """
        self.grid_size = grid_size
        self.grid: Dict[Tuple[int, int], List[SpatialRegion]] = defaultdict(list)
        self.region_count = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _get_grid_key(self, ra: float, dec: float) -> Tuple[int, int]:
        """Get grid key for given coordinates."""
        grid_ra = int(ra / self.grid_size)
        grid_dec = int(dec / self.grid_size)
        return (grid_ra, grid_dec)
    
    def _get_grid_keys_for_bounds(self, bounds: SpatialBounds) -> List[Tuple[int, int]]:
        """Get all grid keys that overlap with given bounds."""
        min_grid_ra = int(bounds.min_ra / self.grid_size)
        max_grid_ra = int(bounds.max_ra / self.grid_size)
        min_grid_dec = int(bounds.min_dec / self.grid_size)
        max_grid_dec = int(bounds.max_dec / self.grid_size)
        
        keys = []
        for grid_ra in range(min_grid_ra, max_grid_ra + 1):
            for grid_dec in range(min_grid_dec, max_grid_dec + 1):
                keys.append((grid_ra, grid_dec))
        
        return keys
    
    def add_region(self, region: SpatialRegion):
        """Add a spatial region to the index."""
        try:
            # Get all grid keys that this region overlaps
            grid_keys = self._get_grid_keys_for_bounds(region.bounds)
            
            # Add region to all overlapping grid cells
            for key in grid_keys:
                self.grid[key].append(region)
            
            self.region_count += 1
            
        except Exception as e:
            self.logger.error(f"Failed to add region to index: {e}")
    
    def query_point(self, ra: float, dec: float) -> List[SpatialRegion]:
        """Query regions that contain a given point."""
        try:
            grid_key = self._get_grid_key(ra, dec)
            candidates = self.grid.get(grid_key, [])
            
            # Filter to regions that actually contain the point
            results = []
            for region in candidates:
                if region.bounds.contains(ra, dec):
                    results.append(region)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query point ({ra}, {dec}): {e}")
            return []
    
    def query_region(self, bounds: SpatialBounds) -> List[SpatialRegion]:
        """Query regions that overlap with given bounds."""
        try:
            grid_keys = self._get_grid_keys_for_bounds(bounds)
            
            # Collect all candidate regions
            candidates = set()
            for key in grid_keys:
                candidates.update(self.grid.get(key, []))
            
            # Filter to regions that actually overlap
            results = []
            for region in candidates:
                if region.bounds.overlaps(bounds):
                    results.append(region)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query region: {e}")
            return []
    
    def query_radius(self, ra: float, dec: float, radius: float) -> List[SpatialRegion]:
        """Query regions within a radius (in degrees) of a point."""
        try:
            # Create bounds for the circular region (approximate)
            bounds = SpatialBounds(
                min_ra=ra - radius,
                max_ra=ra + radius,
                min_dec=dec - radius,
                max_dec=dec + radius
            )
            
            # Get candidate regions
            candidates = self.query_region(bounds)
            
            # Filter by actual distance
            results = []
            for region in candidates:
                region_ra, region_dec = region.center()
                
                # Calculate angular distance (simplified)
                delta_ra = abs(region_ra - ra)
                delta_dec = abs(region_dec - dec)
                
                # Approximate distance
                distance = math.sqrt(delta_ra**2 + delta_dec**2)
                
                if distance <= radius:
                    results.append(region)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query radius around ({ra}, {dec}): {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        total_cells = len(self.grid)
        occupied_cells = sum(1 for cell in self.grid.values() if cell)
        max_regions_per_cell = max(len(cell) for cell in self.grid.values()) if self.grid else 0
        avg_regions_per_cell = self.region_count / total_cells if total_cells > 0 else 0
        
        return {
            'type': 'grid',
            'grid_size': self.grid_size,
            'total_cells': total_cells,
            'occupied_cells': occupied_cells,
            'total_regions': self.region_count,
            'max_regions_per_cell': max_regions_per_cell,
            'avg_regions_per_cell': avg_regions_per_cell,
            'fill_ratio': occupied_cells / total_cells if total_cells > 0 else 0
        }


class QuadTreeNode:
    """Node in a quadtree spatial index."""
    
    def __init__(self, bounds: SpatialBounds, max_depth: int = 8, max_regions: int = 10):
        self.bounds = bounds
        self.max_depth = max_depth
        self.max_regions = max_regions
        self.regions: List[SpatialRegion] = []
        self.children: Optional[List['QuadTreeNode']] = None
        self.is_leaf = True
    
    def subdivide(self):
        """Subdivide this node into four children."""
        if not self.is_leaf:
            return
        
        mid_ra = (self.bounds.min_ra + self.bounds.max_ra) / 2
        mid_dec = (self.bounds.min_dec + self.bounds.max_dec) / 2
        
        # Create four child nodes
        self.children = [
            QuadTreeNode(SpatialBounds(self.bounds.min_ra, mid_ra, self.bounds.min_dec, mid_dec),
                        self.max_depth - 1, self.max_regions),  # SW
            QuadTreeNode(SpatialBounds(mid_ra, self.bounds.max_ra, self.bounds.min_dec, mid_dec),
                        self.max_depth - 1, self.max_regions),  # SE
            QuadTreeNode(SpatialBounds(self.bounds.min_ra, mid_ra, mid_dec, self.bounds.max_dec),
                        self.max_depth - 1, self.max_regions),  # NW
            QuadTreeNode(SpatialBounds(mid_ra, self.bounds.max_ra, mid_dec, self.bounds.max_dec),
                        self.max_depth - 1, self.max_regions)   # NE
        ]
        
        # Redistribute regions to children
        for region in self.regions:
            for child in self.children:
                if child.bounds.overlaps(region.bounds):
                    child.add_region(region)
        
        self.regions.clear()
        self.is_leaf = False
    
    def add_region(self, region: SpatialRegion):
        """Add a region to this node."""
        if self.is_leaf:
            self.regions.append(region)
            
            # Subdivide if we exceed capacity and can go deeper
            if len(self.regions) > self.max_regions and self.max_depth > 0:
                self.subdivide()
        else:
            # Add to appropriate children
            for child in self.children:
                if child.bounds.overlaps(region.bounds):
                    child.add_region(region)
    
    def query_point(self, ra: float, dec: float) -> List[SpatialRegion]:
        """Query regions that contain a point."""
        if not self.bounds.contains(ra, dec):
            return []
        
        if self.is_leaf:
            return [region for region in self.regions if region.bounds.contains(ra, dec)]
        else:
            results = []
            for child in self.children:
                results.extend(child.query_point(ra, dec))
            return results
    
    def query_region(self, bounds: SpatialBounds) -> List[SpatialRegion]:
        """Query regions that overlap with given bounds."""
        if not self.bounds.overlaps(bounds):
            return []
        
        if self.is_leaf:
            return [region for region in self.regions if region.bounds.overlaps(bounds)]
        else:
            results = []
            for child in self.children:
                results.extend(child.query_region(bounds))
            return results


class QuadTreeSpatialIndex:
    """QuadTree-based spatial index for hierarchical lookup."""
    
    def __init__(self, bounds: SpatialBounds, max_depth: int = 8, max_regions: int = 10):
        """
        Initialize quadtree spatial index.
        
        Parameters
        ----------
        bounds : SpatialBounds
            Overall bounds of the spatial domain
        max_depth : int
            Maximum depth of the quadtree
        max_regions : int
            Maximum regions per leaf node before subdivision
        """
        self.root = QuadTreeNode(bounds, max_depth, max_regions)
        self.region_count = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def add_region(self, region: SpatialRegion):
        """Add a spatial region to the index."""
        try:
            self.root.add_region(region)
            self.region_count += 1
        except Exception as e:
            self.logger.error(f"Failed to add region to quadtree: {e}")
    
    def query_point(self, ra: float, dec: float) -> List[SpatialRegion]:
        """Query regions that contain a given point."""
        try:
            return self.root.query_point(ra, dec)
        except Exception as e:
            self.logger.error(f"Failed to query point ({ra}, {dec}): {e}")
            return []
    
    def query_region(self, bounds: SpatialBounds) -> List[SpatialRegion]:
        """Query regions that overlap with given bounds."""
        try:
            return self.root.query_region(bounds)
        except Exception as e:
            self.logger.error(f"Failed to query region: {e}")
            return []
    
    def query_radius(self, ra: float, dec: float, radius: float) -> List[SpatialRegion]:
        """Query regions within a radius of a point."""
        try:
            bounds = SpatialBounds(
                min_ra=ra - radius,
                max_ra=ra + radius,
                min_dec=dec - radius,
                max_dec=dec + radius
            )
            
            candidates = self.query_region(bounds)
            
            # Filter by actual distance
            results = []
            for region in candidates:
                region_ra, region_dec = region.center()
                delta_ra = abs(region_ra - ra)
                delta_dec = abs(region_dec - dec)
                distance = math.sqrt(delta_ra**2 + delta_dec**2)
                
                if distance <= radius:
                    results.append(region)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query radius around ({ra}, {dec}): {e}")
            return []


class SpatialIndexManager:
    """
    Manages spatial indexing for tract/patch lookup and optimization.
    
    This class provides efficient spatial queries for LSST data access
    with support for multiple indexing strategies.
    """
    
    def __init__(self, 
                 coordinate_converter: CoordinateConverter,
                 index_type: SpatialIndexType = SpatialIndexType.GRID,
                 **index_params):
        """
        Initialize spatial index manager.
        
        Parameters
        ----------
        coordinate_converter : CoordinateConverter
            Coordinate conversion utilities
        index_type : SpatialIndexType
            Type of spatial index to use
        **index_params
            Additional parameters for the specific index type
        """
        self.coordinate_converter = coordinate_converter
        self.index_type = index_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize spatial index
        self.spatial_index = self._create_spatial_index(index_type, **index_params)
        
        # Performance metrics
        self.query_times: List[float] = []
        self.build_time: Optional[float] = None
        
        # Build index
        self._build_index()
    
    def _create_spatial_index(self, index_type: SpatialIndexType, **params):
        """Create spatial index of specified type."""
        if index_type == SpatialIndexType.GRID:
            grid_size = params.get('grid_size', 1.0)
            return GridSpatialIndex(grid_size)
        
        elif index_type == SpatialIndexType.QUADTREE:
            # Default bounds for LSST sky coverage
            bounds = params.get('bounds', SpatialBounds(0, 360, -90, 90))
            max_depth = params.get('max_depth', 8)
            max_regions = params.get('max_regions', 10)
            return QuadTreeSpatialIndex(bounds, max_depth, max_regions)
        
        else:
            raise ValueError(f"Unsupported spatial index type: {index_type}")
    
    def _build_index(self):
        """Build spatial index from skymap information."""
        try:
            start_time = time.time()
            
            skymap = self.coordinate_converter.skymap
            
            # Iterate through all tracts
            for tract_info in skymap:
                tract_id = tract_info.tract_id
                
                # Get all patches in this tract
                patch_info_list = tract_info.patch_info
                
                for patch_info in patch_info_list:
                    patch_index = patch_info.index
                    patch_str = f"{patch_index[0]},{patch_index[1]}"
                    
                    try:
                        # Get bounds for this tract/patch
                        bounds = self._get_tract_patch_bounds(tract_id, patch_str)
                        
                        # Create spatial region
                        region = SpatialRegion(
                            bounds=bounds,
                            tract=tract_id,
                            patch=patch_str,
                            data={
                                'tract_id': tract_id,
                                'patch_index': patch_index,
                                'patch_str': patch_str
                            }
                        )
                        
                        # Add to spatial index
                        self.spatial_index.add_region(region)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to index tract {tract_id} patch {patch_str}: {e}")
                        continue
            
            self.build_time = time.time() - start_time
            
            self.logger.info(f"Built spatial index with {getattr(self.spatial_index, 'region_count', 0)} regions in {self.build_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to build spatial index: {e}")
    
    def _get_tract_patch_bounds(self, tract: int, patch: str) -> SpatialBounds:
        """Get spatial bounds for a tract/patch."""
        try:
            min_ra, max_ra, min_dec, max_dec = self.coordinate_converter.get_tract_patch_bounds(tract, patch)
            return SpatialBounds(min_ra, max_ra, min_dec, max_dec)
        except Exception as e:
            # Fallback to center point with small bounds
            ra, dec = self.coordinate_converter.tract_patch_to_radec(tract, patch)
            margin = 0.1  # degrees
            return SpatialBounds(ra - margin, ra + margin, dec - margin, dec + margin)
    
    def find_tract_patch(self, ra: float, dec: float) -> Optional[Tuple[int, str]]:
        """Find tract/patch for given coordinates using spatial index."""
        try:
            start_time = time.time()
            
            # Query spatial index
            regions = self.spatial_index.query_point(ra, dec)
            
            elapsed_time = time.time() - start_time
            self.query_times.append(elapsed_time)
            
            if regions:
                # Return the first matching region
                region = regions[0]
                return (region.tract, region.patch)
            
            # Fallback to coordinate converter
            self.logger.debug(f"No indexed region found for ({ra}, {dec}), using fallback")
            return self.coordinate_converter.radec_to_tract_patch(ra, dec)
            
        except Exception as e:
            self.logger.error(f"Failed to find tract/patch for ({ra}, {dec}): {e}")
            return None
    
    def find_tract_patches_in_radius(self, ra: float, dec: float, radius: float) -> List[Tuple[int, str]]:
        """Find all tract/patch combinations within radius."""
        try:
            start_time = time.time()
            
            # Query spatial index
            regions = self.spatial_index.query_radius(ra, dec, radius)
            
            elapsed_time = time.time() - start_time
            self.query_times.append(elapsed_time)
            
            # Extract tract/patch pairs
            results = [(region.tract, region.patch) for region in regions]
            
            # Remove duplicates while preserving order
            unique_results = []
            seen = set()
            for tract, patch in results:
                key = (tract, patch)
                if key not in seen:
                    unique_results.append(key)
                    seen.add(key)
            
            return unique_results
            
        except Exception as e:
            self.logger.error(f"Failed to find tract/patches in radius around ({ra}, {dec}): {e}")
            return []
    
    def find_tract_patches_in_region(self, min_ra: float, max_ra: float, 
                                   min_dec: float, max_dec: float) -> List[Tuple[int, str]]:
        """Find all tract/patch combinations in a rectangular region."""
        try:
            start_time = time.time()
            
            bounds = SpatialBounds(min_ra, max_ra, min_dec, max_dec)
            regions = self.spatial_index.query_region(bounds)
            
            elapsed_time = time.time() - start_time
            self.query_times.append(elapsed_time)
            
            # Extract tract/patch pairs
            results = [(region.tract, region.patch) for region in regions]
            
            # Remove duplicates
            unique_results = list(set(results))
            
            return unique_results
            
        except Exception as e:
            self.logger.error(f"Failed to find tract/patches in region: {e}")
            return []
    
    def get_neighboring_regions(self, tract: int, patch: str, radius: float = 0.5) -> List[Tuple[int, str]]:
        """Get neighboring tract/patch regions."""
        try:
            # Get center coordinates of the given tract/patch
            ra, dec = self.coordinate_converter.tract_patch_to_radec(tract, patch)
            
            # Find regions within radius
            neighbors = self.find_tract_patches_in_radius(ra, dec, radius)
            
            # Remove the original tract/patch from results
            neighbors = [(t, p) for t, p in neighbors if not (t == tract and p == patch)]
            
            return neighbors
            
        except Exception as e:
            self.logger.error(f"Failed to get neighbors for tract {tract} patch {patch}: {e}")
            return []
    
    def optimize_query_order(self, coordinates: List[Tuple[float, float]]) -> List[Tuple[int, Tuple[float, float]]]:
        """Optimize query order to minimize spatial index lookups."""
        try:
            # Group coordinates by spatial proximity
            clustered_coords = []
            
            for i, (ra, dec) in enumerate(coordinates):
                # Find tract/patch for this coordinate
                tract_patch = self.find_tract_patch(ra, dec)
                if tract_patch:
                    clustered_coords.append((i, (ra, dec), tract_patch))
            
            # Sort by tract first, then patch
            clustered_coords.sort(key=lambda x: (x[2][0], x[2][1]))
            
            # Return reordered indices with coordinates
            return [(idx, coords) for idx, coords, _ in clustered_coords]
            
        except Exception as e:
            self.logger.error(f"Failed to optimize query order: {e}")
            return list(enumerate(coordinates))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the spatial index."""
        avg_query_time = sum(self.query_times) / len(self.query_times) if self.query_times else 0
        
        stats = {
            'index_type': self.index_type.value,
            'build_time': self.build_time,
            'total_queries': len(self.query_times),
            'avg_query_time': avg_query_time,
            'min_query_time': min(self.query_times) if self.query_times else 0,
            'max_query_time': max(self.query_times) if self.query_times else 0
        }
        
        # Add index-specific statistics
        if hasattr(self.spatial_index, 'get_statistics'):
            stats.update(self.spatial_index.get_statistics())
        
        return stats
    
    def clear_performance_stats(self):
        """Clear performance statistics."""
        self.query_times.clear()
        self.logger.info("Cleared spatial index performance statistics")
    
    def rebuild_index(self):
        """Rebuild the spatial index."""
        self.logger.info("Rebuilding spatial index...")
        self.query_times.clear()
        self._build_index()
        self.logger.info("Spatial index rebuilt")