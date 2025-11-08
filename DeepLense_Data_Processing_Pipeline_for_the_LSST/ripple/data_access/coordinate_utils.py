"""
Coordinate conversion utilities for RIPPLe.

This module provides coordinate system conversion between sky coordinates
(RA/Dec) and LSST tract/patch system using skymap information.
"""

from typing import Tuple, List, Optional
import logging
import math

# LSST imports
from lsst.geom import SpherePoint, degrees, Point2D, Box2I, Point2I, Extent2I
from lsst.skymap import BaseSkyMap

# RIPPLe imports
from .exceptions import CoordinateConversionError

logger = logging.getLogger(__name__)


class CoordinateConverter:
    """
    Coordinate conversion utilities using LSST skymap.
    
    This class provides methods to convert between different coordinate
    systems used in LSST data processing.
    """
    
    def __init__(self, skymap: BaseSkyMap):
        """Initialize with skymap."""
        self.skymap = skymap
    
    def radec_to_tract_patch(self, ra: float, dec: float) -> Tuple[int, str]:
        """Convert RA/Dec to tract/patch."""
        try:
            # Create SpherePoint from RA/Dec
            coord = SpherePoint(ra * degrees, dec * degrees)
            
            # Find tract containing the coordinate
            tract_info = self.skymap.findTract(coord)
            tract_id = tract_info.tract_id
            
            # Find patch containing the coordinate
            patch_info = tract_info.findPatch(coord)
            patch_str = f"{patch_info.index[0]},{patch_info.index[1]}"
            
            return (tract_id, patch_str)
            
        except Exception as e:
            raise CoordinateConversionError(
                f"Failed to convert RA/Dec ({ra}, {dec}) to tract/patch: {e}",
                from_coords=(ra, dec)
            )
    
    def radec_to_tract_patch_radius(self, ra: float, dec: float, 
                                   radius: float) -> List[Tuple[int, str]]:
        """Find all tract/patch combinations within radius."""
        try:
            # Create SpherePoint from RA/Dec
            center_coord = SpherePoint(ra * degrees, dec * degrees)
            
            # Find all tracts that overlap with the circular region
            # This is a simplified implementation - in practice, you might want
            # to use more sophisticated geometric algorithms
            
            tract_patches = []
            
            # Get the central tract/patch
            central_tract, central_patch = self.radec_to_tract_patch(ra, dec)
            tract_patches.append((central_tract, central_patch))
            
            # For small radii, we might only need adjacent patches
            # For larger radii, we need to check multiple tracts
            radius_degrees = radius
            
            # Sample points around the circle and find their tracts/patches
            n_samples = 8  # Sample 8 points around the circle
            for i in range(n_samples):
                angle = 2 * 3.14159 * i / n_samples
                # Approximate offset in degrees (simplified)
                offset_ra = radius_degrees * 0.7071 * (1 if i % 2 == 0 else -1)
                offset_dec = radius_degrees * 0.7071 * (1 if i < 4 else -1)
                
                sample_ra = ra + offset_ra
                sample_dec = dec + offset_dec
                
                # Ensure coordinates are within valid ranges
                if -90 <= sample_dec <= 90:
                    try:
                        tract, patch = self.radec_to_tract_patch(sample_ra, sample_dec)
                        if (tract, patch) not in tract_patches:
                            tract_patches.append((tract, patch))
                    except CoordinateConversionError:
                        # Skip coordinates that can't be converted
                        continue
            
            return tract_patches
            
        except Exception as e:
            raise CoordinateConversionError(
                f"Failed to find tract/patch within radius {radius} of ({ra}, {dec}): {e}",
                from_coords=(ra, dec)
            )
    
    def tract_patch_to_radec(self, tract: int, patch: str) -> Tuple[float, float]:
        """Convert tract/patch to central RA/Dec."""
        try:
            # Parse patch string (e.g., "5,7")
            if ',' in patch:
                patch_x, patch_y = map(int, patch.split(','))
            else:
                # Handle alternative patch formats
                patch_x, patch_y = int(patch), 0
            
            # Get tract info
            tract_info = self.skymap.findTract(tract)
            
            # Get patch info
            patch_info = tract_info.findPatch((patch_x, patch_y))
            
            # Get the center coordinate of the patch
            center_coord = patch_info.getInnerBBox().getCenter()
            
            # Convert pixel coordinates to sky coordinates using tract WCS
            wcs = tract_info.getWcs()
            sky_coord = wcs.pixelToSky(center_coord)
            
            # Extract RA/Dec in degrees
            ra = sky_coord.getRa().asDegrees()
            dec = sky_coord.getDec().asDegrees()
            
            return (ra, dec)
            
        except Exception as e:
            raise CoordinateConversionError(
                f"Failed to convert tract/patch ({tract}, {patch}) to RA/Dec: {e}",
                from_coords=(tract, patch)
            )
    
    def sky_to_pixel(self, ra: float, dec: float, wcs) -> Tuple[float, float]:
        """Convert sky coordinates to pixel coordinates using WCS."""
        try:
            sky_coord = SpherePoint(ra * degrees, dec * degrees)
            pixel_coord = wcs.skyToPixel(sky_coord)
            return (pixel_coord.x, pixel_coord.y)
        except Exception as e:
            raise CoordinateConversionError(
                f"Failed to convert sky to pixel: {e}",
                from_coords=(ra, dec)
            )
    
    def pixel_to_sky(self, x: float, y: float, wcs) -> Tuple[float, float]:
        """Convert pixel coordinates to sky coordinates using WCS."""
        try:
            pixel_coord = Point2D(x, y)
            sky_coord = wcs.pixelToSky(pixel_coord)
            return (sky_coord.getRa().asDegrees(), sky_coord.getDec().asDegrees())
        except Exception as e:
            raise CoordinateConversionError(
                f"Failed to convert pixel to sky: {e}",
                from_coords=(x, y)
            )
    
    def calculate_bbox_from_radec(self, ra: float, dec: float, size: int, wcs) -> Box2I:
        """Calculate pixel bounding box from RA/Dec center and size."""
        try:
            # Convert center coordinates to pixels
            center_x, center_y = self.sky_to_pixel(ra, dec, wcs)
            
            # Calculate half-size
            half_size = size // 2
            
            # Create bounding box
            bbox = Box2I(
                Point2I(int(center_x - half_size), int(center_y - half_size)),
                Extent2I(size, size)
            )
            
            return bbox
            
        except Exception as e:
            raise CoordinateConversionError(
                f"Failed to calculate bbox for RA/Dec ({ra}, {dec}): {e}",
                from_coords=(ra, dec)
            )
    
    def validate_coordinates(self, ra: float, dec: float) -> bool:
        """Validate RA/Dec coordinates."""
        try:
            # Check RA range (0-360 degrees)
            if not (0 <= ra <= 360):
                return False
                
            # Check Dec range (-90 to 90 degrees) 
            if not (-90 <= dec <= 90):
                return False
                
            return True
            
        except Exception:
            return False
    
    def angular_separation(self, ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        """Calculate angular separation between two sky coordinates in degrees."""
        try:
            coord1 = SpherePoint(ra1 * degrees, dec1 * degrees)
            coord2 = SpherePoint(ra2 * degrees, dec2 * degrees)
            
            separation = coord1.separation(coord2)
            return separation.asDegrees()
            
        except Exception as e:
            raise CoordinateConversionError(
                f"Failed to calculate angular separation: {e}",
                from_coords=(ra1, dec1, ra2, dec2)
            )
    
    def get_tract_patch_bounds(self, tract: int, patch: str) -> Tuple[float, float, float, float]:
        """Get the RA/Dec bounds of a tract/patch."""
        try:
            # Parse patch string
            if ',' in patch:
                patch_x, patch_y = map(int, patch.split(','))
            else:
                patch_x, patch_y = int(patch), 0
            
            # Get tract info
            tract_info = self.skymap.findTract(tract)
            
            # Get patch info  
            patch_info = tract_info.findPatch((patch_x, patch_y))
            
            # Get patch bounding box
            bbox = patch_info.getInnerBBox()
            
            # Get WCS for coordinate conversion
            wcs = tract_info.getWcs()
            
            # Convert corners to sky coordinates
            corners = [
                bbox.getMin(),
                Point2D(bbox.getMaxX(), bbox.getMinY()),
                bbox.getMax(),
                Point2D(bbox.getMinX(), bbox.getMaxY())
            ]
            
            ra_coords = []
            dec_coords = []
            
            for corner in corners:
                sky_coord = wcs.pixelToSky(corner)
                ra_coords.append(sky_coord.getRa().asDegrees())
                dec_coords.append(sky_coord.getDec().asDegrees())
            
            return (min(ra_coords), max(ra_coords), min(dec_coords), max(dec_coords))
            
        except Exception as e:
            raise CoordinateConversionError(
                f"Failed to get bounds for tract/patch ({tract}, {patch}): {e}",
                from_coords=(tract, patch)
            )