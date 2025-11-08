"""
Utility functions for Butler repository management.
"""

import os
import subprocess
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def check_lsst_environment() -> bool:
    """
    Check if LSST environment is properly set up.
    
    Returns
    -------
    bool
        True if LSST stack is available
    """
    try:
        # Check for eups command
        result = subprocess.run(
            ["eups", "list", "lsst_distrib"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            logger.info("LSST environment is properly configured")
            return True
        else:
            logger.error("LSST environment not found. Please run: source loadLSST.sh && setup lsst_distrib")
            return False
            
    except FileNotFoundError:
        logger.error("LSST stack not found. Please activate the LSST environment.")
        return False
    except Exception as e:
        logger.error(f"Error checking LSST environment: {e}")
        return False


def find_data_files(
    root_path: Path,
    patterns: List[str],
    max_files: Optional[int] = None
) -> List[Path]:
    """
    Find files matching patterns under root path.
    
    Parameters
    ----------
    root_path : Path
        Root directory to search
    patterns : List[str]
        Glob patterns to match
    max_files : int, optional
        Maximum number of files to return
        
    Returns
    -------
    List[Path]
        List of matching file paths
    """
    files = []
    
    for pattern in patterns:
        matches = list(root_path.glob(pattern))
        files.extend(matches)
        
        if max_files and len(files) >= max_files:
            files = files[:max_files]
            break
    
    return files


def detect_instrument_from_fits(fits_path: Path) -> Optional[str]:
    """
    Try to detect instrument from FITS header.
    
    Parameters
    ----------
    fits_path : Path
        Path to FITS file
        
    Returns
    -------
    Optional[str]
        Detected instrument name or None
    """
    try:
        # Try using LSST tools
        from lsst.afw.fits import readMetadata
        
        metadata = readMetadata(str(fits_path))
        
        # Check common instrument keywords
        instrument_keywords = ['INSTRUME', 'INSTRUMENT', 'TELESCOP']
        
        for keyword in instrument_keywords:
            if keyword in metadata:
                value = metadata[keyword].upper()
                
                # Map to standard names
                if 'HSC' in value or 'SUBARU' in value:
                    return 'HSC'
                elif 'LSST' in value:
                    return 'LSSTCam'
                elif 'DECAM' in value or 'CTIO' in value:
                    return 'DECam'
                elif 'LATISS' in value:
                    return 'LATISS'
                    
    except Exception:
        # Fallback to astropy if available
        try:
            from astropy.io import fits
            
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                
                for keyword in ['INSTRUME', 'INSTRUMENT', 'TELESCOP']:
                    if keyword in header:
                        value = str(header[keyword]).upper()
                        
                        if 'HSC' in value:
                            return 'HSC'
                        elif 'LSST' in value:
                            return 'LSSTCam'
                        elif 'DECAM' in value:
                            return 'DECam'
                            
        except Exception:
            pass
    
    return None


def get_instrument_info(instrument_name: str) -> Dict[str, Any]:
    """
    Get instrument configuration information.
    
    Parameters
    ----------
    instrument_name : str
        Instrument name
        
    Returns
    -------
    Dict[str, Any]
        Instrument information
    """
    instruments = {
        "HSC": {
            "class_name": "lsst.obs.subaru.HyperSuprimeCam",
            "filters": ["g", "r", "i", "z", "y", "NB0387", "NB0816", "NB0921"],
            "detectors": list(range(104)),
            "skymap": "hsc_rings_v1"
        },
        "LSSTCam": {
            "class_name": "lsst.obs.lsst.LsstCam",
            "filters": ["u", "g", "r", "i", "z", "y"],
            "detectors": list(range(189)),
            "skymap": "lsst_cells_v1"
        },
        "DECam": {
            "class_name": "lsst.obs.decam.DarkEnergyCamera",
            "filters": ["u", "g", "r", "i", "z", "Y", "VR"],
            "detectors": list(range(62)),
            "skymap": "decam_rings_v1"
        },
        "LATISS": {
            "class_name": "lsst.obs.lsst.Latiss",
            "filters": ["empty", "g", "r", "i", "z", "y"],
            "detectors": [0],
            "skymap": "latiss_v1"
        }
    }
    
    return instruments.get(instrument_name, {})


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human readable string.
    
    Parameters
    ----------
    size_bytes : int
        Size in bytes
        
    Returns
    -------
    str
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def estimate_repository_size(data_path: Path) -> Dict[str, Any]:
    """
    Estimate the size requirements for a Butler repository.
    
    Parameters
    ----------
    data_path : Path
        Path to data files
        
    Returns
    -------
    Dict[str, Any]
        Size estimates
    """
    total_size = 0
    file_count = 0
    
    # Count FITS files
    for fits_file in data_path.rglob("*.fits"):
        total_size += fits_file.stat().st_size
        file_count += 1
    
    # Estimate registry size (rough approximation)
    registry_size = file_count * 10_000  # ~10KB per file in registry
    
    return {
        "data_size": total_size,
        "data_size_formatted": format_size(total_size),
        "file_count": file_count,
        "estimated_registry_size": registry_size,
        "estimated_registry_size_formatted": format_size(registry_size),
        "total_size": total_size + registry_size,
        "total_size_formatted": format_size(total_size + registry_size)
    }


def validate_butler_command() -> bool:
    """
    Check if butler command is available.
    
    Returns
    -------
    bool
        True if butler command is available
    """
    try:
        result = subprocess.run(
            ["butler", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_butler_version() -> Optional[str]:
    """
    Get Butler version information.
    
    Returns
    -------
    Optional[str]
        Butler version string or None
    """
    try:
        # Try to get version from the butler package
        import lsst.daf.butler
        version = getattr(lsst.daf.butler, "__version__", None)
        if version:
            return f"Butler {version}"
        
        # Fallback to checking if butler command exists
        result = subprocess.run(
            ["butler", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return "Butler (version unknown)"
        
    except Exception:
        pass
    
    return None