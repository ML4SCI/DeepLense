"""
Butler repository creation utilities.

This module provides functions for creating and initializing
LSST Butler Gen3 repositories.
"""

import os
import subprocess
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from .config_handler import RepoConfig, ButlerConfig

logger = logging.getLogger(__name__)


def create_butler_repository(
    repo_path: str,
    butler_config: ButlerConfig,
    overwrite: bool = False
) -> bool:
    """
    Create a new Butler Gen3 repository.
    
    Parameters
    ----------
    repo_path : str
        Path where the repository will be created
    butler_config : ButlerConfig
        Butler configuration options
    overwrite : bool, optional
        Whether to overwrite existing repository
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    repo_path = Path(repo_path)
    
    # Check if repository already exists
    if repo_path.exists() and (repo_path / "butler.yaml").exists():
        if not overwrite:
            logger.warning(f"Butler repository already exists at {repo_path}")
            return True
        else:
            logger.info(f"Overwriting existing repository at {repo_path}")
    
    # Create directory if needed
    repo_path.mkdir(parents=True, exist_ok=True)
    
    # Build butler create command
    cmd = ["butler", "create", str(repo_path)]
    
    # Add configuration options
    if butler_config.seed_config:
        cmd.extend(["--seed-config", butler_config.seed_config])
    
    if butler_config.dimension_config:
        cmd.extend(["--dimension-config", butler_config.dimension_config])
    
    if butler_config.standalone:
        cmd.append("--standalone")
    
    if butler_config.override:
        cmd.append("--override")
    
    # Execute command
    try:
        logger.info(f"Creating Butler repository: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully created Butler repository at {repo_path}")
            
            # Create PostgreSQL seed config if needed
            if butler_config.registry_db == "postgresql" and butler_config.postgres_url:
                _create_postgres_seed_config(repo_path, butler_config.postgres_url)
            
            return True
        else:
            logger.error(f"Failed to create repository: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Butler create command failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating repository: {e}")
        return False


def register_instrument(
    repo_path: str,
    instrument_class: str
) -> bool:
    """
    Register an instrument in the Butler repository.
    
    Parameters
    ----------
    repo_path : str
        Path to the Butler repository
    instrument_class : str
        Fully qualified instrument class name
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd = [
        "butler", "register-instrument",
        str(repo_path),
        instrument_class
    ]
    
    try:
        logger.info(f"Registering instrument: {instrument_class}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully registered instrument: {instrument_class}")
            return True
        else:
            logger.error(f"Failed to register instrument: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        # Check if instrument is already registered
        if "already exists" in e.stderr:
            logger.info(f"Instrument {instrument_class} already registered")
            return True
        logger.error(f"Failed to register instrument: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error registering instrument: {e}")
        return False


def write_curated_calibrations(
    repo_path: str,
    instrument_name: str
) -> bool:
    """
    Write curated calibrations for an instrument.
    
    Parameters
    ----------
    repo_path : str
        Path to the Butler repository
    instrument_name : str
        Instrument name (e.g., 'HSC', 'LSSTCam')
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd = [
        "butler", "write-curated-calibrations",
        str(repo_path),
        instrument_name
    ]
    
    try:
        logger.info(f"Writing curated calibrations for {instrument_name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully wrote curated calibrations")
            return True
        else:
            logger.error(f"Failed to write curated calibrations: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to write curated calibrations: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error writing calibrations: {e}")
        return False


def create_collection_chain(
    repo_path: str,
    chain_name: str,
    collections: List[str],
    mode: str = "create"
) -> bool:
    """
    Create or update a collection chain.
    
    Parameters
    ----------
    repo_path : str
        Path to the Butler repository
    chain_name : str
        Name of the collection chain
    collections : List[str]
        Collections to include in the chain
    mode : str, optional
        Chain creation mode ('create', 'extend', 'prepend', 'redefine')
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd = [
        "butler", "collection-chain",
        str(repo_path),
        chain_name,
        "--mode", mode
    ] + collections
    
    try:
        logger.info(f"Creating collection chain: {chain_name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully created collection chain: {chain_name}")
            return True
        else:
            logger.error(f"Failed to create collection chain: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create collection chain: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating collection chain: {e}")
        return False


def make_discrete_skymap(
    repo_path: str,
    instrument_name: str,
    collections: Optional[List[str]] = None,
    skymap_name: str = "discrete"
) -> bool:
    """
    Create a discrete skymap based on available data.
    
    Parameters
    ----------
    repo_path : str
        Path to the Butler repository
    instrument_name : str
        Instrument name
    collections : List[str], optional
        Collections to search for data
    skymap_name : str, optional
        Name for the skymap
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd = [
        "butler", "make-discrete-skymap",
        str(repo_path),
        instrument_name
    ]
    
    if collections:
        cmd.extend(["--collections", ",".join(collections)])
    
    if skymap_name:
        cmd.extend(["-c", f"name='{skymap_name}'"])
    
    try:
        logger.info(f"Creating discrete skymap for {instrument_name}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully created discrete skymap")
            return True
        else:
            logger.error(f"Failed to create skymap: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create skymap: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating skymap: {e}")
        return False


def initialize_repository(
    config: RepoConfig,
    repo_path: str,
    overwrite: bool = False
) -> bool:
    """
    Initialize a complete Butler repository.
    
    Parameters
    ----------
    config : RepoConfig
        Repository configuration
    repo_path : str
        Path to the repository
    overwrite : bool, optional
        Whether to overwrite existing repository
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Create repository
    if not create_butler_repository(repo_path, config.butler, overwrite):
        return False
    
    # Register instrument
    if not register_instrument(repo_path, config.instrument.class_name):
        return False
    
    # Write curated calibrations if requested
    if config.ingestion.write_curated_calibrations:
        if not write_curated_calibrations(repo_path, config.instrument.name):
            logger.warning("Failed to write curated calibrations, continuing...")
    
    # Skip collection chain creation for now - let the import handle it
    logger.info("Skipping collection chain creation - will be created during import")
    
    logger.info(f"Repository initialized successfully at {repo_path}")
    return True


def _create_postgres_seed_config(repo_path: Path, postgres_url: str) -> None:
    """Create seed configuration for PostgreSQL."""
    seed_config = {
        "registry": {
            "db": postgres_url
        }
    }
    
    import yaml
    seed_path = repo_path / "butler-seed.yaml"
    with open(seed_path, 'w') as f:
        yaml.dump(seed_config, f)
    
    logger.info(f"Created PostgreSQL seed configuration at {seed_path}")


def verify_repository(repo_path: str) -> Dict[str, Any]:
    """
    Verify that a Butler repository is properly initialized.
    
    Parameters
    ----------
    repo_path : str
        Path to the Butler repository
        
    Returns
    -------
    Dict[str, Any]
        Verification results
    """
    repo_path = Path(repo_path)
    results = {
        "exists": False,
        "has_butler_yaml": False,
        "has_registry": False,
        "collections": [],
        "instruments": [],
        "errors": []
    }
    
    # Check if repository exists
    if not repo_path.exists():
        results["errors"].append(f"Repository path does not exist: {repo_path}")
        return results
    
    results["exists"] = True
    
    # Check for butler.yaml
    if (repo_path / "butler.yaml").exists():
        results["has_butler_yaml"] = True
    else:
        results["errors"].append("Missing butler.yaml file")
    
    # Check for registry database
    if (repo_path / "gen3.sqlite3").exists():
        results["has_registry"] = True
    else:
        # Might be using PostgreSQL
        if (repo_path / "butler.yaml").exists():
            results["has_registry"] = True
    
    # Try to query collections
    try:
        cmd = ["butler", "query-collections", str(repo_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            results["collections"] = result.stdout.strip().split('\n')
    except Exception as e:
        results["errors"].append(f"Failed to query collections: {e}")
    
    return results