"""
Configuration handler for Butler repository management.

This module handles loading, validation, and management of configuration
for Butler repository creation and data ingestion.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data source."""
    type: str  # 'butler_repo', 'butler_server', 'data_folder'
    path: Optional[str] = None
    server_url: Optional[str] = None
    collections: List[str] = field(default_factory=list)
    create_if_missing: bool = True


@dataclass
class InstrumentConfig:
    """Configuration for instrument."""
    name: str  # e.g., 'HSC', 'LSSTCam', 'DECam'
    class_name: str  # e.g., 'lsst.obs.subaru.HyperSuprimeCam'
    filters: List[str] = field(default_factory=list)
    detector_list: Optional[List[int]] = None


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    raw_data_pattern: Optional[str] = None  # Glob pattern for raw data
    calibration_path: Optional[str] = None
    reference_catalog_path: Optional[str] = None
    transfer_mode: str = "symlink"  # symlink, copy, move, direct
    define_visits: bool = True
    write_curated_calibrations: bool = True
    skip_existing: bool = True
    processes: int = 1  # Number of parallel processes


@dataclass
class ButlerConfig:
    """Butler-specific configuration."""
    dimension_config: Optional[str] = None
    seed_config: Optional[str] = None
    standalone: bool = False
    override: bool = False
    registry_db: str = "sqlite"  # sqlite or postgresql
    postgres_url: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    cutout_size: int = 64
    batch_size: int = 32
    max_workers: int = 4
    cache_size: int = 1000
    enable_performance_monitoring: bool = True
    output_dir: Optional[str] = None


@dataclass
class RepoConfig:
    """Complete repository configuration."""
    data_source: DataSourceConfig
    instrument: InstrumentConfig
    ingestion: IngestionConfig
    butler: ButlerConfig
    processing: ProcessingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RepoConfig':
        """Create RepoConfig from dictionary."""
        return cls(
            data_source=DataSourceConfig(**config_dict.get('data_source', {})),
            instrument=InstrumentConfig(**config_dict.get('instrument', {})),
            ingestion=IngestionConfig(**config_dict.get('ingestion', {})),
            butler=ButlerConfig(**config_dict.get('butler', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {}))
        )


def load_config(config_path: Union[str, Path]) -> RepoConfig:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to configuration YAML file
        
    Returns
    -------
    RepoConfig
        Loaded configuration object
        
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If config file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Expand environment variables
        config_dict = _expand_env_vars(config_dict)
        
        # Create config object
        config = RepoConfig.from_dict(config_dict)
        
        # Validate configuration
        validate_config(config)
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def validate_config(config: RepoConfig) -> None:
    """
    Validate configuration.
    
    Parameters
    ----------
    config : RepoConfig
        Configuration to validate
        
    Raises
    ------
    ValueError
        If configuration is invalid
    """
    # Validate data source
    if config.data_source.type not in ['butler_repo', 'butler_server', 'data_folder']:
        raise ValueError(f"Invalid data source type: {config.data_source.type}")
    
    if config.data_source.type == 'butler_repo' and not config.data_source.path:
        raise ValueError("Butler repository path required for type 'butler_repo'")
    
    if config.data_source.type == 'butler_server' and not config.data_source.server_url:
        raise ValueError("Server URL required for type 'butler_server'")
    
    if config.data_source.type == 'data_folder' and not config.data_source.path:
        raise ValueError("Data folder path required for type 'data_folder'")
    
    # Validate instrument
    if not config.instrument.name:
        raise ValueError("Instrument name is required")
    
    if not config.instrument.class_name:
        raise ValueError("Instrument class name is required")
    
    # Validate ingestion
    if config.ingestion.transfer_mode not in ['symlink', 'copy', 'move', 'direct', 'hardlink']:
        raise ValueError(f"Invalid transfer mode: {config.ingestion.transfer_mode}")
    
    # Validate butler config
    if config.butler.registry_db not in ['sqlite', 'postgresql']:
        raise ValueError(f"Invalid registry database: {config.butler.registry_db}")
    
    if config.butler.registry_db == 'postgresql' and not config.butler.postgres_url:
        raise ValueError("PostgreSQL URL required when using PostgreSQL registry")
    
    # Validate paths exist if specified
    if config.data_source.path:
        path = Path(config.data_source.path)
        if config.data_source.type == 'butler_repo' and not config.data_source.create_if_missing:
            if not path.exists():
                raise ValueError(f"Butler repository path does not exist: {path}")
    
    logger.info("Configuration validated successfully")


def _expand_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in configuration."""
    if isinstance(config_dict, dict):
        return {k: _expand_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [_expand_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        return os.path.expandvars(config_dict)
    else:
        return config_dict


def get_default_config() -> RepoConfig:
    """Get default configuration."""
    return RepoConfig(
        data_source=DataSourceConfig(
            type='data_folder',
            path='./data',
            collections=['raw/all', 'calib', 'refcats'],
            create_if_missing=True
        ),
        instrument=InstrumentConfig(
            name='HSC',
            class_name='lsst.obs.subaru.HyperSuprimeCam',
            filters=['g', 'r', 'i', 'z', 'y']
        ),
        ingestion=IngestionConfig(
            raw_data_pattern='**/*.fits',
            transfer_mode='symlink',
            define_visits=True,
            write_curated_calibrations=True,
            skip_existing=True,
            processes=1
        ),
        butler=ButlerConfig(
            standalone=False,
            override=False,
            registry_db='sqlite'
        ),
        processing=ProcessingConfig(
            cutout_size=64,
            batch_size=32,
            max_workers=4,
            cache_size=1000,
            enable_performance_monitoring=True
        )
    )


def save_config(config: RepoConfig, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Parameters
    ----------
    config : RepoConfig
        Configuration to save
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path)
    
    # Convert to dictionary
    config_dict = {
        'data_source': {
            'type': config.data_source.type,
            'path': config.data_source.path,
            'server_url': config.data_source.server_url,
            'collections': config.data_source.collections,
            'create_if_missing': config.data_source.create_if_missing
        },
        'instrument': {
            'name': config.instrument.name,
            'class_name': config.instrument.class_name,
            'filters': config.instrument.filters,
            'detector_list': config.instrument.detector_list
        },
        'ingestion': {
            'raw_data_pattern': config.ingestion.raw_data_pattern,
            'calibration_path': config.ingestion.calibration_path,
            'reference_catalog_path': config.ingestion.reference_catalog_path,
            'transfer_mode': config.ingestion.transfer_mode,
            'define_visits': config.ingestion.define_visits,
            'write_curated_calibrations': config.ingestion.write_curated_calibrations,
            'skip_existing': config.ingestion.skip_existing,
            'processes': config.ingestion.processes
        },
        'butler': {
            'dimension_config': config.butler.dimension_config,
            'seed_config': config.butler.seed_config,
            'standalone': config.butler.standalone,
            'override': config.butler.override,
            'registry_db': config.butler.registry_db,
            'postgres_url': config.butler.postgres_url
        },
        'processing': {
            'cutout_size': config.processing.cutout_size,
            'batch_size': config.processing.batch_size,
            'max_workers': config.processing.max_workers,
            'cache_size': config.processing.cache_size,
            'enable_performance_monitoring': config.processing.enable_performance_monitoring,
            'output_dir': config.processing.output_dir
        }
    }
    
    # Remove None values
    config_dict = _remove_none_values(config_dict)
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {output_path}")


def _remove_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively remove None values from dictionary."""
    if isinstance(d, dict):
        return {k: _remove_none_values(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [_remove_none_values(item) for item in d]
    else:
        return d