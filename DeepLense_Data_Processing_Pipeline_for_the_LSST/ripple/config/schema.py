"""
Configuration Schema and Validation

Provides functionality for loading and validating RIPPLe configuration files.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .models import RippleConfig


logger = logging.getLogger(__name__)


class ConfigSchema:
    """
    Configuration schema validator and loader for RIPPLe pipeline.
    """
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration template.
        
        Returns:
            Dictionary containing default configuration structure
        """
        return {
            "version": "1.0",
            "description": "RIPPLe pipeline configuration",
            "butler": {
                "repo_path": "/path/to/butler/repository",
                "collections": None  # null = auto-discover
            },
            "data_selection": {
                "filters": ["r"],  # Photometric bands
                "visits": {
                    "ranges": [[903342, 903342]]  # Visit ID ranges
                },
                "detectors": None,  # null = all available
                "sky_region": {
                    "ra_range": None,   # [min_ra, max_ra] in degrees
                    "dec_range": None   # [min_dec, max_dec] in degrees
                }
            },
            "processing": {
                "mode": "batch",  # "batch" or "individual"
                "batch_size": 10,
                "data_products": {
                    "required": ["calexp"],
                    "optional": ["src", "postISRCCD"]
                },
                "cutout_size": None,  # For DeepLense processing
                "preprocessing_steps": []
            },
            "output": {
                "output_directory": "./output",
                "dataset_name": "ripple_dataset",
                "format": "fits",  # "fits", "hdf5", "parquet"
                "create_directories": True
            }
        }
    
    @staticmethod
    def validate_config_dict(config_dict: Dict[str, Any]) -> bool:
        """
        Validate configuration dictionary structure.
        
        Args:
            config_dict: Configuration dictionary loaded from YAML
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required top-level sections
        if 'butler' not in config_dict:
            raise ValueError("Configuration must contain 'butler' section")
        
        # Check required butler fields (support both single and multiple repos)
        butler_config = config_dict['butler']
        
        # Check if using single repo format (backward compatibility)
        single_repo = butler_config.get('repo_path')
        multiple_repos = butler_config.get('repositories')
        
        if single_repo and multiple_repos:
            raise ValueError("Cannot specify both 'repo_path' and 'repositories'. Use one or the other.")
        
        if not single_repo and not multiple_repos:
            raise ValueError("Must specify either 'repo_path' or 'repositories'")
        
        # Validate single repository format
        if single_repo:
            repo_path = Path(single_repo)
            if not repo_path.exists():
                raise ValueError(f"Butler repository path does not exist: {repo_path}")
        
        # Validate multiple repositories format
        if multiple_repos:
            if not isinstance(multiple_repos, list) or len(multiple_repos) == 0:
                raise ValueError("'repositories' must be a non-empty list")
            
            for i, repo in enumerate(multiple_repos):
                if not isinstance(repo, dict):
                    raise ValueError(f"Repository {i+1} must be a dictionary")
                
                if 'name' not in repo or 'repo_path' not in repo:
                    raise ValueError(f"Repository {i+1} must have 'name' and 'repo_path' fields")
                
                repo_path = Path(repo['repo_path'])
                if not repo_path.exists():
                    raise ValueError(f"Repository '{repo['name']}' path does not exist: {repo_path}")
                
                # Validate priority if specified
                if 'priority' in repo and (not isinstance(repo['priority'], int) or repo['priority'] < 1):
                    raise ValueError(f"Repository '{repo['name']}' priority must be a positive integer")
        
        # Validate processing mode if specified
        if 'processing' in config_dict:
            processing = config_dict['processing']
            mode = processing.get('mode', 'batch')
            if mode not in ['batch', 'individual']:
                raise ValueError("processing.mode must be 'batch' or 'individual'")
            
            batch_size = processing.get('batch_size', 10)
            if not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("processing.batch_size must be a positive integer")
        
        # Validate output format if specified
        if 'output' in config_dict:
            output = config_dict['output']
            if 'format' in output:
                format_type = output['format']
                if format_type not in ['fits', 'hdf5', 'parquet']:
                    raise ValueError("output.format must be 'fits', 'hdf5', or 'parquet'")
        
        return True


def load_config(config_path: str) -> RippleConfig:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        RippleConfig object with validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration validation fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
    
    if not config_dict:
        raise ValueError("Configuration file is empty or invalid")
    
    # Validate configuration structure
    ConfigSchema.validate_config_dict(config_dict)
    
    # Create RippleConfig object
    try:
        config = RippleConfig.from_dict(config_dict)
        logger.info("Configuration loaded and validated successfully")
        return config
    except Exception as e:
        raise ValueError(f"Failed to create configuration object: {e}")


def validate_config(config: RippleConfig) -> bool:
    """
    Validate a RippleConfig object.
    
    Args:
        config: RippleConfig object to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Validate butler repository paths
    repositories = config.butler.get_repositories()
    if not repositories:
        raise ValueError("No Butler repositories configured")
    
    for repo_config in repositories:
        repo_path = Path(repo_config.repo_path)
        if not repo_path.exists():
            raise ValueError(f"Butler repository path does not exist: {repo_path}")
        
        # Check for butler.yaml file
        butler_yaml = repo_path / "butler.yaml"
        if not butler_yaml.exists():
            raise ValueError(f"Butler configuration file not found: {butler_yaml}")
    
    # Validate processing configuration
    if config.processing.mode not in ['batch', 'individual']:
        raise ValueError("processing.mode must be 'batch' or 'individual'")
    
    if config.processing.batch_size < 1:
        raise ValueError("processing.batch_size must be positive")
    
    # Validate output configuration if present
    if config.output:
        if config.output.format not in ['fits', 'hdf5', 'parquet']:
            raise ValueError("output.format must be 'fits', 'hdf5', or 'parquet'")
    
    logger.info("Configuration validation successful")
    return True


def create_sample_config(output_path: str, template: str = "default") -> None:
    """
    Create a sample configuration file.
    
    Args:
        output_path: Path where to save the sample configuration
        template: Template type ("default", "minimal", "full")
    """
    if template == "minimal":
        config = {
            "butler": {
                "repo_path": "demo_data/pipelines_check-29.1.1/DATA_REPO"
            }
        }
    elif template == "full":
        config = ConfigSchema.get_default_config()
        # Add more detailed examples for full template
        config["data_selection"]["filters"] = ["g", "r", "i", "z", "y"]
        config["data_selection"]["visits"]["ranges"] = [[903342, 903342], [903343, 903345]]
        config["processing"]["preprocessing_steps"] = ["normalize", "resize", "augment"]
    else:  # default
        config = ConfigSchema.get_default_config()
        # Use our demo repository as default
        config["butler"]["repo_path"] = "demo_data/pipelines_check-29.1.1/DATA_REPO"
    
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Sample configuration created: {output_path}")