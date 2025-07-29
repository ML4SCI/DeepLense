"""
Configuration Data Models

Defines the data models and structures for RIPPLe pipeline configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path


@dataclass
class ButlerRepoConfig:
    """Configuration for a single Butler repository."""
    name: str
    repo_path: str
    collections: Optional[List[str]] = None  # None = auto-discover
    priority: int = 1  # Higher number = higher priority when data exists in multiple repos
    
    def __post_init__(self):
        """Validate and normalize the repository path."""
        self.repo_path = str(Path(self.repo_path).resolve())


@dataclass
class ButlerConfig:
    """Configuration for Butler repository access (supports single or multiple repos)."""
    # Support both single repo (backward compatibility) and multiple repos
    repo_path: Optional[str] = None
    collections: Optional[List[str]] = None
    repositories: Optional[List[ButlerRepoConfig]] = None
    
    def __post_init__(self):
        """Validate and normalize Butler configuration."""
        # Ensure either single repo or multiple repos specified, not both
        single_repo_specified = self.repo_path is not None
        multiple_repos_specified = self.repositories is not None and len(self.repositories) > 0
        
        if single_repo_specified and multiple_repos_specified:
            raise ValueError("Cannot specify both 'repo_path' and 'repositories'. Use one or the other.")
        
        if not single_repo_specified and not multiple_repos_specified:
            raise ValueError("Must specify either 'repo_path' or 'repositories'")
        
        # Convert single repo to multiple repo format for consistency
        if single_repo_specified:
            self.repositories = [ButlerRepoConfig(
                name="default",
                repo_path=self.repo_path,
                collections=self.collections,
                priority=1
            )]
            # Clear single repo fields for clarity
            self.repo_path = None
            self.collections = None
    
    def get_repositories(self) -> List[ButlerRepoConfig]:
        """Get list of repository configurations."""
        return self.repositories or []
    
    def get_primary_repository(self) -> ButlerRepoConfig:
        """Get the primary (highest priority) repository."""
        if not self.repositories:
            raise ValueError("No repositories configured")
        return max(self.repositories, key=lambda r: r.priority)


@dataclass  
class DataSelection:
    """Configuration for data selection criteria."""
    filters: Optional[List[str]] = None  # e.g., ["r", "g", "i"]
    visits: Optional[Dict[str, Any]] = None  # {"ranges": [[903342, 903342]]}
    detectors: Optional[List[int]] = None  # [10, 12, 14] or None for all
    sky_region: Optional[Dict[str, Any]] = None  # {"ra_range": [min, max], "dec_range": [min, max]}
    
    def __post_init__(self):
        """Validate data selection parameters."""
        if self.visits and not isinstance(self.visits.get('ranges'), list):
            raise ValueError("visits.ranges must be a list of [min, max] pairs")


@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""
    mode: str = "batch"  # "batch" or "individual"  
    batch_size: int = 10
    data_products: Optional[Dict[str, List[str]]] = None
    cutout_size: Optional[int] = None  # For DeepLense processing
    preprocessing_steps: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate processing configuration."""
        if self.mode not in ["batch", "individual"]:
            raise ValueError("mode must be 'batch' or 'individual'")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        
        # Set default data products if not specified
        if self.data_products is None:
            self.data_products = {
                "required": ["calexp"],
                "optional": ["src", "postISRCCD"]
            }


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    output_directory: str
    dataset_name: str
    format: str = "fits"  # "fits", "hdf5", "parquet"
    create_directories: bool = True
    
    def __post_init__(self):
        """Validate and normalize output configuration."""
        self.output_directory = str(Path(self.output_directory).resolve())
        if self.format not in ["fits", "hdf5", "parquet"]:
            raise ValueError("format must be 'fits', 'hdf5', or 'parquet'")


@dataclass
class RippleConfig:
    """Main configuration class for RIPPLe pipeline."""
    butler: ButlerConfig
    data_selection: DataSelection = field(default_factory=DataSelection)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: Optional[OutputConfig] = None
    
    # Metadata
    version: str = "1.0"
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RippleConfig':
        """Create RippleConfig from dictionary (loaded from YAML)."""
        
        # Extract Butler config (support both single and multiple repos)
        butler_dict = config_dict.get('butler', {})
        
        # Handle multiple repositories format
        if 'repositories' in butler_dict:
            repo_configs = []
            for repo_data in butler_dict['repositories']:
                if not isinstance(repo_data, dict):
                    raise ValueError("Each repository must be a dictionary with name and repo_path")
                if 'name' not in repo_data or 'repo_path' not in repo_data:
                    raise ValueError("Each repository must have 'name' and 'repo_path' fields")
                
                repo_configs.append(ButlerRepoConfig(
                    name=repo_data['name'],
                    repo_path=repo_data['repo_path'],
                    collections=repo_data.get('collections'),
                    priority=repo_data.get('priority', 1)
                ))
            
            butler_config = ButlerConfig(repositories=repo_configs)
        
        # Handle single repository format (backward compatibility)
        elif 'repo_path' in butler_dict:
            butler_config = ButlerConfig(
                repo_path=butler_dict['repo_path'],
                collections=butler_dict.get('collections')
            )
        else:
            raise ValueError("Butler configuration must contain either 'repo_path' or 'repositories'")
        
        # Extract optional data selection
        data_selection_dict = config_dict.get('data_selection', {})
        data_selection = DataSelection(
            filters=data_selection_dict.get('filters'),
            visits=data_selection_dict.get('visits'),
            detectors=data_selection_dict.get('detectors'),
            sky_region=data_selection_dict.get('sky_region')
        )
        
        # Extract processing config
        processing_dict = config_dict.get('processing', {})
        processing = ProcessingConfig(
            mode=processing_dict.get('mode', 'batch'),
            batch_size=processing_dict.get('batch_size', 10),
            data_products=processing_dict.get('data_products'),
            cutout_size=processing_dict.get('cutout_size'),
            preprocessing_steps=processing_dict.get('preprocessing_steps', [])
        )
        
        # Extract output config (optional)
        output = None
        if 'output' in config_dict:
            output_dict = config_dict['output']
            output = OutputConfig(
                output_directory=output_dict['output_directory'],
                dataset_name=output_dict['dataset_name'],
                format=output_dict.get('format', 'fits'),
                create_directories=output_dict.get('create_directories', True)
            )
        
        return cls(
            butler=butler_config,
            data_selection=data_selection,
            processing=processing,
            output=output,
            version=config_dict.get('version', '1.0'),
            description=config_dict.get('description')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RippleConfig to dictionary for serialization."""
        result = {
            'version': self.version,
            'butler': {
                'repo_path': self.butler.repo_path,
                'collections': self.butler.collections
            },
            'data_selection': {
                'filters': self.data_selection.filters,
                'visits': self.data_selection.visits,
                'detectors': self.data_selection.detectors,
                'sky_region': self.data_selection.sky_region
            },
            'processing': {
                'mode': self.processing.mode,
                'batch_size': self.processing.batch_size,
                'data_products': self.processing.data_products,
                'cutout_size': self.processing.cutout_size,
                'preprocessing_steps': self.processing.preprocessing_steps
            }
        }
        
        if self.output:
            result['output'] = {
                'output_directory': self.output.output_directory,
                'dataset_name': self.output.dataset_name,
                'format': self.output.format,
                'create_directories': self.output.create_directories
            }
        
        if self.description:
            result['description'] = self.description
            
        return result