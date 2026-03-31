"""
Configuration Manager for DeepLense Hyperparameters
Addresses Issue #208: Abstract hyperparameters into YAML/JSON configuration files

Provides a unified configuration system for managing hyperparameters across
all DeepLense sub-projects, enabling reproducibility and agentic automation.
"""

import json
import os
import copy
from typing import Any, Dict, Optional, Union


# Default configurations for common DeepLense tasks
DEFAULT_CLASSIFICATION_CONFIG = {
    "model": {
        "architecture": "resnet18",
        "num_classes": 3,
        "pretrained": True,
        "input_channels": 1,
        "input_size": 64
    },
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "scheduler_params": {
            "T_max": 50,
            "eta_min": 1e-6
        }
    },
    "data": {
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "normalize": True,
        "augmentation": True,
        "num_workers": 4,
        "pin_memory": True
    },
    "logging": {
        "log_interval": 10,
        "save_checkpoints": True,
        "checkpoint_dir": "./checkpoints",
        "use_wandb": False,
        "wandb_project": "deeplense",
        "wandb_entity": None
    },
    "seed": 42
}

DEFAULT_REGRESSION_CONFIG = {
    "model": {
        "architecture": "resnet18",
        "num_outputs": 1,
        "pretrained": True,
        "input_channels": 1,
        "input_size": 64
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "reduce_on_plateau",
        "scheduler_params": {
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-7
        }
    },
    "data": {
        "train_split": 0.7,
        "val_split": 0.15,
        "test_split": 0.15,
        "normalize": True,
        "augmentation": False,
        "num_workers": 4,
        "pin_memory": True
    },
    "logging": {
        "log_interval": 10,
        "save_checkpoints": True,
        "checkpoint_dir": "./checkpoints",
        "use_wandb": False,
        "wandb_project": "deeplense",
        "wandb_entity": None
    },
    "seed": 42
}


class ConfigManager:
    """
    Manages hyperparameter configurations for DeepLense experiments.
    
    Supports loading from JSON/YAML files, merging with defaults,
    validation, and export for reproducibility.
    
    Usage:
        # Load from file
        config = ConfigManager.from_file("experiment_config.json")
        
        # Access parameters
        lr = config.get("training.learning_rate")
        
        # Use defaults
        config = ConfigManager.from_defaults("classification")
        
        # Override specific values
        config.set("training.epochs", 100)
        config.set("model.architecture", "efficientnet_v2_s")
    """
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize with optional configuration dictionary."""
        self._config = config_dict or {}
    
    @classmethod
    def from_defaults(cls, task_type: str = "classification") -> "ConfigManager":
        """
        Create configuration from built-in defaults.
        
        Args:
            task_type: One of 'classification' or 'regression'
            
        Returns:
            ConfigManager instance with default configuration
        """
        if task_type == "classification":
            return cls(copy.deepcopy(DEFAULT_CLASSIFICATION_CONFIG))
        elif task_type == "regression":
            return cls(copy.deepcopy(DEFAULT_REGRESSION_CONFIG))
        else:
            raise ValueError(
                f"Unknown task type '{task_type}'. "
                f"Supported: 'classification', 'regression'"
            )
    
    @classmethod
    def from_file(cls, filepath: str, merge_defaults: bool = True,
                  task_type: str = "classification") -> "ConfigManager":
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            filepath: Path to the configuration file (.json or .yaml/.yml)
            merge_defaults: If True, merge with default config (file overrides)
            task_type: Default task type for merging
            
        Returns:
            ConfigManager instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".json":
            with open(filepath, "r") as f:
                file_config = json.load(f)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
                with open(filepath, "r") as f:
                    file_config = yaml.safe_load(f)
            except ImportError:
                raise ImportError(
                    "PyYAML is required to load YAML files. "
                    "Install it with: pip install pyyaml"
                )
        else:
            raise ValueError(
                f"Unsupported configuration file format: {ext}. "
                f"Use .json, .yaml, or .yml"
            )
        
        if merge_defaults:
            if task_type == "classification":
                defaults = copy.deepcopy(DEFAULT_CLASSIFICATION_CONFIG)
            else:
                defaults = copy.deepcopy(DEFAULT_REGRESSION_CONFIG)
            merged = cls._deep_merge(defaults, file_config)
            return cls(merged)
        
        return cls(file_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ConfigManager":
        """Create configuration from a dictionary."""
        return cls(copy.deepcopy(config_dict))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., 'training.learning_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., 'training.learning_rate')
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Return the full configuration as a dictionary."""
        return copy.deepcopy(self._config)
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to a JSON or YAML file.
        
        Args:
            filepath: Output file path
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        if ext == ".json":
            with open(filepath, "w") as f:
                json.dump(self._config, f, indent=2, default=str)
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
                with open(filepath, "w") as f:
                    yaml.dump(self._config, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML required. Install: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported format: {ext}")
    
    def validate(self) -> list:
        """
        Validate the configuration for common errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check training parameters
        lr = self.get("training.learning_rate")
        if lr is not None and (lr <= 0 or lr > 1):
            errors.append(f"learning_rate ({lr}) should be in (0, 1]")
        
        epochs = self.get("training.epochs")
        if epochs is not None and (not isinstance(epochs, int) or epochs < 1):
            errors.append(f"epochs ({epochs}) must be a positive integer")
        
        bs = self.get("training.batch_size")
        if bs is not None and (not isinstance(bs, int) or bs < 1):
            errors.append(f"batch_size ({bs}) must be a positive integer")
        
        # Check data splits
        train = self.get("data.train_split", 0)
        val = self.get("data.val_split", 0)
        test = self.get("data.test_split", 0)
        total = train + val + test
        if abs(total - 1.0) > 0.01:
            errors.append(
                f"Data splits should sum to 1.0, got {total:.2f} "
                f"(train={train}, val={val}, test={test})"
            )
        
        # Check seed
        seed = self.get("seed")
        if seed is not None and not isinstance(seed, int):
            errors.append(f"seed must be an integer, got {type(seed).__name__}")
        
        # Check model
        num_classes = self.get("model.num_classes")
        if num_classes is not None and (not isinstance(num_classes, int) or num_classes < 2):
            errors.append(f"num_classes ({num_classes}) must be >= 2")
        
        return errors
    
    def diff(self, other: "ConfigManager") -> Dict:
        """
        Compare two configurations and return differences.
        
        Args:
            other: Another ConfigManager to compare with
            
        Returns:
            Dictionary of differences {key: (self_value, other_value)}
        """
        return self._find_diffs(self._config, other._config)
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries. Override values take precedence."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    
    @staticmethod
    def _find_diffs(d1: Dict, d2: Dict, prefix: str = "") -> Dict:
        """Recursively find differences between two dictionaries."""
        diffs = {}
        all_keys = set(list(d1.keys()) + list(d2.keys()))
        
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in d1:
                diffs[full_key] = (None, d2[key])
            elif key not in d2:
                diffs[full_key] = (d1[key], None)
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                nested = ConfigManager._find_diffs(d1[key], d2[key], full_key)
                diffs.update(nested)
            elif d1[key] != d2[key]:
                diffs[full_key] = (d1[key], d2[key])
        
        return diffs
    
    def __repr__(self) -> str:
        return f"ConfigManager({json.dumps(self._config, indent=2, default=str)})"
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)
