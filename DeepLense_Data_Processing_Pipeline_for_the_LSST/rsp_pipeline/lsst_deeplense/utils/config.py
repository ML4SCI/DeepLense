"""
config.py
---------
YAML configuration loader for the LSST-DeepLense pipeline.

Config files follow the schema defined in ``configs/pipeline_config.yaml``.
All values can be overridden programmatically after loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML pipeline configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.yaml`` config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If the config file does not exist.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for config loading.  "
            "Install with:  pip install pyyaml"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    # Validate required top-level keys
    _validate_config(cfg, path)
    return cfg


def _validate_config(cfg: Dict[str, Any], path: Path) -> None:
    required_keys = {"data", "preprocessing", "pipeline"}
    missing = required_keys - set(cfg.keys())
    if missing:
        raise ValueError(
            f"Config '{path}' is missing required top-level keys: {missing}"
        )
