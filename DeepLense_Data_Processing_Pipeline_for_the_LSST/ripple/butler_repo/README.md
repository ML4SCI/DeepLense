# RIPPLe Butler Repository Management

This module provides automated Butler repository creation and management for the RIPPLe pipeline.

## Overview

The butler_repo module handles:
- Creating new Butler Gen3 repositories
- Ingesting raw data, calibrations, and reference catalogs
- Managing repository configuration
- Setting up collections and instruments
- Integrating with RIPPLe data access

## Usage

### 1. Using the Main Pipeline

The recommended way is through the main RIPPLe pipeline:

```bash
# Activate LSST environment
source ~/RIPPLe/lsst_stack/loadLSST.sh
setup lsst_distrib

# Generate a default configuration
python -m ripple.main --generate-config my_config.yaml

# Edit the configuration file
# Then run the pipeline
python -m ripple.main my_config.yaml
```

### 2. Using the Butler Repo Manager Directly

For more control over repository creation:

```bash
# Create repository from configuration
python -m ripple.butler_repo.repo_manager config.yaml

# Or with command-line options
python -m ripple.butler_repo.repo_manager \
    --data-path /path/to/data \
    --instrument HSC \
    --transfer symlink
```

### 3. Programmatic Usage

```python
from ripple.butler_repo import ButlerRepoManager, load_config

# Load configuration
config = load_config("config.yaml")

# Create and setup repository
manager = ButlerRepoManager(config)
success, repo_path = manager.setup_repository()

if success:
    # Get data fetcher for immediate use
    data_fetcher = manager.get_data_fetcher()
```

## Configuration

The system uses YAML configuration files with these main sections:

### Data Source
```yaml
data_source:
  type: data_folder  # or butler_repo, butler_server
  path: /path/to/data
  create_if_missing: true
```

### Instrument
```yaml
instrument:
  name: HSC
  class_name: lsst.obs.subaru.HyperSuprimeCam
  filters: [g, r, i, z, y]
```

### Ingestion
```yaml
ingestion:
  raw_data_pattern: "**/*.fits"
  transfer_mode: symlink
  define_visits: true
```

## Workflow

1. **Repository Creation**
   - Creates Butler repository structure
   - Registers instrument
   - Writes curated calibrations

2. **Data Detection**
   - Searches for export.yaml files
   - Falls back to manual pattern matching
   - Detects calibration types automatically

3. **Data Ingestion**
   - Ingests raw exposures
   - Defines visits
   - Ingests calibrations (bias, dark, flat)
   - Registers and ingests reference catalogs

4. **Collection Setup**
   - Creates default collection chains
   - Organizes data by type
   - Sets up search paths

## Examples

### Demo Data Setup
```yaml
# config_demo.yaml
data_source:
  type: data_folder
  path: /home/user/RIPPLe/demo_data/pipelines_check-29.1.1/input_data

instrument:
  name: HSC
  class_name: lsst.obs.subaru.HyperSuprimeCam
```

### Existing Repository
```yaml
data_source:
  type: butler_repo
  path: /datasets/DC2/repo
  collections: [2.2i/runs/DP0.2]
```

### Creating from Raw Data
```yaml
data_source:
  type: data_folder
  path: /data/observations/2024

ingestion:
  raw_data_pattern: "*/raw/*.fits"
  calibration_path: /data/calibrations
  reference_catalog_path: /data/refcats
```

## Module Structure

- `config_handler.py` - Configuration management
- `create_repo.py` - Repository creation utilities
- `ingest_data.py` - Data ingestion logic
- `repo_manager.py` - Main orchestrator
- `utils.py` - Helper utilities

## Error Handling

The system handles:
- Missing data gracefully
- Partial ingestion failures
- Existing repository detection
- Invalid configurations
- LSST environment issues

All operations are logged with detailed error messages.

## Requirements

- LSST Science Pipelines (v28.0.0+)
- Active LSST environment
- Write permissions for repository location
- Sufficient disk space for data

## Troubleshooting

### Environment Not Found
```bash
source /path/to/lsst_stack/loadLSST.sh
setup lsst_distrib
```

### Permission Denied
Ensure write permissions for repository location:
```bash
chmod -R u+w /path/to/repo
```

### Ingestion Failures
Check log files for specific errors. Common issues:
- Invalid FITS files
- Missing required headers
- Incorrect instrument specification

### Memory Issues
Reduce batch size in configuration:
```yaml
processing:
  batch_size: 16
  max_workers: 2
```