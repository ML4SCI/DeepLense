# RIPPLe - Rubin Image Preparation and Processing Lensing Engine

**Production-scale pipeline bridging LSST data access with DeepLense deep learning workflows**

[![LSST Version](https://img.shields.io/badge/LSST-v28.0.1%20|%20v29.1.1-blue)](lsst_stack/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)](environment.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12-blue)](environment.yml)
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Project Overview

RIPPLe (Rubin Image Preparation and Processing Lensing engine) is a comprehensive data processing pipeline designed to interface the Large Synoptic Survey Telescope (LSST) data products with DeepLense machine learning applications. This pipeline enables efficient data retrieval, preprocessing, and adaptation for gravitational lensing analysis at unprecedented scale.

### Key Applications

1. **Automated Strong Lens Finding** - Process ~100,000 expected lenses from LSST
2. **Dark Matter Substructure Classification** - Extract physics insights from lensing patterns
3. **Image Super-Resolution** - Enhance ground-based observations using deep learning

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RIPPLe Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────┐ │
│  │Data Ingestion│───▶│ Preprocessing │───▶│ML Inference│───▶│ Output  │ │
│  │   (Butler)   │    │   Pipeline    │    │  (Models)  │    │Handler  │ │
│  └─────────────┘    └──────────────┘    └────────────┘    └─────────┘ │
│         │                    │                   │               │       │
│         ▼                    ▼                   ▼               ▼       │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────┐ │
│  │LSST Butler  │    │ Image/WCS    │    │ DeepLense  │    │Results  │ │
│  │Data Products│    │ Processing   │    │  PyTorch   │    │Storage  │ │
│  └─────────────┘    └──────────────┘    └────────────┘    └─────────┘ │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Orchestrator & Configuration                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary
```
Input → LsstDataFetcher → Preprocessor → ModelInterface → ResultHandler → Output
  ↑                                                                          ↓
  └────────────────────── Orchestrator (coordinates flow) ──────────────────┘
```

## Quick Start

### Prerequisites

- Linux system with CUDA support (optional but recommended)
- Anaconda/Miniconda installed
- 32GB+ RAM recommended
- NVIDIA GPU with 8GB+ VRAM (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ML4SCI/DeepLense_Data_Processing_Pipeline_for_the_LSST.git
cd DeepLense_Data_Processing_Pipeline_for_the_LSST
```

2. **Set up LSST Science Pipelines**

We provide multiple LSST versions for compatibility:

```bash
# Option 1: Use stable v28.0.1
source lsst_stack/loadLSST.bash
setup lsst_distrib

# Option 2: Use latest v29.1.1 (recommended)
source lsst_stack3/loadLSST.bash
setup lsst_distrib
```

3. **Verify installation**
```bash
python manual_tests/run_tests.py
```

### Basic Usage

```python
from ripple.data_access import LsstDataFetcher
from ripple.pipeline import PipelineOrchestrator

# Initialize pipeline
config_path = "config_demo.yaml"
pipeline = PipelineOrchestrator(config_path)

# Process a list of targets
targets = [
    {"ra": 150.1, "dec": 2.3, "name": "lens_001"},
    {"ra": 150.2, "dec": 2.4, "name": "lens_002"}
]

results = pipeline.process_targets(targets)
```

## Project Structure

```
DeepLense_Data_Processing_Pipeline_for_the_LSST/
├── ripple/                    # Main Python package
│   ├── butler/                # Butler wrapper utilities
│   ├── butler_repo/           # Repository management
│   ├── data_access/           # Data fetching and caching
│   ├── preprocessing/         # Image preprocessing
│   ├── pipeline/              # Pipeline orchestration
│   ├── models/                # Model interfaces
│   └── utils/                 # Utility functions
├── lsst_stack/                # LSST v28.0.1
├── lsst_stack2/               # LSST v29.0.x
├── lsst_stack3/               # LSST v29.1.1 (latest)
├── manual_tests/              # Test suite
├── demo_data/                 # Sample Butler repository
├── rc2_subset/                # RC2 subset data
├── testdata_decam/            # DECam test data
├── presentations/             # Project presentations
└── config_demo.yaml           # Example configuration
```

## Core Components

### 1. Data Access Layer (`ripple.data_access`)

Efficient data retrieval from LSST Butler repositories with advanced features:

- **Smart caching** for repeated queries
- **Coordinate conversion** (RA/Dec to tract/patch)
- **Batch retrieval** optimization
- **Error handling** with retry logic
- **Performance monitoring**

```python
from ripple.data_access import LsstDataFetcher

fetcher = LsstDataFetcher(butler_config)
cutout = fetcher.fetch_cutout(
    ra=150.1, 
    dec=2.3, 
    size=64,
    filters=['g', 'r', 'i']
)
```

### 2. Preprocessing Module (`ripple.preprocessing`)

Standardized preprocessing pipeline for LSST images:

- **WCS-aware cutout extraction**
- **Multi-band alignment and stacking**
- **Flexible normalization** (MinMax, ZScore, Asinh)
- **PSF matching** (optional)
- **Background subtraction**

### 3. Pipeline Orchestrator (`ripple.pipeline`)

End-to-end workflow management:

- **Batch processing** with GPU optimization
- **Error recovery** and fault tolerance
- **Progress tracking** and logging
- **Configurable processing chains**

### 4. Model Integration (`ripple.models`)

Unified interface for DeepLense models:

- **Lens detection** (binary classification)
- **Substructure analysis** (multi-class)
- **Super-resolution** (2x-4x upsampling)
- **Custom model support**

## Configuration

RIPPLe uses YAML configuration files for flexibility:

```yaml
# config_demo.yaml
data:
  butler:
    repo: "./demo_data/pipelines_check-29.1.1/DATA_REPO"
    collections: ["HSC/RC2/defaults"]
  dataset_type: "deepCoadd"
  filters: ["g", "r", "i"]
  
preprocessing:
  normalization: "minmax"
  cutout_size: 64
  
models:
  task: "lens_finding"
  checkpoint: "models/lens_finder.pth"
  device: "cuda:0"
  
pipeline:
  batch_size: 32
  num_workers: 4
```

## Testing

### Run all tests
```bash
python manual_tests/run_tests.py
```

### Individual test modules
```bash
python manual_tests/01_environment_setup.py      # Environment verification
python manual_tests/02_configuration_tests.py    # Configuration validation
python manual_tests/03_butler_connection_tests.py # Butler connectivity
python manual_tests/04_data_availability_tests.py # Data access tests
```

### Example test scripts
```bash
python test_butler_creator.py        # Test Butler repository creation
python test_fixed_pipeline.py        # Test end-to-end pipeline
python test_raw_detection.py         # Test raw data detection
```


## Project Timeline

- **Phase 0** (Completed): Environment Setup & Infrastructure
- **Phase 1** (Completed): Data Access Layer & Butler Integration
- **Phase 2** (In Progress): Preprocessing Pipeline & Model Integration
- **Phase 3** (Upcoming): Production Deployment & Optimization



