"""
RIPPLe (Rubin Image Preparation and Processing Lensing engine)

A production-scale data processing pipeline that bridges LSST data access tools 
with DeepLense deep learning workflows for gravitational lensing analysis.

Copyright (c) 2025 ML4SCI - Machine Learning for Science
"""

__version__ = "1.0.0-dev"
__author__ = "Kartik Mandar"
__email__ = "kartik4321mandar@gmail.com"

from .data_access import LsstDataFetcher
from .preprocessing import Preprocessor
from .pipeline import PipelineOrchestrator
from .models import ModelInterface

__all__ = [
    "LsstDataFetcher",
    "Preprocessor", 
    "PipelineOrchestrator",
    "ModelInterface",
]