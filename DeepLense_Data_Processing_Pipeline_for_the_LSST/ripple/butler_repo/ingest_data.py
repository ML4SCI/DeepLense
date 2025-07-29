"""
Data ingestion utilities for Butler repositories.

This module provides functions for ingesting raw data, calibrations,
and reference catalogs into Butler Gen3 repositories.
"""

import os
import subprocess
import logging
import glob
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

from .config_handler import RepoConfig, IngestionConfig

logger = logging.getLogger(__name__)


class DataIngestor:
    """Handles data ingestion into Butler repositories."""
    
    def __init__(self, repo_path: str, config: RepoConfig):
        """
        Initialize data ingestor.
        
        Parameters
        ----------
        repo_path : str
            Path to Butler repository
        config : RepoConfig
            Repository configuration
        """
        self.repo_path = Path(repo_path)
        self.config = config
        self.instrument = config.instrument.name
        
    def ingest_all(self) -> Dict[str, Any]:
        """
        Ingest all data according to configuration.
        
        Returns
        -------
        Dict[str, Any]
            Ingestion results
        """
        results = {
            "raw_data": {"success": False, "count": 0, "errors": []},
            "calibrations": {"success": False, "count": 0, "errors": []},
            "reference_catalogs": {"success": False, "count": 0, "errors": []},
            "visits_defined": False
        }
        
        # Ingest raw data
        if self.config.ingestion.raw_data_pattern:
            logger.info("Ingesting raw data...")
            raw_results = self.ingest_raw_data()
            results["raw_data"] = raw_results
            
            # Define visits if successful and requested
            if raw_results["success"] and self.config.ingestion.define_visits:
                logger.info("Defining visits...")
                results["visits_defined"] = self.define_visits()
        
        # Ingest calibrations
        if self.config.ingestion.calibration_path:
            logger.info("Ingesting calibrations...")
            results["calibrations"] = self.ingest_calibrations()
        
        # Ingest reference catalogs
        if self.config.ingestion.reference_catalog_path:
            logger.info("Ingesting reference catalogs...")
            results["reference_catalogs"] = self.ingest_reference_catalogs()
        
        return results
    
    def ingest_raw_data(self) -> Dict[str, Any]:
        """
        Ingest raw science data.
        
        Returns
        -------
        Dict[str, Any]
            Ingestion results
        """
        results = {"success": False, "count": 0, "errors": []}
        
        # Find data files
        data_path = Path(self.config.data_source.path)
        pattern = self.config.ingestion.raw_data_pattern or "**/*.fits"
        
        raw_files = []
        if data_path.is_dir():
            raw_files = list(data_path.glob(pattern))
        
        if not raw_files:
            results["errors"].append(f"No raw data files found matching pattern: {pattern}")
            return results
        
        logger.info(f"Found {len(raw_files)} raw data files")
        
        # Ingest in batches
        batch_size = 100
        total_ingested = 0
        
        for i in range(0, len(raw_files), batch_size):
            batch = raw_files[i:i+batch_size]
            
            cmd = [
                "butler", "ingest-raws",
                str(self.repo_path)
            ] + [str(f) for f in batch]
            
            # Add options
            cmd.extend(["--transfer", self.config.ingestion.transfer_mode])
            cmd.extend(["--output-run", f"{self.instrument}/raw/all"])
            
            # Note: ingest-raws doesn't have a skip-existing option
            
            if self.config.ingestion.processes > 1:
                cmd.extend(["-j", str(self.config.ingestion.processes)])
            
            # Execute ingestion
            try:
                logger.info(f"Ingesting batch {i//batch_size + 1}/{(len(raw_files) + batch_size - 1)//batch_size}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Count successful ingestions
                if "Ingested" in result.stdout:
                    # Parse output to get count
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "Ingested" in line:
                            try:
                                count = int(line.split()[1])
                                total_ingested += count
                            except:
                                total_ingested += len(batch)
                else:
                    total_ingested += len(batch)
                    
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to ingest batch: {e.stderr}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                # Continue with next batch
        
        results["success"] = total_ingested > 0
        results["count"] = total_ingested
        
        if results["success"]:
            logger.info(f"Successfully ingested {total_ingested} raw files")
        
        return results
    
    def define_visits(self) -> bool:
        """
        Define visits from ingested exposures.
        
        Returns
        -------
        bool
            True if successful
        """
        cmd = [
            "butler", "define-visits",
            str(self.repo_path),
            self.config.instrument.class_name,
            "--collections", f"{self.instrument}/raw/all"
        ]
        
        try:
            logger.info("Defining visits from exposures...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.returncode == 0:
                logger.info("Successfully defined visits")
                return True
            else:
                logger.error(f"Failed to define visits: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to define visits: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error defining visits: {e}")
            return False
    
    def ingest_calibrations(self) -> Dict[str, Any]:
        """
        Ingest calibration data (bias, dark, flat).
        
        Returns
        -------
        Dict[str, Any]
            Ingestion results
        """
        results = {"success": False, "count": 0, "errors": []}
        
        calib_path = Path(self.config.ingestion.calibration_path)
        if not calib_path.exists():
            results["errors"].append(f"Calibration path does not exist: {calib_path}")
            return results
        
        # Define calibration types and their patterns
        calib_types = {
            "bias": ["**/bias*.fits", "**/zero*.fits"],
            "dark": ["**/dark*.fits"],
            "flat": ["**/flat*.fits", "**/skyflat*.fits", "**/domeflat*.fits"]
        }
        
        total_ingested = 0
        
        for calib_type, patterns in calib_types.items():
            # Find calibration files
            calib_files = []
            for pattern in patterns:
                calib_files.extend(calib_path.glob(pattern))
            
            if not calib_files:
                logger.warning(f"No {calib_type} calibration files found")
                continue
            
            logger.info(f"Found {len(calib_files)} {calib_type} files")
            
            # Ingest calibrations
            cmd = [
                "butler", "ingest-raws",
                str(self.repo_path)
            ] + [str(f) for f in calib_files]
            
            cmd.extend(["--transfer", self.config.ingestion.transfer_mode])
            cmd.extend(["--output-run", f"{self.instrument}/calib/{calib_type}"])
            
            # Note: ingest-raws doesn't have a skip-existing option
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Count ingested files
                count = len(calib_files)
                total_ingested += count
                logger.info(f"Ingested {count} {calib_type} files")
                
                # Certify calibrations
                if self._certify_calibrations(calib_type):
                    logger.info(f"Certified {calib_type} calibrations")
                
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to ingest {calib_type}: {e.stderr}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        results["success"] = total_ingested > 0
        results["count"] = total_ingested
        
        return results
    
    def _certify_calibrations(self, calib_type: str) -> bool:
        """Certify calibrations with validity range."""
        # Get current date for validity range
        today = datetime.now()
        begin_date = today.strftime("%Y-%m-%dT00:00:00")
        end_date = today.strftime("%Y-%m-%dT23:59:59")
        
        cmd = [
            "butler", "certify-calibrations",
            str(self.repo_path),
            f"{self.instrument}/calib/{calib_type}",
            f"{self.instrument}/calib",
            calib_type,
            "--begin-date", begin_date,
            "--end-date", end_date
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.returncode == 0
        except:
            return False
    
    def ingest_reference_catalogs(self) -> Dict[str, Any]:
        """
        Ingest reference catalogs.
        
        Returns
        -------
        Dict[str, Any]
            Ingestion results
        """
        results = {"success": False, "count": 0, "errors": []}
        
        refcat_path = Path(self.config.ingestion.reference_catalog_path)
        if not refcat_path.exists():
            results["errors"].append(f"Reference catalog path does not exist: {refcat_path}")
            return results
        
        # Find reference catalog types
        catalog_types = self._detect_reference_catalogs(refcat_path)
        
        if not catalog_types:
            results["errors"].append("No reference catalogs found")
            return results
        
        total_ingested = 0
        
        for catalog_name, catalog_files in catalog_types.items():
            logger.info(f"Ingesting {catalog_name} reference catalog ({len(catalog_files)} files)")
            
            # Register dataset type
            if self._register_refcat_dataset_type(catalog_name):
                # Ingest files
                if self._ingest_refcat_files(catalog_name, catalog_files):
                    total_ingested += len(catalog_files)
                    
                    # Add to refcats collection
                    self._add_to_refcats_collection(catalog_name)
                else:
                    results["errors"].append(f"Failed to ingest {catalog_name}")
            else:
                results["errors"].append(f"Failed to register dataset type for {catalog_name}")
        
        results["success"] = total_ingested > 0
        results["count"] = total_ingested
        
        return results
    
    def _detect_reference_catalogs(self, refcat_path: Path) -> Dict[str, List[Path]]:
        """Detect reference catalog types and files."""
        catalogs = {}
        
        # Common reference catalog patterns
        catalog_patterns = {
            "gaia_dr2": ["**/gaia*dr2*.fits", "**/gaia*dr2*.fits"],
            "gaia_dr3": ["**/gaia*dr3*.fits"],
            "ps1_pv3": ["**/ps1*.fits", "**/panstarrs*.fits"],
            "2mass": ["**/2mass*.fits", "**/twomass*.fits"]
        }
        
        # Search for catalog files
        for catalog_type, patterns in catalog_patterns.items():
            files = []
            for pattern in patterns:
                files.extend(refcat_path.glob(pattern))
            
            if files:
                # Extract version info from filename if possible
                sample_file = files[0].name
                import re  # Import at the beginning of the block
                
                if "gaia_dr2" in sample_file.lower():
                    # Extract date version if present
                    match = re.search(r'(\d{8})', sample_file)
                    if match:
                        catalog_name = f"gaia_dr2_{match.group(1)}"
                    else:
                        catalog_name = "gaia_dr2_20200414"  # Default
                elif "ps1" in sample_file.lower():
                    match = re.search(r'(\d{8})', sample_file)
                    if match:
                        catalog_name = f"ps1_pv3_3pi_{match.group(1)}"
                    else:
                        catalog_name = "ps1_pv3_3pi_20170110"  # Default
                else:
                    catalog_name = catalog_type
                
                catalogs[catalog_name] = files
        
        return catalogs
    
    def _register_refcat_dataset_type(self, catalog_name: str) -> bool:
        """Register reference catalog dataset type."""
        cmd = [
            "butler", "register-dataset-type",
            str(self.repo_path),
            catalog_name,
            "SimpleCatalog",
            "htm7"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            if "already exists" in e.stderr:
                return True
            logger.error(f"Failed to register dataset type: {e.stderr}")
            return False
    
    def _ingest_refcat_files(self, catalog_name: str, files: List[Path]) -> bool:
        """Ingest reference catalog files."""
        cmd = [
            "butler", "ingest-files",
            "-t", "direct",
            str(self.repo_path),
            catalog_name,
            "refcats"
        ] + [str(f) for f in files]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to ingest reference catalog: {e.stderr}")
            return False
    
    def _add_to_refcats_collection(self, catalog_name: str) -> bool:
        """Add reference catalog to refcats collection chain."""
        cmd = [
            "butler", "collection-chain",
            str(self.repo_path),
            "--mode", "extend",
            "refcats",
            "refcats"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False  # Don't check return code
            )
            return True
        except:
            return False
    
    def import_from_export(self, export_file: str, data_dir: str) -> bool:
        """
        Import data from an export file.
        
        Parameters
        ----------
        export_file : str
            Path to export.yaml file
        data_dir : str
            Directory containing the data
            
        Returns
        -------
        bool
            True if successful
        """
        cmd = [
            "butler", "import",
            str(self.repo_path),
            str(data_dir),
            "--export-file", str(export_file),
            "--transfer", self.config.ingestion.transfer_mode
        ]
        
        try:
            logger.info(f"Importing data from {export_file}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.returncode == 0:
                logger.info("Successfully imported data")
                return True
            else:
                logger.error(f"Failed to import data: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to import data: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error importing data: {e}")
            return False