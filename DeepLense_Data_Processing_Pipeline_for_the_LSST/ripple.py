#!/usr/bin/env python3
"""
RIPPLe: LSST-DeepLense Data Processing Pipeline

Main entry point for the RIPPLe pipeline that bridges LSST astronomical data
with DeepLense deep learning workflows for gravitational lensing research.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Optional

from ripple.config import load_config, validate_config, create_sample_config
from ripple.butler import ButlerRepoValidator
from ripple.butler.creator import ButlerRepoCreator, DataDiscoveryResult


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the pipeline.
    
    Args:
        verbose: Enable verbose logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_for_astronomical_data(path: str) -> Optional[DataDiscoveryResult]:
    """
    Check if a directory contains astronomical data files.
    
    Args:
        path: Directory path to check
        
    Returns:
        DataDiscoveryResult if astronomical data found, None otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_dir():
            return None
        
        # Use ButlerRepoCreator to discover data files
        temp_creator = ButlerRepoCreator("/tmp/temp_analysis", instrument=None)
        discovery = temp_creator.discover_data_files(path, recursive=True)
        
        # Consider it astronomical data if we found FITS files and detected dataset types
        if discovery.total_files > 0 and discovery.data_files:
            logger.debug(f"Found {discovery.total_files} FITS files with dataset types: {list(discovery.data_files.keys())}")
            return discovery
            
        return None
        
    except Exception as e:
        logger.debug(f"Error checking for astronomical data in {path}: {e}")
        return None


def ask_user_create_butler_repo(path: str, discovery: DataDiscoveryResult) -> bool:
    """
    Ask user interactively if they want to create a Butler repository.
    
    Args:
        path: Directory path
        discovery: Data discovery results
        
    Returns:
        True if user wants to create repository, False otherwise
    """
    print(f"\n{'='*60}")
    print("ASTRONOMICAL DATA DETECTED")
    print(f"{'='*60}")
    print(f"The path '{path}' is not a Butler repository, but contains astronomical data:")
    print(f"  • {discovery.total_files} FITS files found")
    print(f"  • Dataset types: {', '.join(discovery.data_files.keys())}")
    
    if discovery.supported_instruments:
        print(f"  • Detected instruments: {', '.join(discovery.supported_instruments)}")
    
    print(f"\nData breakdown:")
    for dataset_type, count in discovery.file_patterns.items():
        print(f"  • {dataset_type}: {count} files")
    
    print(f"\n{'='*60}")
    
    while True:
        response = input("Would you like to create a Butler repository automatically? [y/N]: ").strip().lower()
        
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def create_butler_repository_interactively(path: str, discovery: DataDiscoveryResult) -> bool:
    """
    Create a Butler repository interactively with user confirmation.
    
    Args:
        path: Directory path to create repository in
        discovery: Data discovery results
        
    Returns:
        True if repository created successfully, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Determine instrument to use
    instrument = None
    if discovery.supported_instruments:
        instrument = list(discovery.supported_instruments)[0]
        print(f"\nAuto-detected instrument: {instrument}")
    
    # Ask for confirmation of creation parameters
    print(f"\nRepository creation parameters:")
    print(f"  • Path: {path}")
    print(f"  • Instrument: {instrument or 'auto-detect'}")
    print(f"  • Data files to process: {discovery.total_files}")
    
    confirm = input("\nProceed with repository creation? [Y/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("Repository creation cancelled.")
        return False
    
    # Create the repository
    print(f"\nCreating Butler repository...")
    try:
        creator = ButlerRepoCreator(path, instrument=instrument)
        
        # Create empty repository first
        repo_result = creator.create_repository(overwrite=False)
        
        if not repo_result.success:
            logger.error(f"Failed to create Butler repository: {repo_result.error_message}")
            return False
        
        print(f"✓ Butler repository created successfully at: {path}")
        
        if repo_result.warnings:
            print("Warnings:")
            for warning in repo_result.warnings:
                print(f"  • {warning}")
        
        # Note: We don't ingest data automatically since that could be time-consuming
        # The repository is created empty and ready for use
        print(f"\nNote: Repository created empty. Data ingestion can be done separately if needed.")
        print(f"The repository is now ready for LSST Butler operations.")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create Butler repository: {e}")
        return False


def validate_butler_repositories(config, allow_interactive: bool = True) -> bool:
    """
    Validate the Butler repositories specified in the configuration.
    
    Args:
        config: RippleConfig object
        allow_interactive: Whether to allow interactive Butler repository creation
        
    Returns:
        True if at least one repository is valid and has usable data
        
    Raises:
        SystemExit: If no repositories are valid
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("BUTLER REPOSITORY VALIDATION")
    logger.info("=" * 60)
    
    repositories = config.butler.get_repositories()
    logger.info(f"Validating {len(repositories)} Butler repository(ies)...")
    
    valid_repos = []
    all_available_products = {}
    all_alternative_data = {}
    all_instruments = set()
    
    # Validate each repository
    for i, repo_config in enumerate(repositories, 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"Repository {i}/{len(repositories)}: {repo_config.name}")
        logger.info(f"{'='*40}")
        logger.info(f"Path: {repo_config.repo_path}")
        logger.info(f"Collections: {repo_config.collections or 'auto-discover'}")
        logger.info(f"Priority: {repo_config.priority}")
        
        # Initialize validator for this repository
        try:
            validator = ButlerRepoValidator(repo_config.repo_path)
        except Exception as e:
            logger.error(f"Failed to initialize Butler validator: {e}")
            
            # Check if this path contains astronomical data that we could convert to a Butler repo
            logger.info("Checking if path contains astronomical data...")
            discovery = check_for_astronomical_data(repo_config.repo_path)
            
            if discovery:
                if allow_interactive:
                    # Ask user if they want to create a Butler repository
                    if ask_user_create_butler_repo(repo_config.repo_path, discovery):
                        # Create the Butler repository
                        if create_butler_repository_interactively(repo_config.repo_path, discovery):
                            # Retry validator initialization after creating repository
                            try:
                                logger.info("Retrying Butler validator initialization...")
                                validator = ButlerRepoValidator(repo_config.repo_path)
                                logger.info("✓ Butler validator initialized successfully after repository creation")
                            except Exception as retry_e:
                                logger.error(f"Failed to initialize Butler validator even after creating repository: {retry_e}")
                                continue
                        else:
                            logger.error("Butler repository creation failed")
                            continue
                    else:
                        logger.info("User declined to create Butler repository")
                        continue
                else:
                    logger.info("Astronomical data detected but interactive mode disabled (--no-interactive)")
                    logger.info(f"Found {discovery.total_files} FITS files with dataset types: {', '.join(discovery.data_files.keys())}")
                    logger.info("Consider creating a Butler repository manually or running without --no-interactive")
                    continue
            else:
                logger.error(f"Path does not contain recognizable astronomical data: {repo_config.repo_path}")
                continue
        
        # Validate repository
        logger.info("Validating repository...")
        result = validator.validate_repository()
        
        if not result.is_valid:
            logger.error(f"Repository validation FAILED: {result.error_message}")
            continue
        
        logger.info("Repository validation PASSED")
        logger.info(f"   Collections: {len(result.collections)}")
        logger.info(f"   Dataset types: {len(result.dataset_types)}")
        logger.info(f"   Instruments: {result.instruments}")
        
        # Store valid repository info
        valid_repos.append((repo_config, validator, result))
        all_instruments.update(result.instruments)
        
        # Merge data products from this repository
        for product_name, product_info in result.data_products.items():
            if product_name not in all_available_products:
                all_available_products[product_name] = []
            all_available_products[product_name].append({
                'repo_name': repo_config.name,
                'info': product_info,
                'priority': repo_config.priority
            })
    
    # Check if we have any valid repositories
    if not valid_repos:
        logger.error("No valid Butler repositories found")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MULTI-REPOSITORY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Valid repositories: {len(valid_repos)}")
    logger.info(f"Combined instruments: {all_instruments}")
    
    # Analyze data products with smart defaults and suggestions
    required_products = config.processing.data_products.get('required', ['calexp'])
    optional_products = config.processing.data_products.get('optional', [])
    
    logger.info("\nData Product Analysis:")
    
    missing_required = []
    available_products = {}
    suggestions = []
    alternative_data = {}  # Initialize outside the if block for scope
    
    # Check configured data products across all repositories
    for product in required_products + optional_products:
        if product in all_available_products:
            # Use the highest priority repository's info for this product
            repo_infos = sorted(all_available_products[product], key=lambda x: x['priority'], reverse=True)
            best_info = repo_infos[0]['info']
            available_products[product] = best_info
            status = "REQUIRED" if product in required_products else "optional"
            repo_name = repo_infos[0]['repo_name']
            logger.info(f"   {status}: {product} ({best_info.available_count} available, {best_info.coverage_percentage:.1f}% coverage) [from {repo_name}]")
        else:
            if product in required_products:
                missing_required.append(product)
            status = "REQUIRED" if product in required_products else "optional"
            logger.info(f"   {status}: {product} (not found)")
    
    # Smart defaults: analyze what IS available and provide suggestions
    if missing_required:
        logger.warning(f"\nMissing required data products: {missing_required}")
        
        # Check for alternative data products and suggest processing pipelines
        for missing_product in missing_required:
            if missing_product == 'calexp':
                # Check for raw data that could be processed to calexp
                if 'raw' in all_available_products:
                    repo_infos = sorted(all_available_products['raw'], key=lambda x: x['priority'], reverse=True)
                    raw_info = repo_infos[0]['info']
                    repo_name = repo_infos[0]['repo_name']
                    alternative_data['raw'] = raw_info
                    suggestions.append(f"Found {raw_info.available_count} raw images in {repo_name} that could be processed to calexp using ISR + calibrateImage pipeline")
                
                # Check for postISRCCD that could be processed to calexp
                if 'postISRCCD' in all_available_products:
                    repo_infos = sorted(all_available_products['postISRCCD'], key=lambda x: x['priority'], reverse=True)
                    isr_info = repo_infos[0]['info']
                    repo_name = repo_infos[0]['repo_name']
                    alternative_data['postISRCCD'] = isr_info
                    suggestions.append(f"Found {isr_info.available_count} postISRCCD images in {repo_name} that could be processed to calexp using calibrateImage pipeline")
            
            elif missing_product == 'src':
                # Check for calexp that could be processed to src
                if 'calexp' in all_available_products:
                    repo_infos = sorted(all_available_products['calexp'], key=lambda x: x['priority'], reverse=True)
                    calexp_info = repo_infos[0]['info']
                    repo_name = repo_infos[0]['repo_name']
                    suggestions.append(f"Found {calexp_info.available_count} calexp images in {repo_name} that could be processed to src catalogs using source detection pipeline")
        
        # Show additional available data products that weren't in config
        logger.info("\nOther Available Data Products:")
        other_products = []
        key_products = ['raw', 'postISRCCD', 'calexp', 'src', 'deepCoadd', 'objectTable', 'visitTable']
        
        for product in key_products:
            if product not in (required_products + optional_products) and product in all_available_products:
                repo_infos = sorted(all_available_products[product], key=lambda x: x['priority'], reverse=True)
                info = repo_infos[0]['info']
                repo_name = repo_infos[0]['repo_name']
                other_products.append(product)
                logger.info(f"   available: {product} ({info.available_count} items, {info.coverage_percentage:.1f}% coverage) [from {repo_name}]")
        
        # Provide helpful suggestions
        if suggestions:
            logger.info("\nSuggested Processing Options:")
            for i, suggestion in enumerate(suggestions, 1):
                logger.info(f"   {i}. {suggestion}")
        
        if alternative_data:
            logger.info(f"\nAlternative: Consider updating your config to use available data products: {list(alternative_data.keys())}")
        
        # Only exit if no viable alternatives and not in validation-only mode
        if not alternative_data and not other_products:
            logger.error("\nNo viable data products found for processing")
            logger.error("This repository may need data processing pipelines to be run first")
            return False
        else:
            logger.warning("Proceeding with available data analysis...")
    
    else:
        logger.info("All required data products found")
    
    # Generate coverage report if data selection criteria specified
    if any([config.data_selection.filters, 
            config.data_selection.visits,
            config.data_selection.detectors]):
        
        logger.info("\nCoverage Report for Data Selection:")
        
        # Smart coverage reporting: use available data products instead of missing ones
        coverage_dataset_types = []
        if available_products:
            # Use available products for coverage analysis
            coverage_dataset_types = list(available_products.keys())
            logger.info(f"Analyzing coverage for available products: {coverage_dataset_types}")
        elif alternative_data:
            # Use alternative products (like raw instead of calexp)
            coverage_dataset_types = list(alternative_data.keys()) 
            logger.info(f"Analyzing coverage for alternative products: {coverage_dataset_types}")
        else:
            # Fall back to required products (will show 0% but still informative)
            coverage_dataset_types = required_products
            logger.info(f"Analyzing coverage for requested products: {coverage_dataset_types}")
        
        try:
            # Use the primary (highest priority) repository's validator for coverage analysis
            primary_repo = config.butler.get_primary_repository()
            primary_validator = None
            for repo_config, validator, result in valid_repos:
                if repo_config.name == primary_repo.name:
                    primary_validator = validator
                    break
            
            if not primary_validator:
                # Fall back to first valid repository
                primary_validator = valid_repos[0][1]
            
            coverage = primary_validator.get_data_coverage(
                dataset_types=coverage_dataset_types,
                collections=primary_repo.collections,
                filters=config.data_selection.filters,
                visit_ranges=config.data_selection.visits.get('ranges') if config.data_selection.visits else None,
                instruments=list(all_instruments)
            )
            
            logger.info(f"   Total data IDs matching criteria: {coverage.total_data_ids}")
            logger.info(f"   Available: {coverage.available_data_ids}")
            logger.info(f"   Missing: {coverage.missing_data_ids}")
            logger.info(f"   Coverage: {coverage.coverage_percentage:.1f}%")
            
            if coverage.instruments:
                logger.info(f"   Instruments: {coverage.instruments}")
            if coverage.filters:
                logger.info(f"   Filters: {coverage.filters}")
            
            if coverage.available_data_ids == 0:
                logger.warning("No data available matching the specified criteria")
                logger.warning("Consider adjusting data selection parameters")
                
        except Exception as e:
            logger.warning(f"Failed to generate coverage report: {e}")
    
    logger.info("\nButler repository validation completed successfully")
    return True


def main() -> None:
    """Main entry point for the RIPPLe pipeline."""
    
    parser = argparse.ArgumentParser(
        description="RIPPLe: LSST-DeepLense Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline with configuration file
  python ripple.py --config config.yaml
  
  # Create sample configuration
  python ripple.py --create-config sample_config.yaml
  
  # Validate Butler repository only
  python ripple.py --config config.yaml --validate-only
  
  # Run with verbose logging
  python ripple.py --config config.yaml --verbose
  
  # Run without interactive Butler repository creation prompts
  python ripple.py --config config.yaml --no-interactive
  
  # Interactive Butler repository creation:
  # If your config specifies a path with astronomical data that's not a Butler repo,
  # RIPPLe will detect the data and offer to create a Butler repository automatically
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--create-config',
        type=str,
        metavar='OUTPUT_PATH',
        help='Create a sample configuration file at the specified path'
    )
    
    parser.add_argument(
        '--template',
        choices=['minimal', 'default', 'full'],
        default='default',
        help='Template type for sample configuration (default: default)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate Butler repository, do not process data'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive Butler repository creation prompts'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle create-config command
    if args.create_config:
        try:
            create_sample_config(args.create_config, args.template)
            logger.info(f"Sample configuration created: {args.create_config}")
            return
        except Exception as e:
            logger.error(f"Failed to create sample configuration: {e}")
            sys.exit(1)
    
    # Require config file for all other operations
    if not args.config:
        logger.error("Configuration file is required. Use --config or --create-config")
        parser.print_help()
        sys.exit(1)
    
    # Load and validate configuration
    try:
        logger.info("Starting RIPPLe Pipeline")
        logger.info("=" * 60)
        
        config = load_config(args.config)
        validate_config(config)
        
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Version: {config.version}")
        if config.description:
            logger.info(f"Description: {config.description}")
        logger.info(f"Processing mode: {config.processing.mode}")
        logger.info(f"Batch size: {config.processing.batch_size}")
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Validate Butler repository
    try:
        validation_success = validate_butler_repositories(config, allow_interactive=not args.no_interactive)
        if not validation_success:
            logger.error("Butler repository validation failed")
            if not args.validate_only:
                logger.error("Cannot proceed with data processing")
                sys.exit(1)
            else:
                logger.info("Repository analysis complete despite missing data products")
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Butler validation error: {e}")
        sys.exit(1)
    
    # Exit if validation-only mode
    if args.validate_only:
        logger.info("Validation complete. Exiting (--validate-only specified)")
        return
    
    # TODO: Implement data processing pipeline
    logger.info("\n" + "=" * 60)
    logger.info("DATA PROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info("Data processing pipeline not yet implemented")
    logger.info("Butler repository validation successful - ready for implementation!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)