# utils/config_loader.py
import yaml
import argparse
import sys

def load_config(config_path):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

def get_config():
    """Parses command line arguments to get the config file path."""
    parser = argparse.ArgumentParser(description="DeepLense Agentic AI Training Pipeline")
    
    # This allows an AI agent to run: python train.py --config configs/my_custom_config.yaml
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/transformer_sim.yaml', 
        help='Path to the YAML configuration file'
    )
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    return config
