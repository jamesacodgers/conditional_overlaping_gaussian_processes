"""
Configuration utilities for SSVI experiments.
Handles YAML config loading, command line parsing, and configuration management.
"""

import yaml
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict, fields
import torch

# Import the config class from the oil flow script
sys.path.insert(0, os.getcwd())


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict


def save_config_to_yaml(config_dict: Dict[str, Any], config_path: str) -> None:
    """Save configuration dictionary to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def dataclass_to_dict(config_obj) -> Dict[str, Any]:
    """Convert dataclass to dictionary, handling special types."""
    config_dict = asdict(config_obj)
    
    # Convert torch tensors to lists for YAML serialization
    for key, value in config_dict.items():
        if isinstance(value, torch.Tensor):
            config_dict[key] = value.tolist()
    
    return config_dict


def dict_to_dataclass(config_dict: Dict[str, Any], config_class):
    """Convert dictionary to dataclass instance, handling special types."""
    # Get field names and types from the dataclass
    field_names = {f.name for f in fields(config_class)}
    field_types = {f.name: f.type for f in fields(config_class)}
    
    # Filter out keys not in the dataclass
    filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
    
    # Convert lists back to torch tensors where needed
    for key, value in filtered_dict.items():
        if key in field_types and field_types[key] == torch.Tensor and isinstance(value, list):
            filtered_dict[key] = torch.tensor(value)
        elif key == 'beta' and value is not None and isinstance(value, list):
            filtered_dict[key] = torch.tensor(value)
        elif key == 'sigma2_s' and value is not None and isinstance(value, list):
            filtered_dict[key] = torch.tensor(value)
    
    return config_class(**filtered_dict)


def get_config_name_from_path(config_path: str) -> str:
    """Extract config name from file path for result naming."""
    # Remove extension and directory path
    config_name = Path(config_path).stem
    
    # Remove common prefixes/suffixes
    if config_name.startswith('ssvi_'):
        config_name = config_name[5:]
    if config_name.endswith('_config'):
        config_name = config_name[:-7]
    
    return config_name


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for SSVI scripts."""
    parser = argparse.ArgumentParser(
        description='Run SSVI Oil Flow GP-LVM experiment with configurable parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML configuration file. If not provided, uses fast_config by default.'
    )
    
    parser.add_argument(
        '--config-name',
        type=str,
        default=None,
        help='Override config name for result file naming (auto-detected from config path if not provided)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default=None,
        help='Override device setting from config'
    )
    
    parser.add_argument(
        '--training-iterations',
        type=int,
        default=None,
        help='Override number of training iterations'
    )
    
    parser.add_argument(
        '--test-iterations',
        type=int,
        default=None,
        help='Override number of test iterations'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--local-lr',
        type=float,
        default=None,
        help='Override local learning rate (encoder parameters)'
    )
    
    parser.add_argument(
        '--global-lr',
        type=float,
        default=None,
        help='Override global learning rate (GP hyperparameters)'
    )
    
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Disable metrics computation'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation and saving'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable saving results to disk'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    return parser


def load_config_with_overrides(config_path: Optional[str] = None, 
                              config_class = None,
                              args: Optional[argparse.Namespace] = None) -> tuple:
    """
    Load configuration with command line overrides.
    
    Returns:
        tuple: (config_object, config_name)
    """
    if config_class is None:
        raise ValueError("config_class must be provided")
    
    # Default config name
    config_name = "fast"
    
    if config_path is not None and os.path.exists(config_path):
        # Load from YAML file
        print(f"Loading configuration from: {config_path}")
        config_dict = load_config_from_yaml(config_path)
        config = dict_to_dataclass(config_dict, config_class)
        config_name = get_config_name_from_path(config_path)
        print(f"Using config name: {config_name}")
    else:
        # Use default fast config
        if config_path is not None:
            print(f"Warning: Config file {config_path} not found, using default fast config")
        else:
            print("No config file specified, using default fast config")
        
        # Import and use get_fast_config function
        from ssvi_scripts_experiments.oil_flow_GPLVM_SSVI_AM_INF import get_fast_config
        config = get_fast_config()
        config_name = "fast"
    
    # Apply command line overrides if provided
    if args is not None:
        if args.config_name is not None:
            config_name = args.config_name
            
        if args.device is not None:
            config.device = args.device
            
        if args.training_iterations is not None:
            config.training_iterations = args.training_iterations
            
        if args.test_iterations is not None:
            config.test_iterations = args.test_iterations

        if args.batch_size is not None:
            config.batch_size = args.batch_size
            
        if args.local_lr is not None:
            config.local_lr = args.local_lr
            
        if args.global_lr is not None:
            config.global_lr = args.global_lr
            
        if args.no_metrics:
            config.enable_metrics = False
            
        if args.no_plots:
            config.plot_training = False
            config.plot_latent_space = False
            config.plot_hyperparams = False
            config.save_training_plots = False
            
        if args.no_save:
            config.save_results = False
        
        if args.quiet:
            config.log_frequency = config.training_iterations // 5  # Reduce log frequency
    
    return config, config_name


def get_base_config_path() -> str:
    """Get the path to the base configuration file."""
    if os.path.basename(os.getcwd()) == "ssvi_scripts_experiments":
        return "ssvi_configs/ssvi_base_config.yaml"
    else:
        return "ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml"


def get_config_dir() -> str:
    """Get the configuration directory path."""
    if os.path.basename(os.getcwd()) == "ssvi_scripts_experiments":
        return "ssvi_configs"
    else:
        return "ssvi_scripts_experiments/ssvi_configs"


def ensure_config_dir_exists():
    """Ensure the configuration directory exists."""
    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)


def list_available_configs() -> list:
    """List all available configuration files."""
    config_dir = get_config_dir()
    if not os.path.exists(config_dir):
        return []
    
    config_files = []
    for file in os.listdir(config_dir):
        if file.endswith('.yaml') or file.endswith('.yml'):
            config_files.append(os.path.join(config_dir, file))
    
    return sorted(config_files)


def print_available_configs():
    """Print available configuration files."""
    configs = list_available_configs()
    if configs:
        print("Available configuration files:")
        for config in configs:
            config_name = get_config_name_from_path(config)
            print(f"  - {config} (name: {config_name})")
    else:
        config_dir = get_config_dir()
        print(f"No configuration files found in {config_dir}/")
        print("The script will use the default fast_config.")
