"""
Configuration utilities for original GP-LVM experiments.
Handles YAML config loading, command line parsing, and configuration management for GP-LVM.
"""

import yaml
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict, fields, dataclass
import torch

# Import shared utilities from SSVI config
from src.ssvi_config_utils import (
    load_config_from_yaml, save_config_to_yaml, dataclass_to_dict, 
    dict_to_dataclass, get_config_name_from_path
)


@dataclass
class OilFlowGPLVMConfig:
    """Configuration for Oil Flow Original GP-LVM"""
    
    total_size = None  # To be set after data loading

    # === Device Configuration ===
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"
    
    # === Data Configuration ===
    data_url: str = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    latent_dim: int = 6  # A = 6 from original (same as SSVI for comparison)
    
    # === Model Architecture ===
    n_inducing_points: int = 12  # M in the original script
    
    # === Training Configuration ===
    training_iterations: int = 1000
    learning_rate: float = 1e-2  # Single optimizer for original GP-LVM
    
    # === GP Hyperparameters ===
    sigma2: float = 1.0
    beta: Any = None         # Lengthscales (None = auto)
    sigma2_s: Any = None     # Signal variances (None = auto)
    P: int = 8               # Number of samples for structured GP-LVM
    trainable_V: bool = True
    
    # === Visualization ===
    plot_training: bool = True
    plot_latent_space: bool = True
    plot_hyperparams: bool = True
    
    # === Progress Reporting ===
    log_frequency: int = 100  # Log every N iterations
    
    # === Advanced Logging ===
    enable_gradient_logging: bool = False  # Log gradients and parameter changes
    gradient_log_frequency: int = 50       # Log gradients every N iterations
    save_results: bool = True              # Save results to folder
    
    # === Metrics Configuration ===
    enable_metrics: bool = True            # Compute evaluation metrics
    metrics_frequency: int = 100           # Compute metrics every N iterations
    test_size: float = 0.2                 # Fraction of data for testing
    split_random_seed: int = 42            # Seed for train/test split (separate from training seed)
    
    # === Visualization Configuration ===
    save_training_plots: bool = True       # Save training result plots during training
    plot_save_frequency: int = 100         # Save plots every N iterations
    
    # === Test Optimization Configuration ===
    test_optimization_steps: int = 50      # Number of optimization steps for test data preparation
    
    # === Missing Data Configuration ===
    enable_missing_data: bool = False      # Enable missing data reconstruction capability
    missing_data_ratio: float = 0.3        # Fraction of dimensions to artificially mask (for testing)
    reconstruction_samples: int = 50       # Number of samples for missing data reconstruction  
    missing_data_seed: int = 123           # Seed for reproducible missing data patterns
    
    def __post_init__(self):
        """Set default values and resolve device if not specified"""
        # Resolve device configuration
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def print_config(self):
        """Print configuration summary"""
        print("=== Oil Flow Original GP-LVM Configuration ===")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0) if self.device.startswith('cuda') else 'Not using CUDA'}")
        else:
            print("CUDA available: No")
        print(f"Model: Structured GP-LVM with {self.n_inducing_points} inducing points")
        print(f"Latent dimensions: {self.latent_dim}")
        print(f"Training: {self.training_iterations} iter, lr={self.learning_rate}")
        print(f"Structured GP-LVM: P={self.P} samples")
        print(f"Logging: Save results={self.save_results}, Gradients={self.enable_gradient_logging}")
        if self.enable_gradient_logging:
            print(f"  Gradient log frequency: every {self.gradient_log_frequency} iterations")
        print("=" * 50)


def create_gplvm_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for GP-LVM scripts."""
    parser = argparse.ArgumentParser(
        description='Run Original GP-LVM Oil Flow experiment with configurable parameters',
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
        '--learning-rate',
        type=float,
        default=None,
        help='Override learning rate'
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


def load_gplvm_config_with_overrides(config_path: Optional[str] = None, 
                                    config_class = None,
                                    args: Optional[argparse.Namespace] = None) -> tuple:
    """
    Load GP-LVM configuration with command line overrides.
    
    Returns:
        tuple: (config_object, config_name)
    """
    if config_class is None:
        config_class = OilFlowGPLVMConfig
    
    # Default config name
    config_name = "fast"
    
    if config_path is not None and os.path.exists(config_path):
        # Load from YAML file
        print(f"Loading GP-LVM configuration from: {config_path}")
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
        
        # Import and use get_fast_gplvm_config function
        from ssvi_scripts_experiments.oil_flow_GPLVM import get_fast_gplvm_config
        config = get_fast_gplvm_config()
        config_name = "fast"
    
    # Apply command line overrides if provided
    if args is not None:
        if args.config_name is not None:
            config_name = args.config_name
            
        if args.device is not None:
            config.device = args.device
            
        if args.training_iterations is not None:
            config.training_iterations = args.training_iterations
            
        if args.learning_rate is not None:
            config.learning_rate = args.learning_rate
            
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


def get_gplvm_base_config_path() -> str:
    """Get the path to the base GP-LVM configuration file."""
    if os.path.basename(os.getcwd()) == "ssvi_scripts_experiments":
        return "original_gp_lvm_configs/base_config.yaml"
    else:
        return "ssvi_scripts_experiments/original_gp_lvm_configs/base_config.yaml"


def get_gplvm_config_dir() -> str:
    """Get the GP-LVM configuration directory path."""
    if os.path.basename(os.getcwd()) == "ssvi_scripts_experiments":
        return "original_gp_lvm_configs"
    else:
        return "ssvi_scripts_experiments/original_gp_lvm_configs"


def ensure_gplvm_config_dir_exists():
    """Ensure the GP-LVM configuration directory exists."""
    config_dir = get_gplvm_config_dir()
    os.makedirs(config_dir, exist_ok=True)


def list_available_gplvm_configs() -> list:
    """List all available GP-LVM configuration files."""
    config_dir = get_gplvm_config_dir()
    if not os.path.exists(config_dir):
        return []
    
    config_files = []
    for file in os.listdir(config_dir):
        if file.endswith('.yaml') or file.endswith('.yml'):
            config_files.append(os.path.join(config_dir, file))
    
    return sorted(config_files)


def print_available_gplvm_configs():
    """Print available GP-LVM configuration files."""
    configs = list_available_gplvm_configs()
    if configs:
        print("Available GP-LVM configuration files:")
        for config in configs:
            config_name = get_config_name_from_path(config)
            print(f"  - {config} (name: {config_name})")
    else:
        config_dir = get_gplvm_config_dir()
        print(f"No configuration files found in {config_dir}/")
        print("The script will use the default fast_config.")
