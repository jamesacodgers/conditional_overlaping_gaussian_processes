#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration utilities for Original GP LVM experiments
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
from .original_gplvm_config import OriginalGPLVMConfig, get_fast_config, get_base_config

def get_config_name_from_path(config_path: str) -> str:
    """Extract config name from file path for result naming."""
    # Remove extension and directory path
    config_name = Path(config_path).stem
    
    # Remove common prefixes/suffixes
    if config_name.startswith('original_gplvm_'):
        config_name = config_name[15:]
    if config_name.startswith('orig_gplvm_'):
        config_name = config_name[11:]
    if config_name.endswith('_config'):
        config_name = config_name[:-7]
    
    return config_name

def create_argument_parser():
    """Create argument parser for command line configuration"""
    parser = argparse.ArgumentParser(description='Original GP LVM Experiment')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    parser.add_argument('--config-name', type=str, default=None,
                       help='Name for this configuration (used for results folder)')
    
    # Data configuration
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset to use (oilflow)')
    parser.add_argument('--test-size', type=int, default=None,
                       help='Number of test samples')
    parser.add_argument('--random-seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Model configuration
    parser.add_argument('--latent-dim', type=int, default=None,
                       help='Latent dimension')
    parser.add_argument('--n-inducing', type=int, default=None,
                       help='Number of inducing points')
    parser.add_argument('--pca-init', action='store_true',
                       help='Initialize with PCA')
    
    # GP Hyperparameters
    parser.add_argument('--init-lengthscales', type=float, default=None,
                       help='Initial lengthscales value')
    parser.add_argument('--init-output-noise-scale', type=float, default=None,
                       help='Initial output noise scale')
    
    # Training configuration
    parser.add_argument('--training-steps', type=int, default=None,
                       help='Number of training steps (applied to all models)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    
    # Testing configuration
    parser.add_argument('--test-iterations', type=int, default=None,
                       help='Number of iterations for test phase evaluation')
    
    # Checkpointing configuration
    parser.add_argument('--checkpoint-frequency', type=int, default=None,
                       help='Checkpoint frequency for saving models and plots')
    
    # Results configuration
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable saving results')
    
    # Device configuration
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto, cuda, cpu)')
    
    return parser

def load_config_with_overrides(config_path: str = None, 
                             config_class: type = OriginalGPLVMConfig,
                             args: argparse.Namespace = None) -> Tuple[OriginalGPLVMConfig, str]:
    """
    Load configuration with command line overrides
    
    Args:
        config_path: Path to YAML config file (optional)
        config_class: Configuration class to instantiate
        args: Command line arguments
        
    Returns:
        Tuple of (config_instance, config_name)
    """
    # Default config name
    config_name = "fast"
    
    # Start with default config
    if config_path is None:
        config = get_fast_config()
    else:
        # Load from YAML file
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nn_layers from list to tuple if needed (YAML doesn't support tuples natively)
        if 'nn_layers' in config_dict and isinstance(config_dict['nn_layers'], list):
            config_dict['nn_layers'] = tuple(config_dict['nn_layers'])
        
        config = config_class(**config_dict)
        config_name = get_config_name_from_path(config_path)
        print(f"Using config name: {config_name}")
        
        # Automatically set results_dir to the same directory as the config file
        # This ensures results are saved in the same folder as the config for easy tracking
        config_dir = os.path.dirname(config_path)
        config.results_dir = os.path.join(config_dir, "orig_gplvm_run_results")
        print(f"Results will be saved to config directory: {config.results_dir}")
    
    # Apply command line overrides
    if args is not None:
        # Convert args to dict, filtering out None values
        overrides = {k: v for k, v in vars(args).items() 
                    if v is not None and k not in ['config', 'config_name']}
        
        # Apply overrides (except results_dir which should stay with config-based automatic setting)
        for key, value in overrides.items():
            if hasattr(config, key) and key != 'results_dir':
                setattr(config, key, value)
        
        # Handle special cases
        if args.training_steps is not None:
            # Apply same training steps to all models
            config.training_steps = {model: args.training_steps for model in config.models}
        
        if args.no_save:
            config.save_results = False
        
        # Use config_name from args if provided
        if args.config_name is not None:
            config_name = args.config_name
    
    return config, config_name

def ensure_config_dir_exists():
    """Ensure config directory exists"""
    # Check if we're already in ssvi_scripts_experiments folder
    current_dir = os.getcwd()
    if current_dir.endswith('ssvi_scripts_experiments'):
        # We're already in ssvi_scripts_experiments, so just create original_gplvm_configs
        config_dir = "original_gplvm_configs"
    else:
        # We're in project root, create full path
        config_dir = "ssvi_scripts_experiments/original_gplvm_configs"
    
    os.makedirs(config_dir, exist_ok=True)
    return config_dir

def print_available_configs():
    """Print available configuration files"""
    config_dir = ensure_config_dir_exists()
    
    # Check if config directory exists and has files
    if os.path.exists(config_dir):
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        
        if config_files:
            print("Available configuration files:")
            for config_file in sorted(config_files):
                print(f"  {config_file}")
        else:
            print("No configuration files found.")
            print(f"Config directory: {config_dir}")
    else:
        print(f"Config directory does not exist: {config_dir}")

def save_config_to_yaml(config: OriginalGPLVMConfig, filepath: str):
    """Save configuration to YAML file"""
    config_dict = {}
    for field_name, field_value in config.__dict__.items():
        if field_value is not None:
            config_dict[field_name] = field_value
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def get_base_config_path():
    """Get path to base configuration file"""
    config_dir = ensure_config_dir_exists()
    return os.path.join(config_dir, "base_lr_0001_s7.yaml")