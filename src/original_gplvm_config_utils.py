#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration utilities for Original GP LVM experiments
"""

import os
import yaml
import argparse
from typing import Dict, Any, Tuple
from .original_gplvm_config import OriginalGPLVMConfig, get_fast_config, get_base_config

def create_argument_parser():
    """Create argument parser for command line configuration"""
    parser = argparse.ArgumentParser(description='Original GP LVM Experiment')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    parser.add_argument('--config-name', type=str, default='fast',
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
    
    # Training configuration
    parser.add_argument('--training-steps', type=int, default=None,
                       help='Number of training steps (applied to all models)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    
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
    # Start with default config
    if config_path is None:
        config = get_fast_config()
        config_name = "fast"
    else:
        # Load from YAML file
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = config_class(**config_dict)
        config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    # Apply command line overrides
    if args is not None:
        # Convert args to dict, filtering out None values
        overrides = {k: v for k, v in vars(args).items() 
                    if v is not None and k not in ['config', 'config_name']}
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Handle special cases
        if args.training_steps is not None:
            # Apply same training steps to all models
            config.training_steps = {model: args.training_steps for model in config.models}
        
        if args.no_save:
            config.save_results = False
        
        # Use config_name from args if provided
        if args.config_name:
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
    return os.path.join(config_dir, "original_gplvm_base_config.yaml")