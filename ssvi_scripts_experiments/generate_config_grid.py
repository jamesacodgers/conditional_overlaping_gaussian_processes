#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Grid Generator for SSVI and Original GP-LVM Experiments

This script generates a grid of configuration files based on specified parameter ranges.
It takes base configurations and creates variations with different parameter combinations.

Usage:
    python generate_config_grid.py
"""

import os
import re
import itertools
from pathlib import Path

def load_base_config_text(config_path):
    """Load base configuration as text to preserve structure and comments"""
    with open(config_path, 'r') as f:
        return f.read()

def save_config_text(config_text, output_path):
    """Save configuration text to YAML file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(config_text)

def update_config_parameter(config_text, param_name, new_value):
    """Update a parameter value in YAML text while preserving structure"""
    # Pattern to match the parameter line with optional comment
    pattern = rf'^({param_name}:\s*)([^\s#]+)(\s*#.*)?$'
    
    def replace_func(match):
        prefix = match.group(1)
        comment = match.group(3) if match.group(3) else ''
        return f"{prefix}{new_value}{comment}"
    
    lines = config_text.split('\n')
    updated_lines = []
    
    for line in lines:
        if re.match(pattern, line):
            updated_lines.append(re.sub(pattern, replace_func, line))
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)

def create_config_name(n_inducing, init_lengthscales, init_output_noise_scale, random_seed):
    """Create a valid filename from parameter values"""
    # Replace dots with nothing for valid filenames
    len_str = str(init_lengthscales).replace('.', '')
    sigm_str = str(init_output_noise_scale).replace('.', '')
    
    return f"n_ind_{n_inducing}_len_{len_str}_sigm_{sigm_str}_s_{random_seed}.yaml"

def generate_config_grid():
    """Generate grid of configurations for both SSVI and Original GP-LVM"""
    
    # Parameter ranges
    n_inducing_values = [8, 32, 128]
    init_lengthscales_values = [0.1, 1.0]
    init_output_noise_scale_values = [0.1, 1.0]
    batch_size_values = [32, 128]
    random_seed_values = [7, 11, 17]
    
    # Base configuration paths
    ssvi_base_config = "ssvi_configs/base_config_many_seeds/base_lr_0001_s7.yaml"
    orig_base_config = "original_gp_lvm_configs/base_config_many_seeds/base_lr_0001_s7.yaml"
    
    # Output folder names
    output_folder_name = "my_exp_configs"
    ssvi_output_folder = f"{output_folder_name}_ssvi"
    orig_output_folder = f"{output_folder_name}_orig_gplvm"
    
    # Load base configurations
    print("Loading base configurations...")
    ssvi_base_text = load_base_config_text(ssvi_base_config)
    orig_base_text = load_base_config_text(orig_base_config)
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        n_inducing_values,
        init_lengthscales_values,
        init_output_noise_scale_values,
        batch_size_values,
        random_seed_values
    ))
    
    print(f"Generating {len(param_combinations)} configuration combinations...")
    
    # Generate SSVI configurations
    ssvi_count = 0
    for n_inducing, init_lengthscales, init_output_noise_scale, batch_size, random_seed in param_combinations:
        # Create new config based on base text
        ssvi_config_text = ssvi_base_text
        
        # Update parameters
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'n_inducing_points', n_inducing)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'init_lengthscales', init_lengthscales)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'init_output_noise_scale', init_output_noise_scale)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'batch_size', batch_size)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'random_seed', random_seed)
        
        # Create filename
        filename = create_config_name(n_inducing, init_lengthscales, init_output_noise_scale, random_seed)
        output_path = os.path.join(ssvi_output_folder, filename)
        
        # Save configuration
        save_config_text(ssvi_config_text, output_path)
        ssvi_count += 1
        
        if ssvi_count % 50 == 0:
            print(f"Generated {ssvi_count} SSVI configurations...")
    
    # Generate Original GP-LVM configurations
    orig_count = 0
    for n_inducing, init_lengthscales, init_output_noise_scale, batch_size, random_seed in param_combinations:
        # Create new config based on base text
        orig_config_text = orig_base_text
        
        # Update parameters (note: Original GP-LVM uses 'n_inducing' not 'n_inducing_points')
        orig_config_text = update_config_parameter(orig_config_text, 'n_inducing', n_inducing)
        orig_config_text = update_config_parameter(orig_config_text, 'init_lengthscales', init_lengthscales)
        orig_config_text = update_config_parameter(orig_config_text, 'init_output_noise_scale', init_output_noise_scale)
        orig_config_text = update_config_parameter(orig_config_text, 'batch_size', batch_size)
        orig_config_text = update_config_parameter(orig_config_text, 'random_seed', random_seed)
        
        # Create filename
        filename = create_config_name(n_inducing, init_lengthscales, init_output_noise_scale, random_seed)
        output_path = os.path.join(orig_output_folder, filename)
        
        # Save configuration
        save_config_text(orig_config_text, output_path)
        orig_count += 1
        
        if orig_count % 50 == 0:
            print(f"Generated {orig_count} Original GP-LVM configurations...")
    
    print(f"\nConfiguration generation completed!")
    print(f"Generated {ssvi_count} SSVI configurations in '{ssvi_output_folder}'")
    print(f"Generated {orig_count} Original GP-LVM configurations in '{orig_output_folder}'")
    print(f"\nTotal parameter combinations: {len(param_combinations)}")
    print(f"Parameter ranges:")
    print(f"  n_inducing: {n_inducing_values}")
    print(f"  init_lengthscales: {init_lengthscales_values}")
    print(f"  init_output_noise_scale: {init_output_noise_scale_values}")
    print(f"  batch_size: {batch_size_values}")
    print(f"  random_seed: {random_seed_values}")

if __name__ == "__main__":
    generate_config_grid()
