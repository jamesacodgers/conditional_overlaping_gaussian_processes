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
    # Handle list values (e.g., [128, 64, 32])
    if isinstance(new_value, list):
        # Format as YAML list
        value_str = '[' + ', '.join(map(str, new_value)) + ']'
    else:
        value_str = str(new_value)
    
    # Pattern to match the parameter line with optional comment
    # Updated to handle list values with spaces: matches everything until # or end of line
    pattern = rf'^({param_name}:\s*)(.+?)(\s*#.*)?$'
    
    def replace_func(match):
        prefix = match.group(1)
        comment = match.group(3) if match.group(3) else ''
        return f"{prefix}{value_str}{comment}"
    
    lines = config_text.split('\n')
    updated_lines = []
    
    for line in lines:
        if re.match(rf'^{param_name}:\s*', line):
            updated_lines.append(re.sub(pattern, replace_func, line))
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)

def update_nested_training_steps(config_text, new_value):
    """Update nested training_steps values for both gauss and nn_gauss models"""
    lines = config_text.split('\n')
    updated_lines = []
    in_training_steps = False
    
    for line in lines:
        # Check if we're entering the training_steps section
        if re.match(r'^training_steps:\s*', line):
            in_training_steps = True
            updated_lines.append(line)
        # Update gauss and nn_gauss values within training_steps
        elif in_training_steps and re.match(r'^\s+(gauss|nn_gauss):\s*', line):
            # Extract the model name and any comment
            match = re.match(r'^(\s+)(gauss|nn_gauss):\s*(.+?)(\s*#.*)?$', line)
            if match:
                indent = match.group(1)
                model = match.group(2)
                comment = match.group(4) if match.group(4) else ''
                updated_lines.append(f"{indent}{model}: {new_value}{comment}")
            else:
                updated_lines.append(line)
        # Exit training_steps section when we hit a non-indented line
        elif in_training_steps and not line.startswith(' ') and line.strip():
            in_training_steps = False
            updated_lines.append(line)
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)

def create_config_name(n_inducing, init_lengthscales, init_output_noise_scale, batch_size, random_seed, learning_rate, n_training_steps, num_inducing_samples, architecture, enable_missing_data, missing_data_ratio, test_size):
    """Create a valid filename from parameter values"""
    # Replace dots with nothing for valid filenames
    len_str = str(init_lengthscales).replace('.', '')
    sigm_str = str(init_output_noise_scale).replace('.', '')
    lr_str = str(learning_rate).replace('.', '')
    n_steps_str = str(n_training_steps).replace('.', '')
    missing_ratio_str = str(missing_data_ratio).replace('.', '')
    test_size_str = str(test_size).replace('.', '')
    
    # Sum of neurons in architecture (e.g., [128, 64, 32] -> 224)
    arch_sum = sum(architecture)
    
    # Missing data flag (t for True, f for False)
    missing_flag = 't' if enable_missing_data else 'f'

    return f"n_ind_{n_inducing}_arch_{arch_sum}_len_{len_str}_sig_{sigm_str}_b_{batch_size}_s_{random_seed}_lr_{lr_str}_n_steps_{n_steps_str}_P_{num_inducing_samples}_miss_{missing_flag}_ratio_{missing_ratio_str}_test_{test_size_str}.yaml"

def generate_config_grid():
    """Generate grid of configurations for both SSVI and Original GP-LVM"""
    
    # Parameter ranges
    n_inducing_values = [32]
    init_lengthscales_values = [1]
    init_output_noise_scale_values = [1]
    batch_size_values = [32]
    # random_seed_values = [0]
    random_seed_values = [0,1,2,3,4]
    learning_rate_values = [0.001]
    n_training_steps = [100_000]
    num_inducing_samples_list = [64] # [64] # For SSVI only
    
    # Architecture parameters
    # For SSVI: encoder_hidden_dims
    # For Original GP-LVM: nn_layers
    # Both use the same architecture options
    architecture_values = [[256, 128]]  # Can add more options like [[256, 128, 64], [64, 32]]
    test_size_values = [0.1]
    
    # Missing data parameters
    enable_missing_data_values = [False]
    missing_data_ratio_values = [0.0]


    # output_folder_name = "num_induing_ablation_v2_16"
    # n_inducing_values = [16]
    # output_folder_name = "num_induing_ablation_v2_32"
    # n_inducing_values = [32]
    output_folder_name = "num_induing_ablation_v2_64"
    n_inducing_values = [64]
    # output_folder_name = "variance_ablation_v1"
    # learning_rate_values = [1e-3,3e-4]
    # num_inducing_samples_list = [4, 64]
    # batch_size_values = [4, 32]
    # output_folder_name = "architecture_ablation_v1"
    # architecture_values = [[16,8],[32,16],[64,32],[128, 64],[256,128]]



    # enable_missing_data_values = [True]
    # missing_data_ratio_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # output_folder_name = "fixed_missing_data_experiment_v5"

    # output_folder_name = "test_fractions_v1"

    ssvi_output_folder = f"oil_flow_related/{output_folder_name}_ssvi"
    orig_output_folder = f"oil_flow_related/{output_folder_name}_orig_gplvm"
    
    # Base configuration paths (now in oil_flow_related directory)
    ssvi_base_config = "oil_flow_related/ssvi_configs/base_lr_0001_s7_test_missing.yaml"
    orig_base_config = "oil_flow_related/original_gp_lvm_configs/base_lr_0001_s7_test_missing.yaml"
    
    
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
        random_seed_values,
        learning_rate_values,
        n_training_steps,
        num_inducing_samples_list,
        architecture_values,
        enable_missing_data_values,
        missing_data_ratio_values,
        test_size_values
    ))
    
    print(f"Generating {len(param_combinations)} configuration combinations...")
    
    # Generate SSVI configurations
    ssvi_count = 0
    for n_inducing, init_lengthscales, init_output_noise_scale, batch_size, random_seed, learning_rate, n_training_steps, num_inducing_samples, architecture, enable_missing_data, missing_data_ratio, test_size in param_combinations:
        # Create new config based on base text
        ssvi_config_text = ssvi_base_text
        
        # Update parameters
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'n_inducing_points', n_inducing)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'encoder_hidden_dims', architecture)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'init_lengthscales', init_lengthscales)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'init_output_noise_scale', init_output_noise_scale)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'batch_size', batch_size)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'random_seed', random_seed)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'local_lr', learning_rate)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'global_lr', learning_rate)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'training_iterations', n_training_steps)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'P', num_inducing_samples)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'enable_missing_data', str(enable_missing_data).lower())
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'missing_data_ratio', missing_data_ratio)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'test_size', test_size)
        
        # Create filename
        filename = create_config_name(n_inducing, init_lengthscales, init_output_noise_scale, batch_size, random_seed, learning_rate, n_training_steps, num_inducing_samples, architecture, enable_missing_data, missing_data_ratio, test_size)
        output_path = os.path.join(ssvi_output_folder, filename)
        
        # Save configuration
        save_config_text(ssvi_config_text, output_path)
        ssvi_count += 1
        
        if ssvi_count % 50 == 0:
            print(f"Generated {ssvi_count} SSVI configurations...")
    
    # Generate Original GP-LVM configurations
    orig_count = 0
    for n_inducing, init_lengthscales, init_output_noise_scale, batch_size, random_seed, learning_rate, n_training_steps, num_inducing_samples, architecture, enable_missing_data, missing_data_ratio, test_size in param_combinations:
        # Create new config based on base text
        orig_config_text = orig_base_text
        
        # Update parameters (note: Original GP-LVM uses 'n_inducing' not 'n_inducing_points' and 'nn_layers' not 'encoder_hidden_dims')
        orig_config_text = update_config_parameter(orig_config_text, 'n_inducing', n_inducing)
        orig_config_text = update_config_parameter(orig_config_text, 'nn_layers', architecture)
        orig_config_text = update_config_parameter(orig_config_text, 'init_lengthscales', init_lengthscales)
        orig_config_text = update_config_parameter(orig_config_text, 'init_output_noise_scale', init_output_noise_scale)
        orig_config_text = update_config_parameter(orig_config_text, 'batch_size', batch_size)
        orig_config_text = update_config_parameter(orig_config_text, 'random_seed', random_seed)
        orig_config_text = update_config_parameter(orig_config_text, 'learning_rate', learning_rate)
        orig_config_text = update_nested_training_steps(orig_config_text, n_training_steps)
        orig_config_text = update_config_parameter(orig_config_text, 'enable_missing_data', str(enable_missing_data).lower())
        orig_config_text = update_config_parameter(orig_config_text, 'missing_data_ratio', missing_data_ratio)
        orig_config_text = update_config_parameter(orig_config_text, 'test_size', test_size)
        
        # Create filename
        filename = create_config_name(n_inducing, init_lengthscales, init_output_noise_scale, batch_size, random_seed, learning_rate, n_training_steps, num_inducing_samples, architecture, enable_missing_data, missing_data_ratio, test_size)
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
    print(f"  architecture: {architecture_values}")
    print(f"  init_lengthscales: {init_lengthscales_values}")
    print(f"  init_output_noise_scale: {init_output_noise_scale_values}")
    print(f"  batch_size: {batch_size_values}")
    print(f"  random_seed: {random_seed_values}")
    print(f"  learning_rate: {learning_rate_values}")
    print(f"  n_training_steps: {n_training_steps}")
    print(f"  num_inducing_samples: {num_inducing_samples_list}")
    print(f"  enable_missing_data: {enable_missing_data_values}")
    print(f"  missing_data_ratio: {missing_data_ratio_values}")
    print(f"  test_size: {test_size_values}")

if __name__ == "__main__":
    generate_config_grid()
