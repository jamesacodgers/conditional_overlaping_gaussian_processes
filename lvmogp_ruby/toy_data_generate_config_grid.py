#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Grid Generator for Toy Data LVMOGP SSVI Experiments

This script generates a grid of configuration files based on specified parameter ranges.
It takes the base configuration and creates variations with different parameter combinations.

Usage:
    # From lvmogp_ruby folder:
    python toy_data_generate_config_grid.py
    
    # From project root:
    python lvmogp_ruby/toy_data_generate_config_grid.py
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
    elif isinstance(new_value, bool):
        # Format as lowercase string for YAML
        value_str = str(new_value).lower()
    else:
        value_str = str(new_value)
    
    # Pattern to match the parameter line with optional comment
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

def create_config_name(n_inducing, architecture, training_iterations, batch_size, 
                       local_lr, global_lr, init_lengthscales, init_signal_variance, 
                       P, random_seed):
    """Create a valid filename from parameter values"""
    # Replace dots with nothing for valid filenames
    local_lr_str = str(local_lr).replace('.', '').replace('-', '')
    global_lr_str = str(global_lr).replace('.', '').replace('-', '')
    len_str = str(init_lengthscales).replace('.', '')
    sig_str = str(init_signal_variance).replace('.', '')
    
    # Sum of neurons in architecture (e.g., [128, 64] -> 192)
    arch_sum = sum(architecture)
    
    return f"n_ind_{n_inducing}_arch_{arch_sum}_iter_{training_iterations}_b_{batch_size}_local_lr_{local_lr_str}_global_lr_{global_lr_str}_len_{len_str}_sig_{sig_str}_P_{P}_s_{random_seed}.yaml"

def generate_config_grid():
    """Generate grid of configurations for toy data LVMOGP SSVI experiments"""
    
    # Model architecture
    n_inducing_points_values = [32, 64]
    encoder_hidden_dims_values = [
        [128, 64]
    ]
    
    # Training parameters
    training_iterations_values = [40000]
    batch_size_values = [256, 332]
    
    # Learning rates (matching base config style)
    local_lr_values = [1e-4, 1e-3]
    global_lr_values = [1e-4, 1e-3]
    
    # GP hyperparameters
    init_lengthscales_values = [1.0]
    init_signal_variance_values = [1.0]
    P_values = [32, 64]  # Number of samples for SSVI
    
    # Random seeds for multiple runs
    random_seed_values = [1]
    
    # ========================================================================
    # ALTERNATIVE PARAMETER RANGES (comment/uncomment as needed)
    # ========================================================================
    # For quick testing (generates fewer configs):
    # n_inducing_points_values = [32]
    # encoder_hidden_dims_values = [[128, 64]]
    # training_iterations_values = [5000]
    # batch_size_values = [324]
    # local_lr_values = [1e-4]
    # global_lr_values = [1e-4]
    # init_lengthscales_values = [1.0]
    # init_signal_variance_values = [1.0]
    # P_values = [32]
    # random_seed_values = [1, 2]
    # 
    # For comprehensive grid search:
    # n_inducing_points_values = [16, 32, 64]
    # encoder_hidden_dims_values = [[64, 32], [128, 64], [256, 128]]
    # training_iterations_values = [5000, 10000, 15000]
    # batch_size_values = [100, 200, 324]
    # local_lr_values = [1e-4, 5e-4, 1e-3]
    # global_lr_values = [1e-4, 5e-4, 1e-3]
    # init_lengthscales_values = [0.5, 1.0, 2.0]
    # init_signal_variance_values = [0.5, 1.0, 2.0]
    # P_values = [16, 32, 64]
    # random_seed_values = [1, 2, 3, 4, 5]
    
    # ========================================================================
    
    # Output folder (relative to lvmogp_ruby directory)
    output_folder = "configs/toy_data_grid"
    
    # Base configuration path (relative to lvmogp_ruby directory)
    base_config = "configs/base_config.yaml"
    
    # Load base configuration
    print("=" * 70)
    print("Toy Data LVMOGP SSVI Configuration Grid Generator")
    print("=" * 70)
    print(f"\nLoading base configuration...")
    print(f"  Base config: {base_config}")
    print(f"  Output folder: {output_folder}")
    
    base_text = load_base_config_text(base_config)
    
    # Calculate total number of combinations
    num_combinations = (
        len(n_inducing_points_values) *
        len(encoder_hidden_dims_values) *
        len(training_iterations_values) *
        len(batch_size_values) *
        len(local_lr_values) *
        len(global_lr_values) *
        len(init_lengthscales_values) *
        len(init_signal_variance_values) *
        len(P_values) *
        len(random_seed_values)
    )
    
    print(f"\nParameter ranges:")
    print(f"  n_inducing_points: {n_inducing_points_values} (n={len(n_inducing_points_values)})")
    print(f"  encoder_hidden_dims: {encoder_hidden_dims_values} (n={len(encoder_hidden_dims_values)})")
    print(f"  training_iterations: {training_iterations_values} (n={len(training_iterations_values)})")
    print(f"  batch_size: {batch_size_values} (n={len(batch_size_values)})")
    print(f"  local_lr: {local_lr_values} (n={len(local_lr_values)})")
    print(f"  global_lr: {global_lr_values} (n={len(global_lr_values)})")
    print(f"  init_lengthscales: {init_lengthscales_values} (n={len(init_lengthscales_values)})")
    print(f"  init_signal_variance: {init_signal_variance_values} (n={len(init_signal_variance_values)})")
    print(f"  P: {P_values} (n={len(P_values)})")
    print(f"  random_seed: {random_seed_values} (n={len(random_seed_values)})")
    
    print(f"\n{'WARNING: ' if num_combinations > 100 else ''}Total configurations to generate: {num_combinations}")
    if num_combinations > 500:
        print("  This is a very large number! Consider reducing parameter ranges.")
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        n_inducing_points_values,
        encoder_hidden_dims_values,
        training_iterations_values,
        batch_size_values,
        local_lr_values,
        global_lr_values,
        init_lengthscales_values,
        init_signal_variance_values,
        P_values,
        random_seed_values
    ))
    
    print(f"\nGenerating configurations...")
    
    # Generate configurations
    count = 0
    for (n_inducing, encoder_dims, training_iter, batch_size, 
         local_lr, global_lr, init_len, init_sig, P, random_seed) in param_combinations:
        
        # Create new config based on base text
        config_text = base_text
        
        # Update parameters
        config_text = update_config_parameter(config_text, 'n_inducing_points', n_inducing)
        config_text = update_config_parameter(config_text, 'encoder_hidden_dims', encoder_dims)
        config_text = update_config_parameter(config_text, 'training_iterations', training_iter)
        config_text = update_config_parameter(config_text, 'batch_size', batch_size)
        config_text = update_config_parameter(config_text, 'local_lr', local_lr)
        config_text = update_config_parameter(config_text, 'global_lr', global_lr)
        config_text = update_config_parameter(config_text, 'init_lengthscales', init_len)
        config_text = update_config_parameter(config_text, 'init_signal_variance', init_sig)
        config_text = update_config_parameter(config_text, 'P', P)
        config_text = update_config_parameter(config_text, 'random_seed', random_seed)
        
        # Update checkpoint frequency to fixed value
        checkpoint_freq = 2500
        config_text = update_config_parameter(config_text, 'checkpoint_frequency', checkpoint_freq)
        
        # Create filename
        filename = create_config_name(
            n_inducing, encoder_dims, training_iter, batch_size,
            local_lr, global_lr, init_len, init_sig, P, random_seed
        )
        output_path = os.path.join(output_folder, filename)
        
        # Save configuration
        save_config_text(config_text, output_path)
        count += 1
        
        if count % 100 == 0:
            print(f"  Generated {count} configurations...")
    
    print(f"\n" + "=" * 70)
    print(f"Configuration generation completed!")
    print(f"Generated {count} configurations in '{output_folder}'")
    print("=" * 70)
    print(f"\nTo run experiments with generated configs:")
    print(f"\n  From lvmogp_ruby folder:")
    print(f"    python toy_data_LVMOGP_SSVI.py --config {output_folder}/<config_name>.yaml")
    print(f"\n  From project root:")
    print(f"    python lvmogp_ruby/toy_data_LVMOGP_SSVI.py --config lvmogp_ruby/{output_folder}/<config_name>.yaml")
    print(f"\nExample:")
    if count > 0:
        # Show an example filename
        example_config = create_config_name(
            n_inducing_points_values[0], 
            encoder_hidden_dims_values[0],
            training_iterations_values[0],
            batch_size_values[0],
            local_lr_values[0],
            global_lr_values[0],
            init_lengthscales_values[0],
            init_signal_variance_values[0],
            P_values[0],
            random_seed_values[0]
        )
        print(f"  python toy_data_LVMOGP_SSVI.py --config {output_folder}/{example_config}")

if __name__ == "__main__":
    generate_config_grid()

