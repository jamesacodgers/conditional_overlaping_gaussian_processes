#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Grid Generator for MNIST/FashionMNIST OOD Detection Experiments

This script generates a grid of configuration files for OOD detection experiments.
It takes base configurations and creates variations with different parameter combinations.

Usage:
    python mnist_ood_generate_config_grid.py
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
    elif new_value is None:
        value_str = 'null'
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


def create_config_name(n_inducing, init_lengthscales, batch_size, random_seed, learning_rate,
                       n_training_steps, num_inducing_samples, architecture, 
                       id_dataset, ood_dataset, test_size, model_type="ssvi"):
    """Create a valid filename from parameter values"""
    # Replace dots with nothing for valid filenames
    len_str = str(init_lengthscales).replace('.', '')
    lr_str = str(learning_rate).replace('.', '')
    n_steps_str = str(n_training_steps).replace('.', '')
    test_size_str = str(test_size).replace('.', '')

    # Sum of neurons in architecture (e.g., [128, 64, 32] -> 224)
    arch_sum = sum(architecture)

    # ID dataset short name
    id_short = "fmnist" if id_dataset == "FashionMNIST" else "mnist"
    # OOD dataset short name
    ood_short = "fmnist" if ood_dataset == "FashionMNIST" else "mnist"

    if model_type == "ssvi":
        return f"ood_{id_short}_id_{ood_short}_ood_n_ind_{n_inducing}_arch_{arch_sum}_len_{len_str}_b_{batch_size}_s_{random_seed}_lr_{lr_str}_n_steps_{n_steps_str}_P_{num_inducing_samples}_test_{test_size_str}.yaml"
    else:
        # Original GP-LVM doesn't use P (num_inducing_samples)
        return f"ood_{id_short}_id_{ood_short}_ood_n_ind_{n_inducing}_arch_{arch_sum}_len_{len_str}_b_{batch_size}_s_{random_seed}_lr_{lr_str}_n_steps_{n_steps_str}_test_{test_size_str}.yaml"


def generate_config_grid():
    """Generate grid of configurations for MNIST/FashionMNIST OOD Detection"""

    # ========================================
    # PARAMETER RANGES - MODIFY AS NEEDED
    # ========================================
    
    # Model Architecture
    n_inducing_values = [64]
    architecture_values = [[512,256]]  # encoder_hidden_dims
    
    # GP Hyperparameters
    init_lengthscales_values = [1.0]
    num_inducing_samples_list = [4]  # P samples for SSVI
    
    # Training Configuration
    batch_size_values = [32]
    learning_rate_values = [0.0001]
    n_training_steps = [200_000]
    
    # Random Seeds
    random_seed_values = [0]
    
    # Data Split
    test_size_values = [0.1]
    
    # Missing Data Configuration
    enable_missing_data_values = [False]  # Enable missing data reconstruction capability
    missing_data_ratio_values = [0.3]  # Fraction of dimensions to artificially mask (for testing)
    
    # ========================================
    # OOD Detection Configuration
    # ========================================
    id_dataset_values = ["MNIST"]  # Options: "MNIST", "FashionMNIST"
    ood_dataset_values = ["FashionMNIST"]  # Options: "MNIST", "FashionMNIST" (must be explicitly specified)
    
    # ========================================
    # ID Dataset Filtering Configuration
    # ========================================
    # Applied to in-distribution dataset
    id_classes_values = [None]  # None = all classes, or list like [0,1,2,3,4,5,6,7] for subset
    id_samples_per_class_values = [1000] # was [500] # Samples per ID class (None = use all)
    
    # ========================================
    # OOD Dataset Filtering Configuration
    # ========================================
    # Applied to out-of-distribution dataset
    ood_classes_values = [None]  # None = all classes, or list like [8,9] for subset
    ood_samples_per_class_values = [500] # was [500]  # Samples per OOD class (None = use all)
    
    # ========================================
    # Output Configuration
    # ========================================
    output_folder_name = "reverse_mnist_ood_experiment_final_v3"
    ssvi_output_folder = f"ood_mnist_related/{output_folder_name}_ssvi"
    orig_output_folder = f"ood_mnist_related/{output_folder_name}_orig_gplvm"
    
    # Base configuration paths
    ssvi_base_config = "ood_mnist_related/mnist_ssvi_configs/mnist_base_lr_0001_s7.yaml"
    orig_base_config = "ood_mnist_related/mnist_original_gp_lvm_config/mnist_base_lr_0001_s7.yaml"

    # Load base configurations
    print("Loading base configurations...")
    ssvi_base_text = load_base_config_text(ssvi_base_config)
    orig_base_text = load_base_config_text(orig_base_config)

    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        n_inducing_values,
        init_lengthscales_values,
        batch_size_values,
        random_seed_values,
        learning_rate_values,
        n_training_steps,
        num_inducing_samples_list,
        architecture_values,
        id_dataset_values,
        ood_dataset_values,
        test_size_values,
        enable_missing_data_values,
        missing_data_ratio_values,
        id_classes_values,
        id_samples_per_class_values,
        ood_classes_values,
        ood_samples_per_class_values
    ))

    print(f"Generating {len(param_combinations)} configuration combinations...")

    # Generate SSVI OOD configurations
    ssvi_count = 0
    for (n_inducing, init_lengthscales, batch_size, random_seed, learning_rate, 
         n_steps, num_inducing_samples, architecture, id_dataset, ood_dataset, test_size,
         enable_missing_data, missing_data_ratio, id_classes, id_samples_per_class, 
         ood_classes, ood_samples_per_class) in param_combinations:
        
        # Create new config based on base text
        ssvi_config_text = ssvi_base_text

        # Update model parameters
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'n_inducing_points', n_inducing)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'encoder_hidden_dims', architecture)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'init_lengthscales', init_lengthscales)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'batch_size', batch_size)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'random_seed', random_seed)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'local_lr', learning_rate)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'global_lr', learning_rate)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'training_iterations', n_steps)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'P', num_inducing_samples)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'test_size', test_size)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'enable_missing_data',
                                                   str(enable_missing_data).lower())
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'missing_data_ratio', missing_data_ratio)
        
        # Update OOD configuration with unified parameters
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'id_dataset', f'"{id_dataset}"')
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'id_classes', id_classes)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'id_samples_per_class', id_samples_per_class)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'ood_dataset', f'"{ood_dataset}"')
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'ood_classes', ood_classes)
        ssvi_config_text = update_config_parameter(ssvi_config_text, 'ood_samples_per_class', ood_samples_per_class)

        # Create filename
        filename = create_config_name(n_inducing, init_lengthscales, batch_size, random_seed,
                                      learning_rate, n_steps, num_inducing_samples, architecture,
                                      id_dataset, ood_dataset, test_size)
        output_path = os.path.join(ssvi_output_folder, filename)

        # Save configuration
        save_config_text(ssvi_config_text, output_path)
        ssvi_count += 1

        if ssvi_count % 10 == 0:
            print(f"Generated {ssvi_count} SSVI OOD configurations...")

    # Generate Original GP-LVM OOD configurations
    orig_count = 0
    for (n_inducing, init_lengthscales, batch_size, random_seed, learning_rate, 
         n_steps, num_inducing_samples, architecture, id_dataset, ood_dataset, test_size,
         enable_missing_data, missing_data_ratio, id_classes, id_samples_per_class, 
         ood_classes, ood_samples_per_class) in param_combinations:
        
        # Create new config based on base text
        orig_config_text = orig_base_text

        # Update model parameters (note: Original GP-LVM uses different param names)
        orig_config_text = update_config_parameter(orig_config_text, 'n_inducing', n_inducing)
        orig_config_text = update_config_parameter(orig_config_text, 'nn_layers', architecture)
        orig_config_text = update_config_parameter(orig_config_text, 'init_lengthscales', init_lengthscales)
        orig_config_text = update_config_parameter(orig_config_text, 'batch_size', batch_size)
        orig_config_text = update_config_parameter(orig_config_text, 'random_seed', random_seed)
        orig_config_text = update_config_parameter(orig_config_text, 'learning_rate', learning_rate)
        orig_config_text = update_nested_training_steps(orig_config_text, n_steps)
        orig_config_text = update_config_parameter(orig_config_text, 'test_size', test_size)
        orig_config_text = update_config_parameter(orig_config_text, 'enable_missing_data',
                                                   str(enable_missing_data).lower())
        orig_config_text = update_config_parameter(orig_config_text, 'missing_data_ratio', missing_data_ratio)
        
        # Update OOD configuration with unified parameters
        orig_config_text = update_config_parameter(orig_config_text, 'id_dataset', f'"{id_dataset}"')
        orig_config_text = update_config_parameter(orig_config_text, 'id_classes', id_classes)
        orig_config_text = update_config_parameter(orig_config_text, 'id_samples_per_class', id_samples_per_class)
        orig_config_text = update_config_parameter(orig_config_text, 'ood_dataset', f'"{ood_dataset}"')
        orig_config_text = update_config_parameter(orig_config_text, 'ood_classes', ood_classes)
        orig_config_text = update_config_parameter(orig_config_text, 'ood_samples_per_class', ood_samples_per_class)

        # Create filename
        filename = create_config_name(n_inducing, init_lengthscales, batch_size, random_seed,
                                      learning_rate, n_steps, num_inducing_samples, architecture,
                                      id_dataset, ood_dataset, test_size, model_type="orig")
        output_path = os.path.join(orig_output_folder, filename)

        # Save configuration
        save_config_text(orig_config_text, output_path)
        orig_count += 1

        if orig_count % 10 == 0:
            print(f"Generated {orig_count} Original GP-LVM OOD configurations...")

    print(f"\nConfiguration generation completed!")
    print(f"Generated {ssvi_count} OOD SSVI configurations in '{ssvi_output_folder}'")
    print(f"Generated {orig_count} Original GP-LVM configurations in '{orig_output_folder}'")
    print(f"\nTotal parameter combinations: {len(param_combinations)}")
    print(f"\nParameter ranges:")
    print(f"  n_inducing: {n_inducing_values}")
    print(f"  architecture: {architecture_values}")
    print(f"  init_lengthscales: {init_lengthscales_values}")
    print(f"  batch_size: {batch_size_values}")
    print(f"  random_seed: {random_seed_values}")
    print(f"  learning_rate: {learning_rate_values}")
    print(f"  n_training_steps: {n_training_steps}")
    print(f"  num_inducing_samples (P): {num_inducing_samples_list}")
    print(f"  test_size: {test_size_values}")
    print(f"\nOOD Configuration:")
    print(f"  id_dataset: {id_dataset_values}")
    print(f"  ood_dataset: {ood_dataset_values}")
    print(f"  id_classes: {id_classes_values} (None = all classes)")
    print(f"  id_samples_per_class: {id_samples_per_class_values}")
    print(f"  ood_classes: {ood_classes_values} (None = all classes)")
    print(f"  ood_samples_per_class: {ood_samples_per_class_values}")


if __name__ == "__main__":
    generate_config_grid()

