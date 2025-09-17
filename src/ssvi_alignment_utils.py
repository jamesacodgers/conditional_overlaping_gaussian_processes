#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to align SSVI GP-LVM outputs with original GP-LVM structure
"""

import sys
import os
import torch
import numpy as np
import pickle as pkl
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add generalised_gplvm_code to path for original utilities
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

def create_standardized_scattered_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, model_type="original", custom_filename=None):
    """
    Create standardized scattered plots that match the original generalised_gplvm_code plot_report function exactly
    but use scattered points instead of error bars.
    
    This function replicates the exact structure and styling of the original plot_report function from
    generalised_gplvm_code/utils/visualisation.py but removes error bars for scattered visualization.
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import pathlib
    
    
    # Use white background styling
    plt.style.use('default')  # Use default matplotlib style with white background
    plt.figure(figsize=(12,4))

    lengthscales_cpu = model.covar_module.base_kernel.lengthscale.detach().cpu()
    inv_lengthscale = (1/lengthscales_cpu).numpy().flatten()
    lengthscales = lengthscales_cpu
    latent_dim = model.X.latent_dim
    
    # Print lengthscales exactly like original plot_report
    print(f"Inverse lengthscales ({model_type}): {inv_lengthscale}")
    
    # Use top dimensions selection (same for both SSVI and original GP-LVM)
    values, indices = torch.topk(lengthscales, k=2, largest=False)
    l1 = indices.cpu().numpy().flatten()[0]
    l2 = indices.cpu().numpy().flatten()[1]
    
    print(f"Selected dimensions: l1={l1}, l2={l2}")
    print(f"Selected lengthscales: {values}")
    
    # Subplot 1: 2D latent subspace (exactly like original plot_report)
    plt.subplot(131)
    plt.title(f'2d latent subspace [{model_name}]', fontsize='small')
    # Use consistent labels for both SSVI and original GP-LVM
    plt.xlabel(f'Latent dim {l1}', fontsize='small')
    plt.ylabel(f'Latent dim {l2}', fontsize='small')

    with torch.no_grad():
        model_name_class = model.X.__class__.__name__
        if model_name_class == 'VariationalLatentVariable':  # 'gauss' model
            # Variational GP-LVM (gauss)
            mu_x = model.X.q_mu.detach().cpu().numpy()
            sigma_x = torch.nn.functional.softplus(model.X.q_log_sigma).detach().cpu().numpy()
        elif model_name_class == 'NNEncoder':  # 'nn_gauss' model
            Y_train_device = Y_train.to(model.X.mu_layers[0].weight.device) if Y_train.device != model.X.mu_layers[0].weight.device else Y_train
            mu_x_result = model.X.mu(Y_train_device)
            sigma_x_result = model.X.sigma(Y_train_device)
            mu_x = mu_x_result.detach().cpu().numpy()
            sigma_x = np.array([torch.sqrt(x.diag()).detach().cpu().numpy() for x in sigma_x_result.detach()])
    
    # Common setup for all model types
    # Handle labels - convert to numpy if it's a tensor
    if hasattr(labels_train, 'cpu'):
        labels_train_np = labels_train.cpu().numpy()
    else:
        labels_train_np = labels_train
    
    # Create color mapping (same for all models)
    c = ["r", "g", "b"]
    colors = [c[int(i)-1] for i in labels_train_np]
    
    # Common plotting parameters
    P_samples = 64  # Match SSVI P value
    alpha = 0.15

    # Original GP-LVM: use same torch.distributions.Normal approach with P_samples
    # Convert numpy arrays to torch tensors for consistent sampling
    mu_x_torch = torch.from_numpy(mu_x)
    sigma_x_torch = torch.from_numpy(sigma_x)

    # Sample P_samples times using torch.distributions.Normal (same as SSVI)
    for p in range(P_samples):
        sample_1 = torch.distributions.Normal(mu_x_torch[:, l1], sigma_x_torch[:, l1]).sample()
        sample_2 = torch.distributions.Normal(mu_x_torch[:, l2], sigma_x_torch[:, l2]).sample()
        plt.scatter(sample_1.detach().cpu().numpy(),
                    sample_2.detach().cpu().numpy(), alpha=alpha, color=colors)
    
    # Subplot 2: Inverse lengthscales (exactly like original plot_report)
    plt.subplot(132)

    plt.bar(np.arange(latent_dim), height=inv_lengthscale, color='teal')
    plt.title('Inverse Lengthscale with SE-ARD kernel', fontsize='small')
    plt.xlabel('Latent dims', fontsize='small')
    
    # Subplot 3: Loss curve (exactly like original plot_report)
    plt.subplot(133)
    # Ensure losses are Python floats for matplotlib
    losses_floats = [float(x) for x in losses]
    plt.plot(losses_floats, label='batch_size=100', color='orange')
    plt.xlabel('Steps', fontsize='small')
    plt.title('Neg. ELBO Loss', fontsize='small')
    
    plt.tight_layout()
    
    # Use model name (gauss/nn_gauss) instead of class name for consistency
    if model_name == 'VariationalLatentVariable':
        model_short_name = 'gauss'
    elif model_name == 'NNEncoder':
        model_short_name = 'nn_gauss'
    else:
        model_short_name = model_name.lower()
    
    # Use custom filename if provided, otherwise use default naming convention
    if custom_filename:
        plot_filename = custom_filename
    else:
        plot_filename = f"oilflow_{model_short_name}_{random_seed}.png"
    
    # Create plots subdirectory only if we're not already in a plots directory
    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        # We're already in a plots directory or checkpoints subdirectory, save directly here
        plot_path = os.path.join(results_dir, plot_filename)
    else:
        # Create plots subdirectory for better organization
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        plot_path = os.path.join(plots_subdir, plot_filename)
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Standardized scattered plots saved to: {plot_path}")


def save_iteration_model(model, model_name, iteration, random_seed, results_dir):
    """
    Save model at specific iteration to pre_trained_models/checkpoints folder
    """
    # Save to pre_trained_models/checkpoints folder for organized checkpoint storage
    pre_trained_dir = os.path.join(results_dir, 'pre_trained_models', 'checkpoints')
    os.makedirs(pre_trained_dir, exist_ok=True)
    
    filename = f'oilflow_{model_name}_iter_{iteration}_seed_{random_seed}.pkl'
    filepath = os.path.join(pre_trained_dir, filename)
    
    with open(filepath, 'wb') as file:
        pkl.dump(model.state_dict(), file)
    
    return filepath


def create_iteration_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, iteration, model_type="original"):
    """
    Create iteration-based plot saved to plots/checkpoints folder
    """
    plot_filename = f"oilflow_{model_name}_iter_{iteration}_seed_{random_seed}.png"
    
    # Create plots/checkpoints subdirectory for organized checkpoint storage
    plots_checkpoints_dir = os.path.join(results_dir, 'plots', 'checkpoints')
    os.makedirs(plots_checkpoints_dir, exist_ok=True)
    
    # Create the standardized plot with custom filename in checkpoints folder
    create_standardized_scattered_plot(
        model, losses, labels_train, config, plots_checkpoints_dir, 
        model_name, random_seed, Y_train, model_type=model_type, 
        custom_filename=plot_filename
    )
    
    return plot_filename