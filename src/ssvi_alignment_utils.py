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

def create_standardized_scattered_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, model_type="original", custom_filename=None, train_indices=None):
    """
    Create standardized scattered plots that match the SSVI plotting exactly.
    
    This function replicates the exact structure and styling of the SSVI visualization
    from src/ssvi_utils.py with the same subplot order and styling.
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import pathlib
    
    # Use default matplotlib style to match SSVI (override any global ggplot style)
    plt.style.use('default')
    
    with torch.no_grad():
        # Color mapping for labels (exactly like SSVI)
        c = ["r", "g", "b"]
        colors = [c[int(i)-1] for i in labels_train]
        
        # Create figure with same layout as SSVI (3 subplots)
        n_plots = 3
        fig_width = 4 * n_plots
        fig = plt.figure(figsize=(fig_width, 4))
        
        plot_idx = 1
        
        # Plot 1: Training curve (exactly like SSVI)
        plt.subplot(1, n_plots, plot_idx)
        plt.plot(losses, 'b-', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('ELBO Training Curve')
        plt.grid(True, alpha=0.3)
        plot_idx += 1
        
        # Plot 2: Latent space visualization (exactly like SSVI)
        plt.subplot(1, n_plots, plot_idx)
        
        # Find the two most informative dimensions (copy original approach exactly)
        inv_lengthscale = 1 / model.covar_module.base_kernel.lengthscale
        values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=2, largest=False)
        dim1 = indices.cpu().numpy().flatten()[0]  # Move to CPU first
        dim2 = indices.cpu().numpy().flatten()[1]  # Move to CPU first
        # For consistency with SSVI, use sqrt(lengthscale) for axis labels too
        lengthscales_cpu = torch.sqrt(inv_lengthscale.flatten()).cpu()  # Move to CPU for string formatting
        
        # Get latent variables (use mu_x directly, no sampling)
        model_name_class = model.X.__class__.__name__
        if model_name_class == 'VariationalLatentVariable':  # 'gauss' model
            # Variational GP-LVM (gauss)
            mu_x = model.X.q_mu.detach().cpu().numpy()
            sigma_x = torch.nn.functional.softplus(model.X.q_log_sigma).detach().cpu().numpy()
        elif model_name_class == 'NNEncoder':  # 'nn_gauss' model
            Y_train_device = Y_train.to(model.X.mu_layers[0].weight.device) if Y_train.device != model.X.mu_layers[0].weight.device else Y_train
            
            # Handle missing data for nn_gauss plotting (same as training)
            if torch.isnan(Y_train_device).any():
                # Replace NaN values with zeros for encoder input (consistent with training)
                Y_train_for_plotting = Y_train_device.clone()
                Y_train_for_plotting[torch.isnan(Y_train_for_plotting)] = -999.0
            else:
                Y_train_for_plotting = Y_train_device
            
            mu_x_result = model.X.mu(Y_train_for_plotting)
            sigma_x_result = model.X.sigma(Y_train_for_plotting)
            mu_x = mu_x_result.detach().cpu().numpy()
        
        # Plot mu_x directly (no sampling) - exactly like SSVI
        plt.scatter(mu_x[:, dim1], mu_x[:, dim2], 
                   c=colors, alpha=0.6, s=20)
        
        plt.xlabel(f'Latent Dimension {dim1} (lengthscale={lengthscales_cpu[dim1]:.2f})')
        plt.ylabel(f'Latent Dimension {dim2} (lengthscale={lengthscales_cpu[dim2]:.2f})') 
        plt.title(f'Latent Space - Top 2 Informative Dims')
        plot_idx += 1
        
        # Plot 3: Hyperparameters (lengthscales) - consistent with SSVI
        plt.subplot(1, n_plots, plot_idx)
        # For mathematical consistency with SSVI version:
        # SSVI: K = exp(-0.5 * d^2 / init_lengthscales)
        # GP-LVM: K = exp(-0.5 * d^2 / lengthscale^2) 
        # For equivalence: lengthscale = sqrt(init_lengthscales)
        lengthscales_plot = torch.sqrt(1 / model.covar_module.base_kernel.lengthscale.flatten()).cpu()
        plt.bar(range(len(lengthscales_plot)), lengthscales_plot, 
               color="blue", alpha=0.7)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Lengthscale')
        plt.title('GP Lengthscales')
        plt.grid(True, alpha=0.3)
        
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
    
    # Generate fractions plot using the same mu_x and top dimensions
    try:
        # Load fractions data - check data directory first, then current directory
        frctn_path = pathlib.Path("data") / "oil_fractions.txt"
        
        if not frctn_path.exists():
            # Fallback to current directory (SSVI approach)
            frctn_path = pathlib.Path(".") / "DataTrnFrctns.txt"

        if not frctn_path.exists():
            raise FileNotFoundError(f"Fractions file not found. Checked: data/oil_fractions.txt and DataTrnFrctns.txt")

        w = np.loadtxt(frctn_path)  # (N, 2)
        w3 = 1.0 - w.sum(axis=1, keepdims=True)  # (N, 1)
        fractions = np.concatenate((w, w3), axis=1)  # (N, 3)
        
        # Debug: print fractions shapes
        print(f"Debug - fractions shape before subsetting: {fractions.shape}")
        print(f"Debug - Y_train shape: {Y_train.shape}")
        
        # Subset fractions data to match training data using train_indices
        train_indices_cpu = train_indices.cpu().numpy()
        fractions = fractions[train_indices_cpu]
        print(f"Debug - fractions shape after subsetting with train_indices: {fractions.shape}")
        
        # Create fractions plot using the same mu_x and top dimensions from above
        fig_frac, ax_frac = plt.subplots(1, 1, figsize=(7.5, 6))
        
        # Use the same top 2 dimensions that were calculated above
        scatter = ax_frac.scatter(mu_x[:, dim1], mu_x[:, dim2], c=fractions, s=20, alpha=0.6)
        ax_frac.set_title("Latent Space Colored by Fractions")
        ax_frac.set_xlabel(f'mu_{dim1}')
        ax_frac.set_ylabel(f'mu_{dim2}')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_frac)
        cbar.set_label('Fractions')
        
        # Ensure equal aspect ratio for square plot
        ax_frac.set_aspect("equal", adjustable='box')
        fig_frac.tight_layout()
        
        # Save fractions plot with same naming pattern as main plot
        fractions_filename = plot_filename.replace('oilflow_', 'latent_space_by_fractions_')
        
        if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
            frac_plot_path = os.path.join(results_dir, fractions_filename)
        else:
            plots_subdir = os.path.join(results_dir, 'plots')
            os.makedirs(plots_subdir, exist_ok=True)
            frac_plot_path = os.path.join(plots_subdir, fractions_filename)
        
        fig_frac.savefig(frac_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved fractions plot to: {frac_plot_path}")
        plt.close(fig_frac)
        
    except FileNotFoundError as e:
        print(f"Warning: Could not create fractions plot: {e}")


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


def create_iteration_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, iteration, model_type="original", train_indices=None):
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
        custom_filename=plot_filename, train_indices=train_indices
    )
    
    return plot_filename