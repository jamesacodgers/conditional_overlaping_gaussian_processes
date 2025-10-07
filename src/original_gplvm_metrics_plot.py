#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function to create metrics progression plot for original GP-LVM
"""

import matplotlib.pyplot as plt
import os
import torch

def compute_effective_latent_dims(model, threshold: float = 0.1):
    """
    Compute number of effective latent dimensions based on lengthscales
    For original GP-LVM models (GPyTorch-based)
    
    This matches the SSVI approach by converting to the same scale:
    - SSVI: lengthscales = 1/beta (direct init_lengthscales scale)
    - GP-LVM: kernel_lengthscales = exp(log_lengthscales) -> convert to init_lengthscales = kernel_lengthscales^2
    - Both now represent informativeness on the same init_lengthscales scale
    - Conversion ensures fair comparison between models
    
    Args:
        model: Original GP-LVM model (GPyTorch-based)
        threshold: Threshold for considering a dimension effective (relative to max)
        
    Returns:
        (effective_dims, total_dims)
    """
    with torch.no_grad():
        # Get lengthscales from GPyTorch model
        if hasattr(model, 'covar_module') and hasattr(model.covar_module, 'base_kernel'):
            # GPyTorch stores lengthscales in log space, so we need to exponentiate
            kernel_lengthscales = model.covar_module.base_kernel.lengthscale.flatten()
            
            # Convert back to init_lengthscales scale for fair comparison with SSVI
            # GP-LVM: lengthscale = sqrt(init_lengthscales) -> init_lengthscales = lengthscale^2
            lengthscales = kernel_lengthscales ** 2
            
            # Normalize by maximum lengthscale (exactly like SSVI)
            max_lengthscale = lengthscales.max()
            relative_lengthscales = lengthscales / max_lengthscale
            
            # Count dimensions above threshold (exactly like SSVI)
            effective_dims = (relative_lengthscales > threshold).sum().item()
            total_dims = len(lengthscales)
        else:
            return 0, 0
        
    return effective_dims, total_dims

def plot_missing_nlpd_metrics_progression(train_missing_nlpd_list, metrics_iterations, 
                                        results_dir, model_name, random_seed):
    """
    Plot missing NLPD metrics progression over training iterations (train only)
    
    Args:
        train_missing_nlpd_list: List of training missing NLPD values
        metrics_iterations: List of iteration numbers when metrics were computed
        results_dir: Directory to save the plot
        model_name: Name of the model
        random_seed: Random seed for filename
    """
    if not train_missing_nlpd_list:
        print("No missing NLPD data available for progression plot")
        return
    
    # Check if we have any valid missing NLPD data (not NaN)
    import numpy as np
    valid_indices = [i for i, x in enumerate(train_missing_nlpd_list) if not np.isnan(x)]
    
    if not valid_indices:
        print("No valid missing NLPD data available for plotting")
        return
    
    # Filter data to only include valid values
    valid_iterations = [metrics_iterations[i] for i in valid_indices]
    valid_missing_nlpd = [train_missing_nlpd_list[i] for i in valid_indices]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot train missing NLPD
    ax.plot(valid_iterations, valid_missing_nlpd, 'b-', label='Train Missing NLPD', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Missing Data NLPD (lower is better)')
    ax.set_title('Missing Data Predictive Density Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'missing_nlpd_metrics_progression_{model_name}.png')
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Missing NLPD metrics progression plot saved to: {plot_path}")


def plot_metrics_progression(train_rmse_list, train_nlpd_list, metrics_iterations, metrics_losses, 
                           effective_dims_list, results_dir, model_name, random_seed):
    """
    Plot metrics progression over training iterations (train metrics only)
    Exactly matches SSVI plot structure and styling
    
    Args:
        train_rmse_list: List of training RMSE values
        train_nlpd_list: List of training NLPD values  
        metrics_iterations: List of iteration numbers when metrics were computed
        metrics_losses: List of losses at the same iterations as metrics
        effective_dims_list: List of effective dimensions at the same iterations
        results_dir: Directory to save the plot
        model_name: Name of the model
        random_seed: Random seed for filename
    """
    if not train_rmse_list or not train_nlpd_list:
        print("No metrics data available for progression plot")
        return
    
    # Create subplots (exactly like SSVI)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: RMSE comparison (train only - exactly like SSVI but only train line)
    ax1.plot(metrics_iterations, train_rmse_list, 'b-', label='Train RMSE', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Reconstruction Error (RMSE)')  # Exact same title as SSVI
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: NLPD comparison (train only - exactly like SSVI but only train line)
    ax2.plot(metrics_iterations, train_nlpd_list, 'b-', label='Train NLPD', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('NLPD')
    ax2.set_title('Predictive Density (NLPD)')  # Exact same title as SSVI
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ELBO Loss progression (exactly like SSVI)
    ax3.plot(metrics_iterations, metrics_losses, 'g-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('ELBO Loss')
    ax3.set_title('Training Loss (ELBO)')  # Exact same title as SSVI
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effective latent dimensions (exactly like SSVI)
    if effective_dims_list:
        ax4.plot(metrics_iterations, effective_dims_list, 'purple', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Effective Dimensions')
        ax4.set_title('Model Complexity (Effective Latent Dims)')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No effective dimensions data', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Model Complexity (Effective Latent Dims)')
    
    plt.tight_layout()
    
    # Save plot to plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'metrics_progression_{model_name}.png')
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics progression plot saved to: {plot_path}")


def plot_elbo_components_progression(likelihood_list, kl_x_list, kl_u_list, metrics_iterations, 
                                    metrics_losses, results_dir, model_name, random_seed):
    """
    Plot ELBO components (likelihood, kl_x, kl_u) progression over training iterations
    
    Args:
        likelihood_list: List of likelihood values
        kl_x_list: List of KL(q(X)||p(X)) values
        kl_u_list: List of KL(q(U)||p(U)) values
        metrics_iterations: List of iteration numbers when metrics were computed
        metrics_losses: List of ELBO losses at the same iterations
        results_dir: Directory to save the plot
        model_name: Name of the model
        random_seed: Random seed for filename
    """
    if not likelihood_list or not kl_x_list or not kl_u_list:
        print("No ELBO components data available for progression plot")
        return
    
    # Check if we have any valid data (not all NaN)
    import numpy as np
    valid_likelihood = [i for i, x in enumerate(likelihood_list) if not np.isnan(x)]
    valid_kl_x = [i for i, x in enumerate(kl_x_list) if not np.isnan(x)]
    valid_kl_u = [i for i, x in enumerate(kl_u_list) if not np.isnan(x)]
    
    if not valid_likelihood and not valid_kl_x and not valid_kl_u:
        print("No valid ELBO components data available for plotting (all NaN)")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Likelihood and ELBO Loss
    if valid_likelihood:
        ax1.plot(metrics_iterations, likelihood_list, 'g-', label='Likelihood', linewidth=2, marker='o')
    ax1.plot(metrics_iterations, metrics_losses, 'k--', label='ELBO Loss', linewidth=2, marker='d')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Likelihood and ELBO Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: KL Divergences
    if valid_kl_x:
        ax2.plot(metrics_iterations, kl_x_list, 'r-', label='KL(q(X)||p(X))', linewidth=2, marker='s')
    if valid_kl_u:
        ax2.plot(metrics_iterations, kl_u_list, 'b-', label='KL(q(U)||p(U))', linewidth=2, marker='^')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('KL Divergence', fontsize=12)
    ax2.set_title('KL Divergence Terms', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'elbo_components_progression_{model_name}.png')
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ELBO components progression plot saved to: {plot_path}")