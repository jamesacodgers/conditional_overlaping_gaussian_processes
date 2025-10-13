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


def plot_missing_rmse_metrics_progression(train_missing_rmse_list, metrics_iterations, 
                                        results_dir, model_name, random_seed):
    """
    Plot missing RMSE metrics progression over training iterations (train only)
    
    Args:
        train_missing_rmse_list: List of training missing RMSE values
        metrics_iterations: List of iteration numbers when metrics were computed
        results_dir: Directory to save the plot
        model_name: Name of the model
        random_seed: Random seed for filename
    """
    if not train_missing_rmse_list:
        print("No missing RMSE data available for progression plot")
        return
    
    # Check if we have any valid missing RMSE data (not NaN)
    import numpy as np
    valid_indices = [i for i, x in enumerate(train_missing_rmse_list) if not np.isnan(x)]
    
    if not valid_indices:
        print("No valid missing RMSE data available for plotting")
        return
    
    # Filter data to only include valid values
    valid_iterations = [metrics_iterations[i] for i in valid_indices]
    valid_missing_rmse = [train_missing_rmse_list[i] for i in valid_indices]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot train missing RMSE
    ax.plot(valid_iterations, valid_missing_rmse, 'r-', label='Train Missing RMSE', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Missing Data RMSE (lower is better)')
    ax.set_title('Missing Data Reconstruction Error Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'missing_rmse_metrics_progression_{model_name}.png')
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Missing RMSE metrics progression plot saved to: {plot_path}")


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


def plot_latent_slice_original_gplvm(model, y_train, model_name, center="zeros", n_idx=None, save_path=None):
    """
    Plot 1D latent slice for original GP-LVM models (gauss and nn_gauss).
    
    Args:
        model: Original BayesianGPLVM model instance (gauss or nn_gauss)
        y_train: Training data [N, M]
        model_name: Type of model ('gauss' or 'nn_gauss')
        center: str, either "zeros" (default) or "data_point". If "data_point", 
                must provide n_idx to specify which data point to use as center.
        n_idx: int, index of data point to use as center when center="data_point"
        save_path: str, optional path to save the plot. If None, displays the plot.
    """
    import numpy as np
    
    # Determine device
    device = next(model.parameters()).device
    
    # Get latent dimension
    if hasattr(model.X, 'latent_dim'):
        Q = model.X.latent_dim
    else:
        raise ValueError("Model does not have latent_dim attribute")
    
    M = y_train.shape[1]  # Observation dimension
    N = y_train.shape[0]  # Number of training points
    
    model.eval()
    
    with torch.no_grad():
        # Get latent representations based on model type
        if model_name == 'gauss':
            # For gauss: latent is variational parameters
            mu_x_mean = model.X.q_mu.detach()  # [N, Q]
        elif model_name == 'nn_gauss':
            # For nn_gauss: encode the training data
            # Handle missing data if present
            y_encoder_input = torch.where(torch.isnan(y_train), 
                                         torch.full_like(y_train, -999.0, dtype=torch.float64), 
                                         y_train)
            mu_x_mean = model.X.mu(y_encoder_input).detach()  # [N, Q]
        else:
            raise ValueError(f"Unsupported model type: {model_name}. Use 'gauss' or 'nn_gauss'")
        
        # 1) Pick most-informative latent dim (smallest lengthscale)
        # For original GP-LVM: lengthscale from kernel
        kernel_lengthscales = model.covar_module.base_kernel.lengthscale.flatten().detach().cpu().numpy()
        # Convert to same scale as SSVI (init_lengthscales scale)
        ells = kernel_lengthscales ** 2  # lengthscale^2 gives init_lengthscales
        j = int(np.argmin(ells))
        
        # 2) Center
        x0 = torch.zeros(Q, dtype=torch.float64, device=device)
        if center == "data_point":
            assert n_idx is not None, "n_idx must be provided when center='data_point'"
            x0 = mu_x_mean[n_idx].to(device)
        
        # 3) Use fixed range for consistent comparison across models
        # Use a standard range of [-3, 3] which covers typical latent space
        lo, hi = -3.0, 3.0
        
        G = 200  # Grid resolution
        
        xs = torch.linspace(lo, hi, G, dtype=torch.float64, device=device)
        Xgrid = x0.repeat(G, 1)
        Xgrid[:, j] = xs
        
        # Get GP predictions using original GP-LVM's forward method
        # model(x) returns MultivariateNormal distribution
        Y_pred = model(Xgrid)  # Returns MultivariateNormal
        
        # Extract predictive mean and variance
        # Y_pred.loc has shape [M, G] (transposed)
        # Y_pred.covariance_matrix has shape [M] list of [G, G] matrices
        f_mean = Y_pred.loc.T  # [G, M] - transpose to match our convention
        
        # Extract diagonal variances from covariance matrices
        # Y_pred.covariance_matrix is a batch of covariance matrices
        f_var = torch.stack([torch.diag(Y_pred.covariance_matrix[m]) for m in range(M)]).T  # [G, M]
        
        # Add observation noise to get full predictive variance
        obs_noise = model.likelihood.noise.item()
        pred_var = f_var + obs_noise  # [G, M]
        pred_sd = torch.sqrt(torch.clamp(pred_var, min=1e-12))  # [G, M]
        
        # 4) Plot
        xnp = xs.detach().cpu().numpy()
        fig, axes = plt.subplots(3, 4, figsize=(12, 7.5), constrained_layout=True)
        axes = axes.ravel()
        
        for m in range(M):
            mu = f_mean[:, m].cpu().numpy()
            sd = pred_sd[:, m].cpu().numpy()
            ax = axes[m]
            ax.plot(xnp, mu, lw=2)
            ax.fill_between(xnp, mu - 2*sd, mu + 2*sd, alpha=0.25)
            ax.set_title(f"y[{m}] vs x_{j} (l={ells[j]:.2g})", fontsize=9)
            ax.set_xlabel(f"x_{j}")
            ax.set_ylabel(f"y[{m}]")
        
        for k in range(M, len(axes)):
            axes[k].axis("off")
        
        if save_path:
            fig.savefig(save_path, dpi=160)
            plt.close(fig)
            print(f"Saved latent slice plot to: {save_path}")
        else:
            plt.show()
    
    model.train()