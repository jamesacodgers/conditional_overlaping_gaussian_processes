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
                           effective_dims_list, results_dir, model_name, random_seed, batch_losses: list = None):
    """
    Plot metrics progression over training iterations (train metrics only)
    Exactly matches SSVI plot structure and styling
    
    Args:
        train_rmse_list: List of training RMSE values
        train_nlpd_list: List of training NLPD values  
        metrics_iterations: List of iteration numbers when metrics were computed
        metrics_losses: List of losses at the same iterations as metrics (full ELBO losses)
        effective_dims_list: List of effective dimensions at the same iterations
        results_dir: Directory to save the plot
        model_name: Name of the model
        random_seed: Random seed for filename
        batch_losses: Optional list of batch ELBO losses (one per iteration) for additional plot
    """
    if not train_rmse_list or not train_nlpd_list:
        print("No metrics data available for progression plot")
        return
    
    # Determine if we need to add batch ELBO plot
    plot_batch_elbo = batch_losses is not None and len(batch_losses) > 0
    
    # Create subplots: 3x2 if batch ELBO (5 plots), otherwise 2x2 (4 plots)
    if plot_batch_elbo:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        ax1 = axes[0, 0]  # RMSE
        ax_batch = axes[0, 1]  # Batch ELBO (second position)
        ax_nlpd = axes[1, 0]  # NLPD
        ax_full_elbo = axes[1, 1]  # Full ELBO
        ax_eff_dims = axes[2, 0]  # Effective dims
        # Hide the last subplot (bottom right)
        axes[2, 1].axis('off')
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        ax_nlpd = ax2
        ax_full_elbo = ax3
        ax_eff_dims = ax4
    
    # Plot 1: RMSE comparison (train only - exactly like SSVI but only train line)
    ax1.plot(metrics_iterations, train_rmse_list, 'b-', label='Train RMSE', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Reconstruction Error (RMSE)')  # Exact same title as SSVI
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Batch ELBO progression (if provided) - second position
    if plot_batch_elbo:
        batch_iterations = list(range(len(batch_losses)))
        ax_batch.plot(batch_iterations, batch_losses, 'orange', linewidth=2)
        ax_batch.set_xlabel('Iteration')
        ax_batch.set_ylabel('Loss')
        ax_batch.set_title('Batch ELBO Progression')
        ax_batch.grid(True, alpha=0.3)
    
    # Plot 3: NLPD comparison (train only - exactly like SSVI but only train line)
    ax_nlpd.plot(metrics_iterations, train_nlpd_list, 'b-', label='Train NLPD', linewidth=2)
    ax_nlpd.set_xlabel('Iteration')
    ax_nlpd.set_ylabel('NLPD')
    ax_nlpd.set_title('Predictive Density (NLPD)')  # Exact same title as SSVI
    ax_nlpd.legend()
    ax_nlpd.grid(True, alpha=0.3)
    
    # Plot 4: Full ELBO Loss progression (exactly like SSVI)
    ax_full_elbo.plot(metrics_iterations, metrics_losses, 'g-', linewidth=2)
    ax_full_elbo.set_xlabel('Iteration')
    ax_full_elbo.set_ylabel('Loss')
    ax_full_elbo.set_title('Training Loss (Full ELBO)')  # Exact same title as SSVI
    ax_full_elbo.grid(True, alpha=0.3)
    
    # Plot 5: Effective latent dimensions (exactly like SSVI)
    if effective_dims_list:
        ax_eff_dims.plot(metrics_iterations, effective_dims_list, 'purple', linewidth=2)
        ax_eff_dims.set_xlabel('Iteration')
        ax_eff_dims.set_ylabel('Effective Dimensions')
        ax_eff_dims.set_title('Model Complexity (Effective Latent Dims)')
        ax_eff_dims.grid(True, alpha=0.3)
    else:
        ax_eff_dims.text(0.5, 0.5, 'No effective dimensions data', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax_eff_dims.transAxes, fontsize=12)
        ax_eff_dims.set_title('Model Complexity (Effective Latent Dims)')
    
    plt.tight_layout()
    
    # Save plot to plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'metrics_progression_{model_name}.png')
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics progression plot saved to: {plot_path}")


def plot_crps_metrics_progression(train_crps_list, metrics_iterations,
                                  results_dir, model_name, random_seed,
                                  test_crps_list=None,
                                  train_coverage_90_list=None,
                                  train_coverage_95_list=None,
                                  test_coverage_90_list=None,
                                  test_coverage_95_list=None,
                                  train_interval_width_90_list=None,
                                  train_interval_width_95_list=None,
                                  test_interval_width_90_list=None,
                                  test_interval_width_95_list=None):
    """
    Plot CRPS, coverage, and mean interval width progression (separate plot).

    Panel 1: CRPS; Panel 2: coverage_90 / coverage_95; Panel 3: interval_width_90 / 95.
    """
    if not train_crps_list:
        print("No CRPS data available for progression plot")
        return

    import numpy as np
    valid_train_idx = [i for i, x in enumerate(train_crps_list) if not np.isnan(x)]
    if not valid_train_idx and not test_crps_list:
        print("No valid CRPS data available for plotting")
        return

    fig, (ax_crps, ax_cov, ax_width) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    if valid_train_idx:
        valid_iterations = [metrics_iterations[i] for i in valid_train_idx]
        ax_crps.plot(valid_iterations, [train_crps_list[i] for i in valid_train_idx],
                     'b-', label='Train CRPS', linewidth=2, marker='o', markersize=4)

    if test_crps_list:
        valid_test_idx = [i for i, x in enumerate(test_crps_list) if not np.isnan(x)]
        if valid_test_idx:
            test_iterations = [metrics_iterations[i] for i in valid_test_idx]
            ax_crps.plot(test_iterations, [test_crps_list[i] for i in valid_test_idx],
                         'r-', label='Test CRPS', linewidth=2, marker='s', markersize=4)

    ax_crps.set_ylabel('CRPS (lower is better)')
    ax_crps.set_title('CRPS, Coverage & Interval Width Progression')
    ax_crps.legend()
    ax_crps.grid(True, alpha=0.3)

    def _plot_series(ax, values, style, label):
        if not values:
            return
        valid_idx = [i for i, x in enumerate(values) if not np.isnan(x)]
        if not valid_idx:
            return
        ax.plot([metrics_iterations[i] for i in valid_idx],
                [values[i] for i in valid_idx],
                style, label=label, linewidth=2, marker='o', markersize=4)

    _plot_series(ax_cov, train_coverage_90_list, 'b-', 'Train coverage_90')
    _plot_series(ax_cov, test_coverage_90_list, 'r-', 'Test coverage_90')
    _plot_series(ax_cov, train_coverage_95_list, 'b--', 'Train coverage_95')
    _plot_series(ax_cov, test_coverage_95_list, 'r--', 'Test coverage_95')
    ax_cov.axhline(0.90, color='gray', linestyle=':', alpha=0.7, label='Nominal 0.90')
    ax_cov.axhline(0.95, color='black', linestyle=':', alpha=0.5, label='Nominal 0.95')
    ax_cov.set_ylabel('Coverage')
    ax_cov.set_ylim(0.0, 1.05)
    ax_cov.legend(fontsize=9)
    ax_cov.grid(True, alpha=0.3)

    _plot_series(ax_width, train_interval_width_90_list, 'b-', 'Train interval_width_90')
    _plot_series(ax_width, test_interval_width_90_list, 'r-', 'Test interval_width_90')
    _plot_series(ax_width, train_interval_width_95_list, 'b--', 'Train interval_width_95')
    _plot_series(ax_width, test_interval_width_95_list, 'r--', 'Test interval_width_95')
    ax_width.set_xlabel('Iteration')
    ax_width.set_ylabel('Mean interval width (lower is sharper)')
    ax_width.legend(fontsize=9)
    ax_width.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'crps_metrics_progression_{model_name}.png')

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"CRPS/coverage/width metrics progression plot saved to: {plot_path}")


def plot_crps_posterior_predictive_progression(train_sample_list, train_mixture_list,
                                               metrics_iterations, results_dir, model_name, random_seed,
                                               test_sample_list=None, test_mixture_list=None):
    """
    Separate progression plot for posterior-predictive CRPS variants (original GPLVM).

    Panel 1: sample CRPS; Panel 2: mixture CRPS.
    """
    import numpy as np

    def _has_valid(values):
        return values is not None and any(not np.isnan(x) for x in values)

    if not (_has_valid(train_sample_list) or _has_valid(train_mixture_list)
            or _has_valid(test_sample_list) or _has_valid(test_mixture_list)):
        print("No valid posterior-predictive CRPS data available for plotting")
        return

    fig, (ax_sample, ax_mix) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    def _plot_series(ax, values, style, label):
        if not values:
            return
        valid_idx = [i for i, x in enumerate(values) if not np.isnan(x)]
        if not valid_idx:
            return
        ax.plot([metrics_iterations[i] for i in valid_idx],
                [values[i] for i in valid_idx],
                style, label=label, linewidth=2, marker='o', markersize=4)

    _plot_series(ax_sample, train_sample_list, 'b-', 'Train sample CRPS')
    _plot_series(ax_sample, test_sample_list, 'r-', 'Test sample CRPS')
    ax_sample.set_ylabel('CRPS (lower is better)')
    ax_sample.set_title('Posterior-Predictive CRPS Variants')
    ax_sample.legend()
    ax_sample.grid(True, alpha=0.3)

    _plot_series(ax_mix, train_mixture_list, 'b-', 'Train mixture CRPS')
    _plot_series(ax_mix, test_mixture_list, 'r-', 'Test mixture CRPS')
    ax_mix.set_xlabel('Iteration')
    ax_mix.set_ylabel('CRPS (lower is better)')
    ax_mix.legend()
    ax_mix.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'crps_posterior_predictive_progression_{model_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Posterior-predictive CRPS progression plot saved to: {plot_path}")


def plot_latent_slice_original_gplvm(model, y_train, model_name, center="zeros", n_idx=None, save_path=None, scaler=None):
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
        scaler: optional scaler for inverse transform (to visualize in original scale)
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
            # Get variance from q_log_sigma
            sigma_x = torch.nn.functional.softplus(model.X.q_log_sigma).detach()  # [N, Q] - standard deviation
        elif model_name == 'nn_gauss':
            # For nn_gauss: encode the training data
            # Handle missing data if present
            y_encoder_input = torch.where(torch.isnan(y_train), 
                                         torch.full_like(y_train, -999.0, dtype=torch.float64), 
                                         y_train)
            mu_x_mean = model.X.mu(y_encoder_input).detach()  # [N, Q]
            # Get variance from encoder
            sigma_x_tensors = model.X.sigma(y_encoder_input).detach()  # List of covariance matrices or tensor
            # Extract diagonal standard deviations
            if isinstance(sigma_x_tensors, list):
                sigma_x = torch.stack([torch.sqrt(x.diag()) for x in sigma_x_tensors])  # [N, Q]
            else:
                sigma_x = torch.sqrt(torch.diagonal(sigma_x_tensors, dim1=-2, dim2=-1))  # [N, Q]
        else:
            raise ValueError(f"Unsupported model type: {model_name}. Use 'gauss' or 'nn_gauss'")
        
        # 1) Pick most-informative latent dim (smallest lengthscale)
        # For original GP-LVM: lengthscale from kernel
        kernel_lengthscales = model.covar_module.base_kernel.lengthscale.flatten().detach().cpu().numpy()
        # Convert to same scale as SSVI (init_lengthscales scale)
        ells = kernel_lengthscales
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
        
        # Inverse transform predictions if scaler provided (for visualization in original scale)
        if scaler is not None:
            f_mean = scaler.inverse_transform(f_mean)  # [G, M]
            # For standard deviation, scale by scaler's std if available
            if hasattr(scaler, 'std'):
                pred_sd = pred_sd * scaler.std  # [G, M]
        
        # Compute Y-axis limits from actual data for consistent scaling across models
        # Get data in original scale (inverse transform if scaler provided)
        y_data_for_limits = y_train.clone()
        if scaler is not None:
            y_data_for_limits = scaler.inverse_transform(y_data_for_limits)
        
        # Compute global Y-axis limits from data range (with padding for uncertainty bands)
        y_min_global = y_data_for_limits.min().item()
        y_max_global = y_data_for_limits.max().item()
        y_range = y_max_global - y_min_global
        y_padding = 1.0 * y_range  # 100% padding (fixed proportion for consistency across scripts, extra space for histograms)
        y_lim_min = y_min_global - y_padding
        y_lim_max = y_max_global + y_padding
        
        # 4) Plot - show first 12 dimensions (or M if M < 12)
        xnp = xs.detach().cpu().numpy()
        
        # Set X-axis limits for consistency (already using [-3, 3] range)
        x_lim_min, x_lim_max = -3.0, 3.0
        
        # Plot only first 12 dimensions (or all if M < 12)
        M_plot = min(M, 12)
        n_cols = 4
        n_rows = int(np.ceil(M_plot / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), constrained_layout=True)
        axes = axes.ravel()
        
        # Sample from q(X) distribution along dimension j for histogram
        # Sample from N(mu_x_mean, sigma_x^2) for each data point
        x_samples_j = torch.normal(mu_x_mean[:, j], torch.clamp(sigma_x[:, j], min=1e-8))  # [N]
        x_j_values = x_samples_j.detach().cpu().numpy()
        
        for m in range(M_plot):
            mu = f_mean[:, m].cpu().numpy()
            sd = pred_sd[:, m].cpu().numpy()
            ax = axes[m]
            ax.plot(xnp, mu, lw=2)
            ax.fill_between(xnp, mu - 2*sd, mu + 2*sd, alpha=0.25)
            ax.set_title(f"y[{m}] vs x_{j} (l={ells[j]:.2g})", fontsize=9)
            ax.set_xlabel(f"x_{j}")
            ax.set_ylabel(f"y[{m}]")
            # Set consistent axis limits
            ax.set_xlim(x_lim_min, x_lim_max)
            ax.set_ylim(y_lim_min, y_lim_max)
            
            # Add histogram directly on the same axes, aligned with x-axis at bottom
            # Compute histogram and scale to make it bigger (use 25% of y-axis range)
            # Use fixed number of bins for consistent granularity across all models
            counts, bins = np.histogram(x_j_values, bins=15, density=True)
            y_range_hist = (y_lim_max - y_lim_min) * 0.25
            max_count = counts.max() if len(counts) > 0 and counts.max() > 0 else 1.0
            counts_scaled = (counts / max_count) * y_range_hist if max_count > 0 else counts * 0
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, counts_scaled, width=(bins[1] - bins[0]), 
                   bottom=y_lim_min, alpha=0.6, color='lightblue', edgecolor='black')
        
        for k in range(M_plot, len(axes)):
            axes[k].axis("off")
        
        if save_path:
            fig.savefig(save_path, dpi=160)
            plt.close(fig)
            print(f"Saved latent slice plot to: {save_path}")
        else:
            plt.show()
    
    model.train()