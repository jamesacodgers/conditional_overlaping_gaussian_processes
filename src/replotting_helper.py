"""
Replotting helper functions for SSVI GP-LVM models.

This module contains functions specifically designed for replotting saved models,
with improved device handling and PDF output support.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import no_grad
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


@no_grad()
def plot_latent_slice_outer_replot(model, y_train, center="zeros", n_idx=None, save_path=None, scaler=None):
    """
    Plot 1D latent slice with outer-expectation uncertainty bands (for replotting).
    
    This function creates a visualization of the GP-LVM's predictive behavior along
    the most informative latent dimension (smallest lengthscale), showing both the
    mean prediction and ELBO-consistent uncertainty bands.
    
    Modified version for replotting that:
    - Ensures all tensors are on the correct device
    - Saves plots as PDF instead of PNG
    
    Args:
        model: GPLVM_SSVI_AM_INF model instance
        y_train: Training data [N, M] - used to encode latent representations
        center: str, either "zeros" (default) or "data_point". If "data_point", 
                must provide n_idx to specify which data point to use as center.
        n_idx: int, index of data point to use as center when center="data_point"
        save_path: str, optional path to save the plot. If None, displays the plot.
        scaler: optional scaler for inverse transform (to visualize in original scale)
    """
    # Get device from model
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)
    
    Q, M, P = model.A, model.M, model.P
    N = y_train.shape[0]  # Number of training points from actual data
    
    # Ensure y_train is on the correct device
    y_train = y_train.to(device)
    
    # Encode training data to get latent representations
    epsilon = model.get_epsilon()
    U_samples = model.get_u_samples(epsilon)
    y_encoder_input = torch.where(torch.isnan(y_train), 
                                   torch.full_like(y_train, -9.0, dtype=torch.float64), 
                                   y_train)
    mu_x_encoded, sigma_x_encoded = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
    # Take first P sample instead of averaging
    mu_x_mean = mu_x_encoded[0]  # [N, Q] - first sample from P samples

    # 1) pick most-informative latent dim (smallest lengthscale)
    betas = model.gp.beta[0].detach().cpu().numpy()  # (Q,)
    ells = 1.0 / np.maximum(betas, 1e-12)
    j = int(np.argmax(ells))

    # 2) center
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

    # Get GP expectation - returns [G, M, C]
    # Ensure Xgrid is on the same device as GP module
    Xgrid = Xgrid.to(device)
    
    # Patch GP module methods to ensure all tensors are created on correct device
    # Get device and dtype from model parameters
    sigma2_s_device = model.gp.sigma2_s.device
    sigma2_s_dtype = model.gp.sigma2_s.dtype
    Vx_device = model.gp.Vx.device
    Vx_dtype = model.gp.Vx.dtype
    
    # Patch get_K_VV_inv_matrix which creates torch.zeros on CPU
    original_get_K_VV_inv_matrix = model.gp.get_K_VV_inv_matrix
    def patched_get_K_VV_inv_matrix():
        K_VV = model.gp.get_K_VV_x_inv()  # C x L x L
        # Create output matrix on correct device
        K_VV_inv_matrix = torch.zeros(model.gp.C, model.gp.C, model.gp.Lx, model.gp.Lx,
                                      dtype=Vx_dtype, device=Vx_device)
        K_VV_inv_matrix[range(model.gp.C), range(model.gp.C), :, :] = K_VV[range(model.gp.C), :, :]
        return K_VV_inv_matrix.transpose(1, 2).reshape(model.gp.C * model.gp.Lx, model.gp.C * model.gp.Lx)
    model.gp.get_K_VV_inv_matrix = patched_get_K_VV_inv_matrix
    
    # Patch get_K_obsV_matrix which creates torch.zeros on CPU
    original_get_K_obsV_matrix = model.gp.get_K_obsV_matrix
    def patched_get_K_obsV_matrix(inputs):
        inputs = inputs.to(Vx_device)
        # Get K_obsV from get_K_xVx (which should work correctly)
        K_obsV = model.gp.get_K_xVx(inputs)  # N x C x L
        # Create output matrix on correct device - match the shape from original
        # Original: torch.zeros(*inputs.shape[:-1], self.C, self.C, self.Lx)
        input_shape = inputs.shape[:-1]  # All dims except last
        K_obsV_matrix = torch.zeros(*input_shape, model.gp.C, model.gp.C, model.gp.Lx,
                                    dtype=Vx_dtype, device=Vx_device)
        # Fill diagonal blocks
        K_obsV_matrix[..., range(model.gp.C), range(model.gp.C), :] = K_obsV[..., range(model.gp.C), :]
        # Reshape to match original output: (*inputs.shape[:-1], self.C, self.Lx*self.C)
        return K_obsV_matrix.reshape(*input_shape, model.gp.C, model.gp.Lx * model.gp.C)
    model.gp.get_K_obsV_matrix = patched_get_K_obsV_matrix
    
    # Patch get_gp_covariance which creates torch.ones on CPU
    original_get_gp_covariance = model.gp.get_gp_covariance
    def patched_get_gp_covariance(inputs):
        # Ensure inputs are on correct device
        inputs = inputs.to(device)
        # Patch the internal method that creates torch.ones - ensure it's on correct device
        K_obsobs_diag_elements = torch.diag_embed(
            torch.ones(inputs.shape[0], model.gp.M, model.gp.C, 
                      dtype=sigma2_s_dtype, device=sigma2_s_device) * 
            model.gp.sigma2_s[None, None, :]
        )  # N x M x C x C
        K_obsV = model.gp.get_K_obsV_matrix(inputs)  # N x CL x CL
        A = model.gp._get_conditional_cov_correction()  # M x CL x CL
        Sigma_s = K_obsobs_diag_elements - (K_obsV[:, None, :, :] @ A[None, :, :, :] @ K_obsV.transpose(-2, -1)[:, None, :, :])
        # Add eye matrix on correct device
        eye_matrix = torch.eye(model.gp.C, dtype=sigma2_s_dtype, device=sigma2_s_device) * 1e-5
        return Sigma_s + eye_matrix
    model.gp.get_gp_covariance = patched_get_gp_covariance
    
    try:
        f_full = model.gp.get_gp_expectation(Xgrid)  # [G, M, C]
        # ELBO-consistent variance using GP covariance method
        gp_cov = model.gp.get_gp_covariance(Xgrid)        # [G, M, C, C]
    finally:
        # Restore original methods
        model.gp.get_gp_covariance = original_get_gp_covariance
        model.gp.get_K_obsV_matrix = original_get_K_obsV_matrix
        model.gp.get_K_VV_inv_matrix = original_get_K_VV_inv_matrix
    
    # Handle C dimension (channels/components)
    C = f_full.shape[-1]
    if C == 1:
        # C=1 case (standard GP-LVM) - squeeze the channel dimension
        f = f_full[..., 0]                                        # [G, M]
        gp_var = gp_cov[..., 0, 0]                                # [G, M]
        pred_var = gp_var + model.sigma2                          # [G, M]
    else:
        # Multi-output case - average over channels for visualization
        f = f_full.mean(dim=-1)                                   # [G, M]
        # For variance, use diagonal and average over channels
        pred_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1).mean(dim=-1) + model.sigma2  # [G, M]
    
    pred_sd = torch.sqrt(torch.clamp(pred_var, min=1e-12))        # [G, M]

    # Inverse transform predictions if scaler provided (for visualization in original scale)
    if scaler is not None:
        f = scaler.inverse_transform(f)  # [G, M]
        # For standard deviation, scale by scaler's std if available
        if hasattr(scaler, 'std'):
            pred_sd = pred_sd * scaler.std  # [G, M]
        # For variance-based scalers, we need to handle differently
        # For now, just scale by std if available, otherwise keep as is

    # Compute Y-axis limits from actual data for consistent scaling across models
    # Get data in original scale (inverse transform if scaler provided)
    y_data_for_limits = y_train.clone()
    if scaler is not None:
        y_data_for_limits = scaler.inverse_transform(y_data_for_limits)
    
    # Filter out NaN and Inf values before computing limits
    y_data_valid = y_data_for_limits[torch.isfinite(y_data_for_limits)]
    
    if len(y_data_valid) == 0:
        # Fallback: use default range if all data is invalid
        y_min_global = -1.0
        y_max_global = 1.0
    else:
        y_min_global = y_data_valid.min().item()
        y_max_global = y_data_valid.max().item()
    
    # Ensure we have valid finite values
    if not np.isfinite(y_min_global) or not np.isfinite(y_max_global):
        y_min_global = -1.0
        y_max_global = 1.0
    
    y_range = y_max_global - y_min_global
    if y_range <= 0 or not np.isfinite(y_range):
        y_range = 2.0  # Default range
    
    y_padding = 1.0 * y_range  # 100% padding (fixed proportion for consistency across scripts, extra space for histograms)
    y_lim_min = y_min_global - y_padding
    y_lim_max = y_max_global + y_padding
    
    # Final validation: ensure limits are finite
    if not np.isfinite(y_lim_min):
        y_lim_min = -2.0
    if not np.isfinite(y_lim_max):
        y_lim_max = 2.0

    # 9) plot - show first 4 dimensions (or M if M < 4)
    xnp = xs.detach().cpu().numpy()
    
    # Set X-axis limits for consistency (already using [-3, 3] range)
    x_lim_min, x_lim_max = -3.0, 3.0
    
    # Plot only first 4 dimensions (or all if M < 4) - single row
    M_plot = min(M, 4)
    n_cols = 4
    n_rows = 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5), constrained_layout=True)
    axes = axes.ravel()
    
    # Sample from q(X) distribution along dimension j for histogram
    # mu_x_encoded is [P, N, Q], sigma_x_encoded is [P, N, Q] (variance)
    # Use first P sample instead of averaging
    sigma_x_first = sigma_x_encoded[0]  # [N, Q] - first sample variance from P samples
    # Sample from N(mu_x_mean, sigma_x_first) for each data point
    x_samples_j = torch.normal(mu_x_mean[:, j], torch.sqrt(torch.clamp(sigma_x_first[:, j], min=1e-8)))  # [N]
    x_j_values = x_samples_j.detach().cpu().numpy()
    
    for m in range(M_plot):
        mu = f[:, m].cpu().numpy()
        sd = pred_sd[:, m].cpu().numpy()
        ax = axes[m]
        ax.plot(xnp, mu, lw=2)
        ax.fill_between(xnp, mu - 2*sd, mu + 2*sd, alpha=0.25)
        ax.set_title(f"y[{m}] vs x[{j}]", fontsize=20)
        ax.set_xlabel(f"x[{j}]", fontsize=20)
        ax.set_ylabel(f"y[{m}]", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
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
        # Change extension to .pdf if it's not already
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Saved latent slice plot to: {save_path}")
    else:
        plt.show()


@no_grad()
def plot_latent_slice_outer_replot_single(model, y_train, center="zeros", n_idx=None, save_path=None, scaler=None):
    """
    Plot 1D latent slice with outer-expectation uncertainty bands - single plot version (for replotting).
    
    This function creates a visualization of the GP-LVM's predictive behavior along
    the most informative latent dimension (smallest lengthscale), showing only y[0] vs x[j]
    (first output dimension vs most informative latent dimension).
    
    Modified version for replotting that:
    - Ensures all tensors are on the correct device
    - Saves plots as PDF instead of PNG
    - Only plots the first output dimension (y[0])
    
    Args:
        model: GPLVM_SSVI_AM_INF model instance
        y_train: Training data [N, M] - used to encode latent representations
        center: str, either "zeros" (default) or "data_point". If "data_point", 
                must provide n_idx to specify which data point to use as center.
        n_idx: int, index of data point to use as center when center="data_point"
        save_path: str, optional path to save the plot. If None, displays the plot.
        scaler: optional scaler for inverse transform (to visualize in original scale)
    """
    # Get device from model
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)
    
    Q, M, P = model.A, model.M, model.P
    N = y_train.shape[0]  # Number of training points from actual data
    
    # Ensure y_train is on the correct device
    y_train = y_train.to(device)
    
    # Encode training data to get latent representations
    epsilon = model.get_epsilon()
    U_samples = model.get_u_samples(epsilon)
    y_encoder_input = torch.where(torch.isnan(y_train), 
                                   torch.full_like(y_train, -9.0, dtype=torch.float64), 
                                   y_train)
    mu_x_encoded, sigma_x_encoded = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
    # Take first P sample instead of averaging
    mu_x_mean = mu_x_encoded[0]  # [N, Q] - first sample from P samples

    # 1) pick most-informative latent dim (smallest lengthscale)
    betas = model.gp.beta[0].detach().cpu().numpy()  # (Q,)
    ells = 1.0 / np.maximum(betas, 1e-12)
    j = int(np.argmax(ells))

    # 2) center
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

    # Get GP expectation - returns [G, M, C]
    # Ensure Xgrid is on the same device as GP module
    Xgrid = Xgrid.to(device)
    
    # Patch GP module methods to ensure all tensors are created on correct device
    # Get device and dtype from model parameters
    sigma2_s_device = model.gp.sigma2_s.device
    sigma2_s_dtype = model.gp.sigma2_s.dtype
    Vx_device = model.gp.Vx.device
    Vx_dtype = model.gp.Vx.dtype
    
    # Patch get_K_VV_inv_matrix which creates torch.zeros on CPU
    original_get_K_VV_inv_matrix = model.gp.get_K_VV_inv_matrix
    def patched_get_K_VV_inv_matrix():
        K_VV = model.gp.get_K_VV_x_inv()  # C x L x L
        # Create output matrix on correct device
        K_VV_inv_matrix = torch.zeros(model.gp.C, model.gp.C, model.gp.Lx, model.gp.Lx,
                                      dtype=Vx_dtype, device=Vx_device)
        K_VV_inv_matrix[range(model.gp.C), range(model.gp.C), :, :] = K_VV[range(model.gp.C), :, :]
        return K_VV_inv_matrix.transpose(1, 2).reshape(model.gp.C * model.gp.Lx, model.gp.C * model.gp.Lx)
    model.gp.get_K_VV_inv_matrix = patched_get_K_VV_inv_matrix
    
    # Patch get_K_obsV_matrix which creates torch.zeros on CPU
    original_get_K_obsV_matrix = model.gp.get_K_obsV_matrix
    def patched_get_K_obsV_matrix(inputs):
        inputs = inputs.to(Vx_device)
        # Get K_obsV from get_K_xVx (which should work correctly)
        K_obsV = model.gp.get_K_xVx(inputs)  # N x C x L
        # Create output matrix on correct device - match the shape from original
        # Original: torch.zeros(*inputs.shape[:-1], self.C, self.C, self.Lx)
        input_shape = inputs.shape[:-1]  # All dims except last
        K_obsV_matrix = torch.zeros(*input_shape, model.gp.C, model.gp.C, model.gp.Lx,
                                    dtype=Vx_dtype, device=Vx_device)
        # Fill diagonal blocks
        K_obsV_matrix[..., range(model.gp.C), range(model.gp.C), :] = K_obsV[..., range(model.gp.C), :]
        # Reshape to match original output: (*inputs.shape[:-1], self.C, self.Lx*self.C)
        return K_obsV_matrix.reshape(*input_shape, model.gp.C, model.gp.Lx * model.gp.C)
    model.gp.get_K_obsV_matrix = patched_get_K_obsV_matrix
    
    # Patch get_gp_covariance which creates torch.ones on CPU
    original_get_gp_covariance = model.gp.get_gp_covariance
    def patched_get_gp_covariance(inputs):
        # Ensure inputs are on correct device
        inputs = inputs.to(device)
        # Patch the internal method that creates torch.ones - ensure it's on correct device
        K_obsobs_diag_elements = torch.diag_embed(
            torch.ones(inputs.shape[0], model.gp.M, model.gp.C, 
                      dtype=sigma2_s_dtype, device=sigma2_s_device) * 
            model.gp.sigma2_s[None, None, :]
        )  # N x M x C x C
        K_obsV = model.gp.get_K_obsV_matrix(inputs)  # N x CL x CL
        A = model.gp._get_conditional_cov_correction()  # M x CL x CL
        Sigma_s = K_obsobs_diag_elements - (K_obsV[:, None, :, :] @ A[None, :, :, :] @ K_obsV.transpose(-2, -1)[:, None, :, :])
        # Add eye matrix on correct device
        eye_matrix = torch.eye(model.gp.C, dtype=sigma2_s_dtype, device=sigma2_s_device) * 1e-5
        return Sigma_s + eye_matrix
    model.gp.get_gp_covariance = patched_get_gp_covariance
    
    try:
        f_full = model.gp.get_gp_expectation(Xgrid)  # [G, M, C]
        # ELBO-consistent variance using GP covariance method
        gp_cov = model.gp.get_gp_covariance(Xgrid)        # [G, M, C, C]
    finally:
        # Restore original methods
        model.gp.get_gp_covariance = original_get_gp_covariance
        model.gp.get_K_obsV_matrix = original_get_K_obsV_matrix
        model.gp.get_K_VV_inv_matrix = original_get_K_VV_inv_matrix
    
    # Handle C dimension (channels/components)
    C = f_full.shape[-1]
    if C == 1:
        # C=1 case (standard GP-LVM) - squeeze the channel dimension
        f = f_full[..., 0]                                        # [G, M]
        gp_var = gp_cov[..., 0, 0]                                # [G, M]
        pred_var = gp_var + model.sigma2                          # [G, M]
    else:
        # Multi-output case - average over channels for visualization
        f = f_full.mean(dim=-1)                                   # [G, M]
        # For variance, use diagonal and average over channels
        pred_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1).mean(dim=-1) + model.sigma2  # [G, M]
    
    pred_sd = torch.sqrt(torch.clamp(pred_var, min=1e-12))        # [G, M]

    # Inverse transform predictions if scaler provided (for visualization in original scale)
    if scaler is not None:
        f = scaler.inverse_transform(f)  # [G, M]
        # For standard deviation, scale by scaler's std if available
        if hasattr(scaler, 'std'):
            pred_sd = pred_sd * scaler.std  # [G, M]
        # For variance-based scalers, we need to handle differently
        # For now, just scale by std if available, otherwise keep as is

    # Compute Y-axis limits from actual data for consistent scaling across models
    # Get data in original scale (inverse transform if scaler provided)
    y_data_for_limits = y_train.clone()
    if scaler is not None:
        y_data_for_limits = scaler.inverse_transform(y_data_for_limits)
    
    # Filter out NaN and Inf values before computing limits
    y_data_valid = y_data_for_limits[torch.isfinite(y_data_for_limits)]
    
    if len(y_data_valid) == 0:
        # Fallback: use default range if all data is invalid
        y_min_global = -1.0
        y_max_global = 1.0
    else:
        y_min_global = y_data_valid.min().item()
        y_max_global = y_data_valid.max().item()
    
    # Ensure we have valid finite values
    if not np.isfinite(y_min_global) or not np.isfinite(y_max_global):
        y_min_global = -1.0
        y_max_global = 1.0
    
    y_range = y_max_global - y_min_global
    if y_range <= 0 or not np.isfinite(y_range):
        y_range = 2.0  # Default range
    
    y_padding = 1.0 * y_range  # 100% padding (fixed proportion for consistency across scripts, extra space for histograms)
    y_lim_min = y_min_global - y_padding
    y_lim_max = y_max_global + y_padding
    
    # Final validation: ensure limits are finite
    if not np.isfinite(y_lim_min):
        y_lim_min = -2.0
    if not np.isfinite(y_lim_max):
        y_lim_max = 2.0

    # Plot - only first dimension (y[0] vs x[j])
    xnp = xs.detach().cpu().numpy()
    
    # Set X-axis limits for consistency (already using [-3, 3] range)
    x_lim_min, x_lim_max = -3.0, 3.0
    
    # Single plot for y[0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    
    # Sample from q(X) distribution along dimension j for histogram
    # mu_x_encoded is [P, N, Q], sigma_x_encoded is [P, N, Q] (variance)
    # Use first P sample instead of averaging
    sigma_x_first = sigma_x_encoded[0]  # [N, Q] - first sample variance from P samples
    # Sample from N(mu_x_mean, sigma_x_first) for each data point
    x_samples_j = torch.normal(mu_x_mean[:, j], torch.sqrt(torch.clamp(sigma_x_first[:, j], min=1e-8)))  # [N]
    x_j_values = x_samples_j.detach().cpu().numpy()
    
    # Plot only m=0 (first output dimension)
    m = 0
    mu = f[:, m].cpu().numpy()
    sd = pred_sd[:, m].cpu().numpy()
    ax.plot(xnp, mu, lw=2)
    ax.fill_between(xnp, mu - 2*sd, mu + 2*sd, alpha=0.25)
    ax.set_title(f"y[{m}] vs x[{j}]", fontsize=20)
    ax.set_xlabel(f"x[{j}]", fontsize=20)
    ax.set_ylabel(f"y[{m}]", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
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
    
    if save_path:
        # Change extension to .pdf if it's not already
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Saved latent slice plot (single) to: {save_path}")
    else:
        plt.show()


@no_grad()
def visualize_ssvi_results_replot(model, y_train, labels_train, losses, config, logger, save_filename: str = "training_results.pdf", show_plot: bool = True, n_optimization_steps: int = None, is_test_data: bool = False, is_checkpoint: bool = False, train_indices: torch.Tensor = None, full_elbo_iterations: list = None, batch_losses: list = None):
    """
    Visualize SSVI training results and latent space (for replotting).
    
    This is a modified version of visualize_ssvi_results that saves all matplotlib plots as PDF.
    HTML plots (3D plots) are still saved as HTML.
    
    Parameters
    ----------
    losses : list
        Full ELBO losses (negative ELBO values)
    full_elbo_iterations : list, optional
        Iteration numbers corresponding to full_elbo_losses. If provided, used for x-axis of full ELBO plot.
    batch_losses : list, optional
        Batch ELBO losses (one per training iteration). If provided, creates an additional plot showing batch ELBO progression.
    """
    
    # Tag filenames to reflect sigma_x -> sqrt(sigma_x) sampling in latent plots
    if isinstance(save_filename, str) and "_sigma_sqrt" not in save_filename:
        if save_filename.endswith(".png"):
            save_filename = save_filename[:-4] + "_sigma_sqrt.png"
        elif save_filename.endswith(".pdf"):
            save_filename = save_filename[:-4] + "_sigma_sqrt.pdf"
        else:
            save_filename = save_filename + "_sigma_sqrt"

    if not (hasattr(config, 'plot_training') and config.plot_training or
            hasattr(config, 'plot_latent_space') and config.plot_latent_space or
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams):
        return

    
    # Color mapping for labels (assuming 3 classes)
    c = ["r", "g", "b"]
    colors = [c[int(i)-1] for i in labels_train]
    
    plot_flags = [
        hasattr(config, 'plot_training') and config.plot_training,
        hasattr(config, 'plot_latent_space') and config.plot_latent_space,
        hasattr(config, 'plot_hyperparams') and config.plot_hyperparams
    ]
    # Only plot latent space and lengthscales (remove ELBO plots)
    n_plots = (1 if plot_flags[1] else 0) + (1 if plot_flags[2] else 0)
    fig_width = 4 * n_plots if n_plots > 0 else 8
    fig = plt.figure(figsize=(fig_width, 4))
    
    plot_idx = 1
    
    # Plot 1: Latent space visualization
    if plot_flags[1]:
        plt.subplot(1, n_plots, plot_idx)
        
        # Find the most informative dimensions (largest lengthscales = smallest beta)
        lengthscales = 1 / model.gp.beta.flatten()
        top_2_dims = torch.argsort(lengthscales, descending=True)[:2]
        top_3_dims = torch.argsort(lengthscales, descending=True)[:3]  # Also compute top 3 for 3D plots
        dim1, dim2 = top_2_dims[0].item(), top_2_dims[1].item()
        dim3 = top_3_dims[2].item()  # Third dimension for 3D plots
        lengthscales_cpu = lengthscales.cpu()  # Move to CPU for string formatting
        # Get latent variables using new explicit training method
        epsilon = model.get_epsilon()
        U = model.get_u_samples(epsilon)  # [P, M, C, Lx]
        
        # Handle missing data if enabled
        if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
            # Preprocess encoder input to handle missing data (replace NaN with -9.0)
            y_encoder_input = torch.where(torch.isnan(y_train), torch.full_like(y_train, -9.0), y_train)
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)  # [P, N, A]
        else:
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, epsilon)  # [P, N, A]
        
        # Keep as torch tensors for sampling, convert to numpy after sampling
        mu_x_cpu = mu_x.cpu()  # Move to CPU but keep as tensor
        sigma_x_cpu = sigma_x.cpu()  # Move to CPU but keep as tensor
        # NOTE: sigma_x returned by model is a variance; convert to std-dev for sampling
        sigma_x_std_cpu = torch.sqrt(torch.clamp(sigma_x_cpu, min=1e-12))
        
        # Plot samples from all P samples with transparency
        for p in range(model.P):
            # Use top 2 most informative latent dimensions
            sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
            plt.scatter(sample_1.detach().cpu().numpy(),
                        sample_2.detach().cpu().numpy(), alpha=0.6, color=colors)
        
        plt.xlabel(f'Latent Dimension {dim1}', fontsize=20)
        plt.ylabel(f'Latent Dimension {dim2}', fontsize=20)
        plt.title(f'Top 2 Informative Dims', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plot_idx += 1
    
    # Generate fractions plot using the same mu_x and top dimensions
    try:
        # Load fractions data - check data directory first, then current directory
        frctn_path = Path("data") / "oil_fractions.txt"
        
        if not frctn_path.exists():
            # Fallback to current directory (SSVI approach)
            frctn_path = Path(".") / "DataTrnFrctns.txt"

        if not frctn_path.exists():
            raise FileNotFoundError(f"Fractions file not found. Checked: data/oil_fractions.txt and DataTrnFrctns.txt")

        w = np.loadtxt(frctn_path)  # (N, 2)
        w3 = 1.0 - w.sum(axis=1, keepdims=True)  # (N, 1)
        fractions = np.concatenate((w, w3), axis=1)  # (N, 3)
        
        # Debug: print fractions shapes
        print(f"Debug - fractions shape before subsetting: {fractions.shape}")
        print(f"Debug - y_train shape: {y_train.shape}")
        
        # Subset fractions data to match training data using train_indices
        train_indices_cpu = train_indices.cpu().numpy()
        fractions = fractions[train_indices_cpu]
        print(f"Debug - fractions shape after subsetting with train_indices: {fractions.shape}")
        
        # Create fractions plot using the same mu_x and top dimensions from above
        # Make figure wider to account for colorbar, but keep plot area square
        fig_frac, ax_frac = plt.subplots(1, 1, figsize=(7.5, 6))
        
        # Use the same top 2 dimensions that were calculated above
        # Plot samples from all P samples with transparency (consistent with latent dims plot)
        for p in range(model.P):
            # Sample from the latent distributions (consistent with main latent space plot)
            sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
            scatter = ax_frac.scatter(sample_1.detach().cpu().numpy(),
                                    sample_2.detach().cpu().numpy(), 
                                    c=fractions, s=20, alpha=0.6)
        ax_frac.set_title("Latent Space Colored by Fractions")
        ax_frac.set_xlabel(f'mu_{dim1}')
        ax_frac.set_ylabel(f'mu_{dim2}')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_frac)
        cbar.set_label('Fractions')
        
        # Ensure equal aspect ratio for square plot
        ax_frac.set_aspect("equal", adjustable='box')
        fig_frac.tight_layout()
        
        # Save fractions plot as PDF
        fractions_filename = save_filename.replace('training_results', 'latent_space_by_fractions')
        # Ensure PDF extension
        if not fractions_filename.endswith('.pdf'):
            fractions_filename = fractions_filename.rsplit('.', 1)[0] + '.pdf'
        if is_checkpoint:
            plot_frac_path = os.path.join(logger.plots_checkpoints_dir, fractions_filename)
        else:
            plot_frac_path = os.path.join(logger.plots_dir, fractions_filename)
        fig_frac.savefig(plot_frac_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig_frac)
        print(f"Fractions plot saved to: {plot_frac_path}")
        
        # Create 3D fractions plot using the same fractions data
        print("Creating 3D fractions plot...")
        
        # Create plotly 3D scatter plot colored by fractions
        fig_3d_frac = go.Figure()
        
        # Plot samples from all P samples with fractions as colors (consistent with 2D fractions plot)
        for p in range(model.P):
            # Sample from the latent distributions (consistent with 2D fractions plot)
            sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
            sample_3 = torch.distributions.Normal(mu_x_cpu[p, :, dim3], sigma_x_std_cpu[p, :, dim3]).sample()
            
            fig_3d_frac.add_trace(go.Scatter3d(
                x=sample_1.detach().cpu().numpy(),
                y=sample_2.detach().cpu().numpy(),
                z=sample_3.detach().cpu().numpy(),
                mode='markers',
                marker=dict(
                    size=4,
                    color=fractions[:, 0],  # Use first fraction component (consistent with 2D plot)
                    colorscale='Viridis',
                    opacity=0.6,
                    colorbar=dict(title="Fraction 1")
                ),
                name='Latent Points',
                showlegend=False  # Hide individual P sample traces in legend
            ))
        
        # Update layout
        fig_3d_frac.update_layout(
            title=f'3D Latent Space Colored by Fractions - Top 3 Dimensions<br>Dim {dim1}, {dim2}, {dim3}',
            scene=dict(
                xaxis_title=f'Dimension {dim1} (lengthscale={lengthscales_cpu[dim1].item():.2f})',
                yaxis_title=f'Dimension {dim2} (lengthscale={lengthscales_cpu[dim2].item():.2f})',
                zaxis_title=f'Dimension {dim3} (lengthscale={lengthscales_cpu[dim3].item():.2f})'
            ),
            width=800,
            height=600
        )
        
        # Save 3D fractions plot as HTML
        plot_3d_frac_filename = save_filename.replace('training_results', '3d_latent_space_by_fractions')
        plot_3d_frac_filename = plot_3d_frac_filename.replace('.pdf', '.html').replace('.png', '.html')
        
        # Save to appropriate directory (checkpoints or main plots)
        if is_checkpoint:
            plot_3d_frac_path = os.path.join(logger.plots_checkpoints_dir, plot_3d_frac_filename)
        else:
            plot_3d_frac_path = os.path.join(logger.plots_dir, plot_3d_frac_filename)
        
        fig_3d_frac.write_html(plot_3d_frac_path)
        print(f"3D fractions plot saved to: {plot_3d_frac_path}")
        
        if show_plot:
            fig_3d_frac.show()
        
    except FileNotFoundError as e:
        print(f"Warning: Could not create fractions plot: {e}")
    
    # Plot 2: Hyperparameters (lengthscales)
    if plot_flags[2]:
        plt.subplot(1, n_plots, plot_idx)
        lengthscales = (1 / model.gp.beta.flatten()).cpu()
        plt.bar(range(len(lengthscales)), lengthscales, 
               color="blue", alpha=0.7)
        plt.xlabel('Latent Dimension', fontsize=20)
        plt.ylabel('')
        plt.title('Inverse SE-ARD Lengthscales', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    
    # Save plot as PNG with 800 DPI
    # Ensure PNG extension
    if not save_filename.endswith('.png'):
        save_filename = save_filename.rsplit('.', 1)[0] + '.png'
    if is_checkpoint:
        plot_path = os.path.join(logger.plots_checkpoints_dir, save_filename)
    else:
        plot_path = os.path.join(logger.plots_dir, save_filename)
    fig.savefig(plot_path, format='png', bbox_inches='tight', dpi=800)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close figure to free memory
    
    # Additional plots: 1D histogram and 3D scatter plot
    print("Creating additional plots: 1D histogram and 3D scatter plot...")
    
    # Ensure labels_train is on CPU for numpy operations
    labels_train_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train
    
    # Check if we can reuse existing computations from latent space plot
    if plot_flags[1] and 'mu_x_cpu' in locals() and 'sigma_x_cpu' in locals():
        # Reuse existing computations from latent space plot
        # We already have mu_x_cpu, sigma_x_cpu, and lengthscales_cpu from the latent space plot
        # Just need to compute top_3_dims for the 3D plot
        lengthscales = lengthscales_cpu  # Use existing lengthscales
    else:
        # Need to compute the data (latent space plot wasn't computed)
        # Find the most important dimensions (largest lengthscales = smallest beta)
        lengthscales = (1 / model.gp.beta.flatten()).cpu()  # Move to CPU
        
        # Get latent variables
        epsilon = model.get_epsilon()
        U = model.get_u_samples(epsilon)  # [P, M, C, Lx]
        
        # Handle missing data if enabled
        if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
            y_encoder_input = torch.where(torch.isnan(y_train), torch.full_like(y_train, -9.0), y_train)
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)  # [P, N, A]
        else:
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, epsilon)  # [P, N, A]
        
        # Move to CPU for plotting
        mu_x_cpu = mu_x.cpu()
        sigma_x_cpu = sigma_x.cpu()
    
    # Use dimensions already computed in latent space plot section
    most_important_dim = dim1  # Use the first (most important) dimension for 1D histogram
    
    # 1. Create 1D histogram for the most important dimension
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 6))
    
    # Collect samples from all P samples for the most important dimension
    all_samples = []
    all_labels = []
    for p in range(model.P):
        samples = torch.distributions.Normal(mu_x_cpu[p, :, most_important_dim], 
                                           sigma_x_std_cpu[p, :, most_important_dim]).sample()
        # Ensure samples are on CPU before converting to numpy
        all_samples.append(samples.detach().cpu().numpy())
        # Repeat labels for each P sample
        all_labels.append(labels_train_cpu)
    
    all_samples = np.concatenate(all_samples)
    all_labels = np.concatenate(all_labels)
    
    # Create histogram with different colors for different classes
    for class_idx in range(3):  # Assuming 3 classes
        class_mask = all_labels == (class_idx + 1)
        if np.any(class_mask):
            class_samples = all_samples[class_mask]
            ax_hist.hist(class_samples, bins=30, alpha=0.7, 
                       label=f'Class {class_idx + 1}', 
                       color=['red', 'green', 'blue'][class_idx])
    
    ax_hist.set_xlabel(f'Latent Dimension {most_important_dim} (Most Important)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'1D Histogram - Most Important Dimension {most_important_dim}')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Save histogram plot as PDF
    hist_filename = save_filename.replace('training_results', '1d_histogram_most_important_dim')
    # Ensure PDF extension
    if not hist_filename.endswith('.pdf'):
        hist_filename = hist_filename.rsplit('.', 1)[0] + '.pdf'
    if is_checkpoint:
        hist_path = os.path.join(logger.plots_checkpoints_dir, hist_filename)
    else:
        hist_path = os.path.join(logger.plots_dir, hist_filename)
    fig_hist.savefig(hist_path, format='pdf', bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close(fig_hist)
    
    # 2. Create 3D scatter plot for top 3 most important dimensions using plotly
    dim1, dim2, dim3 = top_3_dims[0].item(), top_3_dims[1].item(), top_3_dims[2].item()
    
    # Collect samples for all 3 dimensions
    samples_3d = []
    for p in range(model.P):
        sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
        sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
        sample_3 = torch.distributions.Normal(mu_x_cpu[p, :, dim3], sigma_x_std_cpu[p, :, dim3]).sample()
        
        # Ensure all samples are on CPU before converting to numpy
        samples_3d.append({
            'dim1': sample_1.detach().cpu().numpy(),
            'dim2': sample_2.detach().cpu().numpy(),
            'dim3': sample_3.detach().cpu().numpy(),
            'labels': labels_train_cpu
        })
    
    # Create plotly 3D scatter plot
    fig_3d = go.Figure()
    
    # Color mapping for labels
    colors_3d = ['red', 'green', 'blue']
    color_names = ['Class 1', 'Class 2', 'Class 3']
    
    # Add traces for each class
    for class_idx in range(3):
        class_mask = labels_train_cpu == (class_idx + 1)
        if np.any(class_mask):
            # Combine samples from all P samples for this class
            x_vals = np.concatenate([samples_3d[p]['dim1'][class_mask] for p in range(model.P)])
            y_vals = np.concatenate([samples_3d[p]['dim2'][class_mask] for p in range(model.P)])
            z_vals = np.concatenate([samples_3d[p]['dim3'][class_mask] for p in range(model.P)])
            
            fig_3d.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors_3d[class_idx],
                    opacity=0.6
                ),
                name=color_names[class_idx]
            ))
    
    # Update layout
    fig_3d.update_layout(
        title=f'3D Scatter Plot - Top 3 Most Important Dimensions<br>Dim {dim1}, {dim2}, {dim3}',
        scene=dict(
            xaxis_title=f'Dimension {dim1} (lengthscale={lengthscales[dim1].item():.2f})',
            yaxis_title=f'Dimension {dim2} (lengthscale={lengthscales[dim2].item():.2f})',
            zaxis_title=f'Dimension {dim3} (lengthscale={lengthscales[dim3].item():.2f})'
        ),
        width=800,
        height=600
    )
    
    # Save 3D plot as HTML
    plot_3d_filename = save_filename.replace('training_results', '3d_scatter_top3_dims')
    plot_3d_filename = plot_3d_filename.replace('.pdf', '.html').replace('.png', '.html')
    
    # Save to appropriate directory (checkpoints or main plots)
    if is_checkpoint:
        plot_3d_path = os.path.join(logger.plots_checkpoints_dir, plot_3d_filename)
    else:
        plot_3d_path = os.path.join(logger.plots_dir, plot_3d_filename)
    
    fig_3d.write_html(plot_3d_path)
    print(f"3D scatter plot saved to: {plot_3d_path}")
    
    if show_plot:
        fig_3d.show()
    
    print("Additional plots created successfully!")


def get_consistent_qpcr_colors(labels_train):
    """
    Get consistent color mapping for qPCR labels across SSVI and original GP LVM.
    Ensures the same class always gets the same color regardless of which plotting function is used.
    """
    # Use tab10 colormap with reversed order (same as original qPCR_expt.py)
    cmap_colors = plt.get_cmap("tab10").colors[::-1]
    
    # Create consistent color mapping based on unique labels
    unique_labels = np.unique(labels_train)
    color_map = {}
    
    # Assign colors in order of unique labels to ensure consistency
    for i, label in enumerate(sorted(unique_labels)):
        color_map[label] = cmap_colors[i % len(cmap_colors)]
    
    # Return colors in the same order as labels_train
    colors = [color_map[label] for label in labels_train]
    return colors, color_map


def matplotlib_to_plotly_color(matplotlib_color):
    """Convert matplotlib RGB tuple to Plotly format"""
    if isinstance(matplotlib_color, tuple) and len(matplotlib_color) == 3:
        r, g, b = matplotlib_color
        return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
    return matplotlib_color


@no_grad()
def visualize_qpcr_ssvi_results_replot(model, y_train, labels_train, losses, config, logger, save_filename: str = "training_results.pdf", show_plot: bool = True, n_optimization_steps: int = None, is_test_data: bool = False, is_checkpoint: bool = False, train_indices: torch.Tensor = None, full_elbo_iterations: list = None, batch_losses: list = None):
    """
    Visualize qPCR SSVI training results and latent space (for replotting).
    
    This is a modified version of visualize_ssvi_results from qpcr_plot_helper that saves all matplotlib plots as PDF.
    HTML plots (3D plots) are still saved as HTML.
    
    Parameters
    ----------
    losses : list
        Full ELBO losses (negative ELBO values)
    full_elbo_iterations : list, optional
        Iteration numbers corresponding to full_elbo_losses. If provided, used for x-axis of full ELBO plot.
    batch_losses : list, optional
        Batch ELBO losses (one per training iteration). If provided, creates an additional plot showing batch ELBO progression.
    """
    
    # Tag filenames to reflect sigma_x -> sqrt(sigma_x) sampling in latent plots
    if isinstance(save_filename, str) and "_sigma_sqrt" not in save_filename:
        if save_filename.endswith(".png"):
            save_filename = save_filename[:-4] + "_sigma_sqrt.png"
        elif save_filename.endswith(".pdf"):
            save_filename = save_filename[:-4] + "_sigma_sqrt.pdf"
        else:
            save_filename = save_filename + "_sigma_sqrt"

    if not (hasattr(config, 'plot_training') and config.plot_training or
            hasattr(config, 'plot_latent_space') and config.plot_latent_space or
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams):
        return

    # Color mapping for qPCR labels (exactly like original qPCR_expt.py)
    # qPCR labels are integers 0, 1, 2, ... from load_real_data('gene')
    # Use consistent color mapping across SSVI and original GP LVM
    colors, color_map = get_consistent_qpcr_colors(labels_train)

    plot_flags = [
        hasattr(config, 'plot_training') and config.plot_training,
        hasattr(config, 'plot_latent_space') and config.plot_latent_space,
        hasattr(config, 'plot_hyperparams') and config.plot_hyperparams
    ]
    # Only plot latent space and lengthscales (remove ELBO plots)
    n_plots = (1 if plot_flags[1] else 0) + (1 if plot_flags[2] else 0)
    fig_width = 4 * n_plots if n_plots > 0 else 8
    fig = plt.figure(figsize=(fig_width, 4))

    plot_idx = 1

    # Plot 1: Latent space visualization
    if plot_flags[1]:
        plt.subplot(1, n_plots, plot_idx)

        # Find the most informative dimensions (largest lengthscales = smallest beta)
        lengthscales = 1 / model.gp.beta.flatten()
        top_2_dims = torch.argsort(lengthscales, descending=True)[:2]
        top_3_dims = torch.argsort(lengthscales, descending=True)[:3]  # Also compute top 3 for 3D plots
        dim1, dim2 = top_2_dims[0].item(), top_2_dims[1].item()
        dim3 = top_3_dims[2].item()  # Third dimension for 3D plots
        lengthscales_cpu = lengthscales.cpu()  # Move to CPU for string formatting
        # Get latent variables using new explicit training method
        epsilon = model.get_epsilon()
        U = model.get_u_samples(epsilon)  # [P, M, C, Lx]

        # Handle missing data if enabled
        if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
            # Preprocess encoder input to handle missing data (replace NaN with -9.0)
            y_encoder_input = torch.where(torch.isnan(y_train), torch.full_like(y_train, -9.0, dtype=torch.float64), y_train)
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)  # [P, N, A]
        else:
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, epsilon)  # [P, N, A]

        # Keep as torch tensors for sampling, convert to numpy after sampling
        mu_x_cpu = mu_x.cpu()  # Move to CPU but keep as tensor
        sigma_x_cpu = sigma_x.cpu()  # Move to CPU but keep as tensor
        # NOTE: sigma_x returned by model is a variance; convert to std-dev for sampling
        sigma_x_std_cpu = torch.sqrt(torch.clamp(sigma_x_cpu, min=1e-12))

        # Plot samples from all P samples with transparency
        # Use 48 samples for PNG latent plot
        num_samples = min(48, model.P)  # Use 48 or model.P, whichever is smaller
        for p in range(num_samples):
            # Use top 2 most informative latent dimensions
            sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
            plt.scatter(sample_1.detach().cpu().numpy(),
                        sample_2.detach().cpu().numpy(), alpha=0.6, color=colors)

        plt.xlabel(f'Latent Dimension {dim1}', fontsize=20)
        plt.ylabel(f'Latent Dimension {dim2}', fontsize=20)
        plt.title(f'Top 2 Informative Dims', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plot_idx += 1

    # Plot 2: Hyperparameters (lengthscales)
    if plot_flags[2]:
        plt.subplot(1, n_plots, plot_idx)
        lengthscales = (1 / model.gp.beta.flatten()).cpu()
        plt.bar(range(len(lengthscales)), lengthscales,
                color="blue", alpha=0.7)
        plt.xlabel('Latent Dimension', fontsize=20)
        plt.ylabel('')
        plt.title('Inverse SE-ARD Lengthscales', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout()

    # Save plot as PNG with 800 DPI
    # Ensure PNG extension
    if not save_filename.endswith('.png'):
        save_filename = save_filename.rsplit('.', 1)[0] + '.png'
    if is_checkpoint:
        plot_path = os.path.join(logger.plots_checkpoints_dir, save_filename)
    else:
        plot_path = os.path.join(logger.plots_dir, save_filename)
    fig.savefig(plot_path, format='png', bbox_inches='tight', dpi=800)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close figure to free memory

    # Additional plots: 1D histogram and 3D scatter plot
    print("Creating additional plots: 1D histogram and 3D scatter plot...")

    # Ensure labels_train is on CPU for numpy operations
    labels_train_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train

    # Check if we can reuse existing computations from latent space plot
    if plot_flags[1] and 'mu_x_cpu' in locals() and 'sigma_x_cpu' in locals():
        # Reuse existing computations from latent space plot
        # We already have mu_x_cpu, sigma_x_cpu, and lengthscales_cpu from the latent space plot
        # Just need to compute top_3_dims for the 3D plot
        lengthscales = lengthscales_cpu  # Use existing lengthscales
    else:
        # Need to compute the data (latent space plot wasn't computed)
        # Find the most important dimensions (largest lengthscales = smallest beta)
        lengthscales = (1 / model.gp.beta.flatten()).cpu()  # Move to CPU

        # Get latent variables
        epsilon = model.get_epsilon()
        U = model.get_u_samples(epsilon)  # [P, M, C, Lx]

        # Handle missing data if enabled
        if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
            y_encoder_input = torch.where(torch.isnan(y_train), torch.full_like(y_train, -9.0, dtype=torch.float64), y_train)
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)  # [P, N, A]
        else:
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, epsilon)  # [P, N, A]

        # Move to CPU for plotting
        mu_x_cpu = mu_x.cpu()
        sigma_x_cpu = sigma_x.cpu()

    # Use dimensions already computed in latent space plot section
    most_important_dim = dim1  # Use the first (most important) dimension for 1D histogram

    # 1. Create 1D histogram for the most important dimension
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 6))

    # Collect samples from all P samples for the most important dimension
    # Use 48 samples for PNG latent plot
    num_samples = min(48, model.P)  # Use 48 or model.P, whichever is smaller
    all_samples = []
    all_labels = []
    for p in range(num_samples):
        samples = torch.distributions.Normal(mu_x_cpu[p, :, most_important_dim],
                                             sigma_x_std_cpu[p, :, most_important_dim]).sample()
        # Ensure samples are on CPU before converting to numpy
        all_samples.append(samples.detach().cpu().numpy())
        # Repeat labels for each P sample
        all_labels.append(labels_train_cpu)

    all_samples = np.concatenate(all_samples)
    all_labels = np.concatenate(all_labels)

    # Create histogram with different colors for different classes (using consistent color mapping)
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        class_mask = all_labels == label
        if np.any(class_mask):
            class_samples = all_samples[class_mask]
            ax_hist.hist(class_samples, bins=30, alpha=0.7,
                         label=f'Class {label}',
                         color=color_map[label])

    ax_hist.set_xlabel(f'Latent Dimension {most_important_dim} (Most Important)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'1D Histogram - Most Important Dimension {most_important_dim}')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # Save histogram plot as PDF
    hist_filename = save_filename.replace('training_results', '1d_histogram_most_important_dim')
    # Ensure PDF extension
    if not hist_filename.endswith('.pdf'):
        hist_filename = hist_filename.rsplit('.', 1)[0] + '.pdf'
    if is_checkpoint:
        hist_path = os.path.join(logger.plots_checkpoints_dir, hist_filename)
    else:
        hist_path = os.path.join(logger.plots_dir, hist_filename)
    fig_hist.savefig(hist_path, format='pdf', bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close(fig_hist)

    # 2. Create 3D scatter plot for top 3 most important dimensions using plotly
    dim1, dim2, dim3 = top_3_dims[0].item(), top_3_dims[1].item(), top_3_dims[2].item()

    # Collect samples for all 3 dimensions
    # Use 48 samples for PNG latent plot
    num_samples = min(48, model.P)  # Use 48 or model.P, whichever is smaller
    samples_3d = []
    for p in range(num_samples):
        sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
        sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
        sample_3 = torch.distributions.Normal(mu_x_cpu[p, :, dim3], sigma_x_std_cpu[p, :, dim3]).sample()

        # Ensure all samples are on CPU before converting to numpy
        samples_3d.append({
            'dim1': sample_1.detach().cpu().numpy(),
            'dim2': sample_2.detach().cpu().numpy(),
            'dim3': sample_3.detach().cpu().numpy(),
            'labels': labels_train_cpu
        })

    # Create plotly 3D scatter plot
    fig_3d = go.Figure()

    # Color mapping for labels (using consistent color mapping)
    unique_labels = np.unique(labels_train_cpu)
    
    # Add traces for each class
    for label in unique_labels:
        class_mask = labels_train_cpu == label
        if np.any(class_mask):
            # Combine samples from all P samples for this class
            x_vals = np.concatenate([samples_3d[p]['dim1'][class_mask] for p in range(num_samples)])
            y_vals = np.concatenate([samples_3d[p]['dim2'][class_mask] for p in range(num_samples)])
            z_vals = np.concatenate([samples_3d[p]['dim3'][class_mask] for p in range(num_samples)])
            
            fig_3d.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=4,
                    color=matplotlib_to_plotly_color(color_map[label]),
                    opacity=0.6
                ),
                name=f'Class {label}'
            ))

    # Update layout
    fig_3d.update_layout(
        title=f'3D Scatter Plot - Top 3 Most Important Dimensions<br>Dim {dim1}, {dim2}, {dim3}',
        scene=dict(
            xaxis_title=f'Dimension {dim1} (lengthscale={lengthscales[dim1].item():.2f})',
            yaxis_title=f'Dimension {dim2} (lengthscale={lengthscales[dim2].item():.2f})',
            zaxis_title=f'Dimension {dim3} (lengthscale={lengthscales[dim3].item():.2f})'
        ),
        width=800,
        height=600
    )

    # Save 3D plot as HTML
    plot_3d_filename = save_filename.replace('training_results', '3d_scatter_top3_dims')
    plot_3d_filename = plot_3d_filename.replace('.pdf', '.html').replace('.png', '.html')

    # Save to appropriate directory (checkpoints or main plots)
    if is_checkpoint:
        plot_3d_path = os.path.join(logger.plots_checkpoints_dir, plot_3d_filename)
    else:
        plot_3d_path = os.path.join(logger.plots_dir, plot_3d_filename)

    fig_3d.write_html(plot_3d_path)
    print(f"3D scatter plot saved to: {plot_3d_path}")

    if show_plot:
        fig_3d.show()

    print("Additional plots created successfully!")


# FashionMNIST class names for legend
FASHION_MNIST_CLASSES = {
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def get_label_name(label_val, dataset_name, is_ood=False):
    """
    Get legend label name - just the class name without dataset prefix or ID/OOD status.
    
    Parameters
    ----------
    label_val : int
        The class label (0-9)
    dataset_name : str
        'MNIST' or 'FashionMNIST'
    is_ood : bool
        Whether this is OOD data (ignored, kept for compatibility)
        
    Returns
    -------
    str
        Just the class name like "Trouser" or "1"
    """
    if dataset_name == "MNIST":
        return str(int(label_val))
    elif dataset_name == "FashionMNIST":
        return FASHION_MNIST_CLASSES.get(int(label_val), f'Class {int(label_val)}')
    else:
        return f'Class {int(label_val)}'


def get_consistent_mnist_colors(labels_train):
    """
    Get consistent color mapping for MNIST digit labels (0-9) across SSVI and original GP LVM.
    Ensures the same digit always gets the same color regardless of which plotting function is used.
    """
    # Use tab10 colormap for 10 digit classes (0-9)
    cmap_colors = plt.get_cmap("tab10").colors
    
    # Create consistent color mapping based on unique labels
    unique_labels = np.unique(labels_train)
    color_map = {}
    
    # Assign colors in order of unique labels to ensure consistency
    for i, label in enumerate(sorted(unique_labels)):
        color_map[label] = cmap_colors[i % len(cmap_colors)]
    
    # Return colors in the same order as labels_train
    colors = [color_map[label] for label in labels_train]
    return colors, color_map


@no_grad()
def visualize_mnist_ood_ssvi_results_replot(model, y_train, labels_train, losses, config, logger, save_filename: str = "training_results.pdf", show_plot: bool = True, n_optimization_steps: int = None, is_test_data: bool = False, is_checkpoint: bool = False, train_indices: torch.Tensor = None, dataset_name: str = "MNIST", is_ood: bool = False, full_elbo_iterations: list = None, batch_losses: list = None):
    """
    Visualize MNIST OOD SSVI training results and latent space (for replotting).
    
    This is a modified version of visualize_ssvi_results from mnist_ood_plot_helper that saves all matplotlib plots as PDF.
    HTML plots (3D plots) are still saved as HTML.
    
    Parameters
    ----------
    losses : list
        Full ELBO losses (negative ELBO values)
    full_elbo_iterations : list, optional
        Iteration numbers corresponding to full_elbo_losses. If provided, used for x-axis of full ELBO plot.
    batch_losses : list, optional
        Batch ELBO losses (one per training iteration). If provided, creates a 4th plot showing batch ELBO progression.
    dataset_name : str
        'MNIST' or 'FashionMNIST' - used for legend labels
    is_ood : bool
        Whether this is OOD data - used for legend labels
    """
    
    if not (hasattr(config, 'plot_training') and config.plot_training or
            hasattr(config, 'plot_latent_space') and config.plot_latent_space or
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams):
        return

    # Color mapping for MNIST labels (digits 0-9)
    # MNIST labels are integers 0, 1, 2, ..., 9 from load_real_data('mnist')
    # Use consistent color mapping across SSVI and original GP LVM
    colors, color_map = get_consistent_mnist_colors(labels_train)

    # Determine which plots to show
    plot_latent_space = hasattr(config, 'plot_latent_space') and config.plot_latent_space
    plot_hyperparams = hasattr(config, 'plot_hyperparams') and config.plot_hyperparams
    # Only plot latent space and lengthscales (remove ELBO plots)
    n_plots = (1 if plot_latent_space else 0) + (1 if plot_hyperparams else 0)
    fig_width = 4 * n_plots if n_plots > 0 else 8
    fig = plt.figure(figsize=(fig_width, 4))

    plot_idx = 1

    # Plot 1: Latent space visualization
    if plot_latent_space:
        plt.subplot(1, n_plots, plot_idx)

        # Find the most informative dimensions (largest lengthscales = smallest beta)
        lengthscales = 1 / model.gp.beta.flatten()
        top_2_dims = torch.argsort(lengthscales, descending=True)[:2]
        top_3_dims = torch.argsort(lengthscales, descending=True)[:3]  # Also compute top 3 for 3D plots
        dim1, dim2 = top_2_dims[0].item(), top_2_dims[1].item()
        dim3 = top_3_dims[2].item()  # Third dimension for 3D plots
        lengthscales_cpu = lengthscales.cpu()  # Move to CPU for string formatting
        
        # Get latent variables using batched processing to reduce memory usage
        batch_size = getattr(config, 'batch_size', 128)  # Use config batch_size if available, otherwise default to 128
        epsilon = model.get_epsilon()
        U_samples = model.get_u_samples(epsilon)  # [P, M, C, Lx] - Sample U once (shared across all batches)
        
        # Process data in minibatches
        N = y_train.shape[0]
        mu_x_batches = []
        sigma_x_batches = []
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            y_batch = y_train[start_idx:end_idx]
            
            # Handle missing data if enabled
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                y_batch_clean = torch.where(
                    torch.isnan(y_batch),
                    torch.full_like(y_batch, -9.0, dtype=torch.float64),
                    y_batch
                )
                mu_x_batch, sigma_x_batch = model.get_mu_sigma_x_train_samples(y_batch_clean, epsilon)
            else:
                mu_x_batch, sigma_x_batch = model.get_mu_sigma_x_train_samples(y_batch, epsilon)
            
            mu_x_batches.append(mu_x_batch)
            sigma_x_batches.append(sigma_x_batch)
        
        # Concatenate all batches along the N dimension: [P, batch_size, A] -> [P, N, A]
        mu_x = torch.cat(mu_x_batches, dim=1)
        sigma_x = torch.cat(sigma_x_batches, dim=1)

        # Keep as torch tensors for sampling, convert to numpy after sampling
        mu_x_cpu = mu_x.cpu()  # Move to CPU but keep as tensor
        sigma_x_cpu = sigma_x.cpu()  # Move to CPU but keep as tensor

        # Plot samples from all P samples with transparency
        for p in range(model.P):
            # Use top 2 most informative latent dimensions
            sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
            plt.scatter(sample_1.detach().cpu().numpy(),
                        sample_2.detach().cpu().numpy(), alpha=0.6, color=colors)

        plt.xlabel(f'Latent Dimension {dim1}', fontsize=20)
        plt.ylabel(f'Latent Dimension {dim2}', fontsize=20)
        plt.title(f'Top 2 Informative Dims', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        # Add legend with class names below the plot in a horizontal line
        labels_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train
        from matplotlib.patches import Patch
        legend_handles = [Patch(color=color_map[l], label=get_label_name(l, dataset_name, is_ood)) 
                          for l in sorted(np.unique(labels_cpu))]
        plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=len(legend_handles), fontsize=20, frameon=True)
        plot_idx += 1

    # Plot 2: Hyperparameters (lengthscales)
    if plot_hyperparams:
        plt.subplot(1, n_plots, plot_idx)
        lengthscales = (1 / model.gp.beta.flatten()).cpu()
        plt.bar(range(len(lengthscales)), lengthscales,
                color="blue", alpha=0.7)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Lengthscale')
        plt.title('GP Lengthscales')
        plt.grid(True, alpha=0.3)
        plot_idx += 1

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save plot as PDF
    # Ensure PDF extension
    if not save_filename.endswith('.pdf'):
        save_filename = save_filename.rsplit('.', 1)[0] + '.pdf'
    if is_checkpoint:
        plot_path = os.path.join(logger.plots_checkpoints_dir, save_filename)
    else:
        plot_path = os.path.join(logger.plots_dir, save_filename)
    fig.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close figure to free memory

    # Additional plots: 1D histogram and 3D scatter plot
    if plot_latent_space:
        print("Creating additional plots: 1D histogram and 3D scatter plot...")

        # Ensure labels_train is on CPU for numpy operations
        labels_train_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train

        # We already have mu_x_cpu, sigma_x_cpu, lengthscales_cpu, dim1, dim2, dim3, and top_3_dims from the latent space plot
        lengthscales = lengthscales_cpu  # Use existing lengthscales

        # Use dimensions already computed in latent space plot section
        most_important_dim = dim1  # Use the first (most important) dimension for 1D histogram

        # 1. Create 1D histogram for the most important dimension
        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 6))

    # Collect samples from all P samples for the most important dimension
    # Use 64 samples to match original GP-LVM version
    num_samples = min(48, model.P)  # Use 64 or model.P, whichever is smaller
    all_samples = []
    all_labels = []
    for p in range(num_samples):
        samples = torch.distributions.Normal(mu_x_cpu[p, :, most_important_dim],
                                             sigma_x_cpu[p, :, most_important_dim]).sample()
        # Ensure samples are on CPU before converting to numpy
        all_samples.append(samples.detach().cpu().numpy())
        # Repeat labels for each P sample
        all_labels.append(labels_train_cpu)

    all_samples = np.concatenate(all_samples)
    all_labels = np.concatenate(all_labels)

    # Create histogram with different colors for different classes (using consistent color mapping)
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        class_mask = all_labels == label
        if np.any(class_mask):
            class_samples = all_samples[class_mask]
            # Use get_label_name for OOD-aware legend
            ax_hist.hist(class_samples, bins=30, alpha=0.7,
                         label=get_label_name(label, dataset_name, is_ood),
                         color=color_map[label])

        ax_hist.set_xlabel(f'Latent Dimension {most_important_dim} (Most Important)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'1D Histogram - Most Important Dimension {most_important_dim}')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # Save histogram plot as PDF
        hist_filename = save_filename.replace('training_results', '1d_histogram_most_important_dim')
        # Ensure PDF extension
        if not hist_filename.endswith('.pdf'):
            hist_filename = hist_filename.rsplit('.', 1)[0] + '.pdf'
        if is_checkpoint:
            hist_path = os.path.join(logger.plots_checkpoints_dir, hist_filename)
        else:
            hist_path = os.path.join(logger.plots_dir, hist_filename)
        fig_hist.savefig(hist_path, format='pdf', bbox_inches='tight', dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close(fig_hist)

        # 2. Create 3D scatter plot for top 3 most important dimensions using plotly
        dim1, dim2, dim3 = top_3_dims[0].item(), top_3_dims[1].item(), top_3_dims[2].item()

        # Collect samples for all 3 dimensions
        samples_3d = []
        for p in range(model.P):
            sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
            sample_3 = torch.distributions.Normal(mu_x_cpu[p, :, dim3], sigma_x_std_cpu[p, :, dim3]).sample()

            # Ensure all samples are on CPU before converting to numpy
            samples_3d.append({
                'dim1': sample_1.detach().cpu().numpy(),
                'dim2': sample_2.detach().cpu().numpy(),
                'dim3': sample_3.detach().cpu().numpy(),
                'labels': labels_train_cpu
            })

        # Create plotly 3D scatter plot
        fig_3d = go.Figure()

        # Color mapping for labels (using consistent color mapping)
        unique_labels = np.unique(labels_train_cpu)
        
        # Add traces for each class
        for label in unique_labels:
            class_mask = labels_train_cpu == label
            if np.any(class_mask):
                # Combine samples from all P samples for this class
                x_vals = np.concatenate([samples_3d[p]['dim1'][class_mask] for p in range(model.P)])
                y_vals = np.concatenate([samples_3d[p]['dim2'][class_mask] for p in range(model.P)])
                z_vals = np.concatenate([samples_3d[p]['dim3'][class_mask] for p in range(model.P)])
                
                fig_3d.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=matplotlib_to_plotly_color(color_map[label]),
                        opacity=0.6
                    ),
                    # Use get_label_name for OOD-aware legend
                    name=get_label_name(label, dataset_name, is_ood)
                ))

        # Update layout
        fig_3d.update_layout(
            title=f'3D Scatter Plot - Top 3 Most Important Dimensions<br>Dim {dim1}, {dim2}, {dim3}',
            scene=dict(
                xaxis_title=f'Dimension {dim1} (lengthscale={lengthscales[dim1].item():.2f})',
                yaxis_title=f'Dimension {dim2} (lengthscale={lengthscales[dim2].item():.2f})',
                zaxis_title=f'Dimension {dim3} (lengthscale={lengthscales[dim3].item():.2f})'
            ),
            width=800,
            height=600
        )

        # Save 3D plot as HTML
        plot_3d_filename = save_filename.replace('training_results', '3d_scatter_top3_dims')
        plot_3d_filename = plot_3d_filename.replace('.pdf', '.html').replace('.png', '.html')

        # Save to appropriate directory (checkpoints or main plots)
        if is_checkpoint:
            plot_3d_path = os.path.join(logger.plots_checkpoints_dir, plot_3d_filename)
        else:
            plot_3d_path = os.path.join(logger.plots_dir, plot_3d_filename)

        fig_3d.write_html(plot_3d_path)
        print(f"3D scatter plot saved to: {plot_3d_path}")

        if show_plot:
            fig_3d.show()

        print("Additional plots created successfully!")


def visualize_prior_samples_replot(
    model,
    n_images: int = 24,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    image_shape: tuple = (28, 28),
    scaler=None
):
    """
    Visualize samples from the GP-LVM prior (mean images only) - for replotting.
    
    This function samples images by:
    1. Sampling latent points from the prior p(x) = N(0, I)
    2. Sampling U from the variational posterior q(U)
    3. Getting GP predictive mean f(x) | U
    4. Visualizing images in a grid
    
    Saves as PDF instead of PNG.
    
    Args:
        model: Trained SSVI GP-LVM model
        n_images: Number of prior samples to generate (default: 24)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
        dataset_name: Name of the dataset for plot title (default: "MNIST")
        image_shape: Shape of the output image (default: (28, 28) for MNIST)
    
    Returns:
        f_mean_final: Tensor of generated mean images [n_images, M]
    """
    
    model.eval()
    
    with torch.no_grad():
        # Get model parameters
        latent_dim = model.gp.Vx.shape[1]  # Latent dimensionality
        device = model.gp.Vx.device
        
        # 1. Sample latent points from the prior p(x) = N(0, I)
        x_prior = torch.randn(n_images, latent_dim, dtype=torch.float64, device=device)
        
        # 2. Get U samples from variational posterior q(U)
        epsilon = model.get_epsilon()
        U_samples = model.get_u_samples(epsilon)  # [P, M, C, L]
        
        # 3. Get GP predictive distribution for each latent point
        # Shape: x_prior needs to be [P, n_images, latent_dim] for consistency with U_samples
        x_prior_expanded = x_prior.unsqueeze(0).expand(U_samples.shape[0], -1, -1)  # [P, n_images, latent_dim]
        
        # Get GP predictive mean: [P, n_images, M, C]
        f_mean = model.gp.get_gp_expectation_given_U(x_prior_expanded, U_samples)
        
        # For C=1 case (standard GP-LVM for images), extract mean
        if f_mean.shape[-1] == 1:
            f_mean = f_mean[..., 0]  # [P, n_images, M]
        
        # Take one U sample per image (diagonal: i-th U sample for i-th image)
        # This avoids averaging over P and gives proper samples from the generative model
        P = f_mean.shape[0]
        f_mean_samples = []
        for i in range(n_images):
            # Use i % P to cycle through U samples if n_images > P
            p_idx = i % P
            f_mean_samples.append(f_mean[p_idx, i, :])  # [M]
        
        f_mean_final = torch.stack(f_mean_samples, dim=0)  # [n_images, M]
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        f_mean_final = scaler.inverse_transform(f_mean_final)
    
    img_h, img_w = image_shape
    
    # Create grid layout
    n_cols = min(8, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    
    # Handle edge cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_images):
        row = i // n_cols
        col = i % n_cols
        img_mean = f_mean_final[i].cpu().numpy().reshape(img_h, img_w)
        img_mean = np.clip(img_mean, 0, 1)
        axes[row, col].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'{dataset_name} Prior Samples', fontsize=14)
    plt.tight_layout()
    
    # Save if path provided - ensure PDF extension
    if save_path:
        # Ensure PDF extension
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Prior samples visualization saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Return generated images (mean) for further analysis
    return f_mean_final


def visualize_prior_samples_averaged_replot(
    model,
    n_images: int = 8,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    image_shape: tuple = (28, 28),
    scaler=None
):
    """
    Visualize samples from the GP-LVM prior, averaging over P U samples - for replotting.
    
    Saves as PDF instead of PNG.
    
    Args:
        model: Trained SSVI GP-LVM model
        n_images: Number of prior samples to generate (default: 8)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
        dataset_name: Name of the dataset for plot title (default: "MNIST")
        image_shape: Shape of the output image (default: (28, 28) for MNIST)
    
    Returns:
        f_mean_avg: Tensor of mean images averaged over P [n_images, M]
        pred_var_avg: Tensor of variance averaged over P [n_images, M]
    """
    
    model.eval()
    
    with torch.no_grad():
        # Get model parameters
        latent_dim = model.gp.Vx.shape[1]  # Latent dimensionality
        device = model.gp.Vx.device
        
        # 1. Sample latent points from the prior p(x) = N(0, I)
        x_prior = torch.randn(n_images, latent_dim, dtype=torch.float64, device=device)
        
        # 2. Get U samples from variational posterior q(U)
        epsilon = model.get_epsilon()
        U_samples = model.get_u_samples(epsilon)  # [P, M, C, L]
        
        # 3. Get GP predictive distribution for each latent point
        # Shape: x_prior needs to be [P, n_images, latent_dim] for consistency with U_samples
        x_prior_expanded = x_prior.unsqueeze(0).expand(U_samples.shape[0], -1, -1)  # [P, n_images, latent_dim]
        
        # Get GP predictive mean: [P, n_images, M, C]
        f_mean = model.gp.get_gp_expectation_given_U(x_prior_expanded, U_samples)
        
        # Get GP predictive covariance: [P, n_images, M, C, C]
        f_cov = model.gp.get_gp_covariance_given_U(x_prior_expanded, U_samples)
        
        # For C=1 case (standard GP-LVM for images), extract variance
        if f_mean.shape[-1] == 1:
            f_mean = f_mean[..., 0]  # [P, n_images, M]
            f_var = f_cov[..., 0, 0]  # [P, n_images, M]
        else:
            # Multi-output case: use diagonal
            f_var = torch.diagonal(f_cov, dim1=-2, dim2=-1)  # [P, n_images, M, C]
        
        # Add observation noise to get predictive variance
        pred_var = f_var + model.sigma2  # [P, n_images, M]
        
        # Average over P samples (mean prediction and variance)
        # For variance: use law of total variance = variance of means + mean of variances
        f_mean_avg = f_mean.mean(dim=0)  # [n_images, M]
        pred_var_avg = f_mean.var(dim=0) + pred_var.mean(dim=0)  # [n_images, M]
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        f_mean_avg = scaler.inverse_transform(f_mean_avg)
        # For variance, scale by scaler.std^2 (variance scales as std^2)
        if hasattr(scaler, 'std'):
            pred_var_avg = pred_var_avg * (scaler.std ** 2)
    
    img_h, img_w = image_shape
    
    # Create visualization: 2 rows (mean, variance) x n_images columns
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.8, 4))
    
    # Handle edge case of single image
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_images):
        # Top row: Mean reconstruction (averaged over P)
        img_mean = f_mean_avg[i].cpu().numpy().reshape(img_h, img_w)
        img_mean = np.clip(img_mean, 0, 1)
        axes[0, i].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        
        # Bottom row: Variance (uncertainty, averaged over P)
        img_var = pred_var_avg[i].cpu().numpy().reshape(img_h, img_w)
        im = axes[1, i].imshow(img_var, cmap='viridis')
        axes[1, i].axis('off')
    
    # Add row labels on the left
    fig.text(0.02, 0.72, 'Mean', fontsize=12, rotation=90, va='center', ha='center')
    fig.text(0.02, 0.28, 'Variance', fontsize=12, rotation=90, va='center', ha='center')
    
    # Add colorbar for variance on the right
    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.35])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Variance', fontsize=10)
    
    fig.suptitle(f'{dataset_name} Prior Samples (Averaged over P)', fontsize=14, y=0.98)
    plt.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.05, wspace=0.1, hspace=0.15)
    
    # Save if path provided - ensure PDF extension
    if save_path:
        # Ensure PDF extension
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Prior samples (averaged) visualization saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Return generated images (mean) and variance averaged over P
    return f_mean_avg, pred_var_avg


def visualize_test_image_predictions_replot(
    model,
    y_train: torch.Tensor,
    config,
    n_images: int = 24,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    image_shape: tuple = (28, 28),
    scaler=None
):
    """
    Visualize predictions for selected test images using their latent representations - for replotting.
    
    Saves as PDF instead of PNG.
    
    Args:
        model: Trained SSVI GP-LVM model
        y_train: Test data [N, M]
        config: Configuration object with random_seed attribute
        n_images: Number of images to visualize (default: 24)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
        dataset_name: Name of the dataset for plot title (default: "MNIST")
        image_shape: Shape of the output image (default: (28, 28) for MNIST)
        scaler: Scaler for inverse transform (optional)
    
    Returns:
        f_mean_avg: Tensor of mean predictions [n_images, M]
    """
    
    model.eval()
    
    with torch.no_grad():
        # Get model parameters
        device = model.gp.Vx.device
        
        # 1. Sample image indices from test set using config seed
        N = y_train.shape[0]
        if n_images > N:
            n_images = N
        
        # Use config seed for reproducibility
        rng = np.random.RandomState(config.random_seed)
        image_indices = rng.choice(N, size=n_images, replace=False)
        image_indices = torch.tensor(image_indices, dtype=torch.int64, device=device)
        
        # Get selected images
        y_selected = y_train[image_indices]  # [n_images, M]
        
        # Handle missing data preprocessing for encoder input
        if model.enable_missing_data:
            y_selected_clean = torch.where(
                torch.isnan(y_selected), 
                torch.full_like(y_selected, -9.0, dtype=torch.float64), 
                y_selected
            )
        else:
            y_selected_clean = y_selected
        
        # 2. Get U samples from variational posterior q(U) (same as prior sampling)
        epsilon = model.get_epsilon()
        U_samples = model.get_u_samples(epsilon)  # [P, M, C, L]
        
        # 3. Get latent variables for selected images using amortized inference
        mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_selected_clean, epsilon)
        # mu_x: [P, n_images, latent_dim], sigma_x: [P, n_images, latent_dim]
        
        # 4. Get GP predictive distribution using latent variables
        # Use mu_x directly as x values (mean of latent distribution)
        # mu_x is already [P, n_images, latent_dim], matching the expected input shape
        # Get GP predictive mean: [P, n_images, M, C]
        f_mean = model.gp.get_gp_expectation_given_U(mu_x, U_samples)
        
        # For C=1 case (standard GP-LVM for images), extract mean
        if f_mean.shape[-1] == 1:
            f_mean = f_mean[..., 0]  # [P, n_images, M]
        
        # Average over P samples (mean prediction)
        f_mean_avg = f_mean.mean(dim=0)  # [n_images, M]
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        f_mean_avg = scaler.inverse_transform(f_mean_avg)
    
    img_h, img_w = image_shape
    
    # Create grid layout - 8 images in 2 rows (4 columns)
    if n_images == 8:
        n_cols = 4
        n_rows = 2
    else:
        n_cols = min(8, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    
    # Handle edge cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_images):
        row = i // n_cols
        col = i % n_cols
        img_mean = f_mean_avg[i].cpu().numpy().reshape(img_h, img_w)
        img_mean = np.clip(img_mean, 0, 1)
        axes[row, col].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided - ensure PDF extension
    if save_path:
        # Ensure PDF extension
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Test image predictions visualization saved: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Return mean predictions
    return f_mean_avg

