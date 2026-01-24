"""
Replotting helper functions for SSVI GP-LVM models.

This module contains functions specifically designed for replotting saved models,
with improved device handling and PDF output support.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import no_grad


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
    
    # Compute global Y-axis limits from data range (with padding for uncertainty bands)
    y_min_global = y_data_for_limits.min().item()
    y_max_global = y_data_for_limits.max().item()
    y_range = y_max_global - y_min_global
    y_padding = 1.0 * y_range  # 100% padding (fixed proportion for consistency across scripts, extra space for histograms)
    y_lim_min = y_min_global - y_padding
    y_lim_max = y_max_global + y_padding

    # 9) plot - show first 12 dimensions (or M if M < 12)
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
        # Change extension to .pdf if it's not already
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"Saved latent slice plot to: {save_path}")
    else:
        plt.show()

