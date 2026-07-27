#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Missing Data Reconstruction Visualization

This module provides visualization utilities for missing data reconstruction in MNIST,
showing original images, images with missing data, and multiple reconstruction samples.

Usage:
------
This module is automatically called at the end of mnist_GPLVM_SSVI_AM_INF.py when 
missing data is enabled (config.enable_missing_data=True).

The visualization creates two plots:

1. Simple Visualization (`visualize_mnist_missing_reconstruction`):
   - Shows 10 images in 3 columns:
     * Column 1: Original complete images (ground truth)
     * Column 2: Images with missing mask applied (what the model saw during training)
     * Columns 3-7: 5 different reconstruction samples from the model

2. Detailed Visualization (`visualize_mnist_missing_reconstruction_detailed`):
   - Shows 10 images in more columns:
     * Column 1: Original complete images
     * Column 2: Images with missing mask applied
     * Columns 3-7: 5 different reconstruction samples
     * Column 8: Mean reconstruction with RMSE per image
     * Column 9: Uncertainty map (standard deviation across samples)

The visualizations help assess:
- How well the model reconstructs missing pixel values
- Reconstruction quality across different samples
- Uncertainty in the reconstruction (how much samples vary)
- Per-image reconstruction error (RMSE on missing pixels only)

Example:
--------
To run MNIST with missing data visualization:

    python ssvi_scripts_experiments/mnist_GPLVM_SSVI_AM_INF.py

The visualization will be saved to:
    mnist_results/ssvi_results/plots/missing_data_reconstruction_{seed}.png
    mnist_results/ssvi_results/plots/missing_data_reconstruction_detailed_{seed}.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def visualize_mnist_missing_reconstruction(
    model,
    y_train: torch.Tensor,
    missing_mask: torch.Tensor,
    y_original: torch.Tensor,
    n_images: int = 10,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    scaler=None
):
    """
    Visualize MNIST/FashionMNIST missing data reconstruction with original, masked, and reconstructed images.
    
    Args:
        model: Trained SSVI GP-LVM model with missing data support
        y_train: Training data with missing values (NaN for missing) [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
        y_original: Original complete training data before masking [N, D]
        n_images: Number of images to visualize (default: 10)
        n_samples: Number of reconstruction samples per image (default: 5)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
        dataset_name: Name of the dataset for plot title (default: "MNIST")
    """
    model.eval()
    
    # Select first n_images
    n_images = min(n_images, y_train.shape[0])
    y_batch = y_train[:n_images]
    mask_batch = missing_mask[:n_images]
    y_orig_batch = y_original[:n_images]
    
    # Reconstruct images
    with torch.no_grad():
        # Get latent variables using amortized inference
        epsilon = model.get_epsilon()
        U_samples = model.get_u_samples(epsilon)
        
        # Handle missing data in encoder input (consistent with training)
        y_encoder_input = torch.where(mask_batch, 
                                     torch.full_like(y_batch, -9.0, dtype=torch.float64), 
                                     y_batch)
        
        # Get latent distribution parameters
        mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
        
        # Generate multiple reconstruction samples (consistent with SSVI metrics)
        # Generate exactly n_samples reconstructions by cycling through P samples if needed
        reconstructions = []
        for s in range(n_samples):
            # Cycle through P samples if n_samples > model.P
            p = s % model.P
            mu_sample = mu_x[p, :n_images, :]  # [n_images, latent_dim]
            sigma_sample = sigma_x[p, :n_images, :]  # [n_images, latent_dim]
            
            # Sample from Normal distribution (same as SSVI metrics and plotting)
            x_sample = torch.distributions.Normal(mu_sample, sigma_sample).sample()
            
            # Decode to observation space using GP
            y_mean = model.gp.get_gp_expectation(x_sample)  # [n_images, M, C]
            
            # Handle output dimensions (squeeze if C=1)
            if y_mean.shape[-1] == 1:
                y_mean = y_mean[..., 0]  # [n_images, M]
            
            reconstructions.append(y_mean)
        
        # Stack reconstructions [n_samples, n_images, D]
        reconstructions = torch.stack(reconstructions, dim=0)
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        # Inverse transform y_original and reconstructions to original scale
        y_orig_batch = scaler.inverse_transform(y_orig_batch)
        # Store original shape before reshape
        original_shape = reconstructions.shape
        reconstructions = scaler.inverse_transform(reconstructions.reshape(-1, reconstructions.shape[-1])).reshape(original_shape)
    
    # Compute image dimensions (assume square images)
    img_dim = int(np.sqrt(y_batch.shape[1]))
    
    # Create visualization
    # Layout: n_images rows × (1 original + 1 masked + n_samples reconstructions) columns
    n_cols = 2 + n_samples  # original, masked, and n_samples reconstructions
    fig, axes = plt.subplots(n_images, n_cols, figsize=(n_cols * 1.5, n_images * 1.5))
    
    # Handle single row case
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_images):
        # Column 0: Original image
        img_orig = y_orig_batch[i].cpu().numpy().reshape(img_dim, img_dim)
        axes[i, 0].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=10)
        
        # Column 1: Masked image (with missing data)
        img_masked = y_batch[i].clone()
        # Set missing values to 0 for visualization
        img_masked[mask_batch[i]] = 0.0
        # Inverse transform if scaler provided
        if scaler is not None:
            img_masked = scaler.inverse_transform(img_masked.unsqueeze(0)).squeeze(0)
        img_masked = img_masked.cpu().numpy().reshape(img_dim, img_dim)
        axes[i, 1].imshow(img_masked, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Masked\n(Training)', fontsize=10)
        
        # Columns 2+: Reconstruction samples
        for s in range(n_samples):
            img_recon = reconstructions[s, i].cpu().numpy().reshape(img_dim, img_dim)
            # Clip to valid range [0, 1] for visualization
            img_recon = np.clip(img_recon, 0, 1)
            axes[i, 2 + s].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
            axes[i, 2 + s].axis('off')
            if i == 0:
                axes[i, 2 + s].set_title(f'Recon {s+1}', fontsize=10)
    
    # Add overall title
    missing_pct = (mask_batch.sum() / mask_batch.numel() * 100).item()
    fig.suptitle(f'{dataset_name} Missing Data Reconstruction ({missing_pct:.1f}% missing)', 
                 fontsize=14, y=0.995)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Missing data reconstruction visualization saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Compute and print RMSE for missing pixels
    with torch.no_grad():
        # Use mean of reconstructions for RMSE
        y_recon_mean = reconstructions.mean(dim=0)  # [n_images, D]
        
        # Compute RMSE only on missing pixels
        y_true_missing = y_orig_batch[mask_batch]
        y_pred_missing = y_recon_mean[mask_batch]
        
        rmse_missing = torch.sqrt(torch.mean((y_true_missing - y_pred_missing) ** 2)).item()
        
        print(f"Missing pixels RMSE (first {n_images} images): {rmse_missing:.4f}")
        
        return rmse_missing


def visualize_mnist_missing_reconstruction_detailed(
    model,
    y_train: torch.Tensor,
    missing_mask: torch.Tensor,
    y_original: torch.Tensor,
    n_images: int = 10,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    scaler=None
):
    """
    Enhanced visualization with RMSE per image and uncertainty visualization.
    
    Args:
        model: Trained SSVI GP-LVM model with missing data support
        y_train: Training data with missing values (NaN for missing) [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
        y_original: Original complete training data before masking [N, D]
        n_images: Number of images to visualize (default: 10)
        n_samples: Number of reconstruction samples per image (default: 5)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
        dataset_name: Name of the dataset for plot title (default: "MNIST")
    """
    model.eval()
    
    # Select first n_images
    n_images = min(n_images, y_train.shape[0])
    y_batch = y_train[:n_images]
    mask_batch = missing_mask[:n_images]
    y_orig_batch = y_original[:n_images]
    
    # Reconstruct images
    with torch.no_grad():
        # Get latent variables using amortized inference
        epsilon = model.get_epsilon()
        U_samples = model.get_u_samples(epsilon)
        
        # Handle missing data in encoder input (consistent with training)
        y_encoder_input = torch.where(mask_batch, 
                                     torch.full_like(y_batch, -9.0, dtype=torch.float64), 
                                     y_batch)
        
        # Get latent distribution parameters
        mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
        
        # Generate multiple reconstruction samples (consistent with SSVI metrics)
        # Generate exactly n_samples reconstructions by cycling through P samples if needed
        reconstructions = []
        for s in range(n_samples):
            # Cycle through P samples if n_samples > model.P
            p = s % model.P
            mu_sample = mu_x[p, :n_images, :]  # [n_images, latent_dim]
            sigma_sample = sigma_x[p, :n_images, :]  # [n_images, latent_dim]
            
            # Sample from Normal distribution (same as SSVI metrics and plotting)
            x_sample = torch.distributions.Normal(mu_sample, sigma_sample).sample()
            
            # Decode to observation space using GP
            y_mean = model.gp.get_gp_expectation(x_sample)  # [n_images, M, C]
            
            # Handle output dimensions (squeeze if C=1)
            if y_mean.shape[-1] == 1:
                y_mean = y_mean[..., 0]  # [n_images, M]
            
            reconstructions.append(y_mean)
        
        # Stack reconstructions [n_samples, n_images, D]
        reconstructions = torch.stack(reconstructions, dim=0)
        
        # Compute mean and std for each image
        y_recon_mean = reconstructions.mean(dim=0)  # [n_images, D]
        y_recon_std = reconstructions.std(dim=0)    # [n_images, D]
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        # Inverse transform y_original, reconstructions, mean, and std to original scale
        y_orig_batch = scaler.inverse_transform(y_orig_batch)
        # Store original shape before reshape
        original_shape = reconstructions.shape
        reconstructions = scaler.inverse_transform(reconstructions.reshape(-1, reconstructions.shape[-1])).reshape(original_shape)
        y_recon_mean = scaler.inverse_transform(y_recon_mean)
        # For std, we need to scale by the scaler's std (variance scales as std^2)
        if hasattr(scaler, 'std'):
            y_recon_std = y_recon_std * scaler.std
    
    # Compute image dimensions (assume square images)
    img_dim = int(np.sqrt(y_batch.shape[1]))
    
    # Create visualization with additional columns for mean and uncertainty
    # Layout: n_images rows × (1 original + 1 masked + n_samples reconstructions + 1 mean + 1 uncertainty) columns
    n_cols = 4 + n_samples  # original, masked, n_samples reconstructions, mean, uncertainty
    # Make figure wider to accommodate colorbar
    fig, axes = plt.subplots(n_images, n_cols, figsize=(n_cols * 1.5 + 1.0, n_images * 1.5))
    
    # Handle single row case
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    # Compute per-image RMSEs for display
    rmse_per_image = []
    
    for i in range(n_images):
        # Column 0: Original image
        img_orig = y_orig_batch[i].cpu().numpy().reshape(img_dim, img_dim)
        axes[i, 0].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=10)
        
        # Column 1: Masked image (with missing data)
        img_masked = y_batch[i].clone()
        # Set missing values to 0.5 (gray) for better visualization
        img_masked[mask_batch[i]] = 0.5
        # Inverse transform if scaler provided
        if scaler is not None:
            img_masked = scaler.inverse_transform(img_masked.unsqueeze(0)).squeeze(0)
        img_masked = img_masked.cpu().numpy().reshape(img_dim, img_dim)
        axes[i, 1].imshow(img_masked, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Masked\n(Training)', fontsize=10)
        
        # Columns 2 to 2+n_samples-1: Reconstruction samples
        for s in range(n_samples):
            img_recon = reconstructions[s, i].cpu().numpy().reshape(img_dim, img_dim)
            # Clip to valid range [0, 1] for visualization
            img_recon = np.clip(img_recon, 0, 1)
            axes[i, 2 + s].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
            axes[i, 2 + s].axis('off')
            if i == 0:
                axes[i, 2 + s].set_title(f'Sample {s+1}', fontsize=10)
        
        # Column 2+n_samples: Mean reconstruction
        img_mean = y_recon_mean[i].cpu().numpy().reshape(img_dim, img_dim)
        img_mean = np.clip(img_mean, 0, 1)
        axes[i, 2 + n_samples].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[i, 2 + n_samples].axis('off')
        if i == 0:
            axes[i, 2 + n_samples].set_title('Mean', fontsize=10)
        
        # Compute RMSE for this image (only on missing pixels)
        mask_i = mask_batch[i]
        if mask_i.sum() > 0:
            rmse_i = torch.sqrt(torch.mean((y_orig_batch[i][mask_i] - y_recon_mean[i][mask_i]) ** 2)).item()
            rmse_per_image.append(rmse_i)
            
            # Add Missing RMSE text below mean image
            axes[i, 2 + n_samples].text(0.5, -0.1, f'Missing RMSE: {rmse_i:.3f}', 
                                       transform=axes[i, 2 + n_samples].transAxes,
                                       ha='center', fontsize=8)
        
        # Column 2+n_samples+1: Uncertainty (std of reconstructions)
        img_std = y_recon_std[i].cpu().numpy().reshape(img_dim, img_dim)
        im = axes[i, 2 + n_samples + 1].imshow(img_std, cmap='hot', vmin=0, vmax=img_std.max())
        axes[i, 2 + n_samples + 1].axis('off')
        if i == 0:
            axes[i, 2 + n_samples + 1].set_title('Uncertainty', fontsize=10)
    
    # Add colorbar for uncertainty - place it outside the plot area
    # Create colorbar using the last column (uncertainty column) but place it to the right
    cbar = fig.colorbar(im, ax=axes[:, -1].tolist(), fraction=0.046, pad=0.1, shrink=0.8, aspect=20)
    cbar.set_label('Std Dev', fontsize=10)
    
    # Add overall title
    missing_pct = (mask_batch.sum() / mask_batch.numel() * 100).item()
    mean_rmse = np.mean(rmse_per_image) if rmse_per_image else 0.0
    fig.suptitle(f'{dataset_name} Missing Data Reconstruction ({missing_pct:.1f}% missing, Mean Missing RMSE: {mean_rmse:.4f})', 
                 fontsize=14, y=0.995)
    
    # Use subplots_adjust to leave space for colorbar and avoid tight_layout warning
    plt.subplots_adjust(left=0.05, right=0.88, top=0.93, bottom=0.05, wspace=0.3, hspace=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Detailed missing data reconstruction visualization saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    print(f"Missing pixels mean RMSE (first {n_images} images): {mean_rmse:.4f}")
    
    return mean_rmse


def visualize_prior_samples(
    model,
    n_images: int = 24,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    image_shape: tuple = (28, 28),
    scaler=None
):
    """
    Visualize samples from the GP-LVM prior (mean images only).
    
    This function samples images by:
    1. Sampling latent points from the prior p(x) = N(0, I)
    2. Sampling U from the variational posterior q(U)
    3. Getting GP predictive mean f(x) | U
    4. Visualizing images in a grid
    
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
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prior samples visualization saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Return generated images (mean) for further analysis
    return f_mean_final


def visualize_prior_samples_averaged(
    model,
    n_images: int = 8,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    image_shape: tuple = (28, 28),
    scaler=None
):
    """
    Visualize samples from the GP-LVM prior, averaging over P U samples.
    
    This function samples images by:
    1. Sampling latent points from the prior p(x) = N(0, I)
    2. Sampling U from the variational posterior q(U)
    3. Getting GP predictive distribution f(x) | U
    4. Averaging the predictions over all P U samples
    5. Visualizing mean (top row) and variance (bottom row)
    
    This differs from visualize_prior_samples which takes one U sample per image.
    Here we average predictions across all P samples for each latent point.
    
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
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prior samples (averaged) visualization saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Return generated images (mean) and variance averaged over P
    return f_mean_avg, pred_var_avg


def visualize_test_image_predictions(
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
    Visualize predictions for selected test images using their latent representations.
    
    This function:
    1. Samples n_images indices from the test set using config.random_seed
    2. For each selected image:
       - Gets latent variable (mu_x, sigma_x) using amortized inference
       - Uses the same U samples as in prior sampling
       - Gets f_mean and f_var using the latent variables
    3. Visualizes mean and variance in a grid
    
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
        pred_var_avg: Tensor of predictive variances [n_images, M]
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
        img_mean = f_mean_avg[i].cpu().numpy().reshape(img_h, img_w)
        img_mean = np.clip(img_mean, 0, 1)
        axes[row, col].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'{dataset_name} Test Image Predictions', fontsize=14)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test image predictions visualization saved: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Return mean predictions
    return f_mean_avg


def visualize_prior_samples_original_gplvm(
    model,
    likelihood,
    n_images: int = 24,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    image_shape: tuple = (28, 28),
    scaler=None
):
    """
    Visualize samples from the Original GP-LVM prior (mean images only).
    
    This function samples images by:
    1. Sampling latent points from the prior p(x) = N(0, I)
    2. Passing through the GP to get predictive mean
    3. Visualizing images in a grid
    
    Args:
        model: Trained Original GP-LVM model (BayesianGPLVM)
        likelihood: The Gaussian likelihood from the model
        n_images: Number of prior samples to generate (default: 24)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
        dataset_name: Name of the dataset for plot title (default: "MNIST")
        image_shape: Shape of the output image (default: (28, 28) for MNIST)
    
    Returns:
        f_mean: Tensor of generated mean images [n_images, data_dim]
    """
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        # Get model parameters
        latent_dim = model.variational_strategy.inducing_points.shape[1]
        device = model.variational_strategy.inducing_points.device
        
        # 1. Sample latent points from the prior p(x) = N(0, I)
        x_prior = torch.randn(n_images, latent_dim, dtype=torch.float64, device=device)
        
        # 2. Get GP predictive distribution
        # model(x) returns MultivariateNormal with shape [data_dim, n_images]
        gp_dist = model(x_prior)
        
        # Extract mean (same approach as nlpd_utils.py)
        # gp_dist.loc has shape [data_dim, n_images], transpose to [n_images, data_dim]
        f_mean = gp_dist.loc.T  # [n_images, data_dim]
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        f_mean = scaler.inverse_transform(f_mean)
    
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
        img_mean = f_mean[i].cpu().numpy().reshape(img_h, img_w)
        img_mean = np.clip(img_mean, 0, 1)
        axes[row, col].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'{dataset_name} Prior Samples (Original GP-LVM)', fontsize=14)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prior samples visualization saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Return generated images (mean) for further analysis
    return f_mean


def visualize_prior_samples_original_gplvm_averaged(
    model,
    likelihood,
    n_images: int = 8,
    n_samples: int = 8,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    image_shape: tuple = (28, 28),
    scaler=None
):
    """
    Visualize samples from the Original GP-LVM prior, averaging over multiple GP samples.
    
    This function samples images by:
    1. Sampling latent points from the prior p(x) = N(0, I)
    2. Calling the GP multiple times to get multiple predictive distributions
    3. Averaging the predictions over all samples
    4. Visualizing mean (top row) and variance (bottom row)
    
    This differs from visualize_prior_samples_original_gplvm which uses a single GP call.
    Here we average predictions across multiple GP samples for each latent point.
    
    Args:
        model: Trained Original GP-LVM model (BayesianGPLVM)
        likelihood: The Gaussian likelihood from the model
        n_images: Number of prior samples to generate (default: 8)
        n_samples: Number of GP samples to average over (default: 8)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
        dataset_name: Name of the dataset for plot title (default: "MNIST")
        image_shape: Shape of the output image (default: (28, 28) for MNIST)
    
    Returns:
        Tuple of (mean_images, variance_images) tensors averaged over samples
    """
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        # Get model parameters
        latent_dim = model.variational_strategy.inducing_points.shape[1]
        device = model.variational_strategy.inducing_points.device
        
        # 1. Sample latent points from the prior p(x) = N(0, I)
        x_prior = torch.randn(n_images, latent_dim, dtype=torch.float64, device=device)
        
        # 2. Get GP predictive distribution multiple times and average
        f_mean_samples = []
        f_var_samples = []
        
        for _ in range(n_samples):
            # model(x) returns MultivariateNormal with shape [data_dim, n_images]
            gp_dist = model(x_prior)
            
            # Extract mean: [n_images, data_dim]
            f_mean = gp_dist.loc.T
            
            # Get variance by extracting diagonal from each covariance matrix
            f_var = torch.stack([torch.diag(x) for x in gp_dist.covariance_matrix]).T
            
            f_mean_samples.append(f_mean)
            f_var_samples.append(f_var)
        
        # Stack and average over samples
        f_mean_stacked = torch.stack(f_mean_samples, dim=0)  # [n_samples, n_images, data_dim]
        f_var_stacked = torch.stack(f_var_samples, dim=0)    # [n_samples, n_images, data_dim]
        
        # Add observation noise to each sample's variance first
        obs_noise = likelihood.noise
        pred_var_stacked = f_var_stacked + obs_noise  # [n_samples, n_images, data_dim]
        
        # Average over samples (mean prediction and variance)
        # For variance: use law of total variance = variance of means + mean of variances
        f_mean_avg = f_mean_stacked.mean(dim=0)  # [n_images, data_dim]
        pred_var_avg = f_mean_stacked.var(dim=0) + pred_var_stacked.mean(dim=0)  # [n_images, data_dim]
    
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
        # Top row: Mean reconstruction (averaged over samples)
        img_mean = f_mean_avg[i].cpu().numpy().reshape(img_h, img_w)
        img_mean = np.clip(img_mean, 0, 1)
        axes[0, i].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        
        # Bottom row: Variance (uncertainty, averaged over samples)
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
    
    fig.suptitle(f'{dataset_name} Prior Samples (Averaged over {n_samples})', fontsize=14, y=0.98)
    plt.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.05, wspace=0.1, hspace=0.15)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prior samples (averaged) visualization saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Return generated images (mean) and variance averaged over samples
    return f_mean_avg, pred_var_avg

def visualize_test_image_predictions_original_gplvm(
    model,
    likelihood,
    y_test: torch.Tensor,
    config,
    model_name: str = 'nn_gauss',
    n_images: int = 24,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    dataset_name: str = "MNIST",
    image_shape: tuple = (28, 28),
    scaler=None
):
    """
    Visualize predictions for selected test images using their latent representations.
    
    This function:
    1. Samples n_images indices from the test set using config.random_seed
    2. For each selected image:
       - Gets latent variable using encoder (for nn_gauss)
       - Gets GP predictive mean using the latent variables
    3. Visualizes mean predictions in a grid
    
    Args:
        model: Trained Original GP-LVM model (BayesianGPLVM)
        likelihood: The Gaussian likelihood from the model
        y_test: Test data [N, M]
        config: Configuration object with random_seed and enable_missing_data attributes
        model_name: Model name ('nn_gauss' or 'gauss', default: 'nn_gauss')
        n_images: Number of images to visualize (default: 24)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
        dataset_name: Name of the dataset for plot title (default: "MNIST")
        image_shape: Shape of the output image (default: (28, 28) for MNIST)
        scaler: Scaler for inverse transform (optional)
    
    Returns:
        f_mean: Tensor of mean predictions [n_images, M]
    """
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        # Get model parameters
        device = model.variational_strategy.inducing_points.device
        latent_dim = model.variational_strategy.inducing_points.shape[1]
        
        # 1. Sample image indices from test set using config seed
        N = y_test.shape[0]
        if n_images > N:
            n_images = N
        
        # Use config seed for reproducibility
        rng = np.random.RandomState(config.random_seed)
        image_indices = rng.choice(N, size=n_images, replace=False)
        image_indices = torch.tensor(image_indices, dtype=torch.int64, device=device)
        
        # Get selected images
        y_selected = y_test[image_indices]  # [n_images, M]
        
        # 2. Get latent variables for selected images
        if model_name == 'nn_gauss':
            # For nn_gauss, use encoder
            # Temporarily adjust jitter size
            original_jitter = model.X.jitter
            test_jitter = torch.eye(latent_dim, dtype=torch.float64, device=device).unsqueeze(0) * 1e-5
            model.X.jitter = torch.cat([test_jitter for _ in range(n_images)], axis=0)
            
            # Handle missing data preprocessing for encoder input
            if config.enable_missing_data:
                y_selected_clean = torch.where(
                    torch.isnan(y_selected),
                    torch.full_like(y_selected, -9.0, dtype=torch.float64),
                    y_selected
                )
            else:
                y_selected_clean = y_selected
            
            # Get latent variables using encoder
            X_selected_mean = model.X.mu(y_selected_clean)  # [n_images, latent_dim]
            
            # Restore original jitter
            model.X.jitter = original_jitter
        else:
            # For gauss model, would need predict_latent which is expensive
            # This function assumes nn_gauss for efficiency
            raise ValueError(f"visualize_test_image_predictions_original_gplvm currently only supports nn_gauss model, got {model_name}")
        
        # 3. Get GP predictive distribution using latent variables
        # model(x) returns MultivariateNormal with shape [data_dim, n_images]
        gp_dist = model(X_selected_mean)
        
        # Extract mean (same approach as prior samples)
        # gp_dist.loc has shape [data_dim, n_images], transpose to [n_images, data_dim]
        f_mean = gp_dist.loc.T  # [n_images, data_dim]
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        f_mean = scaler.inverse_transform(f_mean)
    
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
        img_mean = f_mean[i].cpu().numpy().reshape(img_h, img_w)
        img_mean = np.clip(img_mean, 0, 1)
        axes[row, col].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'{dataset_name} Test Image Predictions (Original GP-LVM)', fontsize=14)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Test image predictions visualization saved: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    likelihood.train()
    
    # Return mean predictions
    return f_mean


def visualize_mnist_missing_reconstruction_gauss(
    model,
    y_train: torch.Tensor,
    missing_mask: torch.Tensor,
    y_original: torch.Tensor,
    X_mean: torch.Tensor,
    X_covar: torch.Tensor,
    n_images: int = 10,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    scaler=None
):
    """
    Visualize MNIST missing data reconstruction for Original GP-LVM (gauss) model.
    
    Args:
        model: Trained Original GP-LVM model (gauss)
        y_train: Training data with missing values (NaN for missing) [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
        y_original: Original complete training data before masking [N, D]
        X_mean: Pre-computed latent means [N, latent_dim]
        X_covar: Pre-computed latent covariances (standard deviations for gauss) [N, latent_dim]
        n_images: Number of images to visualize (default: 10)
        n_samples: Number of reconstruction samples per image (default: 5)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
    """
    model.eval()
    
    # Select first n_images
    n_images = min(n_images, y_train.shape[0])
    y_batch = y_train[:n_images]
    mask_batch = missing_mask[:n_images]
    y_orig_batch = y_original[:n_images]
    X_mean_batch = X_mean[:n_images]
    X_covar_batch = X_covar[:n_images]
    
    # Reconstruct images using same approach as unified_rmse_utils.py
    with torch.no_grad():
        # Sample from latent distribution (same as compute_rmse_with_sampling)
        reconstructions = []
        for p in range(n_samples):
            # Sample from Normal distribution (same as unified_rmse_utils.py)
            x_dist = torch.distributions.Normal(X_mean_batch, X_covar_batch)
            x_sample = x_dist.sample()  # [n_images, latent_dim]
            
            # Reconstruct through model.reconstruct_y (same as unified_rmse_utils.py)
            y_recon, _ = model.reconstruct_y(x_sample, y_batch, ae=False, model_name='gauss')
            
            # Transpose to match expected format [N, D] instead of [D, N] (same as unified_rmse_utils.py)
            y_recon_T = y_recon.T
            
            reconstructions.append(y_recon_T)
        
        # Stack reconstructions [n_samples, n_images, D]
        reconstructions = torch.stack(reconstructions, dim=0)
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        # Inverse transform y_original and reconstructions to original scale
        y_orig_batch = scaler.inverse_transform(y_orig_batch)
        reconstructions = scaler.inverse_transform(reconstructions.reshape(-1, reconstructions.shape[-1])).reshape(reconstructions.shape)
    
    # Compute image dimensions (assume square images)
    img_dim = int(np.sqrt(y_batch.shape[1]))
    
    # Create visualization
    # Layout: n_images rows × (1 original + 1 masked + n_samples reconstructions) columns
    n_cols = 2 + n_samples  # original, masked, and n_samples reconstructions
    fig, axes = plt.subplots(n_images, n_cols, figsize=(n_cols * 1.5, n_images * 1.5))
    
    # Handle single row case
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_images):
        # Column 0: Original image
        img_orig = y_orig_batch[i].cpu().numpy().reshape(img_dim, img_dim)
        axes[i, 0].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=10)
        
        # Column 1: Masked image (with missing data)
        img_masked = y_batch[i].clone()
        # Set missing values to 0 for visualization
        img_masked[mask_batch[i]] = 0.0
        # Inverse transform if scaler provided
        if scaler is not None:
            img_masked = scaler.inverse_transform(img_masked.unsqueeze(0)).squeeze(0)
        img_masked = img_masked.cpu().numpy().reshape(img_dim, img_dim)
        axes[i, 1].imshow(img_masked, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Masked\n(Training)', fontsize=10)
        
        # Columns 2+: Reconstruction samples
        for s in range(n_samples):
            img_recon = reconstructions[s, i].cpu().numpy().reshape(img_dim, img_dim)
            # Clip to valid range [0, 1] for visualization
            img_recon = np.clip(img_recon, 0, 1)
            axes[i, 2 + s].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
            axes[i, 2 + s].axis('off')
            if i == 0:
                axes[i, 2 + s].set_title(f'Recon {s+1}', fontsize=10)
    
    # Add overall title
    missing_pct = (mask_batch.sum() / mask_batch.numel() * 100).item()
    fig.suptitle(f'MNIST Missing Data Reconstruction (Gauss) ({missing_pct:.1f}% missing)', 
                 fontsize=14, y=0.995)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Missing data reconstruction visualization (Gauss) saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    # Compute and print RMSE for missing pixels
    with torch.no_grad():
        # Use mean of reconstructions for RMSE
        y_recon_mean = reconstructions.mean(dim=0)  # [n_images, D]
        
        # Compute RMSE only on missing pixels
        y_true_missing = y_orig_batch[mask_batch]
        y_pred_missing = y_recon_mean[mask_batch]
        
        rmse_missing = torch.sqrt(torch.mean((y_true_missing - y_pred_missing) ** 2)).item()
        
        print(f"Missing pixels RMSE (first {n_images} images): {rmse_missing:.4f}")
        
        return rmse_missing


def visualize_mnist_missing_reconstruction_gauss_detailed(
    model,
    y_train: torch.Tensor,
    missing_mask: torch.Tensor,
    y_original: torch.Tensor,
    X_mean: torch.Tensor,
    X_covar: torch.Tensor,
    n_images: int = 10,
    n_samples: int = 5,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    scaler=None
):
    """
    Enhanced visualization for Original GP-LVM (gauss) model with RMSE per image and uncertainty visualization.
    
    Args:
        model: Trained Original GP-LVM model (gauss)
        y_train: Training data with missing values (NaN for missing) [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
        y_original: Original complete training data before masking [N, D]
        X_mean: Pre-computed latent means [N, latent_dim]
        X_covar: Pre-computed latent covariances (standard deviations for gauss) [N, latent_dim]
        n_images: Number of images to visualize (default: 10)
        n_samples: Number of reconstruction samples per image (default: 5)
        save_path: Path to save the visualization (optional)
        show_plot: Whether to display the plot (default: False)
    """
    model.eval()
    
    # Select first n_images
    n_images = min(n_images, y_train.shape[0])
    y_batch = y_train[:n_images]
    mask_batch = missing_mask[:n_images]
    y_orig_batch = y_original[:n_images]
    X_mean_batch = X_mean[:n_images]
    X_covar_batch = X_covar[:n_images]
    
    # Reconstruct images using same approach as unified_rmse_utils.py
    with torch.no_grad():
        # Sample from latent distribution (same as compute_rmse_with_sampling)
        reconstructions = []
        for p in range(n_samples):
            # Sample from Normal distribution (same as unified_rmse_utils.py)
            x_dist = torch.distributions.Normal(X_mean_batch, X_covar_batch)
            x_sample = x_dist.sample()  # [n_images, latent_dim]
            
            # Reconstruct through model.reconstruct_y (same as unified_rmse_utils.py)
            y_recon, _ = model.reconstruct_y(x_sample, y_batch, ae=False, model_name='gauss')
            
            # Transpose to match expected format [N, D] instead of [D, N] (same as unified_rmse_utils.py)
            y_recon_T = y_recon.T
            
            reconstructions.append(y_recon_T)
        
        # Stack reconstructions [n_samples, n_images, D]
        reconstructions = torch.stack(reconstructions, dim=0)
        
        # Compute mean and std for each image
        y_recon_mean = reconstructions.mean(dim=0)  # [n_images, D]
        y_recon_std = reconstructions.std(dim=0)    # [n_images, D]
    
    # Inverse transform for visualization if scaler provided
    if scaler is not None:
        # Inverse transform y_original, reconstructions, mean, and std to original scale
        y_orig_batch = scaler.inverse_transform(y_orig_batch)
        reconstructions = scaler.inverse_transform(reconstructions.reshape(-1, reconstructions.shape[-1])).reshape(reconstructions.shape)
        y_recon_mean = scaler.inverse_transform(y_recon_mean)
        # For std, we need to scale by the scaler's std (variance scales as std^2)
        if hasattr(scaler, 'std'):
            y_recon_std = y_recon_std * scaler.std
    
    # Compute image dimensions (assume square images)
    img_dim = int(np.sqrt(y_batch.shape[1]))
    
    # Create visualization with additional columns for mean and uncertainty
    # Layout: n_images rows × (1 original + 1 masked + n_samples reconstructions + 1 mean + 1 uncertainty) columns
    n_cols = 4 + n_samples  # original, masked, n_samples reconstructions, mean, uncertainty
    # Make figure wider to accommodate colorbar
    fig, axes = plt.subplots(n_images, n_cols, figsize=(n_cols * 1.5 + 1.0, n_images * 1.5))
    
    # Handle single row case
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    # Compute per-image RMSEs for display
    rmse_per_image = []
    
    for i in range(n_images):
        # Column 0: Original image
        img_orig = y_orig_batch[i].cpu().numpy().reshape(img_dim, img_dim)
        axes[i, 0].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=10)
        
        # Column 1: Masked image (with missing data)
        img_masked = y_batch[i].clone()
        # Set missing values to 0.5 (gray) for better visualization
        img_masked[mask_batch[i]] = 0.5
        # Inverse transform if scaler provided
        if scaler is not None:
            img_masked = scaler.inverse_transform(img_masked.unsqueeze(0)).squeeze(0)
        img_masked = img_masked.cpu().numpy().reshape(img_dim, img_dim)
        axes[i, 1].imshow(img_masked, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Masked\n(Training)', fontsize=10)
        
        # Columns 2 to 2+n_samples-1: Reconstruction samples
        for s in range(n_samples):
            img_recon = reconstructions[s, i].cpu().numpy().reshape(img_dim, img_dim)
            # Clip to valid range [0, 1] for visualization
            img_recon = np.clip(img_recon, 0, 1)
            axes[i, 2 + s].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
            axes[i, 2 + s].axis('off')
            if i == 0:
                axes[i, 2 + s].set_title(f'Sample {s+1}', fontsize=10)
        
        # Column 2+n_samples: Mean reconstruction
        img_mean = y_recon_mean[i].cpu().numpy().reshape(img_dim, img_dim)
        img_mean = np.clip(img_mean, 0, 1)
        axes[i, 2 + n_samples].imshow(img_mean, cmap='gray', vmin=0, vmax=1)
        axes[i, 2 + n_samples].axis('off')
        if i == 0:
            axes[i, 2 + n_samples].set_title('Mean', fontsize=10)
        
        # Compute RMSE for this image (only on missing pixels)
        mask_i = mask_batch[i]
        if mask_i.sum() > 0:
            rmse_i = torch.sqrt(torch.mean((y_orig_batch[i][mask_i] - y_recon_mean[i][mask_i]) ** 2)).item()
            rmse_per_image.append(rmse_i)
            
            # Add Missing RMSE text below mean image
            axes[i, 2 + n_samples].text(0.5, -0.1, f'Missing RMSE: {rmse_i:.3f}', 
                                       transform=axes[i, 2 + n_samples].transAxes,
                                       ha='center', fontsize=8)
        
        # Column 2+n_samples+1: Uncertainty (std of reconstructions)
        img_std = y_recon_std[i].cpu().numpy().reshape(img_dim, img_dim)
        im = axes[i, 2 + n_samples + 1].imshow(img_std, cmap='hot', vmin=0, vmax=img_std.max())
        axes[i, 2 + n_samples + 1].axis('off')
        if i == 0:
            axes[i, 2 + n_samples + 1].set_title('Uncertainty', fontsize=10)
    
    # Add colorbar for uncertainty - place it outside the plot area
    # Create colorbar using the last column (uncertainty column) but place it to the right
    cbar = fig.colorbar(im, ax=axes[:, -1].tolist(), fraction=0.046, pad=0.1, shrink=0.8, aspect=20)
    cbar.set_label('Std Dev', fontsize=10)
    
    # Add overall title
    missing_pct = (mask_batch.sum() / mask_batch.numel() * 100).item()
    mean_rmse = np.mean(rmse_per_image) if rmse_per_image else 0.0
    fig.suptitle(f'MNIST Missing Data Reconstruction (Gauss) ({missing_pct:.1f}% missing, Mean Missing RMSE: {mean_rmse:.4f})', 
                 fontsize=14, y=0.995)
    
    # Use subplots_adjust to leave space for colorbar and avoid tight_layout warning
    plt.subplots_adjust(left=0.05, right=0.88, top=0.93, bottom=0.05, wspace=0.3, hspace=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Detailed missing data reconstruction visualization (Gauss) saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    model.train()
    
    print(f"Missing pixels mean RMSE (first {n_images} images): {mean_rmse:.4f}")
    
    return mean_rmse


