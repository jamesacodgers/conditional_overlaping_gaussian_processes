#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLPD Utilities for GP-LVM Models

This module provides mathematically aligned NLPD computation functions that ensure
fair comparison between SSVI and Original GP-LVM models.

Key functions:
- compute_nlpd_ssvi_style: SSVI-style NLPD computation adapted for original GP-LVM models
- Ensures mathematical equivalence between SSVI and Original GP-LVM implementations

Note: RMSE function is imported from utils.metrics (original implementation)
"""

import torch
import numpy as np


def compute_nlpd_ssvi_style(model, Y_true, X_mean, X_covar=None, likelihood=None, n_samples=50):
    """
    Compute Negative Log Predictive Density using SSVI-style Monte Carlo approach
    adapted for original GP-LVM models.
    
    This function provides mathematically equivalent NLPD computation to the SSVI implementation,
    ensuring fair comparison between original GP-LVM and SSVI models.
    
    Args:
        model: GP-LVM model
        Y_true: True observations [N, D]
        X_mean: Latent means [N, latent_dim] 
        X_covar: Latent covariances [N, latent_dim, latent_dim] (optional, for gauss model)
        likelihood: GP likelihood (optional, for gauss model)
        n_samples: Number of Monte Carlo samples
        
    Returns:
        NLPD value (lower is better)
        
    Example:
        # For gauss model (with diagonal latent uncertainty):
        X_covar = torch.diag_embed(torch.exp(model.X.q_log_sigma)**2)
        nlpd = compute_nlpd_ssvi_style(model, Y_test, X_test_mean, X_covar, likelihood)
        
        # For nn_gauss model (with full covariance matrix):
        X_covar = model.X.sigma(Y_test)  # Full covariance matrix from encoder
        nlpd = compute_nlpd_ssvi_style(model, Y_test, X_test_mean, X_covar, likelihood)
    """
    model.eval()
    if likelihood:
        likelihood.eval()
    
    with torch.no_grad():
        N, D = Y_true.shape
        log_probs = []
        
        print(f"[DEBUG] NLPD: Y_true device: {Y_true.device}")
        print(f"[DEBUG] NLPD: Y_true shape: {Y_true.shape}")
        
        # Ensure X_mean is a tensor (convert from numpy if necessary)
        if not isinstance(X_mean, torch.Tensor):
            X_mean = torch.tensor(X_mean, dtype=torch.float32)
        
        # Ensure X_mean is on the same device as Y_true
        if X_mean.device != Y_true.device:
            print(f"[DEBUG] NLPD: Moving X_mean from {X_mean.device} to {Y_true.device}")
            X_mean = X_mean.to(Y_true.device)
        
        print(f"[DEBUG] NLPD: X_mean device: {X_mean.device}")
        print(f"[DEBUG] NLPD: X_mean shape: {X_mean.shape}")
        
        for _ in range(n_samples):
            if X_covar is not None:
                print(f"[DEBUG] NLPD: X_covar device: {X_covar.device}")
                print(f"[DEBUG] NLPD: X_covar shape: {X_covar.shape}")
                
                # Ensure X_covar is on the same device as Y_true
                if X_covar.device != Y_true.device:
                    print(f"[DEBUG] NLPD: Moving X_covar from {X_covar.device} to {Y_true.device}")
                    X_covar = X_covar.to(Y_true.device)
                
                # Sample from variational posterior q(x) = N(mu_x, sigma_x^2) - SSVI style
                # Use diagonal sampling approach like SSVI for consistency
                if X_covar.dim() == 3:
                    # Extract diagonal elements from full covariance matrix
                    sigma_x = torch.sqrt(torch.diagonal(X_covar, dim1=-2, dim2=-1))  # [N, latent_dim]
                else:
                    # Already diagonal covariance case (gauss model)
                    sigma_x = torch.sqrt(torch.diagonal(X_covar, dim1=-2, dim2=-1))  # [N, latent_dim]
                
                print(f"[DEBUG] NLPD: sigma_x device: {sigma_x.device}")
                print(f"[DEBUG] NLPD: sigma_x shape: {sigma_x.shape}")
                
                # SSVI-style sampling: x_sample = mu_x + sigma_x * epsilon
                epsilon = torch.randn_like(X_mean)
                X_sample = X_mean + sigma_x * epsilon
                
                print(f"[DEBUG] NLPD: X_sample device: {X_sample.device}")
                print(f"[DEBUG] NLPD: X_sample shape: {X_sample.shape}")
            else:
                # Use point estimates (no uncertainty in latent)
                X_sample = X_mean
            
            # Get GP predictive distribution using SSVI-style manual computation
            # Get GP predictive mean and variance separately like SSVI
            Y_star = model(X_sample)
            
            print(f"[DEBUG] NLPD: Y_star.mean shape: {Y_star.mean.shape}")
            print(f"[DEBUG] NLPD: Y_star.variance shape: {Y_star.variance.shape}")
            print(f"[DEBUG] NLPD: Y_true shape: {Y_true.shape}")
            print(f"[DEBUG] NLPD: X_sample shape: {X_sample.shape}")
            
            # Extract mean and variance from GP distribution
            f_mean = Y_star.mean.T  # [N, D] - transpose to match SSVI format
            f_var = Y_star.variance.T  # [N, D] - transpose to match SSVI format
            
            print(f"[DEBUG] NLPD: f_mean shape after transpose: {f_mean.shape}")
            print(f"[DEBUG] NLPD: f_var shape after transpose: {f_var.shape}")
            
            # Add observation noise like SSVI does
            if likelihood is not None:
                # Add likelihood noise variance
                f_var = f_var + likelihood.noise_covar.noise
            
            # Compute log probability using SSVI-style manual Gaussian formula
            log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (Y_true - f_mean)**2 / f_var
            
            # Handle missing data if present (like SSVI)
            if torch.isnan(Y_true).any():
                missing_mask = torch.isnan(Y_true)
                observed_mask = ~missing_mask
                
                # Check if we have any observed data
                if observed_mask.sum() == 0:
                    # All data is missing - return NaN
                    model.train()
                    if likelihood:
                        likelihood.train()
                    return float('nan')
                
                # Replace NaN values in Y_true with zeros for computation (they'll be masked out)
                Y_true_clean = torch.where(observed_mask, Y_true, torch.zeros_like(Y_true))
                
                # Recompute log probability with clean data
                log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (Y_true_clean - f_mean)**2 / f_var
                
                # Mask out missing values in log probability computation
                log_prob_masked = log_prob * observed_mask.float()
                log_probs.append(log_prob_masked.sum(dim=-1))  # Sum over observed dimensions only
            else:
                # Standard computation for complete data
                log_probs.append(log_prob.sum(dim=-1))  # Sum over dimensions
        
        # Average log probabilities (log of mixture) - same as SSVI
        log_probs = torch.stack(log_probs)  # [n_samples, N]
        # Use logsumexp for numerical stability
        log_pred_density = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
        
        # Negative log predictive density (average over batch)
        nlpd = -log_pred_density.mean().cpu().item()
    
    model.train()
    if likelihood:
        likelihood.train()
    
    return nlpd


