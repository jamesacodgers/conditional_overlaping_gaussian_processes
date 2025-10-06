#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLPD Utilities for GP-LVM Models
"""

from math import e
import torch
import numpy as np


def compute_nlpd_nll_based(model, Y_true, X_mean, model_name, loss_data, X_covar=None, obs_noise=None):
    """
    Compute NLPD using NLL-based approach from movie_lens_large_expt.py

    Args:
        model: GP-LVM model
        Y_true: Observations [N, D]
        X_mean: Latent means [N, latent_dim]
        model_name: Type of model ('point', 'map', 'gauss', or 'nn_gauss')
        loss_data: Loss list (for point/map models, fallback for gauss/nn_gauss)
        X_covar: Latent uncertainties - standard deviations for gauss [N, latent_dim], 
                covariance matrices for nn_gauss [N, latent_dim, latent_dim]
        obs_noise: Observation noise parameter (scalar tensor)

    Returns:
        NLPD value (lower is better)
    """
    model.eval()

    with torch.no_grad():
        N = len(Y_true)

        if model_name in ('gauss', 'nn_gauss'):
            # Aligned with SSVI approach for proper uncertainty handling for both models
            try:
                n_samples = 64  # Match SSVI default number of samples
                
                # Sample from latent distribution using same approach as SSVI
                # X_covar processing depends on model type:
                # For gauss: X_covar already contains standard deviations
                # For nn_gauss: X_covar is a list of covariance matrices, need sqrt(diag())
                if model_name == 'nn_gauss':
                    X_std = torch.stack([torch.sqrt(x.diag()) for x in X_covar])  # [N, latent_dim]
                else:  # gauss model
                    X_std = X_covar  # Already standard deviations
                
                x_dist = torch.distributions.Normal(X_mean, X_std)
                
                # Sample multiple times and average (match SSVI approach)
                log_probs = []
                for _ in range(n_samples):
                    x_sample = x_dist.rsample()  # [N, latent_dim] - single sample like SSVI
                    
                    # Get GP prediction: Y_star = GP(x_sample)
                    Y_star = model(x_sample)

                    # Extract predictive mean and variance
                    f_mean = Y_star.loc  # [D, N] - GP output is transposed
                    f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])  # [D, N]

                    # Add observation noise to get full predictive variance
                    if obs_noise is not None:
                        f_var_full = f_var + obs_noise  # [D, N]
                    else:
                        f_var_full = f_var

                    # Handle missing data if enabled (match SSVI approach)
                    if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                        # Check if we have missing data
                        missing_mask = torch.isnan(Y_true)
                        observed_mask = ~missing_mask
                        
                        # Check if we have any observed data
                        if observed_mask.sum() == 0:
                            # All data is missing - return NaN
                            model.train()
                            return float('nan')
                        
                        # Replace NaN values in Y_true with zeros for computation (they'll be masked out)
                        Y_true_clean = torch.where(observed_mask, Y_true, torch.zeros_like(Y_true, dtype=torch.float64))
                        
                        # Compute log probability: log N(Y_true_clean | f_mean, f_var_full)
                        # Y_true_clean is [N, D], f_mean is [D, N], so transpose Y_true_clean to [D, N]
                        Y_true_clean_T = Y_true_clean.T  # [D, N]
                        log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (
                                    Y_true_clean_T - f_mean) ** 2 / f_var_full
                        
                        # Mask out missing values in log probability computation
                        observed_mask_T = observed_mask.T  # [D, N]
                        log_prob_masked = log_prob * observed_mask_T.float()
                        
                        # Sum over observed dimensions for each sample (match SSVI approach)
                        log_prob_sum = log_prob_masked.sum(dim=0)  # [N] - sum over D dimensions
                        log_probs.append(log_prob_sum)
                    else:
                        # Standard computation for complete data (match SSVI approach)
                        # Y_true is [N, D], f_mean is [D, N], so transpose Y_true to [D, N]
                        Y_true_T = Y_true.T  # [D, N]
                        log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (
                                    Y_true_T - f_mean) ** 2 / f_var_full
                        
                        # Sum over dimensions for each sample (match SSVI approach)
                        log_prob_sum = log_prob.sum(dim=0)  # [N] - sum over D dimensions
                        log_probs.append(log_prob_sum)

                # Stack log probabilities for averaging (match SSVI approach)
                log_probs = torch.stack(log_probs)  # [n_samples, N]

                # Average log probabilities directly (match SSVI approach)
                log_pred_density = log_probs.mean(dim=0)  # [N] - average over samples

                # Compute NLPD as negative average log predictive density (match SSVI)
                nlpd = -log_pred_density.mean().item()

            except Exception as e:
                print(f"WARNING: NLPD computation failed for {model_name} model: {str(e)}")
                print("Falling back to simple loss-based NLPD computation")
                nlpd = loss_data[-1] / N
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    model.train()
    return nlpd


def compute_missing_data_nlpd_gauss(model, Y_original, missing_mask, X_mean, X_std, obs_noise, n_samples=50):
    """
    Compute NLPD specifically for missing data dimensions for gauss model.
    
    This implements the proper NLPD computation for gauss model with missing data:
    1. Create Y_missing: original data with observed values set to NaN
    2. Sample from latent variational distribution q(x) = N(mu_x, sigma_x)
    3. Get GP predictions for those samples
    4. Compute log probability under GP predictive distribution
    5. Apply missing data mask to only evaluate missing dimensions
    
    Args:
        model: GP-LVM model
        Y_original: Original complete data before missing data preprocessing [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
        X_mean: Latent means [N, latent_dim]
        X_std: Latent standard deviations [N, latent_dim]
        obs_noise: Observation noise parameter (scalar tensor)
        n_samples: Number of samples for Monte Carlo estimation
        
    Returns:
        NLPD value for missing data dimensions only
    """
    model.eval()
    
    with torch.no_grad():
        # Check if we have any missing dimensions to evaluate
        if missing_mask.sum() == 0:
            # No missing data - return NaN
            model.train()
            return float('nan')
        
        # Create Y_missing: original data with observed values set to NaN
        # This way, only the originally missing dimensions will have non-NaN values
        Y_missing = Y_original.clone()
        Y_missing[~missing_mask] = float('nan')  # Set observed values to NaN
        
        N, D = Y_missing.shape
        
        # Sample from latent variational distribution and compute log probabilities
        log_probs = []
        
        for _ in range(n_samples):
            # Sample from latent distribution q(x) = N(mu_x, sigma_x)
            x_dist = torch.distributions.Normal(X_mean, X_std)
            x_sample = x_dist.rsample()  # [N, latent_dim]
            
            # Get GP prediction: Y_star = GP(x_sample)
            Y_star = model(x_sample)
            
            # Extract predictive mean and variance
            f_mean = Y_star.loc  # [D, N] - GP output is transposed
            f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])  # [D, N]
            
            # Add observation noise to get full predictive variance
            f_var_full = f_var + obs_noise  # [D, N]
            
            # Compute log probability: log N(Y_missing | f_mean, f_var_full)
            # Y_missing is [N, D], f_mean is [D, N], so transpose Y_missing to [D, N]
            Y_missing_T = Y_missing.T  # [D, N]
            
            # Replace NaN values with zeros for computation (they'll be masked out)
            Y_missing_clean = torch.where(torch.isfinite(Y_missing_T), Y_missing_T, torch.zeros_like(Y_missing_T, dtype=torch.float64))
            
            # Compute log probability under Gaussian
            log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (
                        Y_missing_clean - f_mean) ** 2 / f_var_full
            
            # Apply missing data mask to only include missing dimensions
            missing_mask_T = missing_mask.T  # [D, N]
            log_prob_masked = log_prob * missing_mask_T.float()
            
            # Sum over missing dimensions for each sample
            log_prob_sum = log_prob_masked.sum(dim=0)  # [N] - sum over D dimensions
            log_probs.append(log_prob_sum)
        
        # Stack log probabilities for averaging
        log_probs = torch.stack(log_probs)  # [n_samples, N]
        
        # Average log probabilities across samples
        log_pred_density = log_probs.mean(dim=0)  # [N] - average over samples
        
        # Compute NLPD as negative average log predictive density
        nlpd = -log_pred_density.mean().item()
        
    model.train()
    return nlpd