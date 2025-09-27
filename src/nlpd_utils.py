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
        model_name: Type of model ('point', 'map', 'gauss', or other)
        loss_data: Loss list (for point/map/gauss models)
        X_covar: Covariance matrices for nn_gauss models [N, latent_dim, latent_dim]
        obs_noise: Observation noise parameter (scalar tensor)

    Returns:
        NLPD value (lower is better)
    """
    model.eval()

    with torch.no_grad():
        N = len(Y_true)

        if model_name in ('point', 'map', 'gauss'):
            # For point, map, and gauss models, use the loss from training/test prediction
            nlpd = loss_data[-1] / N
        elif model_name == 'nn_gauss':
            # This aligns with the SSVI approach for proper uncertainty handling
            try:
                n_samples = 64  # Number of Monte Carlo samples
                log_probs = []
                
                # Check if we have missing data
                missing_mask = torch.isnan(Y_true)
                observed_mask = ~missing_mask
                
                # Check if we have any observed data
                if observed_mask.sum() == 0:
                    # All data is missing - return NaN
                    model.train()
                    return float('nan')
                
                # Replace NaN values in Y_true with zeros for computation (they'll be masked out)
                Y_true_clean = torch.where(observed_mask, Y_true, torch.zeros_like(Y_true))
                
                for _ in range(n_samples):
                    # Sample from latent distribution using univariate approach (like SSVI)
                    # Use same sqrt approach as get_X_scales: [torch.sqrt(x.diag()) for x in X_covar]
                    X_std = torch.stack([torch.sqrt(x.diag()) for x in X_covar])  # [N, latent_dim]
                    normal_dist = torch.distributions.Normal(X_mean, X_std)
                    x_sample = normal_dist.sample()
                    
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
                    
                    # Compute log probability: log N(Y_true_clean | f_mean, f_var_full)
                    # Y_true_clean is [N, D], f_mean is [D, N], so transpose Y_true_clean to [D, N]
                    Y_true_clean_T = Y_true_clean.T  # [D, N]
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (Y_true_clean_T - f_mean)**2 / f_var_full
                    
                    # Mask out missing values in log probability computation
                    observed_mask_T = observed_mask.T  # [D, N]
                    log_prob_masked = log_prob * observed_mask_T.float()
                    
                    # Sum over observed dimensions for each sample
                    log_prob_sum = log_prob_masked.sum(dim=0)  # [N] - sum over D dimensions
                    log_probs.append(log_prob_sum)
                
                # Stack log probabilities for mixture computation
                log_probs = torch.stack(log_probs)  # [n_samples, N]
                
                # Use logsumexp for numerically stable mixture computation
                log_pred_density = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)  # [N]
                
                # Compute NLPD as negative average log predictive density
                nlpd = -log_pred_density.mean().item()
                
            except Exception as e:
                print(f"WARNING: NLPD computation failed for nn_gauss model: {str(e)}")
                print("Falling back to simple loss-based NLPD computation")
                nlpd = loss_data[-1] / N
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    model.train()
    return nlpd

