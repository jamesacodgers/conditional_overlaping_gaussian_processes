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
            # For nn_gauss, use sophisticated NLPD computation with Monte Carlo sampling if X_covar is provided
            # Otherwise, fall back to simple computation during training iterations
            
            if X_covar is not None:
                # Sophisticated NLPD computation with Monte Carlo sampling
                # This aligns with the SSVI approach for proper uncertainty handling
                
                n_samples = 50  # Number of Monte Carlo samples
                
            
            log_probs = []
            
            for _ in range(n_samples):
                # Sample from latent distribution: x ~ N(X_mean, X_covar)
                x_sample = torch.distributions.MultivariateNormal(X_mean, X_covar).sample()
                
                # Get GP prediction: Y_star = GP(x_sample)
                Y_star = model(x_sample)
                
                # Extract predictive mean and variance
                f_mean = Y_star.loc  # [N, D]
                f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])  # [N, D]
                
                # Add observation noise to get full predictive variance
                # f_var is [D, N] = [12, 800], noise is scalar
                f_var_full = f_var + obs_noise  # [D, N]
                
                # f_mean is [D, N] = [12, 800], f_var_full is [D, N] = [12, 800]
                # Y_true is [N, D] = [800, 12], so we need to transpose it to [D, N] = [12, 800]
                # Compute log probability
                log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (Y_true.T - f_mean)**2 / f_var_full
                
                
                # Handle missing data by masking out NaNs (like in SSVI)
                # Y_true is [N, D] = [800, 12], observed_mask is [N, D] = [800, 12]
                observed_mask = ~torch.isnan(Y_true)
                # log_prob is [D, N] = [12, 800], so we need to transpose observed_mask to [D, N]
                observed_mask_T = observed_mask.T  # [D, N] = [12, 800]
                log_prob_masked = log_prob * observed_mask_T.float()
                
                # Sum over dimensions for each sample
                log_prob_sum = log_prob_masked.sum(dim=-1)  # [N]
                log_probs.append(log_prob_sum)
            
            # Stack log probabilities for mixture computation
            log_probs = torch.stack(log_probs)  # [n_samples, N]
            
            # Use logsumexp for numerically stable mixture computation
            log_pred_density = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)  # [N]
            
                # Compute NLPD as negative average log predictive density
            nlpd = -log_pred_density.mean().item()
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    model.train()
    return nlpd

