#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLPD Utilities for GP-LVM Models
"""

import torch

def compute_nlpd_nll_based(model, Y_true, X_mean, model_name, loss_data):
    """
    Compute NLPD using NLL-based approach from movie_lens_large_expt.py

    Args:
        model: GP-LVM model
        Y_true: Observations [N, D]
        X_mean: Latent means [N, latent_dim]
        model_name: Type of model ('point', 'map', 'gauss', or other)
        loss_data: Loss list (for point/map/gauss models)

    Returns:
        NLPD value (lower is better)
    """
    model.eval()

    with torch.no_grad():
        N = len(Y_true)

        if model_name in ('point', 'map', 'gauss'):
            # For point, map, and gauss models, use the loss from training/test prediction
            nlpd = loss_data[-1] / N
        else:
            # For other models (like nn_gauss), compute NLL using model prediction
            Y_star = model(X_mean)
            
            # Handle missing data consistently with SSVI implementation
            if torch.isnan(Y_true).any():
                # Create mask for observed values
                missing_mask = torch.isnan(Y_true)
                observed_mask = ~missing_mask
                
                # Check if we have any observed data
                if observed_mask.sum() == 0:
                    # All data is missing - return NaN
                    model.train()
                    return float('nan')
                
                # For nn_gauss models, we need to handle missing data properly
                # The model's log_prob method expects complete data, so we need to
                # compute log probabilities manually with proper masking
                
                # Get predictive distribution parameters
                pred_mean = Y_star.mean.T  # [12, 800] -> [800, 12] to match Y_true
                pred_var = Y_star.variance.T  # [12, 800] -> [800, 12] to match Y_true
                
                # Replace NaN values in Y_true with zeros for computation (they'll be masked out)
                Y_true_clean = torch.where(observed_mask, Y_true, torch.zeros_like(Y_true))
                
                # Compute log probability under Gaussian for observed dimensions only
                log_prob = -0.5 * torch.log(2 * torch.pi * pred_var) - 0.5 * (Y_true_clean - pred_mean)**2 / pred_var
                
                # Mask out missing dimensions and sum over observed dimensions only
                log_prob_masked = log_prob * observed_mask.float()
                log_prob_per_sample = log_prob_masked.sum(dim=-1)  # [N] - sum over observed dimensions
                
                # Compute NLPD (negative log predictive density)
                nlpd = -log_prob_per_sample.mean().item()
            else:
                # Standard computation for complete data
                nlpd = -torch.sum(Y_star.log_prob(Y_true.T)) / N

    model.train()
    return nlpd

