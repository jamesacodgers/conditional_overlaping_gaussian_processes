#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLPD Utilities for GP-LVM Models
"""

from math import e
import torch
import numpy as np


def compute_nlpd_nll_based(model, Y_true, X_mean, model_name, loss_data, X_covar=None, obs_noise=None, enable_missing_data=False, batch_size=256):
    """
    Compute NLPD using NLL-based approach with mini-batching to avoid OOM.

    Args:
        model: GP-LVM model
        Y_true: Observations [N, D]
        X_mean: Latent means [N, latent_dim]
        model_name: Type of model ('point', 'map', 'gauss', or 'nn_gauss')
        loss_data: Loss list (for point/map models, fallback for gauss/nn_gauss)
        X_covar: Latent uncertainties - standard deviations for gauss [N, latent_dim], 
                covariance matrices for nn_gauss [N, latent_dim, latent_dim]
        obs_noise: Observation noise parameter (scalar tensor)
        enable_missing_data: Whether missing data handling is enabled
        batch_size: Batch size for mini-batch processing

    Returns:
        NLPD value (lower is better)
    """
    model.eval()

    with torch.no_grad():
        N = len(Y_true)

        if model_name in ('gauss', 'nn_gauss'):
            try:
                n_samples = 4  # Reduced for faster metrics computation
                
                # Accumulators for batch-wise computation
                total_log_prob = 0.0
                total_count = 0
                
                for start_idx in range(0, N, batch_size):
                    end_idx = min(start_idx + batch_size, N)
                    current_batch_size = end_idx - start_idx
                    
                    # Get batch data
                    Y_batch = Y_true[start_idx:end_idx]
                    X_mean_batch = X_mean[start_idx:end_idx]
                    
                    # X_covar processing depends on model type
                    if model_name == 'nn_gauss':
                        X_covar_batch = X_covar[start_idx:end_idx]
                        X_std_batch = torch.stack([torch.sqrt(x.diag()) for x in X_covar_batch])
                    else:  # gauss model
                        X_std_batch = X_covar[start_idx:end_idx]
                    
                    x_dist = torch.distributions.Normal(X_mean_batch, X_std_batch)
                    
                    # Sample multiple times and average
                    batch_log_probs = []
                    for _ in range(n_samples):
                        x_sample = x_dist.rsample()  # [batch_size, latent_dim]
                        
                        # Get GP prediction: Y_star = GP(x_sample)
                        Y_star = model(x_sample)

                        # Extract predictive mean and variance
                        f_mean = Y_star.loc  # [D, batch_size] - GP output is transposed
                        f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])  # [D, batch_size]

                        # Add observation noise to get full predictive variance
                        if obs_noise is not None:
                            f_var_full = f_var + obs_noise
                        else:
                            f_var_full = f_var

                        # Handle missing data if enabled
                        if enable_missing_data:
                            missing_mask_batch = torch.isnan(Y_batch)
                            observed_mask_batch = ~missing_mask_batch
                            
                            if observed_mask_batch.sum() == 0:
                                continue
                            
                            Y_batch_clean = torch.where(observed_mask_batch, Y_batch, torch.zeros_like(Y_batch, dtype=torch.float64))
                            Y_batch_clean_T = Y_batch_clean.T  # [D, batch_size]
                            log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (
                                        Y_batch_clean_T - f_mean) ** 2 / f_var_full
                            
                            observed_mask_T = observed_mask_batch.T  # [D, batch_size]
                            log_prob_masked = log_prob * observed_mask_T.float()
                            log_prob_sum = log_prob_masked.sum(dim=0)  # [batch_size]
                            batch_log_probs.append(log_prob_sum)
                        else:
                            Y_batch_T = Y_batch.T  # [D, batch_size]
                            log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (
                                        Y_batch_T - f_mean) ** 2 / f_var_full
                            log_prob_sum = log_prob.sum(dim=0)  # [batch_size]
                            batch_log_probs.append(log_prob_sum)

                    if batch_log_probs:
                        # Stack and average over samples for this batch
                        batch_log_probs = torch.stack(batch_log_probs)  # [n_samples, batch_size]
                        log_pred_density_batch = batch_log_probs.mean(dim=0)  # [batch_size]
                        total_log_prob += log_pred_density_batch.sum().item()
                        total_count += current_batch_size

                # Compute final NLPD
                if total_count == 0:
                    model.train()
                    return float('nan')
                
                nlpd = -total_log_prob / total_count

            except Exception as e:
                print(f"WARNING: NLPD computation failed for {model_name} model: {str(e)}")
                print("Falling back to simple loss-based NLPD computation")
                nlpd = float('nan')
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    model.train()
    return nlpd


def compute_missing_data_nlpd_gauss(model, Y_original, missing_mask, X_mean, X_std, obs_noise, n_samples=50, batch_size=256):
    """
    Compute NLPD specifically for missing data dimensions for gauss model using mini-batching.
    
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
        batch_size: Batch size for mini-batch processing
        
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
        
        N = Y_original.shape[0]
        
        # Accumulators for batch-wise computation
        total_log_prob = 0.0
        total_count = 0
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            current_batch_size = end_idx - start_idx
            
            # Get batch data
            Y_batch = Y_original[start_idx:end_idx]
            mask_batch = missing_mask[start_idx:end_idx]
            X_mean_batch = X_mean[start_idx:end_idx]
            X_std_batch = X_std[start_idx:end_idx]
            
            # Sample from latent variational distribution and compute log probabilities
            batch_log_probs = []
            
            for _ in range(n_samples):
                # Sample from latent distribution q(x) = N(mu_x, sigma_x)
                x_dist = torch.distributions.Normal(X_mean_batch, X_std_batch)
                x_sample = x_dist.rsample()  # [batch_size, latent_dim]
                
                # Get GP prediction: Y_star = GP(x_sample)
                Y_star = model(x_sample)
                
                # Extract predictive mean and variance
                f_mean = Y_star.loc  # [D, batch_size] - GP output is transposed
                f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])  # [D, batch_size]
                
                # Add observation noise to get full predictive variance
                f_var_full = f_var + obs_noise  # [D, batch_size]
                
                # Compute log probability under Gaussian
                Y_batch_T = Y_batch.T  # [D, batch_size]
                log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (
                            Y_batch_T - f_mean) ** 2 / f_var_full
                
                # Apply missing data mask to only include missing dimensions
                mask_batch_T = mask_batch.T  # [D, batch_size]
                log_prob_masked = log_prob * mask_batch_T.float()
                
                # Sum over missing dimensions for each sample
                log_prob_sum = log_prob_masked.sum(dim=0)  # [batch_size] - sum over D dimensions
                batch_log_probs.append(log_prob_sum)
            
            if batch_log_probs:
                # Stack log probabilities for averaging
                batch_log_probs = torch.stack(batch_log_probs)  # [n_samples, batch_size]
                
                # Average log probabilities across samples
                log_pred_density_batch = batch_log_probs.mean(dim=0)  # [batch_size]
                total_log_prob += log_pred_density_batch.sum().item()
                total_count += current_batch_size
        
        # Compute final NLPD
        if total_count == 0:
            model.train()
            return float('nan')
        
        nlpd = -total_log_prob / total_count
        
    model.train()
    return nlpd