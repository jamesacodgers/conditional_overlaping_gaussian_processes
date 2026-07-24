#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLPD / CRPS Utilities for GP-LVM Models
"""

from math import e
import torch
import numpy as np

from src.ssvi_metrics import (
    gaussian_crps,
    sample_crps,
    gaussian_interval_coverage,
    sample_interval_coverage,
    _CRPS_SIGMA_EPS,
    COVERAGE_LEVELS,
)


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


def compute_crps_nll_based(model, Y_true, X_mean, model_name, X_covar=None, obs_noise=None,
                           enable_missing_data=False, batch_size=256, use_sample_estimator=False):
    """
    Compute CRPS mirroring ``compute_nlpd_nll_based`` for original GP-LVM models.

    Uses the identical predictive path as NLPD: sample x ~ q(x), GP predictive
    mean/variance (+ observation noise), then closed-form Gaussian CRPS on each
    MC component. Aggregation matches NLPD:
      - score each MC draw
      - average over MC samples
      - sum over observed dimensions
      - average over datapoints

    Optional sample estimator: E|X − y| − (1/2) E|X − X′|.
    Lower is better.
    """
    model.eval()

    with torch.no_grad():
        N = len(Y_true)

        if model_name not in ('gauss', 'nn_gauss'):
            raise ValueError(f"Invalid model name: {model_name}")

        try:
            n_samples = 4  # Same as compute_nlpd_nll_based

            total_crps = 0.0
            total_count = 0

            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                current_batch_size = end_idx - start_idx

                Y_batch = Y_true[start_idx:end_idx]
                X_mean_batch = X_mean[start_idx:end_idx]

                if model_name == 'nn_gauss':
                    X_covar_batch = X_covar[start_idx:end_idx]
                    X_std_batch = torch.stack([torch.sqrt(x.diag()) for x in X_covar_batch])
                else:
                    X_std_batch = X_covar[start_idx:end_idx]

                if enable_missing_data:
                    observed_mask = ~torch.isnan(Y_batch)
                    if observed_mask.sum() == 0:
                        continue
                    Y_obs = torch.where(observed_mask, Y_batch, torch.zeros_like(Y_batch, dtype=torch.float64))
                    observed_mask_T = observed_mask.T  # [D, batch]
                else:
                    observed_mask = None
                    Y_obs = Y_batch
                    observed_mask_T = None

                Y_obs_T = Y_obs.T  # [D, batch] — matches GP output layout
                x_dist = torch.distributions.Normal(X_mean_batch, X_std_batch)

                batch_crps = []
                means = []
                vars_ = []

                for _ in range(n_samples):
                    x_sample = x_dist.rsample()
                    Y_star = model(x_sample)
                    f_mean = Y_star.loc  # [D, batch]
                    f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])  # [D, batch]
                    if obs_noise is not None:
                        f_var = f_var + obs_noise

                    if use_sample_estimator:
                        means.append(f_mean)
                        vars_.append(f_var)
                    else:
                        f_std = f_var.clamp(min=_CRPS_SIGMA_EPS).sqrt()
                        crps_vals = gaussian_crps(f_mean, f_std, Y_obs_T)  # [D, batch]
                        if observed_mask_T is not None:
                            crps_sum = (crps_vals * observed_mask_T.float()).sum(dim=0)  # [batch]
                        else:
                            crps_sum = crps_vals.sum(dim=0)
                        batch_crps.append(crps_sum)

                if use_sample_estimator:
                    f_mean_stack = torch.stack(means)
                    f_var_stack = torch.stack(vars_)
                    f_std = f_var_stack.clamp(min=_CRPS_SIGMA_EPS).sqrt()
                    y_pred_samples = torch.distributions.Normal(f_mean_stack, f_std).rsample()
                    crps_vals = sample_crps(y_pred_samples, Y_obs_T)  # [D, batch]
                    if observed_mask_T is not None:
                        crps_per_point = (crps_vals * observed_mask_T.float()).sum(dim=0)
                    else:
                        crps_per_point = crps_vals.sum(dim=0)
                else:
                    # Mirror NLPD: average MC sample scores → [batch]
                    crps_per_point = torch.stack(batch_crps).mean(dim=0)

                total_crps += crps_per_point.sum().item()
                total_count += current_batch_size

            if total_count == 0:
                model.train()
                return float('nan')

            crps = total_crps / total_count

        except Exception as e:
            print(f"WARNING: CRPS computation failed for {model_name} model: {str(e)}")
            crps = float('nan')

    model.train()
    return crps


def compute_coverage_nll_based(model, Y_true, X_mean, model_name, X_covar=None, obs_noise=None,
                               enable_missing_data=False, batch_size=256,
                               levels=COVERAGE_LEVELS, use_sample_quantiles=False):
    """
    Predictive interval coverage mirroring ``compute_nlpd_nll_based``.

    Same predictive path as NLPD (x ~ q(x), GP loc/diag-cov + observation noise).
    Moment-matches MC Gaussians to one predictive N(μ, σ²) per entry (σ² includes
    likelihood noise), builds central intervals at each ``level``, and returns:

        coverage = (1 / N_obs) Σ 1[y ∈ [L, U]]

    over observed entries only. Keys: ``coverage_90``, ``coverage_95``, ...
    """
    model.eval()
    results = {f'coverage_{int(round(level * 100))}': float('nan') for level in levels}

    with torch.no_grad():
        N = len(Y_true)

        if model_name not in ('gauss', 'nn_gauss'):
            raise ValueError(f"Invalid model name: {model_name}")

        try:
            n_samples = 4  # Same as NLPD / CRPS

            totals = {f'coverage_{int(round(level * 100))}': 0.0 for level in levels}
            total_count = 0

            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)

                Y_batch = Y_true[start_idx:end_idx]
                X_mean_batch = X_mean[start_idx:end_idx]

                if model_name == 'nn_gauss':
                    X_covar_batch = X_covar[start_idx:end_idx]
                    X_std_batch = torch.stack([torch.sqrt(x.diag()) for x in X_covar_batch])
                else:
                    X_std_batch = X_covar[start_idx:end_idx]

                if enable_missing_data:
                    observed_mask = ~torch.isnan(Y_batch)
                    if observed_mask.sum() == 0:
                        continue
                    Y_obs = torch.where(observed_mask, Y_batch, torch.zeros_like(Y_batch, dtype=torch.float64))
                else:
                    observed_mask = torch.ones_like(Y_batch, dtype=torch.bool)
                    Y_obs = Y_batch

                Y_obs_T = Y_obs.T
                observed_mask_T = observed_mask.T
                x_dist = torch.distributions.Normal(X_mean_batch, X_std_batch)

                means = []
                vars_ = []
                for _ in range(n_samples):
                    x_sample = x_dist.rsample()
                    Y_star = model(x_sample)
                    f_mean = Y_star.loc
                    f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])
                    if obs_noise is not None:
                        f_var = f_var + obs_noise
                    means.append(f_mean)
                    vars_.append(f_var)

                f_mean_stack = torch.stack(means)
                f_var_stack = torch.stack(vars_)
                f_std_stack = f_var_stack.clamp(min=_CRPS_SIGMA_EPS).sqrt()

                if use_sample_quantiles:
                    y_pred_samples = torch.distributions.Normal(f_mean_stack, f_std_stack).rsample()
                    for level in levels:
                        key = f'coverage_{int(round(level * 100))}'
                        inside = sample_interval_coverage(y_pred_samples, Y_obs_T, level)
                        totals[key] += (inside * observed_mask_T.float()).sum().item()
                else:
                    mu_pred = f_mean_stack.mean(dim=0)
                    var_pred = f_var_stack.mean(dim=0) + f_mean_stack.var(dim=0, unbiased=False)
                    sigma_pred = var_pred.clamp(min=_CRPS_SIGMA_EPS).sqrt()
                    for level in levels:
                        key = f'coverage_{int(round(level * 100))}'
                        inside = gaussian_interval_coverage(mu_pred, sigma_pred, Y_obs_T, level)
                        totals[key] += (inside * observed_mask_T.float()).sum().item()

                total_count += observed_mask.sum().item()

            if total_count == 0:
                model.train()
                return results

            for key in totals:
                results[key] = totals[key] / total_count

        except Exception as e:
            print(f"WARNING: Coverage computation failed for {model_name} model: {str(e)}")

    model.train()
    return results


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