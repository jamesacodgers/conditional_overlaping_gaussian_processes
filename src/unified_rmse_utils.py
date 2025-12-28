#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified RMSE computation utilities for Original GP-LVM models

This module provides a unified RMSE computation function that aligns with the SSVI approach
by sampling from the latent distribution and handling both missing and non-missing data scenarios.
"""

import torch
import numpy as np
from typing import Optional, Union, List


def compute_rmse_with_sampling(model, 
                                       y_true: torch.Tensor, 
                                       X_mean: torch.Tensor,
                                       X_covar: Union[torch.Tensor, List[torch.Tensor], None] = None,
                                       model_name: str = None,
                                       n_samples: int = 10,
                                       enable_missing_data: bool = False,
                                       batch_size: int = 256) -> float:
    """
    Compute RMSE using mini-batching to avoid OOM.
    
    Args:
        model: Original GP-LVM model (gauss or nn_gauss)
        y_true: True observations [N, D]
        X_mean: Pre-computed latent means [N, latent_dim]
        X_covar: Pre-computed latent covariances (variances for gauss, cov matrices for nn_gauss)
        model_name: Model type ('gauss' or 'nn_gauss') 
        n_samples: Number of samples for Monte Carlo estimation
        enable_missing_data: Whether missing data handling is enabled
        batch_size: Batch size for mini-batch processing
        
    Returns:
        RMSE value
    """
    model.eval()
    with torch.no_grad():
        N = y_true.shape[0]
        
        # Accumulators for batch-wise computation
        total_squared_error = 0.0
        total_count = 0
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            
            # Get batch data
            y_batch = y_true[start_idx:end_idx]
            mu_x_batch = X_mean[start_idx:end_idx].detach()
            
            # X_covar processing depends on model type:
            if model_name == 'nn_gauss':
                # For nn_gauss, X_covar is a list of covariance matrices
                X_covar_batch = X_covar[start_idx:end_idx]
                sigma_x_batch = torch.stack([torch.sqrt(x.diag()) for x in X_covar_batch])
            else:
                # For gauss, X_covar already contains standard deviations
                sigma_x_batch = X_covar[start_idx:end_idx]
            
            # Sample from latent distribution and reconstruct
            reconstructions = []
            for p in range(n_samples):
                x_dist = torch.distributions.Normal(mu_x_batch, sigma_x_batch)
                x_sample = x_dist.sample()  # [batch_size, latent_dim]
                
                # Reconstruct through model.reconstruct_y
                if model_name == 'nn_gauss':
                    if enable_missing_data:
                        y_for_recon = y_batch.clone()
                        y_for_recon[torch.isnan(y_for_recon)] = -999.0
                        y_recon, _ = model.reconstruct_y(x_sample, y_for_recon, ae=True, model_name=model_name)
                    else:
                        y_recon, _ = model.reconstruct_y(x_sample, y_batch, ae=True, model_name=model_name)
                else:  # gauss model - call model directly to avoid covariance_matrix materialization
                    y_pred = model(x_sample)
                    y_recon = y_pred.loc  # [D, batch_size] - just the mean, no covariance
                
                reconstructions.append(y_recon)
            
            # Average reconstructions over samples 
            y_pred = torch.stack(reconstructions).mean(dim=0)  # [D, batch_size]
            y_pred_T = y_pred.T  # [batch_size, D]
            
            # Compute squared errors for this batch
            if enable_missing_data:
                missing_mask_batch = torch.isnan(y_batch)
                observed_mask_batch = ~missing_mask_batch
                
                if observed_mask_batch.sum() > 0:
                    y_batch_clean = torch.where(observed_mask_batch, y_batch, torch.zeros_like(y_batch))
                    squared_errors = (y_batch_clean - y_pred_T) ** 2
                    observed_squared_errors = squared_errors * observed_mask_batch.float()
                    total_squared_error += observed_squared_errors.sum().item()
                    total_count += observed_mask_batch.sum().item()
            else:
                # Standard computation for complete data
                squared_errors = (y_batch - y_pred_T) ** 2
                total_squared_error += squared_errors.sum().item()
                total_count += y_batch.numel()
        
        # Compute final RMSE
        if total_count == 0:
            model.train()
            return float('nan')
        
        mse = total_squared_error / total_count
        rmse = np.sqrt(mse)
        
    model.train()
    return rmse


def compute_missing_data_rmse_with_sampling(model,
                               Y_original: torch.Tensor,
                               missing_mask: torch.Tensor,
                               X_mean: torch.Tensor,
                               X_covar: Union[torch.Tensor, List[torch.Tensor], None] = None,
                               model_name: str = None,
                               n_samples: int = 10,
                               batch_size: int = 256) -> float:
    """
    Compute RMSE specifically for missing data dimensions using mini-batching.
    
    Does everything exactly the same as compute_rmse_with_sampling,
    but at the end computes RMSE only on missing values (reverse of observed mask).
    
    Args:
        model: Original GP-LVM model (gauss or nn_gauss)
        Y_original: Original complete data before missing data preprocessing [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
        X_mean: Pre-computed latent means [N, latent_dim]
        X_covar: Pre-computed latent covariances (variances for gauss, cov matrices for nn_gauss)
        model_name: Model type ('gauss' or 'nn_gauss')
        n_samples: Number of samples for Monte Carlo estimation
        batch_size: Batch size for mini-batch processing

    Returns:
        RMSE value for missing data dimensions only
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
        total_squared_error = 0.0
        total_missing_count = 0
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            
            # Get batch data
            y_batch = Y_original[start_idx:end_idx]
            mask_batch = missing_mask[start_idx:end_idx]
            mu_x_batch = X_mean[start_idx:end_idx].detach()
            
            # X_covar processing depends on model type
            if model_name == 'nn_gauss':
                X_covar_batch = X_covar[start_idx:end_idx]
                sigma_x_batch = torch.stack([torch.sqrt(x.diag()) for x in X_covar_batch])
            else:
                sigma_x_batch = X_covar[start_idx:end_idx]

            # Create input data with missing values applied
            y_with_missing = y_batch.clone()
            y_with_missing[mask_batch] = float('nan')
            
            # Sample from latent distribution
            reconstructions = []
            for p in range(n_samples):
                x_dist = torch.distributions.Normal(mu_x_batch, sigma_x_batch)
                x_sample = x_dist.sample()  # [batch_size, latent_dim]

                # Reconstruct through model.reconstruct_y
                y_recon, _ = model.reconstruct_y(x_sample, y_with_missing, ae=False, model_name=model_name)
                reconstructions.append(y_recon)

            # Average reconstructions over samples
            y_pred = torch.stack(reconstructions).mean(dim=0)  # [D, batch_size]
            y_pred_T = y_pred.T  # [batch_size, D]

            # Compute squared errors only on MISSING values
            y_true_missing = y_batch[mask_batch]
            y_pred_missing = y_pred_T[mask_batch]
            
            squared_errors = (y_true_missing - y_pred_missing) ** 2
            total_squared_error += squared_errors.sum().item()
            total_missing_count += mask_batch.sum().item()
        
        # Compute final RMSE
        if total_missing_count == 0:
            model.train()
            return float('nan')
        
        mse_missing = total_squared_error / total_missing_count
        rmse_missing = np.sqrt(mse_missing)

    model.train()
    return rmse_missing