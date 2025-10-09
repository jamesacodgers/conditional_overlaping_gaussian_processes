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
                                       enable_missing_data: bool = False) -> float:
    """
    Args:
        model: Original GP-LVM model (gauss or nn_gauss)
        y_true: True observations [N, D]
        X_mean: Pre-computed latent means [N, latent_dim]
        X_covar: Pre-computed latent covariances (variances for gauss, cov matrices for nn_gauss)
        model_name: Model type ('gauss' or 'nn_gauss') 
        n_samples: Number of samples for Monte Carlo estimation
        enable_missing_data: Whether missing data handling is enabled
        
    Returns:
        RMSE value
    """
    model.eval()
    with torch.no_grad():
        # Use pre-computed latent variables from main script (already computed in oil_flow_GPLVM.py)
        mu_x = X_mean.detach()
        
        # X_covar processing depends on model type:
        # For gauss: softplus(q_log_sigma) = actual sigma values (standard deviations)
        # For nn_gauss: model.X.sigma() returns a list of covariance matrices
        if model_name == 'nn_gauss':
            # For nn_gauss, X_covar is a list of covariance matrices
            # Extract diagonal standard deviations for each matrix (same as NLPD and original code)
            sigma_x = torch.stack([torch.sqrt(x.diag()) for x in X_covar])
        else:
            # For gauss, X_covar already contains standard deviations
            sigma_x = X_covar
        
        # Sample from latent distribution (same as create_standardized_scattered_plot sampling approach)
        reconstructions = []
        for p in range(n_samples):
            # Sample from all latent dimensions using same approach as create_standardized_scattered_plot
            x_dist = torch.distributions.Normal(mu_x, sigma_x)
            x_sample = x_dist.sample()  # [batch_size, latent_dim]
            
            # Reconstruct through model.reconstruct_y for consistency with original approach
            if model_name == 'nn_gauss':
                # For nn_gauss with missing data, use cleaned input for encoder (same as main script)
                if enable_missing_data:
                    y_for_recon = y_true.clone()
                    y_for_recon[torch.isnan(y_for_recon)] = -999.0
                    y_recon, _ = model.reconstruct_y(x_sample, y_for_recon, ae=True, model_name=model_name)
                else:
                    y_recon, _ = model.reconstruct_y(x_sample, y_true, ae=True, model_name=model_name)
            else:  # gauss model
                y_recon, _ = model.reconstruct_y(x_sample, y_true, ae=False, model_name=model_name)
            
            reconstructions.append(y_recon)
        
        # Average reconstructions over samples 
        y_pred = torch.stack(reconstructions).mean(dim=0)  # [batch_size, D]
        
        # Compute RMSE - handle missing data if needed
        # Transpose y_pred to match y_true format (same as original code: Y_train_recon.T)
        y_pred_T = y_pred.T
        
        if enable_missing_data:
            # Import the improved missing data RMSE function
            from src.missing_data_utils import rmse_missing_improved
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred_T.detach().cpu().numpy()
            rmse = rmse_missing_improved(y_true_np, y_pred_np)
        else:
            # Standard RMSE computation for complete data
            mse = torch.mean((y_true - y_pred_T) ** 2)
            rmse = torch.sqrt(mse).item()
        
    model.train()
    return rmse


def compute_missing_data_rmse_with_sampling(model,
                               Y_original: torch.Tensor,
                               missing_mask: torch.Tensor,
                               X_mean: torch.Tensor,
                               X_covar: Union[torch.Tensor, List[torch.Tensor], None] = None,
                               model_name: str = None,
                               n_samples: int = 10) -> float:
    """
    Compute RMSE specifically for missing data dimensions.
    
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
        
        # Use pre-computed latent variables from main script (same as compute_rmse_with_sampling)
        mu_x = X_mean.detach()

        # X_covar processing depends on model type (same as compute_rmse_with_sampling)
        if model_name == 'nn_gauss':
            sigma_x = torch.stack([torch.sqrt(x.diag()) for x in X_covar])
        else:
            sigma_x = X_covar

        # Create input data with missing values applied
        # For gauss: set missing values to NaN
        # For nn_gauss: set missing values to -999
        y_with_missing = Y_original.clone()
        y_with_missing[missing_mask] = float('nan')
        
        # Sample from latent distribution (same as compute_rmse_with_sampling)
        reconstructions = []
        for p in range(n_samples):
            x_dist = torch.distributions.Normal(mu_x, sigma_x)
            x_sample = x_dist.sample()  # [batch_size, latent_dim]

            # Reconstruct through model.reconstruct_y (same as compute_rmse_with_sampling)
            y_recon, _ = model.reconstruct_y(x_sample, y_with_missing, ae=False, model_name=model_name)

            reconstructions.append(y_recon)

        # Average reconstructions over samples (same as compute_rmse_with_sampling)
        y_pred = torch.stack(reconstructions).mean(dim=0)  # [batch_size, D]

        # Transpose y_pred to match Y_original format (same as compute_rmse_with_sampling)
        y_pred_T = y_pred.T

        # ONLY DIFFERENCE: Compute RMSE only on MISSING values (reverse of observed mask)
        # Get the true values and predictions only for missing positions
        y_true_missing = Y_original[missing_mask]
        y_pred_missing = y_pred_T[missing_mask]
        
        # Compute MSE and then RMSE on missing values only
        mse_missing = torch.mean((y_true_missing - y_pred_missing) ** 2)
        rmse_missing = torch.sqrt(mse_missing).item()

    model.train()
    return rmse_missing