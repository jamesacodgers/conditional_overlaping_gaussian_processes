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


def compute_unified_rmse(model, 
                                 y_true: torch.Tensor, 
                                 model_name: str,
                                 n_samples: int = 10,
                                 enable_missing_data: bool = False,
                                 Y_train: Optional[torch.Tensor] = None) -> float:
    """
    Improved version of unified RMSE computation with better missing data handling
    
    This version uses the same approach as rmse_missing_improved for better
    missing data reconstruction evaluation.
    
    Args:
        model: Original GP-LVM model
        y_true: True observations [N, D]
        model_name: Model type ('gauss' or 'nn_gauss')
        n_samples: Number of samples for Monte Carlo estimation
        enable_missing_data: Whether missing data handling is enabled
        Y_train: Training data (required for nn_gauss model)
        
    Returns:
        RMSE value
    """
    model.eval()
    with torch.no_grad():
        # Get latent variables based on model type (same as above)
        if model_name == 'gauss':
            mu_x = model.X.q_mu
            sigma_x = torch.nn.functional.softplus(model.X.q_log_sigma)
            
        elif model_name == 'nn_gauss':
            if Y_train is None:
                raise ValueError("Y_train must be provided for nn_gauss model")
            
            if enable_missing_data and torch.isnan(Y_train).any():
                Y_train_for_encoder = Y_train.clone()
                Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -999.0
            else:
                Y_train_for_encoder = Y_train
            
            mu_x_result = model.X.mu(Y_train_for_encoder)
            sigma_covariances = model.X.sigma(Y_train_for_encoder)
            mu_x = mu_x_result
            sigma_x = torch.stack([torch.sqrt(x.diag()) for x in sigma_covariances])
            
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Must be 'gauss' or 'nn_gauss'")
        
        # Sample from latent distribution for reconstruction
        reconstructions = []
        for i in range(n_samples):
            x_dist = torch.distributions.Normal(mu_x, sigma_x)
            x_sample = x_dist.sample()
            
            # Use model.reconstruct_y() for consistency with original approach
            if model_name == 'nn_gauss':
                y_recon, _ = model.reconstruct_y(x_sample, Y_train, ae=True, model_name=model_name)
            else:
                y_recon, _ = model.reconstruct_y(x_sample, Y_train, ae=False, model_name=model_name)
            
            reconstructions.append(y_recon)
        
        y_pred = torch.stack(reconstructions).mean(dim=0)
        
        # Use improved missing data RMSE computation
        if enable_missing_data:
            # Convert to numpy for rmse_missing_improved function
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            
            # Import the improved missing data RMSE function
            from src.missing_data_utils import rmse_missing_improved
            rmse = rmse_missing_improved(y_true_np, y_pred_np)
        else:
            # Standard RMSE computation for complete data
            mse = torch.mean((y_true - y_pred) ** 2)
            rmse = torch.sqrt(mse).item()
        
    model.train()
    return rmse


def compute_rmse_with_sampling(model, 
                                       y_true: torch.Tensor, 
                                       X_mean: torch.Tensor,
                                       X_covar: Union[torch.Tensor, List[torch.Tensor], None] = None,
                                       model_name: str = None,
                                       n_samples: int = 10,
                                       enable_missing_data: bool = False,
                                       Y_train: Optional[torch.Tensor] = None) -> float:
    """
    Args:
        model: Original GP-LVM model (gauss or nn_gauss)
        y_true: True observations [N, D]
        X_mean: Pre-computed latent means [N, latent_dim]
        X_covar: Pre-computed latent covariances (variances for gauss, cov matrices for nn_gauss)
        model_name: Model type ('gauss' or 'nn_gauss') 
        n_samples: Number of samples for Monte Carlo estimation
        enable_missing_data: Whether missing data handling is enabled
        Y_train: Training data (required for nn_gauss model)
        
    Returns:
        RMSE value
    """
    model.eval()
    with torch.no_grad():
        # Use pre-computed latent variables from main script (already computed in oil_flow_GPLVM.py)
        mu_x = X_mean.detach()
        
        # X_covar comes from model.get_X_scales() - already processed correctly:
        # For gauss: softplus(q_log_sigma) = actual sigma values
        # For nn_gauss: [torch.sqrt(x.diag()) for x in sigma_matrices] = standard deviations
        sigma_x = X_covar
        
        # Sample from latent distribution (same as create_standardized_scattered_plot sampling approach)
        reconstructions = []
        for p in range(n_samples):
            # Sample from all latent dimensions using same approach as create_standardized_scattered_plot
            x_dist = torch.distributions.Normal(mu_x, sigma_x)
            x_sample = x_dist.sample()  # [batch_size, latent_dim]
            
            # Reconstruct through model.reconstruct_y for consistency with original approach
            if model_name == 'nn_gauss':
                y_recon, _ = model.reconstruct_y(x_sample, Y_train, ae=True, model_name=model_name)
            else:  # gauss model
                y_recon, _ = model.reconstruct_y(x_sample, Y_train, ae=False, model_name=model_name)
            
            reconstructions.append(y_recon)
        
        # Average reconstructions over samples 
        y_pred = torch.stack(reconstructions).mean(dim=0)  # [batch_size, D]
        
        # Compute RMSE - handle missing data if needed
        if enable_missing_data:
            # Import the improved missing data RMSE function
            from src.missing_data_utils import rmse_missing_improved
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            rmse = rmse_missing_improved(y_true_np, y_pred_np)
        else:
            # Standard RMSE computation for complete data
            mse = torch.mean((y_true - y_pred) ** 2)
            rmse = torch.sqrt(mse).item()
        
    model.train()
    return rmse