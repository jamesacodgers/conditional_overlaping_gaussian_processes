#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint utilities for Original GP LVM experiments
"""

import torch
import numpy as np
import os
import json
from typing import Tuple, List
from .nlpd_utils import compute_nlpd_nll_based

# Import rmse function - need to handle the path properly
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))
from utils.metrics import rmse


def compute_checkpoint_metrics(model, model_name: str, Y_train: torch.Tensor, Y_test: torch.Tensor,
                             likelihood, ae: bool, prior_x_test=None, config=None, 
                             losses_train=None, losses_test=None) -> Tuple[float, float, float, float]:
    """
    Compute RMSE and NLPD metrics for train and test data at a checkpoint
    
    Args:
        model: The trained model
        model_name: Name of the model ('gauss' or 'nn_gauss')
        Y_train: Training data
        Y_test: Test data
        likelihood: Model likelihood
        ae: Whether using autoencoder (True for nn_gauss)
        prior_x_test: Prior for test latent variables
        config: Configuration object
        losses_train: Training loss list (optional, for point/map/gauss models)
        losses_test: Test loss list (optional, for point/map/gauss models)
        
    Returns:
        Tuple of (train_rmse, test_rmse, train_nlpd, test_nlpd)
    """
    with torch.no_grad():
        model.eval()
        likelihood.eval()
        
        # Ensure encoder is in eval mode for nn_gauss
        if hasattr(model.X, 'eval'):
            model.X.eval()
        
        # Get training latent variables and reconstructions
        X_train_mean = model.get_X_mean(Y_train)
        X_train_scales = model.get_X_scales(Y_train)
        
        # Compute training reconstruction
        # Ensure X_train_mean is on correct device and dtype (it's already a tensor)
        X_train_mean_device = X_train_mean.to(dtype=torch.float32) if X_train_mean.dtype != torch.float32 else X_train_mean
        Y_train_recon, Y_train_pred_covar = model.reconstruct_y(
            X_train_mean_device, Y_train, ae=ae, model_name=model_name)
        
        # Compute training RMSE
        train_rmse = rmse(Y_train, Y_train_recon.T)
        
        # Compute training NLPD using NLL-based approach
        train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean_device, model_name, losses_train)
        
        # Get test latent variables
        if ae is True:
            if model_name == 'nn_gauss':
                # Store original jitter
                original_jitter = model.X.jitter
                # Create jitter for test data size
                latent_dim = X_train_mean.shape[1]
                test_jitter = torch.eye(latent_dim).unsqueeze(0) * 1e-5
                model.X.jitter = torch.cat([test_jitter for i in range(len(Y_test))], axis=0)
                
                # Call encoder directly
                X_test_mean = model.X.mu(Y_test)
                X_test_covar = model.X.sigma(Y_test)
                
                # Restore original jitter
                model.X.jitter = original_jitter
            else:
                # For other autoencoder models, use predict_latent
                X_test_mean, X_test_covar = model.predict_latent(Y_train, Y_test, 
                                                                 0.001, likelihood, config.random_seed,
                                                                 prior_x=prior_x_test, ae=True, 
                                                                 model_name='nn_gauss', pca=config.pca_init)
        else:
            # For variational models, use predict_latent
            losses_test, X_test = model.predict_latent(Y_train, Y_test, 0.001, 
                                                      likelihood, config.random_seed, 
                                                      prior_x=prior_x_test, ae=ae, 
                                                      model_name=model_name, pca=config.pca_init, 
                                                      steps=config.test_iterations)
            
            if model_name in ('point', 'map'):
                X_test_mean = X_test.X
            elif model_name == 'gauss':
                X_test_mean = X_test.q_mu.detach()
        
        # Compute test reconstruction
        # Ensure X_test_mean is on correct device and dtype (it's already a tensor)
        X_test_mean_device = X_test_mean.to(dtype=torch.float32) if X_test_mean.dtype != torch.float32 else X_test_mean
        Y_test_recon, Y_test_pred_covar = model.reconstruct_y(
            X_test_mean_device, Y_test, ae=ae, model_name=model_name)
        
        # Compute test RMSE
        test_rmse = rmse(Y_test, Y_test_recon.T)
        
        # Compute test NLPD using NLL-based approach
        test_nlpd = compute_nlpd_nll_based(model, Y_test, X_test_mean_device, model_name, losses_test)
        
        # Convert tensors to floats
        train_rmse_val = float(train_rmse.item()) if hasattr(train_rmse, 'item') else float(train_rmse)
        test_rmse_val = float(test_rmse.item()) if hasattr(test_rmse, 'item') else float(test_rmse)
        train_nlpd_val = float(train_nlpd.item()) if hasattr(train_nlpd, 'item') else float(train_nlpd)
        test_nlpd_val = float(test_nlpd.item()) if hasattr(test_nlpd, 'item') else float(test_nlpd)
        
        return train_rmse_val, test_rmse_val, train_nlpd_val, test_nlpd_val


def save_checkpoint(model, model_name: str, iteration: int, Y_train: torch.Tensor, Y_test: torch.Tensor,
                   likelihood, ae: bool, config, logger, prior_x_test=None, losses_train=None, losses_test=None):
    """
    Save checkpoint metrics and plot
    
    Args:
        model: The trained model
        model_name: Name of the model
        iteration: Current iteration number
        Y_train: Training data
        Y_test: Test data
        likelihood: Model likelihood
        ae: Whether using autoencoder
        config: Configuration object
        logger: Results logger
        prior_x_test: Prior for test latent variables
        losses_train: Training loss list (optional, for point/map/gauss models)
        losses_test: Test loss list (optional, for point/map/gauss models)
    """
    if logger and logger.get_results_dir():
        # Compute metrics
        train_rmse, test_rmse, train_nlpd, test_nlpd = compute_checkpoint_metrics(
            model, model_name, Y_train, Y_test, likelihood, ae, prior_x_test, config, losses_train, losses_test)
        
        # Save metrics to JSON
        logger.save_checkpoint_metrics(model_name, config.random_seed, iteration,
                                     train_rmse, test_rmse, train_nlpd, test_nlpd)
        
        # Load existing checkpoint data to build history
        checkpoint_file = os.path.join(logger.get_results_dir(), "training_results", f"{model_name}_{config.random_seed}_checkpoints.json")
        
        train_rmse_history = []
        test_rmse_history = []
        train_nlpd_history = []
        test_nlpd_history = []
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Build history from existing checkpoints
            for iter_key in sorted(checkpoint_data.keys(), key=int):
                iter_data = checkpoint_data[iter_key]
                train_rmse_history.append(iter_data['train_rmse'])
                test_rmse_history.append(iter_data['test_rmse'])
                train_nlpd_history.append(iter_data['train_nlpd'])
                test_nlpd_history.append(iter_data['test_nlpd'])
        
        # Add current checkpoint
        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)
        train_nlpd_history.append(train_nlpd)
        test_nlpd_history.append(test_nlpd)
        
        # Save plot with complete history
        logger.save_checkpoint_plot(model_name, config.random_seed, iteration,
                                   train_rmse_history, test_rmse_history,
                                   train_nlpd_history, test_nlpd_history)
        
        # Checkpoint saved (removed print to avoid interfering with tqdm progress bar)


def save_final_checkpoint_results(model, model_name: str, seed: int, final_iteration: int,
                                train_rmse: float, test_rmse: float, train_nlpd: float, test_nlpd: float,
                                logger):
    """
    Save final checkpoint results with complete training metrics
    
    Args:
        model: The trained model
        model_name: Name of the model
        seed: Random seed
        final_iteration: Final iteration number
        train_rmse: Final training RMSE
        test_rmse: Final test RMSE
        train_nlpd: Final training NLPD
        test_nlpd: Final test NLPD
        logger: Results logger
    """
    if logger and logger.get_results_dir():
        # Save final checkpoint metrics
        logger.save_checkpoint_metrics(model_name, seed, final_iteration,
                                     train_rmse, test_rmse, train_nlpd, test_nlpd)
        
        # Create final checkpoint plot with single point (final results)
        logger.save_checkpoint_plot(model_name, seed, final_iteration,
                                  [train_rmse], [test_rmse],
                                  [train_nlpd], [test_nlpd])
        
        print(f"  Final checkpoint saved for {model_name} at iteration {final_iteration}")