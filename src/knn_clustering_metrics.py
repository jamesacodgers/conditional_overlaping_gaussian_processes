#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k-NN Clustering Quality Metrics

This module provides utilities for measuring clustering quality in latent space
using k-Nearest Neighbors classification accuracy.

k-NN accuracy is a direct measure of how well classes are separated in the latent space:
- Higher accuracy means better clustering
- Different k values capture different neighborhood structures:
  * k=1: Most local structure (nearest neighbor)
  * k=3: Local neighborhood structure
  * k=7: Medium-range structure
  * k=12: Broader neighborhood structure

IMPORTANT: Latent representations are obtained by sampling from the variational
distribution and averaging (matching the latent space plots). This ensures k-NN
metrics reflect the same latent space visualization shown in plots.
"""

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def compute_knn_accuracies(latent_representations, labels, k_values=[1, 3, 7, 12]):
    """
    Compute k-NN classification accuracy for multiple k values.
    
    This function measures clustering quality by evaluating k-NN classifiers
    on the latent representations. Uses all data (no train/test split) since
    we're measuring clustering quality, not generalization.
    
    Parameters
    ----------
    latent_representations : torch.Tensor or np.ndarray
        Latent representations of shape [N, latent_dim] or [P, N, latent_dim].
        If 3D (from SSVI with P samples), will average across P dimension.
    labels : torch.Tensor or np.ndarray
        Class labels of shape [N]
    k_values : list of int
        List of k values to evaluate (default: [1, 3, 7, 12])
        
    Returns
    -------
    dict
        Dictionary with keys 'knn_{k}': Accuracy for each k value
        
    Examples
    --------
    >>> # For SSVI model with P samples
    >>> mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, U)
    >>> latent = mu_x.mean(dim=0)  # Average across P samples
    >>> results = compute_knn_accuracies(latent, labels)
    >>> print(f"k=3 accuracy: {results['knn_3']:.4f}")
    
    >>> # For original GP-LVM
    >>> latent = model.X.q_mu.detach()
    >>> results = compute_knn_accuracies(latent, labels)
    """
    # Convert to numpy if needed
    if torch.is_tensor(latent_representations):
        latent_np = latent_representations.detach().cpu().numpy()
    else:
        latent_np = np.array(latent_representations)
    
    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.array(labels)
    
    # Handle 3D latent representations (P samples from SSVI)
    if latent_np.ndim == 3:
        # Average across P dimension: [P, N, latent_dim] -> [N, latent_dim]
        latent_np = latent_np.mean(axis=0)
    
    # Ensure 2D shape [N, latent_dim]
    if latent_np.ndim != 2:
        raise ValueError(f"Expected 2D or 3D latent representations, got shape {latent_np.shape}")
    
    results = {}
    
    # Use all data (no split - we're measuring clustering quality)
    X_all = latent_np
    y_all = labels_np
    
    # Compute accuracy for each k value
    for k in k_values:
        # Make sure k is not larger than number of samples
        k_effective = min(k, len(X_all) - 1)
        if k_effective < k:
            print(f"  Warning: k={k} reduced to k={k_effective} (not enough samples)")
        
        # Fit k-NN classifier on all data
        knn = KNeighborsClassifier(n_neighbors=k_effective)
        knn.fit(X_all, y_all)
        
        # Evaluate on all data
        y_pred = knn.predict(X_all)
        acc = accuracy_score(y_all, y_pred)
        results[f'knn_{k}'] = float(acc)
    
    return results


def compute_knn_accuracies_ssvi(model, y_train, labels_train, k_values=[1, 3, 7, 12], batch_size=128):
    """
    Compute k-NN accuracies specifically for SSVI models.
    
    This is a convenience wrapper that handles the SSVI-specific latent extraction.
    
    The latent representation is obtained by sampling from the variational distribution
    (using model.P samples) and averaging. This ensures k-NN metrics are aligned 
    with what the plots show.
    
    Parameters
    ----------
    model : GPLVM_SSVI_AM_INF
        SSVI GP-LVM model
    y_train : torch.Tensor
        Training data of shape [N, M]
    labels_train : torch.Tensor or np.ndarray
        Class labels of shape [N]
    k_values : list of int
        List of k values to evaluate (default: [1, 3, 7, 12])
    batch_size : int
        Batch size for processing data (default: 128, to reduce memory usage)
        
    Returns
    -------
    dict
        Dictionary with k-NN accuracies for each k value
    """
    with torch.no_grad():
        # Sample U once (shared across all batches)
        epsilon = torch.randn(model.P, model.M, model.C*model.Lx)
        U = model.get_u_samples(epsilon)
        
        # Process data in minibatches to reduce memory usage
        N = y_train.shape[0]
        latent_batches = []
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            y_batch = y_train[start_idx:end_idx]
            
            # Handle missing data if enabled
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                y_batch_clean = torch.where(
                    torch.isnan(y_batch),
                    torch.full_like(y_batch, -9.0, dtype=torch.float64),
                    y_batch
                )
                mu_x_batch, sigma_x_batch = model.get_mu_sigma_x_train_samples(y_batch_clean, U)
            else:
                mu_x_batch, sigma_x_batch = model.get_mu_sigma_x_train_samples(y_batch, U)
            
            # Sample from distribution and average for this batch
            sampled_latents_batch = []
            for p in range(model.P):
                latent_sample = torch.distributions.Normal(mu_x_batch[p], sigma_x_batch[p]).sample()
                sampled_latents_batch.append(latent_sample)
            
            # Average across P samples: [P, batch_size, latent_dim] -> [batch_size, latent_dim]
            latent_batch = torch.stack(sampled_latents_batch).mean(dim=0)
            latent_batches.append(latent_batch)
        
        # Concatenate all batches: [N, latent_dim]
        latent = torch.cat(latent_batches, dim=0)
    
    return compute_knn_accuracies(latent, labels_train, k_values)


def compute_knn_accuracies_original_gplvm(model, y_train, labels_train, 
                                           model_name='gauss', k_values=[1, 3, 7, 12], batch_size=128):
    """
    Compute k-NN accuracies specifically for original GP-LVM models.
    
    This is a convenience wrapper that handles the original GP-LVM latent extraction.
    
    The latent representation is obtained by sampling from the variational distribution
    (16 samples) and averaging. This ensures k-NN metrics are aligned with what 
    the plots show.
    
    Parameters
    ----------
    model : BayesianGPLVM
        Original GP-LVM model
    y_train : torch.Tensor
        Training data of shape [N, M]
    labels_train : torch.Tensor or np.ndarray
        Class labels of shape [N]
    model_name : str
        Model type: 'gauss' or 'nn_gauss'
    k_values : list of int
        List of k values to evaluate (default: [1, 3, 7, 12])
    batch_size : int
        Batch size for processing data (default: 128, to reduce memory usage)
        
    Returns
    -------
    dict
        Dictionary with k-NN accuracies for each k value
    """
    with torch.no_grad():
        model_name_class = model.X.__class__.__name__
        
        if model_name_class == 'VariationalLatentVariable':  # 'gauss' model
            # Variational GP-LVM (gauss) - latent already computed, no batching needed
            mu_x = model.X.q_mu.detach()
            sigma_x = torch.nn.functional.softplus(model.X.q_log_sigma).detach()
            
            # Sample from distribution (aligned with plotting)
            # Take 16 samples and average to get robust representation
            n_samples = 16
            sampled_latents = []
            for _ in range(n_samples):
                latent_sample = torch.distributions.Normal(mu_x, sigma_x).sample()
                sampled_latents.append(latent_sample)
            
            # Average across samples: [n_samples, N, latent_dim] -> [N, latent_dim]
            latent = torch.stack(sampled_latents).mean(dim=0)
            
        elif model_name_class == 'NNEncoder':  # 'nn_gauss' model
            # Process data in minibatches to reduce memory usage
            N = y_train.shape[0]
            latent_batches = []
            
            # Store original jitter to restore later (same pattern as test prediction)
            original_jitter = model.X.jitter
            latent_dim = model.X.latent_dim
            device = original_jitter.device
            dtype = original_jitter.dtype
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = y_train[start_idx:end_idx]
                batch_size_actual = end_idx - start_idx
                
                # Create jitter for batch (same pattern as test prediction)
                batch_jitter = torch.eye(latent_dim, dtype=dtype, device=device).unsqueeze(0) * 1e-5
                model.X.jitter = torch.cat([batch_jitter for _ in range(batch_size_actual)], axis=0)
                
                # Handle missing data
                if torch.isnan(y_batch).any():
                    y_batch_clean = y_batch.clone()
                    y_batch_clean[torch.isnan(y_batch_clean)] = -999.0
                    mu_x_batch = model.X.mu(y_batch_clean).detach()
                    sigma_covariances_batch = model.X.sigma(y_batch_clean).detach()
                else:
                    mu_x_batch = model.X.mu(y_batch).detach()
                    sigma_covariances_batch = model.X.sigma(y_batch).detach()
                
                # Extract standard deviations
                sigma_x_batch = torch.stack([torch.sqrt(cov.diag()) for cov in sigma_covariances_batch])
                
                # Sample and average for this batch
                n_samples = 16
                sampled_latents_batch = []
                for _ in range(n_samples):
                    latent_sample = torch.distributions.Normal(mu_x_batch, sigma_x_batch).sample()
                    sampled_latents_batch.append(latent_sample)
                
                latent_batch = torch.stack(sampled_latents_batch).mean(dim=0)
                latent_batches.append(latent_batch)
            
            # Restore original jitter
            model.X.jitter = original_jitter
            
            # Concatenate all batches
            latent = torch.cat(latent_batches, dim=0)
        else:
            raise ValueError(f"Unknown model class: {model_name_class}")
    
    return compute_knn_accuracies(latent, labels_train, k_values)


def print_knn_results(results, prefix=""):
    """
    Pretty print k-NN accuracy results.
    
    Parameters
    ----------
    results : dict
        Dictionary of k-NN accuracies from compute_knn_accuracies
    prefix : str
        Optional prefix for print statements (e.g., "  " for indentation)
    """
    print(f"{prefix}k-NN Clustering Quality Metrics:")
    
    # Extract k values from results
    k_values = sorted(set(int(k.split('_')[-1]) for k in results.keys() if 'knn' in k))
    
    for k in k_values:
        key = f'knn_{k}'
        if key in results:
            print(f"{prefix}  k={k:2d}: Accuracy={results[key]:.4f}")

