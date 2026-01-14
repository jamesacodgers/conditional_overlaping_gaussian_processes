#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOD Detection Metrics for GP-LVM SSVI

Provides functions to compute:
- AUROC (Area Under Receiver Operating Characteristic)
- AUPRC (Area Under Precision-Recall Curve)
- FPR@80 (False Positive Rate at 80% True Positive Rate)
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OODMetrics:
    """Container for OOD detection metrics"""
    auroc: float
    auprc: float
    fpr_at_80_tpr: float
    
    def to_dict(self) -> Dict:
        return {
            'auroc': self.auroc,
            'auprc': self.auprc,
            'fpr_at_80_tpr': self.fpr_at_80_tpr
        }
    
    def __repr__(self):
        return (f"OODMetrics(AUROC={self.auroc:.4f}, "
                f"AUPRC={self.auprc:.4f}, "
                f"FPR@80={self.fpr_at_80_tpr:.4f})")


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Compute Area Under Receiver Operating Characteristic curve.
    
    Parameters
    ----------
    id_scores : np.ndarray
        OOD detection scores for in-distribution samples (lower = more in-distribution)
    ood_scores : np.ndarray
        OOD detection scores for out-of-distribution samples (higher = more OOD)
        
    Returns
    -------
    auroc : float
        AUROC score (0.5 = random, 1.0 = perfect)
    """
    # Create labels: 0 for in-distribution, 1 for OOD
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    return roc_auc_score(labels, scores)


def compute_auprc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    Parameters
    ----------
    id_scores : np.ndarray
        OOD detection scores for in-distribution samples
    ood_scores : np.ndarray
        OOD detection scores for out-of-distribution samples
        
    Returns
    -------
    auprc : float
        AUPRC score
    """
    # Create labels: 0 for in-distribution, 1 for OOD
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def compute_fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, 
                       target_tpr: float = 0.80) -> float:
    """
    Compute False Positive Rate at a given True Positive Rate threshold.
    
    Parameters
    ----------
    id_scores : np.ndarray
        OOD detection scores for in-distribution samples
    ood_scores : np.ndarray
        OOD detection scores for out-of-distribution samples
    target_tpr : float
        Target True Positive Rate (default: 0.80 for 80%)
        
    Returns
    -------
    fpr : float
        False Positive Rate at the given TPR threshold
    """
    # Create labels: 0 for in-distribution, 1 for OOD
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, _ = roc_curve(labels, scores)
    
    # Find FPR at the target TPR
    # Get the index where TPR first exceeds target_tpr
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr):
        return fpr[-1]
    return fpr[idx]


def compute_all_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> OODMetrics:
    """
    Compute all OOD detection metrics.
    
    Filters out NaN scores (e.g., from samples with all missing values) before
    computing metrics, consistent with regular evaluation which skips all-missing batches.
    
    Parameters
    ----------
    id_scores : np.ndarray
        OOD detection scores for in-distribution samples
    ood_scores : np.ndarray
        OOD detection scores for out-of-distribution samples
        
    Returns
    -------
    metrics : OODMetrics
        Container with AUROC, AUPRC, and FPR@80
    """
    # Filter out NaN scores (from samples with all missing values)
    id_valid_mask = np.isfinite(id_scores)
    ood_valid_mask = np.isfinite(ood_scores)
    
    id_scores_filtered = id_scores[id_valid_mask]
    ood_scores_filtered = ood_scores[ood_valid_mask]
    
    if len(id_scores_filtered) == 0 or len(ood_scores_filtered) == 0:
        # If no valid scores, return NaN metrics
        return OODMetrics(auroc=float('nan'), auprc=float('nan'), fpr_at_80_tpr=float('nan'))
    
    auroc = compute_auroc(id_scores_filtered, ood_scores_filtered)
    auprc = compute_auprc(id_scores_filtered, ood_scores_filtered)
    fpr_at_80 = compute_fpr_at_tpr(id_scores_filtered, ood_scores_filtered, target_tpr=0.80)
    
    return OODMetrics(auroc=auroc, auprc=auprc, fpr_at_80_tpr=fpr_at_80)


def compute_nlpd_scores(model, Y: torch.Tensor, n_samples: int = 32) -> np.ndarray:
    """
    Compute per-sample negative log predictive density (NLPD) scores for OOD detection.
    Uses same GP computation as ssvi_metrics.compute_nlpd but returns per-sample scores.
    Handles missing data (NaN values) by only computing over observed dimensions.
    Higher NLPD indicates more OOD-like samples.
    
    Parameters
    ----------
    model : GPLVM_SSVI_AM_INF
        Trained GP-LVM SSVI model
    Y : torch.Tensor
        Observations (N x D) - may contain NaN values for missing data
    n_samples : int
        Number of samples for NLPD computation (unused, kept for API compatibility)
        
    Returns
    -------
    nlpd_scores : np.ndarray
        Per-sample NLPD scores (N,)
    """
    model.eval()
    
    with torch.no_grad():
        # Get U samples (same as ssvi_metrics.compute_nlpd)
        U_samples = model.get_u_samples()
        
        N = Y.shape[0]
        all_nlpd_scores = []
        
        # Process in batches (consistent with ssvi_metrics pattern)
        # Use a reasonable batch size to avoid OOM
        batch_size = 256
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            Y_batch = Y[start_idx:end_idx]
            
            # Preprocess encoder input to handle missing data (same as ssvi_metrics.py line 274)
            y_encoder_input = torch.where(torch.isnan(Y_batch), torch.full_like(Y_batch, -9.0, dtype=torch.float64), Y_batch)
            
            # Get latent representations using SSVI encoder (same as ssvi_metrics.py line 277)
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U_samples)
            
            # Sample latent variables (same as ssvi_metrics.py line 280-281)
            x_dist = torch.distributions.Normal(mu_x, sigma_x)
            x_samples = x_dist.rsample()  # [P, batch_size, latent_dim] or [batch_size, latent_dim]
            
            # Get GP predictive mean and variance (same as ssvi_metrics.compute_nlpd lines 284-295)
            f_mean = model.gp.get_gp_expectation_given_U(x_samples, U_samples)
            gp_cov = model.gp.get_gp_covariance_given_U(x_samples, U_samples)
            
            # Handle C=1 case (standard GP-LVM)
            if gp_cov.shape[-1] == 1:
                f_var = gp_cov[..., 0, 0] + model.sigma2
                f_mean = f_mean[..., 0]
            else:
                f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2
            
            # Handle missing data if enabled (aligned with ssvi_metrics.py lines 298-316)
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                missing_mask_batch = torch.isnan(Y_batch)
                observed_mask_batch = ~missing_mask_batch
                
                # For per-sample scores: handle each sample individually
                # Replace NaN values in Y_batch with zeros for computation (same as ssvi_metrics.py line 308)
                Y_batch_clean = torch.where(observed_mask_batch, Y_batch, torch.zeros_like(Y_batch, dtype=torch.float64))
                
                # Compute log probability under Gaussian (same as ssvi_metrics.py line 311)
                # Shape handling: f_mean and f_var might have P dimension, Y_batch_clean is [batch_size, D]
                if f_mean.dim() == 3:  # [P, batch_size, D]
                    # Expand Y_batch_clean for broadcasting: [batch_size, D] -> [1, batch_size, D]
                    Y_batch_expanded = Y_batch_clean[None, :, :]
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (Y_batch_expanded - f_mean)**2 / f_var
                    # Sum over dimensions: [P, batch_size, D] -> [P, batch_size]
                    log_prob_masked = (log_prob * observed_mask_batch.float()[None, :, :]).sum(dim=-1)
                    # Average over P samples: [P, batch_size] -> [batch_size]
                    log_prob_per_sample = log_prob_masked.mean(dim=0)
                else:  # [batch_size, D]
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (Y_batch_clean - f_mean)**2 / f_var
                    # Sum over observed dimensions: [batch_size, D] -> [batch_size]
                    log_prob_per_sample = (log_prob * observed_mask_batch.float()).sum(dim=-1)
                
                # Handle all-missing samples: set to NaN (consistent with skipping all-missing batches in regular code)
                observed_count_per_sample = observed_mask_batch.sum(dim=-1).float()  # [batch_size]
                batch_nlpd = torch.where(
                    observed_count_per_sample > 0,
                    -log_prob_per_sample,
                    torch.full_like(log_prob_per_sample, float('nan'))  # NaN for all-missing samples
                ).cpu().numpy()
            else:
                # Standard computation for complete data (same as ssvi_metrics.py lines 319-323)
                if f_mean.dim() == 3:  # [P, batch_size, D]
                    Y_batch_expanded = Y_batch[None, :, :]
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (Y_batch_expanded - f_mean)**2 / f_var
                    log_prob_per_sample = log_prob.sum(dim=-1).mean(dim=0)  # [batch_size]
                else:  # [batch_size, D]
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (Y_batch - f_mean)**2 / f_var
                    log_prob_per_sample = log_prob.sum(dim=-1)  # [batch_size]
                
                batch_nlpd = -log_prob_per_sample.cpu().numpy()
            
            all_nlpd_scores.append(batch_nlpd)
        
        # Concatenate all batch results
        nlpd_scores = np.concatenate(all_nlpd_scores)
    
    model.train()
    return nlpd_scores


def compute_reconstruction_error_scores(model, Y: torch.Tensor, 
                                         n_samples: int = 32) -> np.ndarray:
    """
    Compute per-sample reconstruction error (RMSE) scores for OOD detection.
    Uses EXACTLY the same GP computation as ssvi_metrics.compute_rmse but returns per-sample scores.
    Handles missing data (NaN values) by only computing over observed dimensions.
    Higher reconstruction error indicates more OOD-like samples.
    
    Parameters
    ----------
    model : GPLVM_SSVI_AM_INF
        Trained GP-LVM SSVI model
    Y : torch.Tensor
        Observations (N x D) - may contain NaN values for missing data
    n_samples : int
        Number of samples for averaging (unused, kept for API compatibility)
        
    Returns
    -------
    rmse_scores : np.ndarray
        Per-sample RMSE scores (N,)
    """
    model.eval()
    
    with torch.no_grad():
        # === EXACTLY as in ssvi_metrics.compute_rmse ===
        
        # Sample U for both model types (same as ssvi_metrics.py line 95)
        U_samples = model.get_u_samples()
        
        N = Y.shape[0]
        all_rmse_scores = []
        
        # Process in batches (consistent with ssvi_metrics pattern)
        # Use a reasonable batch size to avoid OOM
        batch_size = 256
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            Y_batch = Y[start_idx:end_idx]
            
            # Preprocess encoder input to handle missing data (same as ssvi_metrics.py line 109)
            y_encoder_input = torch.where(torch.isnan(Y_batch), torch.full_like(Y_batch, -9.0, dtype=torch.float64), Y_batch)
            
            # Get latent representations for this batch using SSVI encoder (same as ssvi_metrics.py line 112)
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U_samples)
            
            # Use proper probabilistic sampling like in compute_nlpd (same as ssvi_metrics.py lines 115-116)
            x_dist = torch.distributions.Normal(mu_x, sigma_x)
            x_samples = x_dist.rsample()  # [P, batch_size, latent_dim] or [batch_size, latent_dim]
            
            # Reconstruct through GP mean prediction using proper GP method (same as ssvi_metrics.py line 119)
            y_recon = model.gp.get_gp_expectation_given_U(x_samples, U_samples)  # [P, batch_size, M, C] or [batch_size, M, C]
            
            if y_recon.shape[-1] == 1:
                y_recon = y_recon[..., 0]  # [P, batch_size, M] or [batch_size, M] for C=1 case
            
            # Average reconstructions over samples (same as ssvi_metrics.py line 125)
            if y_recon.dim() == 3:  # [P, batch_size, M]
                y_pred = y_recon.mean(dim=0)  # [batch_size, M]
            else:  # [batch_size, M]
                y_pred = y_recon  # Already averaged or single sample
            
            # Handle missing data if enabled (aligned with ssvi_metrics.py lines 128-139)
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                missing_mask_batch = torch.isnan(Y_batch)
                observed_mask_batch = ~missing_mask_batch
                
                # Compute squared errors only on observed values (same as ssvi_metrics.py lines 135-136)
                Y_batch_clean = torch.where(observed_mask_batch, Y_batch, torch.zeros_like(Y_batch, dtype=torch.float64))
                squared_errors = (Y_batch_clean - y_pred) ** 2
                observed_squared_errors = squared_errors * observed_mask_batch.float()
                
                # For per-sample RMSE: sum over observed dimensions and normalize per sample
                observed_count_per_sample = observed_mask_batch.sum(dim=-1).float()  # [batch_size]
                sum_sq_error_per_sample = observed_squared_errors.sum(dim=-1)  # [batch_size]
                
                # Compute RMSE per sample: sqrt(sum of observed squared errors / count of observed)
                # Handle all-missing samples: set to NaN (consistent with skipping all-missing batches in regular code)
                batch_rmse = torch.where(
                    observed_count_per_sample > 0,
                    torch.sqrt(sum_sq_error_per_sample / observed_count_per_sample),
                    torch.full_like(observed_count_per_sample, float('nan'))  # NaN for all-missing samples
                ).cpu().numpy()
            else:
                # Standard computation for complete data (same as ssvi_metrics.py lines 142-143)
                squared_errors = (Y_batch - y_pred) ** 2
                batch_rmse = torch.sqrt(squared_errors.sum(dim=-1)).cpu().numpy()  # [batch_size]
            
            all_rmse_scores.append(batch_rmse)
        
        # Concatenate all batch results
        rmse_scores = np.concatenate(all_rmse_scores)
    
    model.train()
    return rmse_scores


def compute_latent_variance_scores(model, Y: torch.Tensor) -> np.ndarray:
    """
    Compute latent variance scores for OOD detection.
    Higher variance indicates more uncertainty / OOD-like samples.
    
    Parameters
    ----------
    model : GPLVM_SSVI_AM_INF
        Trained GP-LVM SSVI model
    Y : torch.Tensor
        Observations (N x D)
        
    Returns
    -------
    var_scores : np.ndarray
        Per-sample variance scores (N,) - sum of latent variances
    """
    model.eval()
    
    with torch.no_grad():
        U_samples = model.get_u_samples()
        
        Y_clean = torch.where(torch.isnan(Y), torch.full_like(Y, -9.0, dtype=torch.float64), Y)
        mu_x, sigma_x = model.get_mu_sigma_x_train_samples(Y_clean, U_samples)
        
        # sigma_x may have shape [P, N, latent_dim] or [N, latent_dim]
        # Sum variance across latent dimensions for each sample
        var_scores = sigma_x.sum(dim=-1)  # [P, N] or [N]
        
        # If there's a P dimension, average over it to get per-sample scores [N]
        if var_scores.dim() == 2:
            var_scores = var_scores.mean(dim=0)  # [N]
        
        var_scores = var_scores.cpu().numpy()
    
    model.train()
    return var_scores


def evaluate_ood_detection(model, id_test_data: torch.Tensor, 
                           ood_data: torch.Tensor,
                           score_type: str = 'nlpd',
                           n_samples: int = 32) -> Tuple[OODMetrics, np.ndarray, np.ndarray]:
    """
    Evaluate OOD detection performance.
    
    Parameters
    ----------
    model : GPLVM_SSVI_AM_INF
        Trained GP-LVM SSVI model
    id_test_data : torch.Tensor
        In-distribution test data (N_id x D)
    ood_data : torch.Tensor
        Out-of-distribution data (N_ood x D)
    score_type : str
        Type of OOD score: 'nlpd', 'rmse', or 'variance'
    n_samples : int
        Number of samples for score computation
        
    Returns
    -------
    metrics : OODMetrics
        OOD detection metrics
    id_scores : np.ndarray
        Scores for in-distribution samples
    ood_scores : np.ndarray
        Scores for OOD samples
    """
    # Compute scores based on type
    if score_type == 'nlpd':
        id_scores = compute_nlpd_scores(model, id_test_data, n_samples)
        ood_scores = compute_nlpd_scores(model, ood_data, n_samples)
    elif score_type == 'rmse':
        id_scores = compute_reconstruction_error_scores(model, id_test_data, n_samples)
        ood_scores = compute_reconstruction_error_scores(model, ood_data, n_samples)
    elif score_type == 'variance':
        id_scores = compute_latent_variance_scores(model, id_test_data)
        ood_scores = compute_latent_variance_scores(model, ood_data)
    else:
        raise ValueError(f"Unknown score type: {score_type}. "
                        f"Supported: 'nlpd', 'rmse', 'variance'")
    
    # Compute metrics
    metrics = compute_all_ood_metrics(id_scores, ood_scores)
    
    return metrics, id_scores, ood_scores


# ============================================================================
# Original GP-LVM OOD Score Functions (BayesianGPLVM from gpytorch)
# ============================================================================

def compute_nlpd_scores_original_gplvm(model, Y: torch.Tensor, X_mean: torch.Tensor,
                                        X_covar: torch.Tensor, model_name: str,
                                        obs_noise: torch.Tensor, n_samples: int = 4,
                                        batch_size: int = 128) -> np.ndarray:
    """
    Compute per-sample NLPD scores for Original GP-LVM (BayesianGPLVM).
    Aligned with compute_nlpd_nll_based but returns per-sample scores.
    Uses batched computation to avoid OOM.
    Handles missing data (NaN values) by only computing over observed dimensions.
    
    Parameters
    ----------
    model : BayesianGPLVM
        Trained Original GP-LVM model
    Y : torch.Tensor
        Observations (N x D) - may contain NaN values for missing data
    X_mean : torch.Tensor
        Latent means (N x latent_dim)
    X_covar : torch.Tensor
        Latent covariances - std for gauss, cov matrices for nn_gauss
    model_name : str
        Model type: 'gauss' or 'nn_gauss'
    obs_noise : torch.Tensor
        Observation noise from likelihood
    n_samples : int
        Number of Monte Carlo samples
    batch_size : int
        Batch size for memory-efficient computation
        
    Returns
    -------
    nlpd_scores : np.ndarray
        Per-sample NLPD scores (N,)
    """
    model.eval()
    
    with torch.no_grad():
        N = len(Y)
        all_nlpd_scores = []
        
        # Process in batches to avoid OOM
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            
            # Get batch data
            Y_batch = Y[start_idx:end_idx]
            X_mean_batch = X_mean[start_idx:end_idx]
            
            # Process X_covar to get standard deviations for this batch
            if model_name == 'nn_gauss':
                X_covar_batch = X_covar[start_idx:end_idx]
                X_std_batch = torch.stack([torch.sqrt(x.diag()) for x in X_covar_batch])
            else:  # gauss model
                X_std_batch = X_covar[start_idx:end_idx]
            
            x_dist = torch.distributions.Normal(X_mean_batch, X_std_batch)
            
            # Handle missing data: create mask for observed values (check per batch)
            has_missing_in_batch = torch.isnan(Y_batch).any()
            if has_missing_in_batch:
                missing_mask_batch = torch.isnan(Y_batch)
                observed_mask_batch = ~missing_mask_batch
            else:
                observed_mask_batch = torch.ones_like(Y_batch, dtype=torch.bool)
            
            # Collect per-sample log probs across Monte Carlo samples
            batch_log_probs = []
            for _ in range(n_samples):
                x_sample = x_dist.rsample()  # [batch_size, latent_dim]
                
                # Get GP prediction
                Y_star = model(x_sample)
                f_mean = Y_star.loc  # [D, batch_size] - GP output is transposed
                
                # Extract diagonal of covariance matrix for each output dimension
                f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])  # [D, batch_size]
                
                # Add observation noise
                f_var_full = f_var + obs_noise  # [D, batch_size]
                
                # Compute log probability: log N(Y | f_mean, f_var_full)
                # Y_batch is [batch_size, D], f_mean is [D, batch_size], so transpose Y
                Y_T = Y_batch.T  # [D, batch_size]
                
                if has_missing_in_batch:
                    # Handle missing data: only compute over observed dimensions
                    # Aligned with nlpd_utils.py pattern (lines 87-94)
                    # Clean Y by replacing NaN with zeros (same as nlpd_utils.py line 87)
                    Y_batch_clean = torch.where(observed_mask_batch, Y_batch, torch.zeros_like(Y_batch, dtype=torch.float64))
                    Y_batch_clean_T = Y_batch_clean.T  # [D, batch_size]
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (Y_batch_clean_T - f_mean) ** 2 / f_var_full
                    
                    # Mask log probability to only include observed dimensions (same as nlpd_utils.py lines 92-93)
                    observed_mask_T = observed_mask_batch.T  # [D, batch_size]
                    log_prob_masked = log_prob * observed_mask_T.float()
                    log_prob_sum = log_prob_masked.sum(dim=0)  # [batch_size]
                else:
                    # Standard computation for complete data (same as nlpd_utils.py lines 98-100)
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (Y_T - f_mean) ** 2 / f_var_full
                    log_prob_sum = log_prob.sum(dim=0)  # [batch_size]
                
                batch_log_probs.append(log_prob_sum)
            
            # Stack and average over MC samples for this batch
            batch_log_probs = torch.stack(batch_log_probs)  # [n_samples, batch_size]
            log_pred_density = batch_log_probs.mean(dim=0)  # [batch_size]
            
            # NLPD is negative log probability
            # Handle all-missing samples: if log_pred_density is 0 (all missing), set to NaN
            # (aligned with regular code which skips all-missing batches)
            if has_missing_in_batch:
                observed_count_per_sample = observed_mask_batch.sum(dim=-1).float()  # [batch_size]
                batch_nlpd = torch.where(
                    observed_count_per_sample > 0,
                    -log_pred_density,
                    torch.full_like(log_pred_density, float('nan'))  # NaN for all-missing samples
                ).cpu().numpy()
            else:
                batch_nlpd = -log_pred_density.cpu().numpy()
            
            all_nlpd_scores.append(batch_nlpd)
        
        # Concatenate all batch results
        nlpd_scores = np.concatenate(all_nlpd_scores)
    
    model.train()
    return nlpd_scores


def compute_reconstruction_error_scores_original_gplvm(model, Y: torch.Tensor,
                                                        X_mean: torch.Tensor,
                                                        X_covar: torch.Tensor,
                                                        model_name: str,
                                                        n_samples: int = 4,
                                                        batch_size: int = 128) -> np.ndarray:
    """
    Compute per-sample reconstruction error (RMSE) scores for Original GP-LVM.
    Uses batched computation to avoid OOM.
    Handles missing data (NaN values) by only computing over observed dimensions.
    
    Parameters
    ----------
    model : BayesianGPLVM
        Trained Original GP-LVM model
    Y : torch.Tensor
        Observations (N x D) - may contain NaN values for missing data
    X_mean : torch.Tensor
        Latent means (N x latent_dim)
    X_covar : torch.Tensor
        Latent covariances
    model_name : str
        Model type: 'gauss' or 'nn_gauss'
    n_samples : int
        Number of Monte Carlo samples
    batch_size : int
        Batch size for memory-efficient computation
        
    Returns
    -------
    rmse_scores : np.ndarray
        Per-sample RMSE scores (N,)
    """
    model.eval()
    
    with torch.no_grad():
        N = len(Y)
        all_rmse_scores = []
        
        # Process in batches to avoid OOM
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            
            # Get batch data
            Y_batch = Y[start_idx:end_idx]
            X_mean_batch = X_mean[start_idx:end_idx]
            
            # Process X_covar to get standard deviations for this batch
            if model_name == 'nn_gauss':
                X_covar_batch = X_covar[start_idx:end_idx]
                X_std_batch = torch.stack([torch.sqrt(x.diag()) for x in X_covar_batch])
            else:  # gauss model
                X_std_batch = X_covar[start_idx:end_idx]
            
            x_dist = torch.distributions.Normal(X_mean_batch, X_std_batch)
            
            # Collect reconstructions for this batch
            batch_recons = []
            for _ in range(n_samples):
                x_sample = x_dist.rsample()
                Y_star = model(x_sample)
                y_pred = Y_star.loc.T  # [batch_size, D] - transpose from [D, batch_size]
                batch_recons.append(y_pred)
            
            # Average reconstructions
            y_pred_mean = torch.stack(batch_recons).mean(dim=0)  # [batch_size, D]
            
            # Check if this batch has missing data
            has_missing_in_batch = torch.isnan(Y_batch).any()
            
            # Compute per-sample RMSE for this batch
            if has_missing_in_batch:
                # Handle missing data: only compute over observed dimensions
                # Aligned with unified_rmse_utils.py pattern
                missing_mask_batch = torch.isnan(Y_batch)
                observed_mask_batch = ~missing_mask_batch
                
                # Clean Y_batch by replacing NaN with 0 (same as unified_rmse_utils.py line 92)
                Y_batch_clean = torch.where(observed_mask_batch, Y_batch, torch.zeros_like(Y_batch, dtype=torch.float64))
                
                # Compute squared errors with cleaned data (same as unified_rmse_utils.py line 93)
                squared_errors = (Y_batch_clean - y_pred_mean) ** 2
                
                # Mask to only include observed dimensions (same as unified_rmse_utils.py line 94)
                observed_squared_errors = squared_errors * observed_mask_batch.float()
                
                # For per-sample RMSE: sum over observed dimensions and normalize per sample
                # Count observed dimensions per sample
                observed_count = observed_mask_batch.sum(dim=-1).float()  # [batch_size]
                
                # Sum observed squared errors per sample
                sum_sq_error_per_sample = observed_squared_errors.sum(dim=-1)  # [batch_size]
                
                # Compute RMSE per sample: sqrt(sum of observed squared errors / count of observed)
                # Handle case where all values are missing (set to NaN, will be filtered in metrics)
                batch_rmse = torch.where(
                    observed_count > 0,
                    torch.sqrt(sum_sq_error_per_sample / observed_count),
                    torch.full_like(observed_count, float('nan'))  # NaN for all-missing samples
                ).cpu().numpy()
            else:
                # Standard computation for complete data
                sq_error = (Y_batch - y_pred_mean) ** 2
                batch_rmse = torch.sqrt(sq_error.sum(dim=-1)).cpu().numpy()  # [batch_size]
            
            all_rmse_scores.append(batch_rmse)
        
        # Concatenate all batch results
        rmse_scores = np.concatenate(all_rmse_scores)
    
    model.train()
    return rmse_scores


def compute_latent_variance_scores_original_gplvm(model, Y: torch.Tensor,
                                                   X_covar: torch.Tensor,
                                                   model_name: str) -> np.ndarray:
    """
    Compute latent variance scores for Original GP-LVM OOD detection.
    
    Parameters
    ----------
    model : BayesianGPLVM
        Trained Original GP-LVM model
    Y : torch.Tensor
        Observations (N x D) - needed for nn_gauss encoder
    X_covar : torch.Tensor
        Latent covariances
    model_name : str
        Model type: 'gauss' or 'nn_gauss'
        
    Returns
    -------
    var_scores : np.ndarray
        Per-sample variance scores (N,) - sum of latent variances
    """
    model.eval()
    
    with torch.no_grad():
        if model_name == 'nn_gauss':
            # For nn_gauss, X_covar contains covariance matrices
            # Sum diagonal elements (variances) across latent dimensions
            var_scores = torch.stack([x.diag().sum() for x in X_covar]).cpu().numpy()
        else:  # gauss model
            # X_covar is standard deviations, square to get variances
            var_scores = (X_covar ** 2).sum(dim=-1).cpu().numpy()
    
    model.train()
    return var_scores


def evaluate_ood_detection_original_gplvm(model, id_test_data: torch.Tensor,
                                           ood_data: torch.Tensor,
                                           id_X_mean: torch.Tensor,
                                           id_X_covar: torch.Tensor,
                                           ood_X_mean: torch.Tensor,
                                           ood_X_covar: torch.Tensor,
                                           model_name: str,
                                           obs_noise: torch.Tensor,
                                           score_type: str = 'nlpd',
                                           n_samples: int = 4) -> Tuple[OODMetrics, np.ndarray, np.ndarray]:
    """
    Evaluate OOD detection for Original GP-LVM.
    
    Parameters
    ----------
    model : BayesianGPLVM
        Trained Original GP-LVM model
    id_test_data : torch.Tensor
        In-distribution test data (N_id x D)
    ood_data : torch.Tensor
        Out-of-distribution data (N_ood x D)
    id_X_mean, id_X_covar : torch.Tensor
        Latent representations for ID test data
    ood_X_mean, ood_X_covar : torch.Tensor
        Latent representations for OOD data
    model_name : str
        Model type: 'gauss' or 'nn_gauss'
    obs_noise : torch.Tensor
        Observation noise from likelihood
    score_type : str
        Type of OOD score: 'nlpd', 'rmse', or 'variance'
    n_samples : int
        Number of samples for score computation
        
    Returns
    -------
    metrics : OODMetrics
        OOD detection metrics
    id_scores : np.ndarray
        Scores for in-distribution samples
    ood_scores : np.ndarray
        Scores for OOD samples
    """
    if score_type == 'nlpd':
        id_scores = compute_nlpd_scores_original_gplvm(
            model, id_test_data, id_X_mean, id_X_covar, model_name, obs_noise, n_samples)
        ood_scores = compute_nlpd_scores_original_gplvm(
            model, ood_data, ood_X_mean, ood_X_covar, model_name, obs_noise, n_samples)
    elif score_type == 'rmse':
        id_scores = compute_reconstruction_error_scores_original_gplvm(
            model, id_test_data, id_X_mean, id_X_covar, model_name, n_samples)
        ood_scores = compute_reconstruction_error_scores_original_gplvm(
            model, ood_data, ood_X_mean, ood_X_covar, model_name, n_samples)
    elif score_type == 'variance':
        id_scores = compute_latent_variance_scores_original_gplvm(
            model, id_test_data, id_X_covar, model_name)
        ood_scores = compute_latent_variance_scores_original_gplvm(
            model, ood_data, ood_X_covar, model_name)
    else:
        raise ValueError(f"Unknown score type: {score_type}. "
                        f"Supported: 'nlpd', 'rmse', 'variance'")
    
    # Compute metrics
    metrics = compute_all_ood_metrics(id_scores, ood_scores)
    
    return metrics, id_scores, ood_scores

