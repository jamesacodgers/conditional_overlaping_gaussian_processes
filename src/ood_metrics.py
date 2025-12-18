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
    auroc = compute_auroc(id_scores, ood_scores)
    auprc = compute_auprc(id_scores, ood_scores)
    fpr_at_80 = compute_fpr_at_tpr(id_scores, ood_scores, target_tpr=0.80)
    
    return OODMetrics(auroc=auroc, auprc=auprc, fpr_at_80_tpr=fpr_at_80)


def compute_nlpd_scores(model, Y: torch.Tensor, n_samples: int = 32) -> np.ndarray:
    """
    Compute per-sample negative log predictive density (NLPD) scores for OOD detection.
    Uses same GP computation as ssvi_metrics.compute_nlpd but returns per-sample scores.
    Higher NLPD indicates more OOD-like samples.
    
    Parameters
    ----------
    model : GPLVM_SSVI_AM_INF
        Trained GP-LVM SSVI model
    Y : torch.Tensor
        Observations (N x D)
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
        
        # Preprocess encoder input to handle missing data
        Y_clean = torch.where(torch.isnan(Y), torch.full_like(Y, -9.0, dtype=torch.float64), Y)
        
        # Get latent representations using SSVI encoder
        mu_x, sigma_x = model.get_mu_sigma_x_train_samples(Y_clean, U_samples)
        
        # Sample latent variables
        x_dist = torch.distributions.Normal(mu_x, sigma_x)
        x_samples = x_dist.rsample()
        
        # Get GP predictive mean and variance (same as ssvi_metrics.compute_nlpd)
        f_mean = model.gp.get_gp_expectation_given_U(x_samples, U_samples)
        gp_cov = model.gp.get_gp_covariance_given_U(x_samples, U_samples)
        
        # Handle C=1 case (standard GP-LVM)
        if gp_cov.shape[-1] == 1:
            f_var = gp_cov[..., 0, 0] + model.sigma2
            f_mean = f_mean[..., 0]
        else:
            f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2
        
        # Compute per-sample log probability under Gaussian (SAME as ssvi_metrics.py line 276)
        # log p(y | x) = -0.5 * (log(2*pi*var) + (y - mean)^2 / var)
        log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (Y - f_mean)**2 / f_var
        
        # Sum over M dimension (observation dimensions) - same as ssvi_metrics.py line 277
        log_prob = log_prob.sum(dim=-1)  # [P, N] or [N] depending on whether P exists
        

        if log_prob.dim() == 2:
            log_prob_per_sample = log_prob.mean(dim=0)  # [N] - average over P samples
        else:
            log_prob_per_sample = log_prob  # [N] - already 1D
        
        # NLPD is negative log probability
        nlpd_scores = -log_prob_per_sample.cpu().numpy()
    
    model.train()
    return nlpd_scores


def compute_reconstruction_error_scores(model, Y: torch.Tensor, 
                                         n_samples: int = 32) -> np.ndarray:
    """
    Compute per-sample reconstruction error (RMSE) scores for OOD detection.
    Uses EXACTLY the same GP computation as ssvi_metrics.compute_rmse but returns per-sample scores.
    Higher reconstruction error indicates more OOD-like samples.
    
    Parameters
    ----------
    model : GPLVM_SSVI_AM_INF
        Trained GP-LVM SSVI model
    Y : torch.Tensor
        Observations (N x D)
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
        
        # Sample U for both model types
        U_samples = model.get_u_samples()
        
        # Preprocess encoder input to handle missing data (consistent with training)
        y_encoder_input = torch.where(torch.isnan(Y), torch.full_like(Y, -9.0, dtype=torch.float64), Y)
        
        # Get latent representations for this batch using SSVI encoder
        mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U_samples)
        
        # Use proper probabilistic sampling like in compute_nlpd
        x_dist = torch.distributions.Normal(mu_x, sigma_x)
        x_samples = x_dist.rsample()
        
        # Reconstruct through GP mean prediction using proper GP method
        y_recon = model.gp.get_gp_expectation_given_U(x_samples, U_samples)  # [batch_size, M, C]
        
        if y_recon.shape[-1] == 1:
            y_recon = y_recon[..., 0]  # [batch_size, M] for C=1 case
        
        # Average reconstructions over samples
        y_pred = y_recon.mean(dim=0)  # [batch_size, D]
        
        # === ONLY DIFFERENCE: Return per-sample scores instead of mean ===
        # Compute per-sample squared error, sum over dimensions, then sqrt for RMSE
        sq_error = (Y - y_pred) ** 2
        rmse_scores = torch.sqrt(sq_error.sum(dim=-1)).cpu().numpy()  # (N,)
    
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
                                        obs_noise: torch.Tensor, n_samples: int = 64) -> np.ndarray:
    """
    Compute per-sample NLPD scores for Original GP-LVM (BayesianGPLVM).
    Aligned with compute_nlpd_nll_based but returns per-sample scores.
    
    Parameters
    ----------
    model : BayesianGPLVM
        Trained Original GP-LVM model
    Y : torch.Tensor
        Observations (N x D)
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
        
    Returns
    -------
    nlpd_scores : np.ndarray
        Per-sample NLPD scores (N,)
    """
    model.eval()
    
    with torch.no_grad():
        N = len(Y)
        
        # Process X_covar to get standard deviations
        if model_name == 'nn_gauss':
            X_std = torch.stack([torch.sqrt(x.diag()) for x in X_covar])  # [N, latent_dim]
        else:  # gauss model
            X_std = X_covar  # Already standard deviations
        
        x_dist = torch.distributions.Normal(X_mean, X_std)
        
        # Collect per-sample log probs across Monte Carlo samples
        all_log_probs = []
        for _ in range(n_samples):
            x_sample = x_dist.rsample()  # [N, latent_dim]
            
            # Get GP prediction
            Y_star = model(x_sample)
            f_mean = Y_star.loc  # [D, N] - GP output is transposed
            f_var = torch.stack([torch.diag(x) for x in Y_star.covariance_matrix])  # [D, N]
            
            # Add observation noise
            f_var_full = f_var + obs_noise  # [D, N]
            
            # Compute log probability: log N(Y | f_mean, f_var_full)
            # Y is [N, D], f_mean is [D, N], so transpose Y to [D, N]
            Y_T = Y.T  # [D, N]
            log_prob = -0.5 * torch.log(2 * torch.pi * f_var_full) - 0.5 * (Y_T - f_mean) ** 2 / f_var_full
            
            # Sum over dimensions for each sample [N]
            log_prob_sum = log_prob.sum(dim=0)
            all_log_probs.append(log_prob_sum)
        
        # Stack and average over MC samples
        all_log_probs = torch.stack(all_log_probs)  # [n_samples, N]
        log_pred_density = all_log_probs.mean(dim=0)  # [N]
        
        # NLPD is negative log probability
        nlpd_scores = -log_pred_density.cpu().numpy()
    
    model.train()
    return nlpd_scores


def compute_reconstruction_error_scores_original_gplvm(model, Y: torch.Tensor,
                                                        X_mean: torch.Tensor,
                                                        X_covar: torch.Tensor,
                                                        model_name: str,
                                                        n_samples: int = 64) -> np.ndarray:
    """
    Compute per-sample reconstruction error (RMSE) scores for Original GP-LVM.
    
    Parameters
    ----------
    model : BayesianGPLVM
        Trained Original GP-LVM model
    Y : torch.Tensor
        Observations (N x D)
    X_mean : torch.Tensor
        Latent means (N x latent_dim)
    X_covar : torch.Tensor
        Latent covariances
    model_name : str
        Model type: 'gauss' or 'nn_gauss'
    n_samples : int
        Number of Monte Carlo samples
        
    Returns
    -------
    rmse_scores : np.ndarray
        Per-sample RMSE scores (N,)
    """
    model.eval()
    
    with torch.no_grad():
        # Process X_covar to get standard deviations
        if model_name == 'nn_gauss':
            X_std = torch.stack([torch.sqrt(x.diag()) for x in X_covar])
        else:  # gauss model
            X_std = X_covar
        
        x_dist = torch.distributions.Normal(X_mean, X_std)
        
        # Collect reconstructions
        all_recons = []
        for _ in range(n_samples):
            x_sample = x_dist.rsample()
            Y_star = model(x_sample)
            y_pred = Y_star.loc.T  # [N, D] - transpose from [D, N]
            all_recons.append(y_pred)
        
        # Average reconstructions
        y_pred_mean = torch.stack(all_recons).mean(dim=0)  # [N, D]
        
        # Compute per-sample RMSE
        sq_error = (Y - y_pred_mean) ** 2
        rmse_scores = torch.sqrt(sq_error.sum(dim=-1)).cpu().numpy()  # [N]
    
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
                                           n_samples: int = 64) -> Tuple[OODMetrics, np.ndarray, np.ndarray]:
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

