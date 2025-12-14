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
        
        # Compute per-sample log probability under Gaussian
        # log p(y | x) = -0.5 * (log(2*pi*var) + (y - mean)^2 / var)
        log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (Y - f_mean)**2 / f_var
        
        # Sum over dimensions to get per-sample log prob
        log_prob_per_sample = log_prob.sum(dim=-1)  # (N,)
        
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
        
        Y_clean = torch.where(torch.isnan(Y), torch.full_like(Y, -9.0), Y)
        mu_x, sigma_x = model.get_mu_sigma_x_train_samples(Y_clean, U_samples)
        
        # Sum variance across latent dimensions for each sample
        var_scores = sigma_x.sum(dim=-1).cpu().numpy()
    
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

