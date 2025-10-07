#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for extracting ELBO components from Original GP-LVM (GPyTorch-based)
"""

import torch
import gpytorch
from gpytorch.variational import VariationalStrategy


def extract_elbo_components(model, likelihood, output_batch, Y_batch, num_data):
    """
    Extract ELBO components (likelihood, kl_x, kl_u) from GPyTorch model
    
    This function extracts the three main components of the ELBO:
    - Likelihood: log p(Y|f) scaled to full dataset
    - KL(q(X)||p(X)): KL divergence for latent variables
    - KL(q(u)||p(u)): KL divergence for inducing variables
    
    Args:
        model: GPyTorch BayesianGPLVM model
        likelihood: GPyTorch likelihood
        output_batch: Model output (predictive distribution)
        Y_batch: Target data batch [data_dim, batch_size]
        num_data: Total number of data points
        
    Returns:
        tuple: (likelihood_term, kl_x, kl_u) all scaled appropriately
    """
    with torch.no_grad():
        batch_size = Y_batch.shape[-1]
        
        # 1. Compute expected log likelihood on minibatch, scaled to full dataset
        # Use GPyTorch settings for proper likelihood sampling
        with gpytorch.settings.num_likelihood_samples(1):
            exp_log_prob = likelihood.expected_log_prob(Y_batch, output_batch)  # shape [..., batch_size]
            likelihood_term = exp_log_prob.sum(-1).mean() * num_data
        
        # 2. Get KL(q(X)||p(X)) from model's added loss terms (registered by LatentVariable)
        kl_x = torch.tensor(0.0)
        for _, loss_term in model.named_added_loss_terms():
            kl_x = kl_x + loss_term.loss()
        
        # 3. Get KL(q(u)||p(u)) from all variational strategies (robust for multi-output/deep models)
        # Note: kl_divergence() can return tensor with shape [data_dim] for batch_shape models
        kl_u = torch.tensor(0.0)
        for m in model.modules():
            if isinstance(m, VariationalStrategy):
                kl_div = m.kl_divergence()
                # Sum across all dimensions to get scalar
                kl_u = kl_u + kl_div.sum()
        
        # Divide by num_data to match SSVI normalization (ELBO per data point)
        return (likelihood_term / num_data).item(), (kl_x / num_data).item(), (kl_u / num_data).item()

