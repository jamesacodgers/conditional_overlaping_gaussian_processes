#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLPD Utilities for GP-LVM Models
"""

import torch

def compute_nlpd_nll_based(model, Y_true, X_mean, model_name, loss_data):
    """
    Compute NLPD using NLL-based approach from movie_lens_large_expt.py

    Args:
        model: GP-LVM model
        Y_true: Observations [N, D]
        X_mean: Latent means [N, latent_dim]
        model_name: Type of model ('point', 'map', 'gauss', or other)
        loss_data: Loss list (for point/map/gauss models)

    Returns:
        NLPD value (lower is better)
    """
    model.eval()

    with torch.no_grad():
        N = len(Y_true)

        if model_name in ('point', 'map', 'gauss'):
            # For point, map, and gauss models, use the loss from training/test prediction
            nlpd = loss_data[-1] / N
        else:
            # For other models (like nn_gauss), compute NLL using model prediction
            Y_star = model(X_mean)
            # log_prob returns scalar per sample (already summed over dimensions)
            # So we sum over samples and divide by N for NLPD
            nlpd = -torch.sum(Y_star.log_prob(Y_true.T)) / N

    model.train()
    return nlpd

