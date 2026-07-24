"""
SSVI GP-LVM Evaluation Metrics

Implementation of metrics used in "Generalised Gaussian Process Latent Variable Models (GPLVM) 
with Stochastic Variational Inference" paper for evaluating model performance.

Key metrics:
- RMSE: Root Mean Square Error for reconstruction quality
- NLPD: Negative Log Predictive Density for uncertainty quantification
- CRPS: Continuous Ranked Probability Score for predictive accuracy / uncertainty quality
- Predictive interval coverage and mean interval width (MPIW)
- Train/Test performance comparison for overfitting detection
"""

import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Numerical floor for predictive standard deviation in CRPS / coverage
_CRPS_SIGMA_EPS = 1e-8

# Nominal predictive interval coverage levels
COVERAGE_LEVELS = (0.90, 0.95)


def _normal_z_for_coverage(level: float) -> float:
    """z_{1-α/2} for central (1-α) Gaussian predictive interval, level = 1-α."""
    alpha = 1.0 - level
    return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2.0)))


def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                  eps: float = _CRPS_SIGMA_EPS) -> torch.Tensor:
    """
    Closed-form CRPS for univariate Gaussian predictive distributions.

    CRPS(N(μ, σ²), y) = σ [ z (2 Φ(z) − 1) + 2 φ(z) − 1/√π ],
    where z = (y − μ) / σ, φ and Φ are the standard normal pdf and cdf.
    """
    sigma = sigma.clamp(min=eps)
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - (1.0 / math.sqrt(math.pi)))


def sample_crps(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Sample-based CRPS estimator:
        CRPS = E|X − y| − (1/2) E|X − X′|
    where samples has leading dimension over predictive draws.
    """
    term1 = (samples - y).abs().mean(dim=0)
    # Pairwise absolute differences over the sample axis
    term2 = 0.5 * (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean(dim=(0, 1))
    return term1 - term2


def gaussian_interval_bounds(mu: torch.Tensor, sigma: torch.Tensor, level: float,
                             eps: float = _CRPS_SIGMA_EPS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central Gaussian predictive interval [L, U] at level = 1−α."""
    sigma = sigma.clamp(min=eps)
    z = _normal_z_for_coverage(level)
    lower = mu - z * sigma
    upper = mu + z * sigma
    return lower, upper


def gaussian_interval_coverage(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                               level: float, eps: float = _CRPS_SIGMA_EPS) -> torch.Tensor:
    """
    Indicator that y lies in the central Gaussian predictive interval
    [μ − z_{1−α/2} σ, μ + z_{1−α/2} σ], with level = 1−α.
    """
    lower, upper = gaussian_interval_bounds(mu, sigma, level, eps=eps)
    return ((y >= lower) & (y <= upper)).float()


def gaussian_interval_width(mu: torch.Tensor, sigma: torch.Tensor, level: float,
                            eps: float = _CRPS_SIGMA_EPS) -> torch.Tensor:
    """Width U − L of the central Gaussian predictive interval at ``level``."""
    lower, upper = gaussian_interval_bounds(mu, sigma, level, eps=eps)
    return upper - lower


def sample_interval_bounds(samples: torch.Tensor, level: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Empirical central interval bounds from predictive samples (leading draw dim)."""
    alpha = 1.0 - level
    lower = torch.quantile(samples, alpha / 2.0, dim=0)
    upper = torch.quantile(samples, 1.0 - alpha / 2.0, dim=0)
    return lower, upper


def sample_interval_coverage(samples: torch.Tensor, y: torch.Tensor, level: float) -> torch.Tensor:
    """
    Indicator that y lies in the empirical central interval from predictive samples.
    ``samples`` has leading dimension over draws.
    """
    lower, upper = sample_interval_bounds(samples, level)
    return ((y >= lower) & (y <= upper)).float()


def sample_interval_width(samples: torch.Tensor, level: float) -> torch.Tensor:
    """Width U − L of the empirical central interval from predictive samples."""
    lower, upper = sample_interval_bounds(samples, level)
    return upper - lower


@dataclass
class SSVIMetrics:
    """Container for SSVI GP-LVM evaluation metrics"""
    
    # Reconstruction metrics
    train_rmse: float = 0.0
    test_rmse: float = 0.0
    
    # Uncertainty metrics
    train_nlpd: float = 0.0
    test_nlpd: float = 0.0
    
    # Continuous Ranked Probability Score (lower is better)
    train_crps: float = 0.0
    test_crps: float = 0.0
    # Alias for held-out / test CRPS in evaluation output
    crps: float = 0.0

    # Predictive interval coverage (held-out primary names: coverage_90 / coverage_95)
    train_coverage_90: float = float('nan')
    train_coverage_95: float = float('nan')
    test_coverage_90: float = float('nan')
    test_coverage_95: float = float('nan')
    coverage_90: float = float('nan')
    coverage_95: float = float('nan')

    # Mean prediction interval width / MPIW (held-out: interval_width_90 / interval_width_95)
    train_interval_width_90: float = float('nan')
    train_interval_width_95: float = float('nan')
    test_interval_width_90: float = float('nan')
    test_interval_width_95: float = float('nan')
    interval_width_90: float = float('nan')
    interval_width_95: float = float('nan')
    
    # Missing data uncertainty metrics
    train_missing_nlpd: float = float('nan')
    test_missing_nlpd: float = float('nan')
    
    # Missing data reconstruction metrics
    train_missing_rmse: float = float('nan')
    test_missing_rmse: float = float('nan')
    
    # Model complexity metrics
    effective_latent_dims: int = 0
    total_latent_dims: int = 0
    
    # Training progression
    iteration: int = 0
    elbo_loss: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for logging"""
        return {
            'iteration': self.iteration,
            'train_rmse': self.train_rmse,
            'test_rmse': self.test_rmse,
            'train_nlpd': self.train_nlpd,
            'test_nlpd': self.test_nlpd,
            'train_crps': self.train_crps,
            'test_crps': self.test_crps,
            'crps': self.crps,
            'train_coverage_90': self.train_coverage_90,
            'train_coverage_95': self.train_coverage_95,
            'test_coverage_90': self.test_coverage_90,
            'test_coverage_95': self.test_coverage_95,
            'coverage_90': self.coverage_90,
            'coverage_95': self.coverage_95,
            'train_interval_width_90': self.train_interval_width_90,
            'train_interval_width_95': self.train_interval_width_95,
            'test_interval_width_90': self.test_interval_width_90,
            'test_interval_width_95': self.test_interval_width_95,
            'interval_width_90': self.interval_width_90,
            'interval_width_95': self.interval_width_95,
            'train_missing_nlpd': self.train_missing_nlpd,
            'test_missing_nlpd': self.test_missing_nlpd,
            'train_missing_rmse': self.train_missing_rmse,
            'test_missing_rmse': self.test_missing_rmse,
            'effective_latent_dims': self.effective_latent_dims,
            'total_latent_dims': self.total_latent_dims,
            'elbo_loss': self.elbo_loss
        }


class SSVIMetricsComputer:
    """Compute evaluation metrics for SSVI GP-LVM models"""
    
    def __init__(self, device: str = "cpu", metrics_batch_size: int = 256):
        self.device = device
        self.metrics_history: List[SSVIMetrics] = []
        self.metrics_batch_size = metrics_batch_size  # Batch size for metrics computation
    
    
    
    def compute_rmse(self, 
                     model, 
                     y_true: torch.Tensor, 
                     n_samples: int = 10) -> float:
        """
        Compute Root Mean Square Error for reconstruction using mini-batching.
        
        Args:
            model: SSVI GP-LVM model
            y_true: True observations [N, D]
            n_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            RMSE value
        """
        model.eval()
        with torch.no_grad():
            # Sample U once for all batches
            epsilon = model.get_epsilon()
            U_samples = model.get_u_samples(epsilon)
            
            N = y_true.shape[0]
            batch_size = self.metrics_batch_size
            
            # Accumulators for batch-wise computation
            total_squared_error = 0.0
            total_count = 0
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = y_true[start_idx:end_idx]
                
                # Preprocess encoder input to handle missing data (consistent with training)
                y_encoder_input = torch.where(torch.isnan(y_batch), torch.full_like(y_batch, -9.0, dtype=torch.float64), y_batch)
                
                # Get latent representations for this batch using SSVI encoder
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
                
                # Use proper probabilistic sampling like in compute_nlpd
                x_dist = torch.distributions.Normal(mu_x, sigma_x)
                x_samples = x_dist.rsample()  # [n_samples, batch_size, latent_dim]
                
                # Reconstruct through GP mean prediction using proper GP method
                y_recon = model.gp.get_gp_expectation_given_U(x_samples, U_samples)  # [n_samples, batch_size, M, C]
                
                if y_recon.shape[-1] == 1:
                    y_recon = y_recon[..., 0]  # [n_samples, batch_size, M] for C=1 case
                
                # Average reconstructions over samples
                y_pred = y_recon.mean(dim=0)  # [batch_size, D]
                
                # Compute squared errors for this batch
                if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                    # Handle missing data: create mask for observed values
                    missing_mask_batch = torch.isnan(y_batch)
                    observed_mask_batch = ~missing_mask_batch
                    
                    if observed_mask_batch.sum() > 0:
                        # Compute squared errors only on observed values
                        y_batch_clean = torch.where(observed_mask_batch, y_batch, torch.zeros_like(y_batch))
                        squared_errors = (y_batch_clean - y_pred) ** 2
                        observed_squared_errors = squared_errors * observed_mask_batch.float()
                        total_squared_error += observed_squared_errors.sum().item()
                        total_count += observed_mask_batch.sum().item()
                else:
                    # Standard computation for complete data
                    squared_errors = (y_batch - y_pred) ** 2
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

    def compute_missing_data_rmse(self,
                     model,
                     y_original: torch.Tensor,
                     missing_mask: torch.Tensor,
                     n_samples: int = 10) -> float:
        """
        Compute Root Mean Square Error specifically for missing data dimensions using mini-batching.
        
        This does everything exactly the same as compute_rmse,
        but at the end computes RMSE only on missing values (reverse of observed mask).
        
        Args:
            model: SSVI GP-LVM model
            y_original: Original complete data before missing data preprocessing [N, D]
            missing_mask: Boolean mask [N, D] where True=missing
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
            
            # Sample U once for all batches
            epsilon = model.get_epsilon()
            U_samples = model.get_u_samples(epsilon)
            
            N = y_original.shape[0]
            batch_size = self.metrics_batch_size
            
            # Accumulators for batch-wise computation
            total_squared_error = 0.0
            total_missing_count = 0
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = y_original[start_idx:end_idx]
                mask_batch = missing_mask[start_idx:end_idx]

                # Preprocess encoder input to handle missing data (consistent with training)
                # Create masked data by setting missing dimensions to -9.0
                y_encoder_input = y_batch.clone()
                y_encoder_input[mask_batch] = -9.0

                # Get latent representations for this batch using SSVI encoder
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)

                # Use proper probabilistic sampling like in compute_nlpd
                x_dist = torch.distributions.Normal(mu_x, sigma_x)
                x_samples = x_dist.rsample()  # [n_samples, batch_size, latent_dim]

                # Reconstruct through GP mean prediction using proper GP method
                y_recon = model.gp.get_gp_expectation_given_U(x_samples, U_samples)  # [n_samples, batch_size, M, C]

                if y_recon.shape[-1] == 1:
                    y_recon = y_recon[..., 0]  # [n_samples, batch_size, M] for C=1 case

                # Average reconstructions over samples
                y_pred = y_recon.mean(dim=0)  # [batch_size, D]

                # Compute squared errors only on MISSING values
                y_true_missing = y_batch[mask_batch]
                y_pred_missing = y_pred[mask_batch]
                
                squared_errors = (y_true_missing - y_pred_missing) ** 2
                total_squared_error += squared_errors.sum().item()
                total_missing_count += mask_batch.sum().item()
            
            # Compute final RMSE
            if total_missing_count == 0:
                model.train()
                return float('nan')
            
            mse_missing = total_squared_error / total_missing_count
            rmse = np.sqrt(mse_missing)

        model.train()
        return rmse

    
    def compute_nlpd(self, 
                     model, 
                     y_true: torch.Tensor, 
                     n_samples: int = 50) -> float:
        """
        Compute Negative Log Predictive Density for uncertainty quantification using mini-batching.
        
        Args:
            model: SSVI GP-LVM model
            y_true: True observations [N, D]
            n_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            NLPD value (lower is better)
        """
        model.eval()
        with torch.no_grad():
            # Sample U once for all batches
            epsilon = model.get_epsilon()
            U_samples = model.get_u_samples(epsilon)
            
            N = y_true.shape[0]
            batch_size = self.metrics_batch_size
            
            # Accumulators for batch-wise computation
            total_log_prob = 0.0
            total_count = 0
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = y_true[start_idx:end_idx]
                current_batch_size = end_idx - start_idx
                
                # Preprocess encoder input to handle missing data (consistent with training)
                y_encoder_input = torch.where(torch.isnan(y_batch), torch.full_like(y_batch, -9.0, dtype=torch.float64), y_batch)
                
                # Get latent representations for this batch using SSVI encoder
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
                
                # Compute averaged latent variables once before the loop (optimization)
                x_dist = torch.distributions.Normal(mu_x, sigma_x)
                x_samples = x_dist.rsample()  # [n_samples, batch_size, latent_dim]            
                
                # Get GP predictive mean using GP method directly
                f_mean = model.gp.get_gp_expectation_given_U(x_samples, U_samples)  # [n_samples, batch_size, M, C]
                
                # Get proper GP predictive variance: GP covariance + observation noise
                gp_cov = model.gp.get_gp_covariance_given_U(x_samples, U_samples)  # [n_samples, batch_size, M, C, C]

                # For observed data (C=1 case), extract diagonal and add observation noise
                if gp_cov.shape[-1] == 1:
                    f_var = gp_cov[..., 0, 0] + model.sigma2  # [n_samples, batch_size, M]
                    f_mean = f_mean[..., 0]  # [n_samples, batch_size, M] to match f_var
                else:
                    # For multi-output case, use diagonal elements + observation noise
                    f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2  # [n_samples, batch_size, M, C]
                
                # Handle missing data if enabled
                if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                    # Mask out missing values in log probability computation
                    missing_mask_batch = torch.isnan(y_batch)
                    observed_mask_batch = ~missing_mask_batch
                    
                    if observed_mask_batch.sum() == 0:
                        # Skip this batch if all data is missing
                        continue
                    
                    # Replace NaN values in y_batch with zeros for computation (they'll be masked out)
                    y_batch_clean = torch.where(observed_mask_batch, y_batch, torch.zeros_like(y_batch))
                    
                    # Compute log probability under Gaussian
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (y_batch_clean - f_mean)**2 / f_var
                    
                    # Sum over observed dimensions and average over samples
                    log_prob_masked = (log_prob * observed_mask_batch.float()[None, :, :]).sum(dim=-1).mean(dim=0)  # [batch_size]
                    total_log_prob += log_prob_masked.sum().item()
                    total_count += current_batch_size
                else:
                    # Standard computation for complete data
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (y_batch - f_mean)**2 / f_var
                    # Sum over dimensions and average over samples
                    log_prob_batch = log_prob.sum(dim=-1).mean(dim=0)  # [batch_size]
                    total_log_prob += log_prob_batch.sum().item()
                    total_count += current_batch_size
            
            # Compute final NLPD
            if total_count == 0:
                model.train()
                return float('nan')
            
            # Negative log predictive density (average over all samples)
            nlpd = -total_log_prob / total_count
            
        model.train()
        return nlpd

    def compute_crps(self,
                     model,
                     y_true: torch.Tensor,
                     n_samples: int = 50,
                     use_sample_estimator: bool = False) -> float:
        """
        Compute Continuous Ranked Probability Score mirroring ``compute_nlpd``.

        Uses the identical SSVI predictive path as NLPD (encoder latents, U samples,
        GP mean/variance + observation noise). By default, applies closed-form
        Gaussian CRPS to each MC predictive Gaussian and averages over MC samples
        the same way NLPD averages log-density — then sums over observed dimensions
        and averages over datapoints (NLPD-style reduction).

        If ``use_sample_estimator`` is True, uses
            E|X − y| − (1/2) E|X − X′|
        on draws from those same predictive Gaussians.

        Lower is better.
        """
        model.eval()
        with torch.no_grad():
            # Same U / epsilon sampling as compute_nlpd
            epsilon = model.get_epsilon()
            U_samples = model.get_u_samples(epsilon)

            N = y_true.shape[0]
            batch_size = self.metrics_batch_size

            total_crps = 0.0
            total_count = 0

            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = y_true[start_idx:end_idx]
                current_batch_size = end_idx - start_idx

                # Same encoder preprocessing as compute_nlpd
                y_encoder_input = torch.where(
                    torch.isnan(y_batch),
                    torch.full_like(y_batch, -9.0, dtype=torch.float64),
                    y_batch,
                )

                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
                x_dist = torch.distributions.Normal(mu_x, sigma_x)
                x_samples = x_dist.rsample()  # [n_samples, batch_size, latent_dim]

                f_mean = model.gp.get_gp_expectation_given_U(x_samples, U_samples)
                gp_cov = model.gp.get_gp_covariance_given_U(x_samples, U_samples)

                if gp_cov.shape[-1] == 1:
                    f_var = gp_cov[..., 0, 0] + model.sigma2  # [n_samples, batch, M]
                    f_mean = f_mean[..., 0]
                else:
                    f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2

                f_std = f_var.clamp(min=_CRPS_SIGMA_EPS).sqrt()

                if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                    observed_mask = ~torch.isnan(y_batch)
                    if observed_mask.sum() == 0:
                        continue
                    y_obs = torch.where(observed_mask, y_batch, torch.zeros_like(y_batch))
                    mask_f = observed_mask.float()[None, :, :]  # [1, batch, M]
                else:
                    observed_mask = None
                    y_obs = y_batch
                    mask_f = None

                if use_sample_estimator:
                    y_pred_samples = torch.distributions.Normal(f_mean, f_std).rsample()
                    # sample_crps returns [batch, M]; treat as single score map
                    crps_vals = sample_crps(y_pred_samples, y_obs)  # [batch, M]
                    if mask_f is not None:
                        crps_per_point = (crps_vals * observed_mask.float()).sum(dim=-1)  # [batch]
                    else:
                        crps_per_point = crps_vals.sum(dim=-1)
                else:
                    # Mirror NLPD: score each MC Gaussian, then average over MC samples
                    # gaussian_crps broadcasts over leading sample dim → [n_samples, batch, M]
                    crps_vals = gaussian_crps(f_mean, f_std, y_obs)
                    if mask_f is not None:
                        crps_per_point = (crps_vals * mask_f).sum(dim=-1).mean(dim=0)  # [batch]
                    else:
                        crps_per_point = crps_vals.sum(dim=-1).mean(dim=0)  # [batch]

                total_crps += crps_per_point.sum().item()
                total_count += current_batch_size

            if total_count == 0:
                model.train()
                return float('nan')

            crps = total_crps / total_count

        model.train()
        return crps

    def compute_coverage(self,
                         model,
                         y_true: torch.Tensor,
                         n_samples: int = 50,
                         levels: Tuple[float, ...] = COVERAGE_LEVELS,
                         use_sample_quantiles: bool = False) -> Dict[str, float]:
        """
        Predictive interval coverage and mean width (MPIW), mirroring the SSVI NLPD path.

        Uses encoder latents, U samples, and GP mean/variance + observation noise
        (full posterior predictive variance). Moment-matches MC components to a
        single Gaussian per entry (or empirical quantiles if
        ``use_sample_quantiles``), builds central intervals at each ``level``,
        and returns:

            coverage = (1 / N_obs) Σ_{n,d} 1[y_nd ∈ [L_nd, U_nd]]
            interval_width = (1 / N_obs) Σ_{n,d} (U_nd − L_nd)

        over observed entries. Keys: ``coverage_90``, ``coverage_95``,
        ``interval_width_90``, ``interval_width_95``, ...
        """
        model.eval()
        results = {}
        for level in levels:
            pct = int(round(level * 100))
            results[f'coverage_{pct}'] = float('nan')
            results[f'interval_width_{pct}'] = float('nan')

        with torch.no_grad():
            epsilon = model.get_epsilon()
            U_samples = model.get_u_samples(epsilon)

            N = y_true.shape[0]
            batch_size = self.metrics_batch_size

            cov_totals = {f'coverage_{int(round(level * 100))}': 0.0 for level in levels}
            width_totals = {f'interval_width_{int(round(level * 100))}': 0.0 for level in levels}
            total_count = 0

            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = y_true[start_idx:end_idx]

                y_encoder_input = torch.where(
                    torch.isnan(y_batch),
                    torch.full_like(y_batch, -9.0, dtype=torch.float64),
                    y_batch,
                )

                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
                x_dist = torch.distributions.Normal(mu_x, sigma_x)
                x_samples = x_dist.rsample()

                f_mean = model.gp.get_gp_expectation_given_U(x_samples, U_samples)
                gp_cov = model.gp.get_gp_covariance_given_U(x_samples, U_samples)

                if gp_cov.shape[-1] == 1:
                    f_var = gp_cov[..., 0, 0] + model.sigma2
                    f_mean = f_mean[..., 0]
                else:
                    f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2

                if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                    observed_mask = ~torch.isnan(y_batch)
                    if observed_mask.sum() == 0:
                        continue
                    y_obs = torch.where(observed_mask, y_batch, torch.zeros_like(y_batch))
                else:
                    observed_mask = torch.ones_like(y_batch, dtype=torch.bool)
                    y_obs = y_batch

                f_std = f_var.clamp(min=_CRPS_SIGMA_EPS).sqrt()
                mask_f = observed_mask.float()

                if use_sample_quantiles:
                    y_pred_samples = torch.distributions.Normal(f_mean, f_std).rsample()
                    for level in levels:
                        pct = int(round(level * 100))
                        inside = sample_interval_coverage(y_pred_samples, y_obs, level)
                        width = sample_interval_width(y_pred_samples, level)
                        cov_totals[f'coverage_{pct}'] += (inside * mask_f).sum().item()
                        width_totals[f'interval_width_{pct}'] += (width * mask_f).sum().item()
                else:
                    # One predictive Gaussian per entry via moment matching (includes noise)
                    mu_pred = f_mean.mean(dim=0)
                    var_pred = f_var.mean(dim=0) + f_mean.var(dim=0, unbiased=False)
                    sigma_pred = var_pred.clamp(min=_CRPS_SIGMA_EPS).sqrt()
                    for level in levels:
                        pct = int(round(level * 100))
                        inside = gaussian_interval_coverage(mu_pred, sigma_pred, y_obs, level)
                        width = gaussian_interval_width(mu_pred, sigma_pred, level)
                        cov_totals[f'coverage_{pct}'] += (inside * mask_f).sum().item()
                        width_totals[f'interval_width_{pct}'] += (width * mask_f).sum().item()

                total_count += observed_mask.sum().item()

            if total_count == 0:
                model.train()
                return results

            for key, total in cov_totals.items():
                results[key] = total / total_count
            for key, total in width_totals.items():
                results[key] = total / total_count

        model.train()
        return results    
    def compute_missing_data_nlpd(self, 
                     model, 
                     y_original: torch.Tensor, 
                     missing_mask: torch.Tensor,
                     n_samples: int = 50) -> float:
        """
        Compute Negative Log Predictive Density specifically for missing data dimensions using mini-batching.
        
        This follows the same approach as nlpd_utils.compute_missing_data_nlpd_gauss:
        1. Use y_original (complete data) as encoder input
        2. Apply reverse mask: set observed values to NaN, leave missing values as they were
        3. Compute probabilities only on the originally missing dimensions
        
        Args:
            model: SSVI GP-LVM model
            y_original: Original complete data before missing data preprocessing [N, D]
            missing_mask: Boolean mask [N, D] where True=missing
            n_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            NLPD value for missing data dimensions only (lower is better)
        """
        model.eval()
        with torch.no_grad():
            # Check if we have any missing dimensions to evaluate
            if missing_mask.sum() == 0:
                # No missing data - return NaN
                model.train()
                return float('nan')
            
            # Sample U once for all batches
            epsilon = model.get_epsilon()
            U_samples = model.get_u_samples(epsilon)
            
            N = y_original.shape[0]
            batch_size = self.metrics_batch_size
            
            # Accumulators for batch-wise computation
            total_log_prob = 0.0
            total_count = 0
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = y_original[start_idx:end_idx]
                mask_batch = missing_mask[start_idx:end_idx]
                current_batch_size = end_idx - start_idx
                
                # Preprocess encoder input to handle missing data (consistent with training)
                # Create masked data by setting missing dimensions to -9.0
                y_encoder_input = y_batch.clone()
                y_encoder_input[mask_batch] = -9.0
                
                # Get latent representations for this batch using SSVI encoder
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
                
                # Compute averaged latent variables once before the loop (optimization)
                x_dist = torch.distributions.Normal(mu_x, sigma_x)
                x_samples = x_dist.rsample()  # [n_samples, batch_size, latent_dim]            
                
                # Get GP predictive mean using GP method directly
                f_mean = model.gp.get_gp_expectation_given_U(x_samples, U_samples)  # [n_samples, batch_size, M, C]
                
                # Get proper GP predictive variance: GP covariance + observation noise
                gp_cov = model.gp.get_gp_covariance_given_U(x_samples, U_samples)  # [n_samples, batch_size, M, C, C]

                # For observed data (C=1 case), extract diagonal and add observation noise
                if gp_cov.shape[-1] == 1:
                    f_var = gp_cov[..., 0, 0] + model.sigma2  # [n_samples, batch_size, M]
                    f_mean = f_mean[..., 0]  # [n_samples, batch_size, M] to match f_var
                else:
                    # For multi-output case, use diagonal elements + observation noise
                    f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2  # [n_samples, batch_size, M, C]
                
                # Compute log probability under Gaussian only for missing dimensions
                log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (y_batch - f_mean)**2 / f_var
                
                # Apply missing data mask to only include missing dimensions
                # Sum over missing dimensions and average over samples
                log_prob_masked = (log_prob * mask_batch.float()[None, :, :]).sum(dim=-1).mean(dim=0)  # [batch_size]
                total_log_prob += log_prob_masked.sum().item()
                total_count += current_batch_size
            
            # Compute final NLPD
            if total_count == 0:
                model.train()
                return float('nan')
            
            # Negative log predictive density (average over all samples)
            nlpd = -total_log_prob / total_count
            
        model.train()
        return nlpd
    
    def compute_effective_latent_dims(self, model, threshold: float = 0.1) -> Tuple[int, int]:
        """
        Compute number of effective latent dimensions based on lengthscales
        
        Args:
            model: SSVI GP-LVM model
            threshold: Threshold for considering a dimension effective (relative to max)
            
        Returns:
            (effective_dims, total_dims)
        """
        with torch.no_grad():
            # Get lengthscales (1/beta)
            if hasattr(model.gp, 'beta'):
                lengthscales = 1.0 / model.gp.beta.flatten()
            else:
                # Fallback if beta not available
                return 0, 0
            
            # Normalize by maximum lengthscale
            max_lengthscale = lengthscales.max()
            relative_lengthscales = lengthscales / max_lengthscale
            
            # Count dimensions above threshold
            effective_dims = (relative_lengthscales > threshold).sum().item()
            total_dims = len(lengthscales)
            
        return effective_dims, total_dims
    
    def compute_full_metrics(self, 
                           model, 
                           y_train: torch.Tensor, 
                           y_test: torch.Tensor,
                           iteration: int = 0,
                           elbo_loss: float = 0.0,
                           n_samples: int = 20,
                           y_train_original: torch.Tensor = None,
                           y_test_original: torch.Tensor = None,
                           train_missing_mask: torch.Tensor = None,
                           test_missing_mask: torch.Tensor = None) -> SSVIMetrics:
        """
        Compute all metrics for current model state
        
        Args:
            model: SSVI GP-LVM model
            y_train: Training data (with missing values if enable_missing_data=True)
            y_test: Test data (with missing values if enable_missing_data=True)
            iteration: Current training iteration
            elbo_loss: Current ELBO loss
            n_samples: Number of samples for Monte Carlo
            y_train_original: Original training data before missing data preprocessing (optional)
            y_test_original: Original test data before missing data preprocessing (optional)
            train_missing_mask: Boolean mask for training data missing dimensions (optional)
            test_missing_mask: Boolean mask for test data missing dimensions (optional)
            
        Returns:
            SSVIMetrics object with all computed metrics
        """
        metrics = SSVIMetrics()
        metrics.iteration = iteration
        metrics.elbo_loss = elbo_loss
        
        # Reconstruction metrics using SSVI encoder
        metrics.train_rmse = self.compute_rmse(model, y_train, n_samples)
        metrics.test_rmse = self.compute_rmse(model, y_test, n_samples)
        
        # Uncertainty metrics (computationally expensive, use fewer samples)
        metrics.train_nlpd = self.compute_nlpd(model, y_train, n_samples)
        metrics.test_nlpd = self.compute_nlpd(model, y_test, n_samples)

        # Continuous Ranked Probability Score (Gaussian closed form by default)
        metrics.train_crps = self.compute_crps(model, y_train, n_samples)
        metrics.test_crps = self.compute_crps(model, y_test, n_samples)
        metrics.crps = metrics.test_crps  # held-out scalar for evaluation output

        # Predictive interval coverage + mean width (full predictive variance incl. likelihood noise)
        train_cov = self.compute_coverage(model, y_train, n_samples)
        test_cov = self.compute_coverage(model, y_test, n_samples)
        metrics.train_coverage_90 = train_cov.get('coverage_90', float('nan'))
        metrics.train_coverage_95 = train_cov.get('coverage_95', float('nan'))
        metrics.test_coverage_90 = test_cov.get('coverage_90', float('nan'))
        metrics.test_coverage_95 = test_cov.get('coverage_95', float('nan'))
        metrics.coverage_90 = metrics.test_coverage_90
        metrics.coverage_95 = metrics.test_coverage_95
        metrics.train_interval_width_90 = train_cov.get('interval_width_90', float('nan'))
        metrics.train_interval_width_95 = train_cov.get('interval_width_95', float('nan'))
        metrics.test_interval_width_90 = test_cov.get('interval_width_90', float('nan'))
        metrics.test_interval_width_95 = test_cov.get('interval_width_95', float('nan'))
        metrics.interval_width_90 = metrics.test_interval_width_90
        metrics.interval_width_95 = metrics.test_interval_width_95
        
        # Missing data metrics (only if missing data is enabled and original data/masks are provided)
        if (hasattr(model, 'enable_missing_data') and model.enable_missing_data and
            y_train_original is not None and y_test_original is not None and 
            train_missing_mask is not None and test_missing_mask is not None):
            metrics.train_missing_nlpd = self.compute_missing_data_nlpd(
                model, y_train_original, train_missing_mask, n_samples
            )
            metrics.test_missing_nlpd = self.compute_missing_data_nlpd(
                model, y_test_original, test_missing_mask, n_samples
            )
            metrics.train_missing_rmse = self.compute_missing_data_rmse(
                model, y_train_original, train_missing_mask, n_samples
            )
            metrics.test_missing_rmse = self.compute_missing_data_rmse(
                model, y_test_original, test_missing_mask, n_samples
            )
           

        # Model complexity
        effective_dims, total_dims = self.compute_effective_latent_dims(model)
        metrics.effective_latent_dims = effective_dims
        metrics.total_latent_dims = total_dims
            
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of final metrics"""
        if not self.metrics_history:
            return {}
        
        final_metrics = self.metrics_history[-1]
        
        return {
            'final_train_rmse': final_metrics.train_rmse,
            'final_test_rmse': final_metrics.test_rmse,
            'final_train_nlpd': final_metrics.train_nlpd,
            'final_test_nlpd': final_metrics.test_nlpd,
            'final_train_crps': final_metrics.train_crps,
            'final_test_crps': final_metrics.test_crps,
            'crps': final_metrics.crps,
            'final_train_coverage_90': final_metrics.train_coverage_90,
            'final_train_coverage_95': final_metrics.train_coverage_95,
            'final_test_coverage_90': final_metrics.test_coverage_90,
            'final_test_coverage_95': final_metrics.test_coverage_95,
            'coverage_90': final_metrics.coverage_90,
            'coverage_95': final_metrics.coverage_95,
            'final_train_interval_width_90': final_metrics.train_interval_width_90,
            'final_train_interval_width_95': final_metrics.train_interval_width_95,
            'final_test_interval_width_90': final_metrics.test_interval_width_90,
            'final_test_interval_width_95': final_metrics.test_interval_width_95,
            'interval_width_90': final_metrics.interval_width_90,
            'interval_width_95': final_metrics.interval_width_95,
            'final_train_missing_nlpd': final_metrics.train_missing_nlpd,
            'final_test_missing_nlpd': final_metrics.test_missing_nlpd,
            'final_train_missing_rmse': final_metrics.train_missing_rmse,
            'final_test_missing_rmse': final_metrics.test_missing_rmse,
            'final_effective_dims': final_metrics.effective_latent_dims,
            'total_latent_dims': final_metrics.total_latent_dims,
            'overfitting_ratio': final_metrics.test_rmse / final_metrics.train_rmse if final_metrics.train_rmse > 0 else float('inf'),
            'total_iterations': len(self.metrics_history)
        }
    


def create_train_test_split(y: torch.Tensor, 
                          labels: torch.Tensor,
                          test_size: float = 0.2,
                          random_seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create stratified train/test split preserving class balance using sklearn
    
    Args:
        y: Data tensor [N, D]
        labels: Label tensor [N]
        test_size: Fraction for test set
        random_seed: Random seed for reproducibility (uses sklearn's random_state, no PyTorch seed manipulation)
        
    Returns:
        (y_train, y_test, labels_train, labels_test, train_indices, test_indices)
    """
    # Convert to numpy for sklearn compatibility
    y_np = y.cpu().numpy()
    labels_np = labels.cpu().numpy()
    indices = np.arange(len(y))
    
    # Use sklearn's stratified train_test_split (no PyTorch seed manipulation)
    train_indices, test_indices, y_train_np, y_test_np, labels_train_np, labels_test_np = train_test_split(
        indices, y_np, labels_np,
        test_size=test_size,
        stratify=labels_np,
        random_state=random_seed
    )
    
    # Convert back to tensors on original device
    device = y.device
    dtype = y.dtype  # Preserve original dtype for data tensors
    y_train = torch.from_numpy(y_train_np).to(device=device, dtype=dtype)
    y_test = torch.from_numpy(y_test_np).to(device=device, dtype=dtype)
    labels_train = torch.from_numpy(labels_train_np).to(device=device, dtype=dtype)
    labels_test = torch.from_numpy(labels_test_np).to(device=device, dtype=dtype)
    train_indices = torch.from_numpy(train_indices).to(device=device, dtype=torch.int64)
    test_indices = torch.from_numpy(test_indices).to(device=device, dtype=torch.int64)
    
    return y_train, y_test, labels_train, labels_test, train_indices, test_indices
