"""
SSVI GP-LVM Evaluation Metrics

Implementation of metrics used in "Generalised Gaussian Process Latent Variable Models (GPLVM) 
with Stochastic Variational Inference" paper for evaluating model performance.

Key metrics:
- RMSE: Root Mean Square Error for reconstruction quality
- NLPD: Negative Log Predictive Density for uncertainty quantification
- Train/Test performance comparison for overfitting detection
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


@dataclass
class SSVIMetrics:
    """Container for SSVI GP-LVM evaluation metrics"""
    
    # Reconstruction metrics
    train_rmse: float = 0.0
    test_rmse: float = 0.0
    
    # Uncertainty metrics
    train_nlpd: float = 0.0
    test_nlpd: float = 0.0
    
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
            epsilon = torch.randn(model.P, model.M, model.C*model.Lx)
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
            epsilon = torch.randn(model.P, model.M, model.C*model.Lx)
            # Sample U once for all batches
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
            epsilon = torch.randn(model.P, model.M, model.C*model.Lx)
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
            epsilon = torch.randn(model.P, model.M, model.C*model.Lx)
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
        
        try:
            # Reconstruction metrics using SSVI encoder
            metrics.train_rmse = self.compute_rmse(model, y_train, n_samples)
            metrics.test_rmse = self.compute_rmse(model, y_test, n_samples)
            
            # Uncertainty metrics (computationally expensive, use fewer samples)
            metrics.train_nlpd = self.compute_nlpd(model, y_train, n_samples)
            metrics.test_nlpd = self.compute_nlpd(model, y_test, n_samples)
            
            # Missing data metrics (only if missing data is enabled and original data/masks are provided)
            if (hasattr(model, 'enable_missing_data') and model.enable_missing_data and
                y_train_original is not None and y_test_original is not None and 
                train_missing_mask is not None and test_missing_mask is not None):
                try:
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
                except Exception as e:
                    print(f"Warning: Error computing missing data metrics at iteration {iteration}: {e}")
                    # Keep default values (nan) for failed metrics

            # Model complexity
            effective_dims, total_dims = self.compute_effective_latent_dims(model)
            metrics.effective_latent_dims = effective_dims
            metrics.total_latent_dims = total_dims
            
        except Exception as e:
            print(f"Warning: Error computing metrics at iteration {iteration}: {e}")
            # Keep default values (0.0) for failed metrics
        
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
