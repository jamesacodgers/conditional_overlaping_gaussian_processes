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
            'effective_latent_dims': self.effective_latent_dims,
            'total_latent_dims': self.total_latent_dims,
            'elbo_loss': self.elbo_loss
        }


class SSVIMetricsComputer:
    """Compute evaluation metrics for SSVI GP-LVM models"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.metrics_history: List[SSVIMetrics] = []
    
    def _sample_latent_variables(self, mu_x: torch.Tensor, sigma_x: torch.Tensor, model) -> torch.Tensor:
        """
        Sample latent variables from the posterior distribution q(x) = N(mu_x, sigma_x^2)
        
        Args:
            mu_x: Latent means [P, batch_size, latent_dim]
            sigma_x: Latent standard deviations [P, batch_size, latent_dim]
            model: GP-LVM model (used to get P value)
            
        Returns:
            x_sample: Sampled latent variables [batch_size, latent_dim]
        """
        # Sample one of the P samples randomly
        sample_idx = torch.randint(0, model.P, (1,)).item()
        mu_sample = mu_x[sample_idx]  # [batch_size, latent_dim]
        sigma_sample = sigma_x[sample_idx]  # [batch_size, latent_dim]
        
        # Sample from the latent variable distribution q(x) = N(mu_x, sigma_x^2)
        epsilon = torch.randn_like(sigma_sample)
        x_sample = mu_sample + sigma_sample * epsilon
        
        return x_sample
    
    def compute_rmse(self, 
                     model, 
                     y_true: torch.Tensor, 
                     batch_indices: Optional[torch.Tensor] = None,
                     n_samples: int = 10,
                     is_test_data: bool = False) -> float:
        """
        Compute Root Mean Square Error for reconstruction
        
        Args:
            model: GP-LVM model (SSVI or structured)
            y_true: True observations [N, D] or [batch_size, D]
            batch_indices: Optional indices if using subset of data
            n_samples: Number of samples for Monte Carlo estimation
            is_test_data: Whether this is test data (determines which method to call)
            
        Returns:
            RMSE value
        """
        model.eval()
        with torch.no_grad():
            if batch_indices is not None:
                # Get subset of data
                y_batch = y_true[batch_indices]
            else:
                # Use full dataset
                y_batch = y_true
                N = y_true.shape[0]
                batch_indices = torch.arange(N, device=y_true.device)
            
            # Sample U for both model types
            U_samples = model.get_u_samples()
            
            # Preprocess encoder input to handle missing data (consistent with training)
            y_encoder_input = torch.where(torch.isnan(y_batch), torch.zeros_like(y_batch), y_batch)
            
            # Get latent representations for this batch using explicit method
            if is_test_data:
                mu_x, sigma_x = model.get_mu_sigma_x_test_samples(y_encoder_input, U_samples)
            else:
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U_samples)
            
            # Sample from latent distribution for reconstruction
            reconstructions = []
            for i in range(n_samples):
                # Sample latent variables using helper function
                x_sample = self._sample_latent_variables(mu_x, sigma_x, model)
                
                # Reconstruct through GP mean prediction using proper GP method
                y_recon = model.gp.get_gp_expectation(x_sample)  # [batch_size, M, C]
                
                if y_recon.shape[-1] == 1:
                    y_recon = y_recon[..., 0]  # [batch_size, M] for C=1 case
                
                reconstructions.append(y_recon)
            
            # Average reconstructions
            y_pred = torch.stack(reconstructions).mean(dim=0)  # [batch_size, D]
            
            # Compute RMSE - handle missing data if enabled
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                # Handle missing data: create mask for observed values
                missing_mask = torch.isnan(y_batch)
                observed_mask = ~missing_mask
                
                # Check if we have any observed data
                if observed_mask.sum() == 0:
                    # All data is missing - return NaN
                    model.train()
                    return float('nan')
                
                # Compute RMSE only on observed values
                # Replace NaN values in y_batch with zeros for computation (they'll be masked out)
                y_batch_clean = torch.where(observed_mask, y_batch, torch.zeros_like(y_batch))
                squared_errors = (y_batch_clean - y_pred) ** 2
                observed_squared_errors = squared_errors * observed_mask.float()
                mse = observed_squared_errors.sum() / observed_mask.sum()
            else:
                # Standard RMSE computation for complete data
                mse = torch.mean((y_batch - y_pred) ** 2)
            
            rmse = torch.sqrt(mse).item()
            
        model.train()
        return rmse
    
    def compute_nlpd(self, 
                     model, 
                     y_true: torch.Tensor, 
                     batch_indices: Optional[torch.Tensor] = None,
                     n_samples: int = 50,
                     is_test_data: bool = False) -> float:
        """
        Compute Negative Log Predictive Density for uncertainty quantification
        
        Args:
            model: GP-LVM model (SSVI or structured)
            y_true: True observations [N, D] or [batch_size, D]
            batch_indices: Optional indices if using subset of data
            n_samples: Number of samples for Monte Carlo estimation
            is_test_data: Whether this is test data (determines which method to call)
            
        Returns:
            NLPD value (lower is better)
        """
        model.eval()
        with torch.no_grad():
            if batch_indices is not None:
                y_batch = y_true[batch_indices]
            else:
                y_batch = y_true
                N = y_true.shape[0]
                batch_indices = torch.arange(N, device=y_true.device)
            
            # Sample U for both model types
            U_samples = model.get_u_samples()
            
            # Preprocess encoder input to handle missing data (consistent with training)
            y_encoder_input = torch.where(torch.isnan(y_batch), torch.zeros_like(y_batch), y_batch)
            
            # Get latent representations for this batch using explicit method
            if is_test_data:
                mu_x, sigma_x = model.get_mu_sigma_x_test_samples(y_encoder_input, U_samples)
            else:
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U_samples)
            
            log_probs = []
            
            for _ in range(n_samples):
                # Sample latent variables using helper function
                x_sample = self._sample_latent_variables(mu_x, sigma_x, model)
                
                # Compute predictive distribution through GP using proper GP methods
                
                # Get GP predictive mean using GP method directly
                f_mean = model.gp.get_gp_expectation(x_sample)  # [batch_size, M, C]
                
                # Get proper GP predictive variance: GP covariance + observation noise
                gp_cov = model.gp.get_gp_covariance(x_sample)  # [batch_size, M, C, C]
                
                # For observed data (C=1 case), extract diagonal and add observation noise
                if gp_cov.shape[-1] == 1:
                    f_var = gp_cov[..., 0, 0] + model.sigma2  # [batch_size, M]
                    f_mean = f_mean[..., 0]  # [batch_size, M] to match f_var
                else:
                    # For multi-output case, use diagonal elements + observation noise
                    f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2  # [batch_size, M, C]
                
                # Handle missing data if enabled
                if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                    # Mask out missing values in log probability computation
                    missing_mask = torch.isnan(y_batch)
                    observed_mask = ~missing_mask
                    
                    # Check if we have any observed data
                    if observed_mask.sum() == 0:
                        # All data is missing - return NaN
                        model.train()
                        return float('nan')
                    
                    # Replace NaN values in y_batch with zeros for computation (they'll be masked out)
                    y_batch_clean = torch.where(observed_mask, y_batch, torch.zeros_like(y_batch))
                    
                    # Compute log probability under Gaussian
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (y_batch_clean - f_mean)**2 / f_var
                    
                    log_prob_masked = log_prob * observed_mask.float()
                    log_probs.append(log_prob_masked.sum(dim=-1))  # Sum over observed dimensions only
                else:
                    # Standard computation for complete data
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (y_batch - f_mean)**2 / f_var
                    log_probs.append(log_prob.sum(dim=-1))  # Sum over dimensions
                    
            
            # Average log probabilities (log of mixture)
            log_probs = torch.stack(log_probs)  # [n_samples, batch_size]
            # Use logsumexp for numerical stability
            log_pred_density = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
            
            # Negative log predictive density (average over batch)
            nlpd = -log_pred_density.mean().item()
            
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
                           train_indices: Optional[torch.Tensor] = None,
                           test_indices: Optional[torch.Tensor] = None,
                           iteration: int = 0,
                           elbo_loss: float = 0.0,
                           n_samples: int = 20) -> SSVIMetrics:
        """
        Compute all metrics for current model state
        
        Args:
            model: SSVI GP-LVM model
            y_train: Training data
            y_test: Test data  
            train_indices: Optional training indices for subset
            test_indices: Optional test indices for subset
            iteration: Current training iteration
            elbo_loss: Current ELBO loss
            n_samples: Number of samples for Monte Carlo
            
        Returns:
            SSVIMetrics object with all computed metrics
        """
        metrics = SSVIMetrics()
        metrics.iteration = iteration
        metrics.elbo_loss = elbo_loss
        
        try:
            # Reconstruction metrics with explicit train/test flags
            metrics.train_rmse = self.compute_rmse(model, y_train, train_indices, n_samples, is_test_data=False)
            metrics.test_rmse = self.compute_rmse(model, y_test, test_indices, n_samples, is_test_data=True)
            
            # Uncertainty metrics (computationally expensive, use fewer samples)
            uncertainty_samples = min(10, n_samples)
            metrics.train_nlpd = self.compute_nlpd(model, y_train, train_indices, uncertainty_samples, is_test_data=False)
            metrics.test_nlpd = self.compute_nlpd(model, y_test, test_indices, uncertainty_samples, is_test_data=True)
            
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
    y_train = torch.from_numpy(y_train_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)
    labels_train = torch.from_numpy(labels_train_np).to(device)
    labels_test = torch.from_numpy(labels_test_np).to(device)
    train_indices = torch.from_numpy(train_indices).to(device)
    test_indices = torch.from_numpy(test_indices).to(device)
    
    return y_train, y_test, labels_train, labels_test, train_indices, test_indices
