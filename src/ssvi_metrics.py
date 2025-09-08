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
    
    def compute_rmse(self, 
                     model, 
                     y_true: torch.Tensor, 
                     batch_indices: Optional[torch.Tensor] = None,
                     n_samples: int = 10) -> float:
        """
        Compute Root Mean Square Error for reconstruction
        
        Args:
            model: SSVI GP-LVM model
            y_true: True observations [N, D] or [batch_size, D]
            batch_indices: Optional indices if using subset of data
            n_samples: Number of samples for Monte Carlo estimation
            
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
            
            # Sample U for amortized inference
            U_samples = model.get_u_samples()
            
            # Get latent representations for this batch
            mu_x, sigma_x = model.get_mu_sigma_x_samples(y_batch, U_samples)
            
            # Sample from latent distribution for reconstruction
            reconstructions = []
            for _ in range(n_samples):
                # Sample latent variables
                if hasattr(model, 'P') and model.P > 1:
                    # Use one of the SSVI samples
                    sample_idx = torch.randint(0, model.P, (1,)).item()
                    x_sample = mu_x[sample_idx]  # [batch_size, latent_dim]
                else:
                    # Standard VAE-style sampling
                    epsilon = torch.randn_like(sigma_x)
                    x_sample = mu_x + sigma_x * epsilon
                
                # Reconstruct through GP mean prediction
                y_recon = model.get_ssvi_mean_prediction(x_sample)  # [batch_size, D]
                reconstructions.append(y_recon)
            
            # Average reconstructions
            y_pred = torch.stack(reconstructions).mean(dim=0)  # [batch_size, D]
            
            # Compute RMSE
            mse = torch.mean((y_batch - y_pred) ** 2)
            rmse = torch.sqrt(mse).item()
            
        model.train()
        return rmse
    
    def compute_nlpd(self, 
                     model, 
                     y_true: torch.Tensor, 
                     batch_indices: Optional[torch.Tensor] = None,
                     n_samples: int = 50) -> float:
        """
        Compute Negative Log Predictive Density for uncertainty quantification
        
        Args:
            model: SSVI GP-LVM model
            y_true: True observations [N, D] or [batch_size, D]
            batch_indices: Optional indices if using subset of data
            n_samples: Number of samples for Monte Carlo estimation
            
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
            
            # Sample U for amortized inference
            U_samples = model.get_u_samples()
            
            # Get latent representations for this batch
            mu_x, sigma_x = model.get_mu_sigma_x_samples(y_batch, U_samples)
            
            log_probs = []
            
            for _ in range(n_samples):
                # Sample latent variables
                if hasattr(model, 'P') and model.P > 1:
                    sample_idx = torch.randint(0, model.P, (1,)).item()
                    x_sample = mu_x[sample_idx]
                else:
                    epsilon = torch.randn_like(sigma_x)
                    x_sample = mu_x + sigma_x * epsilon
                
                # Compute predictive distribution through GP
                try:
                    # Get GP predictive mean using built-in method
                    f_mean = model.get_ssvi_mean_prediction(x_sample)  # [batch_size, D]
                    
                    # For GP predictive variance, use observation noise as approximation
                    f_var = model.sigma2 * torch.ones_like(f_mean)
                    
                    # Compute log probability under Gaussian
                    log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (y_batch - f_mean)**2 / f_var
                    log_probs.append(log_prob.sum(dim=-1))  # Sum over dimensions
                    
                except Exception as e:
                    # Fallback: use simpler computation with mean only
                    f_mean = model.get_ssvi_mean_prediction(x_sample)
                    log_prob = -0.5 * torch.sum((y_batch - f_mean)**2, dim=-1) / model.sigma2
                    log_probs.append(log_prob)
            
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
            # Reconstruction metrics
            metrics.train_rmse = self.compute_rmse(model, y_train, train_indices, n_samples)
            metrics.test_rmse = self.compute_rmse(model, y_test, test_indices, n_samples)
            
            # Uncertainty metrics (computationally expensive, use fewer samples)
            uncertainty_samples = min(10, n_samples)
            metrics.train_nlpd = self.compute_nlpd(model, y_train, train_indices, uncertainty_samples)
            metrics.test_nlpd = self.compute_nlpd(model, y_test, test_indices, uncertainty_samples)
            
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
    
    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()


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
