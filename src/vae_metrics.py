"""
VAE-specific Evaluation Metrics

Metrics tailored for VAE models (RMSE, ELBO, KL, LL).
No NLPD or missing data support - those are for GP models.
Preserves same output structure as SSVI for easy comparison.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class VAEMetrics:
    """Container for VAE evaluation metrics"""
    
    # Reconstruction metrics
    train_rmse: float = 0.0
    test_rmse: float = 0.0
    
    # VAE-specific metrics
    train_elbo: float = 0.0
    test_elbo: float = 0.0
    train_kl: float = 0.0
    test_kl: float = 0.0
    train_ll: float = 0.0
    test_ll: float = 0.0
    
    # Training progression
    iteration: int = 0
    elbo_loss: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for logging"""
        return {
            'iteration': self.iteration,
            'train_rmse': self.train_rmse,
            'test_rmse': self.test_rmse,
            'train_elbo': self.train_elbo,
            'test_elbo': self.test_elbo,
            'train_kl': self.train_kl,
            'test_kl': self.test_kl,
            'train_ll': self.train_ll,
            'test_ll': self.test_ll,
            'elbo_loss': self.elbo_loss
        }


class VAEMetricsComputer:
    """Compute evaluation metrics for VAE models"""
    
    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Device to use for computations
        """
        self.device = device
        self.metrics_history: List[VAEMetrics] = []
    
    def compute_rmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Compute Root Mean Square Error.
        
        Args:
            y_true: (N, M) true observations
            y_pred: (N, M) predicted observations
            
        Returns:
            RMSE value
        """
        mse = torch.mean((y_true - y_pred) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()
    
    def compute_vae_metrics(
        self,
        vae_model,
        y: torch.Tensor,
        num_samples: int = 32
    ) -> Tuple[float, float, float, float]:
        """
        Compute VAE-specific metrics (ELBO, KL, LL, RMSE).
        
        Args:
            vae_model: VAE model instance
            y: (N, M) observations (flat format)
            num_samples: Number of samples for ELBO estimation
            
        Returns:
            elbo, kl, ll, rmse
        """
        vae_model.eval()
        with torch.no_grad():
            # Compute ELBO (VAE accepts flat vectors directly for MLP)
            elbo, metrics = vae_model.elbo(y, num_samples=num_samples)
            
            # Extract metrics
            elbo_val = metrics['elbo'] if 'elbo' in metrics else metrics.get('beta_nll_elbo', 0.0)
            kl_val = metrics['kl']
            ll_val = metrics['ll']
            rmse_val = metrics['RMSE']
            
            # Convert to floats
            elbo_mean = elbo_val.item() if hasattr(elbo_val, 'item') else float(elbo_val)
            kl_mean = kl_val.item() if hasattr(kl_val, 'item') else float(kl_val)
            ll_mean = ll_val.item() if hasattr(ll_val, 'item') else float(ll_val)
            rmse_mean = rmse_val.item() if hasattr(rmse_val, 'item') else float(rmse_val)
        
        vae_model.train()
        return elbo_mean, kl_mean, ll_mean, rmse_mean
    
    def compute_full_metrics(
        self,
        vae_model,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        iteration: int,
        elbo_loss: float,
        n_samples: int = 32
    ) -> VAEMetrics:
        """
        Compute comprehensive metrics on train and test sets.
        
        Args:
            vae_model: VAE model instance
            y_train: (N_train, M) training observations
            y_test: (N_test, M) test observations
            iteration: Current training iteration
            elbo_loss: Current ELBO loss
            n_samples: Number of samples for metrics computation
            
        Returns:
            VAEMetrics instance with all computed metrics
        """
        # Compute train metrics
        train_elbo, train_kl, train_ll, train_rmse = self.compute_vae_metrics(
            vae_model, y_train, num_samples=n_samples
        )
        
        # Compute test metrics
        test_elbo, test_kl, test_ll, test_rmse = self.compute_vae_metrics(
            vae_model, y_test, num_samples=n_samples
        )
        
        # Create metrics object
        metrics = VAEMetrics(
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            train_elbo=train_elbo,
            test_elbo=test_elbo,
            train_kl=train_kl,
            test_kl=test_kl,
            train_ll=train_ll,
            test_ll=test_ll,
            iteration=iteration,
            elbo_loss=elbo_loss
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_metrics_summary(self) -> Dict:
        """
        Get summary of metrics across training.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}
        
        final_metrics = self.metrics_history[-1]
        
        summary = {
            'final_train_rmse': final_metrics.train_rmse,
            'final_test_rmse': final_metrics.test_rmse,
            'final_train_elbo': final_metrics.train_elbo,
            'final_test_elbo': final_metrics.test_elbo,
            'final_train_kl': final_metrics.train_kl,
            'final_test_kl': final_metrics.test_kl,
            'final_train_ll': final_metrics.train_ll,
            'final_test_ll': final_metrics.test_ll,
            'overfitting_ratio': final_metrics.test_rmse / final_metrics.train_rmse if final_metrics.train_rmse > 0 else 1.0,
        }
        
        # Add progression statistics
        if len(self.metrics_history) > 1:
            train_rmse_values = [m.train_rmse for m in self.metrics_history]
            test_rmse_values = [m.test_rmse for m in self.metrics_history]
            
            summary['best_train_rmse'] = min(train_rmse_values)
            summary['best_test_rmse'] = min(test_rmse_values)
            summary['worst_train_rmse'] = max(train_rmse_values)
            summary['worst_test_rmse'] = max(test_rmse_values)
        
        return summary


def create_train_test_split(
    y_full: torch.Tensor,
    labels_full: torch.Tensor,
    test_size: float = 0.1,
    random_seed: int = 7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create train/test split for data.
    Same interface as SSVI for consistency.
    
    Args:
        y_full: (N, M) full dataset
        labels_full: (N,) labels
        test_size: Fraction for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        y_train, y_test, labels_train, labels_test, train_indices, test_indices
    """
    N = y_full.shape[0]
    device = y_full.device
    
    # Create indices and split
    indices = torch.arange(N, device=device)
    train_indices, test_indices = train_test_split(
        indices.cpu().numpy(),
        test_size=test_size,
        random_state=random_seed
    )
    
    # Convert back to tensors
    train_indices = torch.tensor(train_indices, dtype=torch.long, device=device)
    test_indices = torch.tensor(test_indices, dtype=torch.long, device=device)
    
    # Split data
    y_train = y_full[train_indices]
    y_test = y_full[test_indices]
    labels_train = labels_full[train_indices]
    labels_test = labels_full[test_indices]
    
    return y_train, y_test, labels_train, labels_test, train_indices, test_indices

