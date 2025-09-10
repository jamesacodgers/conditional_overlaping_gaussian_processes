"""
Missing Data Utilities for SSVI GP-LVM

Implementation of missing data handling capabilities for GP-LVM models,
following the approach from "Generalised Gaussian Process Latent Variable Models (GPLVM) 
with Stochastic Variational Inference" paper.

Key features:
- Gaussian missing data likelihood that marginalizes over unobserved dimensions
- Data masking utilities for creating missing data patterns
- Reconstruction methods for recovering missing dimensions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class GaussianMissingDataLikelihood(nn.Module):
    """
    Gaussian likelihood that can handle missing data by marginalizing over unobserved dimensions.
    
    Based on the approach from the GPLVM SVI paper where missing dimensions are handled by:
    1. Only computing likelihood terms for observed dimensions
    2. Using conditional independence given inducing variables
    3. Marginalizing out missing dimensions automatically
    """
    
    def __init__(self, sigma2: float = 1.0):
        super().__init__()
        self.sigma2 = sigma2
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                missing_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Gaussian log likelihood, handling missing data.
        
        Args:
            y_pred: Predicted values [batch_size, output_dim]
            y_true: True values [batch_size, output_dim] (may contain NaN for missing)
            missing_mask: Boolean mask [batch_size, output_dim] where True=missing (optional)
            
        Returns:
            log_likelihood: Scalar log likelihood summed over observed dimensions only
        """
        # Create missing mask from NaN values if not provided
        if missing_mask is None:
            missing_mask = torch.isnan(y_true)
        
        # Replace NaN values with zeros for computation (they'll be masked out)
        y_true_clean = torch.where(missing_mask, torch.zeros_like(y_true), y_true)
        
        # Compute squared errors
        squared_errors = (y_pred - y_true_clean) ** 2
        
        # Only include observed dimensions in likelihood computation
        observed_mask = ~missing_mask
        
        # Sum over observed dimensions only
        observed_squared_errors = squared_errors * observed_mask.float()
        
        # Count observed dimensions for normalization
        n_observed = observed_mask.sum()
        
        if n_observed == 0:
            # All data missing - return zero likelihood
            return torch.zeros(1, device=y_pred.device)
        
        # Gaussian log likelihood (without constant terms)
        log_likelihood = -0.5 * observed_squared_errors.sum() / self.sigma2
        
        return log_likelihood


def create_missing_data_pattern(data: torch.Tensor, 
                               missing_ratio: float = 0.3,
                               pattern_type: str = "random",
                               seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create missing data patterns for testing reconstruction capabilities.
    
    Args:
        data: Original data [N, D]
        missing_ratio: Fraction of dimensions to mask per data point
        pattern_type: Type of missing pattern ("random", "block", "structured")
        seed: Random seed for reproducibility
        
    Returns:
        masked_data: Data with missing values as NaN [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    N, D = data.shape
    missing_mask = torch.zeros_like(data, dtype=torch.bool)
    
    if pattern_type == "random":
        # Random missing pattern - each data point has random missing dimensions
        for i in range(N):
            n_missing = int(missing_ratio * D)
            missing_dims = torch.randperm(D)[:n_missing]
            missing_mask[i, missing_dims] = True
            
    elif pattern_type == "block":
        # Block missing pattern - contiguous blocks of missing dimensions
        for i in range(N):
            n_missing = int(missing_ratio * D)
            start_dim = torch.randint(0, D - n_missing + 1, (1,)).item()
            missing_mask[i, start_dim:start_dim + n_missing] = True
            
    elif pattern_type == "structured":
        # Structured missing - some dimensions always missing, others always observed
        always_missing_dims = torch.randperm(D)[:int(missing_ratio * D)]
        missing_mask[:, always_missing_dims] = True
        
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")
    
    # Create masked data with NaN for missing values
    masked_data = data.clone()
    masked_data[missing_mask] = float('nan')
    
    return masked_data, missing_mask


def reconstruct_missing_dimensions(model, 
                                 y_observed: torch.Tensor,
                                 missing_mask: torch.Tensor,
                                 n_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct missing dimensions using the trained GP-LVM model.
    
    Args:
        model: Trained SSVI GP-LVM model
        y_observed: Data with missing values as NaN [batch_size, output_dim]
        missing_mask: Boolean mask [batch_size, output_dim] where True=missing
        n_samples: Number of samples for Monte Carlo reconstruction
        
    Returns:
        y_reconstructed: Complete data with reconstructed missing values [batch_size, output_dim]
        reconstruction_std: Standard deviation of reconstructions [batch_size, output_dim]
    """
    model.eval()
    
    with torch.no_grad():
        # Get latent variables using amortized inference (encoder handles missing data)
        # The encoder should be robust to missing data through its input preprocessing
        U_samples = model.get_u_samples()
        
        # Handle missing data in encoder input
        y_encoder_input = torch.where(missing_mask, 
                                     torch.zeros_like(y_observed), 
                                     y_observed)
        
        # Get latent distribution parameters
        mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U_samples)
        
        # Sample from latent space and reconstruct
        reconstructions = []
        
        for _ in range(n_samples):
            # Sample latent variables
            sample_idx = torch.randint(0, model.P, (1,)).item()
            mu_sample = mu_x[sample_idx]  # [batch_size, latent_dim]
            sigma_sample = sigma_x[sample_idx]  # [batch_size, latent_dim]
            
            epsilon = torch.randn_like(sigma_sample)
            x_sample = mu_sample + sigma_sample * epsilon
            
            # Decode to observation space using GP
            y_mean = model.gp.get_gp_expectation(x_sample)  # [batch_size, M, C]
            
            # Handle output dimensions (squeeze if C=1)
            if y_mean.shape[-1] == 1:
                y_mean = y_mean[..., 0]  # [batch_size, M]
            
            reconstructions.append(y_mean)
        
        # Stack reconstructions and compute statistics
        reconstructions = torch.stack(reconstructions, dim=0)  # [n_samples, batch_size, output_dim]
        
        # Compute mean and std across samples
        y_reconstructed_mean = reconstructions.mean(dim=0)
        y_reconstructed_std = reconstructions.std(dim=0)
        
        # For complete reconstruction, use observed values where available
        y_complete = torch.where(missing_mask, 
                                y_reconstructed_mean, 
                                torch.where(torch.isnan(y_observed), 
                                           y_reconstructed_mean, 
                                           y_observed))
    
    return y_complete, y_reconstructed_std


def evaluate_reconstruction_quality(y_true: torch.Tensor,
                                   y_reconstructed: torch.Tensor,
                                   missing_mask: torch.Tensor,
                                   reconstruction_std: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Evaluate reconstruction quality for missing dimensions only.
    
    Args:
        y_true: Original complete data [batch_size, output_dim]
        y_reconstructed: Reconstructed data [batch_size, output_dim]
        missing_mask: Boolean mask [batch_size, output_dim] where True=missing
        reconstruction_std: Reconstruction uncertainty [batch_size, output_dim] (optional)
        
    Returns:
        metrics: Dictionary with reconstruction quality metrics
    """
    # Extract missing values only
    missing_indices = missing_mask
    
    if missing_indices.sum() == 0:
        return {"rmse": 0.0, "mae": 0.0, "relative_error": 0.0, "mean_uncertainty": 0.0}
    
    y_true_missing = y_true[missing_indices].cpu().numpy()
    y_recon_missing = y_reconstructed[missing_indices].cpu().numpy()
    
    # Compute metrics for missing dimensions only
    rmse = np.sqrt(np.mean((y_true_missing - y_recon_missing)**2))
    mae = np.mean(np.abs(y_true_missing - y_recon_missing))
    
    # Relative error (normalized by true values magnitude)
    relative_error = np.mean(np.abs(y_true_missing - y_recon_missing) / 
                            (np.abs(y_true_missing) + 1e-8))
    
    # Compute uncertainty metrics if reconstruction_std is provided
    mean_uncertainty = 0.0
    if reconstruction_std is not None:
        std_missing = reconstruction_std[missing_indices].cpu().numpy()
        mean_uncertainty = np.mean(std_missing)
    
    return {
        "rmse": float(rmse),
        "mae": float(mae), 
        "relative_error": float(relative_error),
        "mean_uncertainty": float(mean_uncertainty),
        "n_missing": int(missing_indices.sum().item())
    }


def visualize_reconstruction_results(y_true: torch.Tensor,
                                   y_observed: torch.Tensor,
                                   y_reconstructed: torch.Tensor,
                                   missing_mask: torch.Tensor,
                                   save_path: Optional[str] = None,
                                   n_samples: int = 5) -> plt.Figure:
    """
    Visualize missing data reconstruction results.
    
    Args:
        y_true: Original complete data [N, D]
        y_observed: Data with missing values [N, D]
        y_reconstructed: Reconstructed complete data [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
        save_path: Optional path to save the figure
        n_samples: Number of data points to visualize
        
    Returns:
        fig: Matplotlib figure object
    """
    # Select random samples to visualize
    N, D = y_true.shape
    sample_indices = np.random.choice(N, min(n_samples, N), replace=False)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Extract data for this sample
        true_vals = y_true[idx].cpu().numpy()
        observed_vals = y_observed[idx].cpu().numpy()
        recon_vals = y_reconstructed[idx].cpu().numpy()
        mask = missing_mask[idx].cpu().numpy()
        
        dimensions = np.arange(D)
        
        # Plot true values (baseline)
        ax.plot(dimensions, true_vals, 'g-', label='Ground Truth', linewidth=2, alpha=0.8)
        
        # Plot observed values
        observed_dims = dimensions[~mask]
        if len(observed_dims) > 0:
            ax.scatter(observed_dims, true_vals[~mask], color='blue', 
                      label='Observed', s=50, zorder=5)
        
        # Plot missing values and reconstructions
        missing_dims = dimensions[mask]
        if len(missing_dims) > 0:
            ax.scatter(missing_dims, true_vals[mask], color='red', 
                      label='Missing (True)', s=50, zorder=5, alpha=0.7)
            ax.scatter(missing_dims, recon_vals[mask], color='orange', 
                      label='Reconstructed', s=50, zorder=5, marker='x')
            
            # Connect missing true values to reconstructions
            for dim in missing_dims:
                ax.plot([dim, dim], [true_vals[dim], recon_vals[dim]], 
                       'k--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample {idx}: Missing Data Reconstruction')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction visualization saved to: {save_path}")
    
    return fig


def preprocess_data_for_missing_training(data: torch.Tensor,
                                       config) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Preprocess data for missing data training experiments.
    
    Args:
        data: Original complete data [N, D]
        config: Configuration object with missing data settings
        
    Returns:
        processed_data: Data with artificial missing values [N, D]
        missing_mask: Boolean mask [N, D] where True=missing
        info: Dictionary with preprocessing information
    """
    if not config.enable_missing_data:
        # No missing data - return original data
        return data, torch.zeros_like(data, dtype=torch.bool), {"n_missing": 0}
    
    print(f"Creating missing data pattern...")
    print(f"  Missing ratio: {config.missing_data_ratio}")
    print(f"  Pattern type: random")
    print(f"  Seed: {config.missing_data_seed}")
    
    # Create missing data pattern
    masked_data, missing_mask = create_missing_data_pattern(
        data, 
        missing_ratio=config.missing_data_ratio,
        pattern_type="random",  # Could be made configurable
        seed=config.missing_data_seed
    )
    
    n_missing = missing_mask.sum().item()
    total_elements = data.numel()
    missing_percentage = (n_missing / total_elements) * 100
    
    info = {
        "n_missing": n_missing,
        "total_elements": total_elements,
        "missing_percentage": missing_percentage,
        "pattern_type": "random"
    }
    
    print(f"  Missing elements: {n_missing}/{total_elements} ({missing_percentage:.1f}%)")
    
    return masked_data, missing_mask, info
