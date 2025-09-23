"""
Missing Data Utilities for SSVI GP-LVM

Implementation of missing data handling capabilities for GP-LVM models,
following the approach from "Generalised Gaussian Process Latent Variable Models (GPLVM) 
with Stochastic Variational Inference" paper.

Key features:
- Data masking utilities for creating missing data patterns
- Reconstruction methods for recovering missing dimensions
- Evaluation metrics for missing data reconstruction quality
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def rmse_missing_improved(Y_test, Y_recon):
    """
    Improved RMSE computation for missing data that handles NaN values properly.
    
    Args:
        Y_test: Original data [N, D] (numpy array)
        Y_recon: Reconstructed data [N, D] (numpy array)
        
    Returns:
        RMSE value as torch tensor
    """
    mse_per_sample = np.nanmean(np.square(Y_test - Y_recon), axis=1)
    # Handle case where all values might be NaN
    valid_mse = mse_per_sample[~np.isnan(mse_per_sample)]
    if len(valid_mse) == 0:
        return torch.tensor(float('inf'))
    return torch.sqrt(torch.mean(torch.Tensor(valid_mse)))


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
        
        # Handle missing data in encoder input (consistent with training)
        y_encoder_input = torch.where(missing_mask, 
                                     torch.full_like(y_observed, -9.0), 
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
    print(f"  Seed: {config.random_seed}")
    
    # Create missing data pattern
    masked_data, missing_mask = create_missing_data_pattern(
        data, 
        missing_ratio=config.missing_data_ratio,
        pattern_type="random",  # Could be made configurable
        seed=config.random_seed
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
