# %% 
"""
Oil Flow VAE (Variational Autoencoder) with MLP architecture

Usage examples:
    # Run with default fast config
    python ssvi_scripts_experiments/oil_flow_VAE.py
    
    # Run with base config
    python ssvi_scripts_experiments/oil_flow_VAE.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml
    
    # Run with base config and overrides
    python ssvi_scripts_experiments/oil_flow_VAE.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml --training-iterations 100 --device cuda
    
    # Run with custom config name for results
    python ssvi_scripts_experiments/oil_flow_VAE.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml --config-name my_experiment
"""

import torch 
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np
import urllib.request
import tarfile
import os
import sys
import json
from datetime import datetime
import time

# Get the project root directory (parent of ssvi_scripts_experiments)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add both project root and generalised_gplvm_code to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))
sys.path.insert(0, os.path.join(project_root, 'vae_code'))

# Import VAE components (clean MLP version for tabular data)
from vae_mlp import VAE_MLP

# Import src utilities
from src.ssvi_utils import SSVIResultsLogger
from src.vae_metrics import create_train_test_split
from src.vae_utils import train_vae_with_metrics, save_vae_final_results
from src.vae_plots import visualize_vae_results
from src.ssvi_config_utils import (
    load_config_with_overrides, create_argument_parser, 
    get_base_config_path, ensure_config_dir_exists, print_available_configs
)

# Device will be set in main() from config

@dataclass
class VAEConfig:
    """General Configuration for VAE (Variational Autoencoder) - NOT dataset specific"""
    
    total_size: int = None  # To be set after data loading
    data_dim: int = None    # To be set after data loading (M)

    # === Device Configuration ===
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"
    
    # === Data Configuration ===
    latent_dim: int = 6  # Latent dimension
    test_size: float = 0.1  # Fraction of data for testing
    random_seed: int = 7  # Random seed for reproducibility
    
    # === VAE Architecture (MLP only for tabular data) ===
    # MLP hidden dimensions
    enc_mlp_hidden_dims: Optional[List[int]] = None  # Will be set to [256, 128] if None
    dec_mlp_hidden_dims: Optional[List[int]] = None  # Will be set to [128, 256] if None
    
    # Posterior and Likelihood
    posterior_form: str = "diagonal_gaussian"  # 'diagonal_gaussian' or 'full_covariance_gaussian'
    likelihood: str = "gaussian"  # 'gaussian' or 'bernoulli'
    noise: str = "heteroscedastic"  # 'heteroscedastic' or 'homoscedastic'
    noise_std: float = 0.1  # Initial noise std for homoscedastic noise
    train_noise: bool = False  # Train homoscedastic noise variance
    prior_std: float = 1.0  # Std of isotropic Gaussian prior
    
    # Activation functions (stored as strings, converted to nn.Module in __post_init__)
    likelihood_activation: str = "sigmoid"  # 'sigmoid', 'identity', etc.
    nonlinearity: str = "relu"  # 'relu', 'tanh', etc.
    
    # === Training Configuration ===
    training_iterations: int = 1000  # Number of training epochs (called iterations to match SSVI)
    batch_size: int = 100
    lr: float = 1e-3  # Learning rate (called lr to match VAE code, maps to local_lr in SSVI)
    num_samples: int = 1  # Number of samples for ELBO estimation (maps to P in SSVI)
    
    # Early stopping parameters (from VAE utils)
    early_stopping: bool = False
    min_es_iters: int = 300
    ref_es_iters: int = 100
    smooth_es_iters: int = 100
    es_thresh: float = 1e-2
    
    # Beta-NLL weighting (from VAE code)
    beta_nll: bool = False
    beta: float = 0.5
    
    # === Progress Reporting ===
    log_frequency: int = 100  # Log every N iterations
    checkpoint_frequency: int = 100  # Save checkpoint models every N iterations
    
    # === Advanced Logging ===
    save_results: bool = True              # Save results to folder
    results_dir: str = "oil_flow_results/vae_results"  # Results directory (relative to current working directory)

    # === Metrics Configuration ===
    enable_metrics: bool = True            # Compute evaluation metrics
    metrics_frequency: int = 100           # Compute metrics every N iterations
    
    # === Visualization Configuration ===
    save_training_plots: bool = True       # Save training result plots during training (saved with checkpoints)
    plot_training: bool = True             # Plot training progress
    plot_latent_space: bool = True         # Plot latent space visualization  
    
    def __post_init__(self):
        """Set default architectures and resolve device if not specified"""
        # Set default encoder/decoder hidden dimensions for MLP
        if self.enc_mlp_hidden_dims is None:
            self.enc_mlp_hidden_dims = [256, 128]
        if self.dec_mlp_hidden_dims is None:
            self.dec_mlp_hidden_dims = [128, 256]
        
        # Resolve device configuration
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_activation(self, name: str):
        """Convert activation name to torch.nn.Module"""
        import torch.nn as nn
        activations = {
            'sigmoid': nn.Sigmoid(),
            'identity': nn.Identity(),
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }
        return activations.get(name.lower(), nn.Identity())
    
    def print_config(self):
        """Print configuration summary"""
        print("=== VAE-MLP Configuration (Tabular Data) ===")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0) if self.device.startswith('cuda') else 'Not using CUDA'}")
        else:
            print("CUDA available: No")
        print(f"Model: latent_dim={self.latent_dim}")
        print(f"Encoder hidden dims: {self.enc_mlp_hidden_dims}")
        print(f"Decoder hidden dims: {self.dec_mlp_hidden_dims}")
        print(f"Posterior: {self.posterior_form}, Likelihood: {self.likelihood}")
        print(f"Noise: {self.noise}, Noise std: {self.noise_std}")
        print(f"Training: {self.training_iterations} epochs, batch={self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"ELBO samples: {self.num_samples}")
        print(f"Random seed: {self.random_seed}")
        print(f"Logging: Save results={self.save_results}")
        print("=" * 50)

def load_oil_flow_data(config: VAEConfig):
    """Load and preprocess oil flow data with optional train/test split - same as SSVI"""
    print("Loading oil flow data from data folder...")
    
    # Use the same data loading approach as Original GP-LVM and SSVI
    from src.data_utils import download_oilflow_data_if_needed
    from utils.data import load_real_data
    
    # Download data if needed (same as Original GP-LVM)
    download_oilflow_data_if_needed()
    
    # Load data using the same function as Original GP-LVM
    N, d, q, X, Y, labels = load_real_data('oilflow')
    
    # Convert to tensors and move to device (matching Original GP-LVM approach)
    y_full = Y.to(config.device)
    labels_full = torch.tensor(labels, dtype=torch.float64, device=config.device)
    
    # Create train/test split if metrics enabled
    print(f"Creating train/test split (test_size={config.test_size})...")
    y_train, y_test, labels_train, labels_test, train_indices, test_indices = create_train_test_split(
        y_full, labels_full,
        test_size=config.test_size,
        random_seed=config.random_seed
    )
    print(f"Data split shapes - Train: {y_train.shape}, Test: {y_test.shape}")
    print(f"Labels split shapes - Train: {labels_train.shape}, Test: {labels_test.shape}")

    # Use training data for model initialization
    y = y_train
    labels = labels_train
    split_data = {
        'y_train': y_train,
        'y_test': y_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'train_indices': train_indices,
        'test_indices': test_indices
    }
    
    print(f"Data loaded: {y.shape[0]} samples, {y.shape[1]} dimensions")
    config.total_size = y.shape[0]
    config.data_dim = y.shape[1]  # Store data dimension for VAE initialization

    return y, labels, split_data


def create_vae_model(config: VAEConfig):
    """
    Create VAE-MLP model from config for tabular data.
    
    Clean interface:
    - Input: (batch, data_dim) flat vectors
    - Output: (batch, data_dim) flat vectors
    - No image/color/pixel concepts
    """
    import torch.nn as nn
    
    # Get activation modules
    likelihood_activation = config.get_activation(config.likelihood_activation)
    nonlinearity = config.get_activation(config.nonlinearity)
    
    # Create clean MLP VAE for tabular data
    model = VAE_MLP(
        data_dim=config.data_dim,
        latent_dim=config.latent_dim,
        enc_hidden_dims=config.enc_mlp_hidden_dims,
        dec_hidden_dims=config.dec_mlp_hidden_dims,
        posterior_form=config.posterior_form,
        likelihood=config.likelihood,
        noise=config.noise,
        noise_std=config.noise_std,
        train_noise=config.train_noise,
        prior_std=torch.tensor(config.prior_std),
        likelihood_activation=likelihood_activation,
        nonlinearity=nonlinearity,
    )
    
    # Move model to device
    model = model.to(config.device)
    
    return model

# === Pre-defined Configurations ===

def get_fast_config():
    """Fast configuration for VAE-MLP testing - matches SSVI fast config structure"""
    return VAEConfig(
        # === Data Configuration ===
        latent_dim=6,
        test_size=0.1,
        random_seed=7,
        
        # === VAE-MLP Architecture ===
        enc_mlp_hidden_dims=[256, 128],
        dec_mlp_hidden_dims=[128, 256],
        
        # Posterior and Likelihood
        posterior_form="diagonal_gaussian",
        likelihood="gaussian",
        noise="heteroscedastic",
        noise_std=0.1,
        train_noise=False,
        prior_std=1.0,
        
        # Activation functions
        likelihood_activation="identity",  # No activation for regression
        nonlinearity="relu",
        
        # === Training Configuration ===
        training_iterations=200,
        batch_size=50,
        lr=1e-3,
        num_samples=1,
        
        # Early stopping
        early_stopping=False,
        
        # Beta-NLL
        beta_nll=False,
        beta=0.5,
        
        # === Progress Reporting ===
        log_frequency=50,
        checkpoint_frequency=100,
        
        # === Logging ===
        save_results=True,
        results_dir="oil_flow_results/vae_results",
        
        # === Metrics ===
        enable_metrics=True,  # Enable for comparison with SSVI
        metrics_frequency=50,
        
        # === Visualization ===
        save_training_plots=True,
        plot_training=True,
        plot_latent_space=True,
    )

def main(config: VAEConfig = None, config_name: str = "fast"):
    """Main function for Oil Flow VAE - preserves same structure as SSVI"""
    
    # Use default config if none provided
    if config is None:
        config = get_fast_config()
    
    # Set device and dtype from config
    torch.set_default_device(config.device)
    torch.set_default_dtype(torch.float64)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)

    # Print configuration
    config.print_config()
    
    # Load oil flow data (same preprocessing as SSVI)
    y_train, labels_train, split_data = load_oil_flow_data(config)
    
    # Create VAE-MLP model
    print("\nInitializing VAE-MLP model...")
    model = create_vae_model(config)
    
    print(f"Model initialized:")
    print(f"  Architecture: MLP (tabular data)")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Data dimension: {config.data_dim}")
    print(f"  Encoder: {config.data_dim} → {config.enc_mlp_hidden_dims} → {config.latent_dim}")
    print(f"  Decoder: {config.latent_dim} → {config.dec_mlp_hidden_dims} → {config.data_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize logger with config-based naming (same as SSVI)
    experiment_name = f"vae_{config_name}"
    logger = SSVIResultsLogger(config, experiment_name=experiment_name)
    if logger.results_dir:
        print(f"Results will be saved to: {logger.results_dir}")
        # Log initial info
        logger.log_training("=== Oil Flow VAE-MLP Training Session ===")
        logger.log_training(f"Model: latent_dim={config.latent_dim}, data_dim={config.data_dim}")
        logger.log_training(f"Training: {config.training_iterations} iter, batch={config.batch_size}")
        logger.log_training(f"Learning rate: {config.lr}")
        logger.log_training(f"Architecture: MLP (tabular data)")
    
    # Train model (uses same structure as SSVI)
    print(f"\nTraining model...")
    training_history, metrics_computer = train_vae_with_metrics(
        vae_model=model,
        y_train=y_train,
        config=config,
        logger=logger,
        split_data=split_data,
        labels_train=labels_train
    )
    
    # Visualize final results (VAE-specific plots)
    print(f"\nVisualizing results...")
    visualize_vae_results(
        vae_model=model,
        y_train=y_train,
        labels_train=labels_train,
        losses=training_history['losses'],
        config=config,
        logger=logger,
        show_plot=False,
        train_indices=split_data['train_indices']
    )
    
    # Save all final results (metrics, plots, etc.) - same structure as SSVI
    save_vae_final_results(
        vae_model=model,
        training_history=training_history,
        config=config,
        logger=logger,
        metrics_computer=metrics_computer,
        split_data=split_data
    )
    
    return model, training_history, config, metrics_computer

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Ensure config directory exists
    ensure_config_dir_exists()
    
    # Show available configs if no config specified
    if args.config is None:
        print("No config file specified. Available configurations:")
        print_available_configs()
        print("\nUsing default fast_config...\n")
    
    # Load configuration with overrides
    config, config_name = load_config_with_overrides(
        config_path=args.config,
        config_class=VAEConfig,
        args=args
    )
    
    print(f"Using configuration: {config_name}")
    if args.config:
        print(f"Config file: {args.config}")
    
    # Run main function with loaded config
    main(config, config_name)
    
