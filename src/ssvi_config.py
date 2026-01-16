"""
SSVI GP-LVM Configuration Classes

Shared configuration classes for SSVI GP-LVM experiments.
"""

import torch
from dataclasses import dataclass
from typing import Any


@dataclass
class OilFlowSSVIConfig:
    """Configuration for Oil Flow SSVI GP-LVM with Amortized Inference"""
    
    total_size = None  # To be set after data loading

    # === Device Configuration ===
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"
    
    # === Data Configuration ===
    # data_url: str = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"  # Now using data folder approach
    latent_dim: int = 6  # A = 6 from original
    test_size_int: int = 50  # Number of test samples (like original GP LVM)
    random_seed: int = 7  # Random seed for reproducibility
    
    # === Model Architecture ===
    n_inducing_points: int = 64
    encoder_hidden_dims: list = None  # Will be set to [256, 128, 64] if None
    
    # === Training Configuration ===
    training_iterations: int = 1000
    batch_size: int = 100
    
    # === Learning Rates ===
    local_lr: float = 1e-3   # Encoder parameters
    global_lr: float = 1e-2  # GP hyperparameters (higher like in original)
    
    # === GP Hyperparameters ===
    sigma2_s: Any = None     # Signal variances (None = auto)
    init_lengthscales: float = 1.0  # Initial lengthscales value (1/beta) - default matches current standard
    init_output_noise_scale: float = 1.0  # Initial output noise scale (sigma2) - default matches current standard
    P: int = 8               # Number of samples for SSVI (like original)
    trainable_V: bool = True
    
    # === Progress Reporting ===
    log_frequency: int = 100  # Log every N iterations
    checkpoint_frequency: int = 100  # Save checkpoint models every N iterations
    
    # === Advanced Logging ===
    enable_gradient_logging: bool = False  # Log gradients and parameter changes
    gradient_log_frequency: int = 50       # Log gradients every N iterations
    save_results: bool = True              # Save results to folder
    results_dir: str = "oil_flow_results/ssvi_results"  # Results directory (relative to current working directory)

    # === Metrics Configuration ===
    enable_metrics: bool = True            # Compute evaluation metrics
    metrics_frequency: int = 4000           # Compute metrics every N iterations
    test_size: float = 0.1                 # Fraction of data for testing
    
    # === Visualization Configuration ===
    save_training_plots: bool = True       # Save training result plots during training (saved with checkpoints)
    plot_training: bool = True             # Plot training progress
    plot_latent_space: bool = True         # Plot latent space visualization  
    plot_hyperparams: bool = True          # Plot model hyperparameters
    plot_latent_slice: bool = True         # Plot 1D latent slice with uncertainty bands (adaptive grid from data)
    
    # === Missing Data Configuration ===
    enable_missing_data: bool = False      # Enable missing data reconstruction capability
    missing_data_ratio: float = 0.3        # Fraction of dimensions to artificially mask (for testing)
    reconstruction_samples: int = 50       # Number of samples for missing data reconstruction
    
    # === MNIST Data Filtering Configuration ===
    mnist_digits: list = None           # List of digits to include (e.g., [0, 1, 2]). None = all digits (0-9)
    samples_per_digit: int = None          # Number of samples per digit. None = use all available samples
    
    def __post_init__(self):
        """Set default encoder architecture and resolve device if not specified"""
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [256, 128, 64]
        
        # Resolve device configuration
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def print_config(self):
        """Print configuration summary"""
        print("=== Oil Flow SSVI GP-LVM Configuration ===")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0) if self.device.startswith('cuda') else 'Not using CUDA'}")
        else:
            print("CUDA available: No")
        print(f"Model: {self.n_inducing_points} inducing points, latent_dim={self.latent_dim}")
        print(f"Encoder: {self.encoder_hidden_dims} hidden dims")
        print(f"Training: {self.training_iterations} iter, batch={self.batch_size}")
        print(f"Learning rates: Local={self.local_lr}, Global={self.global_lr}")
        print(f"Random seed: {self.random_seed}")
        print(f"Init lengthscales: {self.init_lengthscales}")
        print(f"Init output noise scale: {self.init_output_noise_scale}")
        print(f"SSVI: P={self.P} samples")
        print(f"Logging: Save results={self.save_results}, Gradients={self.enable_gradient_logging}")
        if self.enable_gradient_logging:
            print(f"  Gradient log frequency: every {self.gradient_log_frequency} iterations")

        print("=" * 50)

