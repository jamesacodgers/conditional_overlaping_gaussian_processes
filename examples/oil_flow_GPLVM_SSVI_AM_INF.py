# %% 
import torch 
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Any
import numpy as np
import urllib.request
import tarfile
import os
import sys
import json
from datetime import datetime
import time

sys.path.insert(0, os.getcwd())

from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF
from src.ssvi_utils import SSVIResultsLogger, visualize_ssvi_results

# Device will be set in main() from config

@dataclass
class OilFlowSSVIConfig:
    """Configuration for Oil Flow SSVI GP-LVM with Amortized Inference"""
    
    total_size = None  # To be set after data loading

    # === Device Configuration ===
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"
    
    # === Data Configuration ===
    data_url: str = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    latent_dim: int = 6  # A = 6 from original
    
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
    sigma2: float = 1.0
    beta: Any = None         # Lengthscales (None = auto)
    sigma2_s: Any = None     # Signal variances (None = auto)
    P: int = 8               # Number of samples for SSVI (like original)
    trainable_V: bool = True
    
    # === Visualization ===
    plot_training: bool = True
    plot_latent_space: bool = True
    plot_hyperparams: bool = True
    
    # === Progress Reporting ===
    log_frequency: int = 100  # Log every N iterations
    
    # === Advanced Logging ===
    enable_gradient_logging: bool = False  # Log gradients and parameter changes
    gradient_log_frequency: int = 50       # Log gradients every N iterations
    save_results: bool = True              # Save results to folder
    
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
        print(f"SSVI: P={self.P} samples")
        print(f"Logging: Save results={self.save_results}, Gradients={self.enable_gradient_logging}")
        if self.enable_gradient_logging:
            print(f"  Gradient log frequency: every {self.gradient_log_frequency} iterations")

        print("=" * 50)

def load_oil_flow_data(config: OilFlowSSVIConfig):
    """Load and preprocess oil flow data"""
    print("Downloading and loading oil flow data...")
    
    # Download data
    urllib.request.urlretrieve(config.data_url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')
    
    # Load data and move to device
    y = torch.Tensor(np.loadtxt(fname='DataTrn.txt')).to(config.device)
    labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt')).to(config.device)
    labels = (labels @ torch.diag(torch.tensor([1, 2, 3], device=config.device, dtype=labels.dtype))).sum(axis=1)

    # Initialize inducing points and latent variables (following original)
    Vx = torch.randn(config.n_inducing_points, config.latent_dim, device=config.device)
    mu_x, _, _ = torch.pca_lowrank(y, config.latent_dim)
    mu_x = mu_x / mu_x.std(dim=0)
    
    print(f"Data loaded: {y.shape[0]} samples, {y.shape[1]} dimensions")
    print(f"Latent initialization: {mu_x.shape}")
    print(f"Inducing points: {Vx.shape}")
    config.total_size = y.shape[0]

    return y, labels, Vx, mu_x

class OilFlow_SSVI_GPLVM(GPLVM_SSVI_AM_INF):
    """Oil Flow SSVI GP-LVM with Amortized Inference"""
    
    def __init__(self, config: OilFlowSSVIConfig, Vx, mu_x, M):
        # Initialize underlying SSVI model
        super().__init__(
            Vx=Vx,
            M=M,  # Observation dimension
            mu_x=mu_x,
            sigma2=config.sigma2,
            beta=config.beta,
            sigma2_s=config.sigma2_s,
            P=config.P,
            trainable_V=config.trainable_V,
            encoder_hidden_dims=config.encoder_hidden_dims,
            local_lr=config.local_lr,
            global_lr=config.global_lr,
            device=config.device
        )
        
        # Store config
        self.config = config

    def train_step(self, Y_batch: torch.Tensor, batch_indices: torch.Tensor, U_samples: torch.Tensor = None):
        """
        Modified train_step: proper SSVI separation of local/global optimization
        """
        # Zero gradients for local optimizer (encoder) only initially
        self.optimizer_local.zero_grad()

        # Sample U before amortized inference step
        U_samples = self.get_u_samples()

        # Get local parameters via amortized inference (gradients only to encoder)
        mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_samples(Y_batch, U_samples)
        
        # Now zero gradients for global optimizer before global computation
        self.optimizer_global.zero_grad()
        
        # Compute global ELBO loss (gradients flow to all parameters)
        loss = -self.gplvm_elbo(Y_batch,
                               mu_n_batch,
                               sigma_n_batch,
                               U_samples=U_samples,
                               batch_size=self.config.batch_size,
                               total_size=self.config.total_size
                               )
        
        # Backward pass (gradients computed for all parameters)
        loss.backward()
        
        # Update parameters with different learning rates
        self.optimizer_local.step()
        self.optimizer_global.step()
        
        return loss.item()

    @property 
    def mu_x(self): 
        return self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return self.Sigma_x_train
    


def train_oil_flow_ssvi(model: OilFlow_SSVI_GPLVM, y, config: OilFlowSSVIConfig, logger: SSVIResultsLogger = None):
    """Training function for Oil Flow SSVI GP-LVM with optional logging"""
    losses = []
    
    print(f"Training Oil Flow SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")
    
    start_time = time.time()
    
    for i in range(config.training_iterations):
        # Sample batch indices
        N = y.shape[0]
        batch_indices = torch.randperm(N, device=y.device)[:config.batch_size]
        y_batch = y[batch_indices]
        
        # Training step - U sampling happens inside automatically each step
        loss = model.train_step(y_batch, batch_indices)
        losses.append(loss)
        
        # Gradient logging (only if logger provided and enabled)
        if logger and config.enable_gradient_logging and i % config.gradient_log_frequency == 0:
            gradient_info = model.get_gradient_info()
            logger.log_gradients(i, loss, gradient_info)
        
        # Progress reporting
        if i % config.log_frequency == 0:
            elapsed = time.time() - start_time
            message = f"Iteration {i}/{config.training_iterations}: Loss={loss:.4f}, Time={elapsed:.1f}s"
            print(f"Iteration {i}/{config.training_iterations}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            # Log to file
            if logger:
                logger.log_training(message)
    
    total_time = time.time() - start_time
    final_message = f"Training completed! Final Loss: {losses[-1]:.4f}, Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    print(f"Training completed!")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)")
    # Log to file
    if logger:
        logger.log_training(final_message)
    
    return losses

# === Pre-defined Configurations ===

def get_fast_config():
    """Fast configuration for testing"""
    return OilFlowSSVIConfig(
        training_iterations=500,
        batch_size=50,
        n_inducing_points=8,
        encoder_hidden_dims=[128, 64],
        local_lr=2e-3,
        global_lr=1e-2,
        P=4,
        log_frequency=50,
        enable_gradient_logging=True,
        gradient_log_frequency=25
    )

def get_standard_config():
    """Standard configuration (matches original oil flow example scale)"""
    return OilFlowSSVIConfig(
        training_iterations=1000,
        batch_size=100,
        n_inducing_points=12,
        encoder_hidden_dims=[256, 128, 64],
        local_lr=1e-3,
        global_lr=1e-2,
        P=8,
        log_frequency=100
    )

def get_high_quality_config():
    """High-quality configuration for best results"""
    return OilFlowSSVIConfig(
        training_iterations=2000,
        batch_size=200,
        n_inducing_points=16,
        encoder_hidden_dims=[512, 256, 128, 64],
        local_lr=5e-4,
        global_lr=5e-3,
        P=16,
        log_frequency=100
    )

def get_ablation_config():
    """Minimal configuration for ablation studies"""
    return OilFlowSSVIConfig(
        training_iterations=300,
        batch_size=32,
        n_inducing_points=6,
        encoder_hidden_dims=[64, 32],
        local_lr=5e-3,
        global_lr=2e-2,
        P=4,
        log_frequency=30
    )

def main(config: OilFlowSSVIConfig = None):
    """Main function for Oil Flow SSVI GP-LVM example"""
    
    # Use default config if none provided
    if config is None:
        config = get_standard_config()
    
    # Set device from config
    torch.set_default_device(config.device)
    
    # Print configuration
    config.print_config()
    
    # Load oil flow data
    y, labels, Vx, mu_x = load_oil_flow_data(config)
    
    # Create model
    print("\nInitializing Oil Flow SSVI GP-LVM...")
    model = OilFlow_SSVI_GPLVM(
        config=config,
        Vx=Vx,
        mu_x=mu_x,
        M=y.shape[1]  # Observation dimension
    )
    
    print(f"Model initialized:")
    print(f"  Encoder: {model.encoder}")
    print(f"  Inducing points: {model.gp.Vx.shape}")
    print(f"  Latent variables: {model.mu_x.shape}")
    
    # Initialize logger
    logger = SSVIResultsLogger(config, experiment_name="oil_flow_gp_lvm_ssvi")
    if logger.results_dir:
        print(f"Results will be saved to: {logger.results_dir}")
        # Log initial info
        logger.log_training("=== Oil Flow SSVI GP-LVM Training Session ===")
        logger.log_training(f"Model: {config.n_inducing_points} inducing points, latent_dim={config.latent_dim}")
        logger.log_training(f"Training: {config.training_iterations} iter, batch={config.batch_size}")
        logger.log_training(f"Learning rates: Local={config.local_lr}, Global={config.global_lr}")
    
    # Train model
    print(f"\nTraining model...")
    losses = train_oil_flow_ssvi(model, y, config, logger)
    
    # Test single step functionality
    print(f"\nTesting single step...")
    test_indices = torch.randperm(y.shape[0])[:10]
    test_y = y[test_indices]
    test_loss = model.train_step(test_y, test_indices)
    print(f"Test step loss: {test_loss:.4f}")
    
    # Visualize results
    print(f"\nVisualizing results...")
    visualize_ssvi_results(model, y, labels, losses, config, logger)
    

    
    # Print final model parameters
    print(f"\nFinal Model Parameters:")
    print(f"  Observation noise (sigma2): {model.sigma2:.4f}")
    print(f"  Signal variances (sigma2_s): {model.gp.sigma2_s}")
    print(f"  Lengthscales (1/beta): {(1/model.gp.beta).flatten()}")
    
    return model, losses, config

if __name__ == "__main__":
    # === Best Learning Rate Configurations ===
    # Based on experimental results: Config 1 achieved 175.60 loss, Config 2 achieved 181.82 loss
    
    # Generate diverse initial lengthscales (same for all experiments for fair comparison)
    # Sample initial lengthscales with significant variation across dimensions
    initial_lengthscales = torch.ones(10)/4 #torch.exp(torch.randn(6) * 0.8 + 0.5)  # Mean ~1.6, significant variation
    initial_beta = torch.tensor([1.]) #(1.0 / initial_lengthscales).detach()  # Convert to beta and detach
    print(f"Initial lengthscales for all experiments: {initial_lengthscales.tolist()}")
    print(f"Initial beta values: {initial_beta.tolist()}")
    
    # Configuration: Low local, High global (Final Loss: 181.82)
    second_best_config = OilFlowSSVIConfig(
        training_iterations=100,
        batch_size=1000,
        local_lr=1e-2,
        global_lr=1e-2,
        beta=initial_beta,
        encoder_hidden_dims=[128, 64, 32],
        enable_gradient_logging=True,
        gradient_log_frequency=50,
        n_inducing_points=16,
        latent_dim=10,
        P=8
    )
    main(second_best_config)
