# %% 
import torch 
import matplotlib.pyplot as plt
from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF
from dataclasses import dataclass
from typing import Any
import numpy as np
import urllib.request
import tarfile
import time
import os
import json
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

class ResultsLogger:
    """Logger for oil flow SSVI GP-LVM results and gradients"""
    
    def __init__(self, config: 'OilFlowSSVIConfig'):
        self.config = config
        
        if config.save_results:
            # Create results directory with timestamp in existing results folder
            timestamp = datetime.now().strftime("%d_%m_%H_%M")
            self.results_dir = f"results/oil_flow_gp_lvm_ssvi_results/oil_flow_res_{timestamp}"
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Initialize log files
            self.training_log_path = os.path.join(self.results_dir, "training_log.txt")
            self.gradient_log_path = os.path.join(self.results_dir, "gradient_log.jsonl")
            
            # Save config
            config_path = os.path.join(self.results_dir, "config.json")
            with open(config_path, 'w') as f:
                # Convert config to dict, handling non-serializable fields
                config_dict = {}
                for key, value in config.__dict__.items():
                    try:
                        json.dumps(value)  # Test if serializable
                        config_dict[key] = value
                    except:
                        config_dict[key] = str(value)
                json.dump(config_dict, f, indent=2)
        else:
            self.results_dir = None
    
    def log_training(self, message: str):
        """Log training progress to file only (console prints handled elsewhere)"""
        if self.results_dir:
            with open(self.training_log_path, 'a') as f:
                f.write(f"{message}\n")
    
    def log_gradients(self, iteration: int, loss: float, gradient_info: dict):
        """Log gradient information to file"""
        if self.results_dir and self.config.enable_gradient_logging:
            log_entry = {
                'iteration': iteration,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                **gradient_info
            }
            with open(self.gradient_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def save_plots(self, figure, filename: str):
        """Save matplotlib figure to results directory"""
        if self.results_dir:
            plot_path = os.path.join(self.results_dir, filename)
            figure.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.log_training(f"Plot saved: {filename}")
    
    def get_results_dir(self):
        """Get results directory path"""
        return self.results_dir

@dataclass
class OilFlowSSVIConfig:
    """Configuration for Oil Flow SSVI GP-LVM with Amortized Inference"""
    
    # === Data Configuration ===
    data_url: str = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    latent_dim: int = 6  # A = 6 from original
    
    # === Model Architecture ===
    n_inducing_points: int = 12
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
        """Set default encoder architecture if not specified"""
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [256, 128, 64]
    
    def print_config(self):
        """Print configuration summary"""
        print("=== Oil Flow SSVI GP-LVM Configuration ===")
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
    
    # Load data
    y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
    labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
    labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)
    
    # Initialize inducing points and latent variables (following original)
    Vx = torch.randn(config.n_inducing_points, config.latent_dim)
    mu_x, _, _ = torch.pca_lowrank(y, config.latent_dim)
    mu_x = mu_x / mu_x.std(dim=0)
    
    print(f"Data loaded: {y.shape[0]} samples, {y.shape[1]} dimensions")
    print(f"Latent initialization: {mu_x.shape}")
    print(f"Inducing points: {Vx.shape}")
    
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
            global_lr=config.global_lr
        )
        
        # Store config
        self.config = config
        
        # Fix dimensionality inconsistency: ensure Lx = L
        self.Lx = self.L

    def train_step(self, Y_batch: torch.Tensor, batch_indices: torch.Tensor, U_samples: torch.Tensor = None):
        """
        Modified train_step: sample U inside for proper gradient flow
        """
        # Zero gradients for both optimizers BEFORE forward pass
        self.optimizer_local.zero_grad()
        self.optimizer_global.zero_grad()
        
        # Sample U inside train_step for proper gradient flow to q(u)
        if U_samples is None:
            U_samples = self.get_u_samples_ssvi(self.P)
        
        # Get local parameters via amortized inference (with gradients to all params)
        mu_n_batch, sigma_n_batch = self.amortized_inference_step(Y_batch, batch_indices)
        
        # Compute global ELBO loss (gradients flow to all parameters)
        loss = -self.gplvm_elbo(Y_batch,
                               mu_n_batch.unsqueeze(0).expand(self.P, -1, -1),
                               sigma_n_batch.unsqueeze(0).expand(self.P, -1, -1),
                               U_samples=U_samples)
        
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
    
    def gplvm_elbo(self, y, mu_x_train, Sigma_x_train, U_samples: torch.Tensor = None):
        """GPLVM ELBO for oil flow data"""
        R_train = torch.ones((y.shape[0], 1), device=y.device, dtype=y.dtype)
        return self.elbo(y, R_train, mu_x_train=mu_x_train, Sigma_x_train=Sigma_x_train, U_samples=U_samples)

def train_oil_flow_ssvi(model: OilFlow_SSVI_GPLVM, y, config: OilFlowSSVIConfig, logger: ResultsLogger = None):
    """Training function for Oil Flow SSVI GP-LVM with optional logging"""
    losses = []
    
    print(f"Training Oil Flow SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")
    
    start_time = time.time()
    
    for i in range(config.training_iterations):
        # Sample batch indices
        N = y.shape[0]
        batch_indices = torch.randperm(N)[:config.batch_size]
        y_batch = y[batch_indices]
        
        # Training step - U sampling now happens inside for proper gradient flow
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

def visualize_results(model: OilFlow_SSVI_GPLVM, y, labels, losses, config: OilFlowSSVIConfig, logger: ResultsLogger):
    """Visualize training results and latent space"""
    
    if not (config.plot_training or config.plot_latent_space or config.plot_hyperparams):
        return
    
    with torch.no_grad():
        # Color mapping for labels
        c = ["r", "g", "b"]
        colors = [c[int(i)-1] for i in labels]
        
        n_plots = sum([config.plot_training, config.plot_latent_space, config.plot_hyperparams])
        fig_width = 4 * n_plots
        fig = plt.figure(figsize=(fig_width, 4))
        
        plot_idx = 1
        
        # Plot 1: Training curve
        if config.plot_training:
            plt.subplot(1, n_plots, plot_idx)
            plt.plot(losses, 'b-', alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('SSVI Training Curve')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 2: Latent space visualization
        if config.plot_latent_space:
            plt.subplot(1, n_plots, plot_idx)
            
            # Plot samples from all P particles with transparency
            for p in range(model.P):
                # Use first two latent dimensions for visualization
                mu_x_p = model.mu_x[p, :, :2]  # [N, 2]
                plt.scatter(mu_x_p[:, 0], mu_x_p[:, 1], 
                           c=colors, alpha=0.15, s=10)
            
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2') 
            plt.title(f'Latent Space (P={model.P} particles)')
            plot_idx += 1
        
        # Plot 3: Hyperparameters (lengthscales)
        if config.plot_hyperparams:
            plt.subplot(1, n_plots, plot_idx)
            lengthscales = 1 / model.gp.beta.flatten()
            plt.bar(range(len(lengthscales)), lengthscales, 
                   color="blue", alpha=0.7)
            plt.xlabel('Latent Dimension')
            plt.ylabel('Lengthscale')
            plt.title('GP Lengthscales')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        logger.save_plots(fig, "training_results.png")
        plt.show()

def visualize_latent_samples(model: OilFlow_SSVI_GPLVM, labels, logger: ResultsLogger, dims=(4, 0)):
    """Visualize samples from latent space uncertainty"""
    with torch.no_grad():
        c = ["r", "g", "b"]
        colors = [c[int(i)-1] for i in labels]
        
        fig = plt.figure(figsize=(8, 6))
        
        for p in range(model.P):
            # Sample from latent distribution
            dim1, dim2 = dims
            sample_1 = torch.distributions.Normal(
                model.mu_x[p, :, dim1], 
                model.Sigma_x[p, :, dim1].sqrt()
            ).sample()
            sample_2 = torch.distributions.Normal(
                model.mu_x[p, :, dim2], 
                model.Sigma_x[p, :, dim2].sqrt()
            ).sample()
            
            plt.scatter(sample_1, sample_2, alpha=0.1, c=colors, s=10)
        
        plt.xlabel(f'Latent Dimension {dim1}')
        plt.ylabel(f'Latent Dimension {dim2}')
        plt.title(f'Latent Space Samples (Dims {dim1} vs {dim2})')
        
        # Save plot
        logger.save_plots(fig, f"latent_samples_dims_{dim1}_{dim2}.png")
        plt.show()

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
    logger = ResultsLogger(config)
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
    visualize_results(model, y, labels, losses, config, logger)
    
    # Additional latent space visualization
    if config.plot_latent_space:
        print(f"Showing latent space samples...")
        visualize_latent_samples(model, labels, logger, dims=(4, 0))
    
    # Print final model parameters
    print(f"\nFinal Model Parameters:")
    print(f"  Observation noise (sigma2): {model.sigma2:.4f}")
    print(f"  Signal variances (sigma2_s): {model.gp.sigma2_s}")
    print(f"  Lengthscales (1/beta): {(1/model.gp.beta).flatten()}")
    
    return model, losses, config

if __name__ == "__main__":
    # === Usage Examples ===
    
    # 1. Standard configuration (default)
    # main()
    
    # 2. Fast configuration for testing
     main(get_fast_config())
    
    # 3. High-quality configuration
    # main(get_high_quality_config())
    
    # 4. Ablation study configuration
    # main(get_ablation_config())
    
    # 5. Custom configuration
    # custom_config = OilFlowSSVIConfig(
    #     training_iterations=1500,
    #     batch_size=150,
    #     local_lr=1e-3,
    #     global_lr=5e-3,
    #     encoder_hidden_dims=[256, 128]
    # )
    # main(custom_config)
