# %% 
"""
Oil Flow SSVI GP-LVM with Amortized Inference

Usage examples:
    # Run with default fast config
    python ssvi_scripts_experiments/oil_flow_GPLVM_SSVI_AM_INF.py

    # Run with base config
    python ssvi_scripts_experiments/oil_flow_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml

    # Run with base config and overrides
    python ssvi_scripts_experiments/oil_flow_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml --training-iterations 100 --device cuda

    # Run with custom config name for results
    python ssvi_scripts_experiments/oil_flow_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml --config-name my_experiment
"""

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

# Get the project root directory (parent of ssvi_scripts_experiments)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add both project root and generalised_gplvm_code to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF
from src.ssvi_utils import visualize_ssvi_results, plot_metrics_history, \
    save_metrics_to_json
from src.ssvi_alignment_utils import SSVIOriginalGPLVMResultsLogger
from src.ssvi_metrics import SSVIMetricsComputer, create_train_test_split
from src.ssvi_config_utils import (
    load_config_with_overrides, create_argument_parser,
    get_base_config_path, ensure_config_dir_exists, print_available_configs
)
from src.ssvi_alignment_utils import (
    load_oilflow_data_for_ssvi, integrate_original_gplvm_outputs, SSVIOriginalGPLVMResultsLogger
)


# Device will be set in main() from config

@dataclass
class OilFlowSSVIConfig:
    """Configuration for Oil Flow SSVI GP-LVM with Amortized Inference"""

    total_size = None  # To be set after data loading

    # === Device Configuration ===
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"

    # === Data Configuration ===
    data_url: str = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    dataset: str = "oilflow"  # Dataset name for original GP LVM compatibility
    latent_dim: int = 6  # A = 6 from original
    n_inducing: int = 64  # Alias for n_inducing_points for compatibility
    test_size_int: int = 50  # Number of test samples (like original GP LVM)
    random_seed: int = 7  # Random seed for reproducibility
    learning_rate: float = 1e-3  # Alias for local_lr for compatibility
    plot_colors: list = None  # Will be set to ['r', 'b', 'g'] if None

    # === Model Architecture ===
    n_inducing_points: int = 64
    encoder_hidden_dims: list = None  # Will be set to [256, 128, 64] if None

    # === Training Configuration ===
    training_iterations: int = 1000
    batch_size: int = 100

    # === Learning Rates ===
    local_lr: float = 1e-3  # Encoder parameters
    global_lr: float = 1e-2  # GP hyperparameters (higher like in original)

    # === GP Hyperparameters ===
    sigma2: float = 1.0
    beta: Any = None  # Lengthscales (None = auto)
    sigma2_s: Any = None  # Signal variances (None = auto)
    P: int = 8  # Number of samples for SSVI (like original)
    trainable_V: bool = True

    # === Checkpoint Configuration ===
    checkpoint_frequency: int = 100  # Save models, plots, metrics, and log progress every N iterations

    # === Advanced Logging ===
    enable_gradient_logging: bool = False  # Log gradients and parameter changes
    gradient_log_frequency: int = 50  # Log gradients every N iterations
    save_results: bool = True  # Save results to folder
    results_dir: str = "oil_flow_results/ssvi_results"  # Results directory (relative to current working directory)

    # === Metrics Configuration ===
    enable_metrics: bool = True  # Compute evaluation metrics
    split_random_seed: int = 42  # Seed for train/test split (separate from training seed)

    # === Visualization Configuration ===
    save_training_plots: bool = True  # Save training result plots during training
    plot_training: bool = True  # Plot training progress
    plot_latent_space: bool = True  # Plot latent space visualization
    plot_hyperparams: bool = True  # Plot model hyperparameters

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
            print(
                f"CUDA available: {torch.cuda.get_device_name(0) if self.device.startswith('cuda') else 'Not using CUDA'}")
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
    """Load and preprocess oil flow data with optional train/test split"""
    print("Downloading and loading oil flow data...")

    # Download data
    urllib.request.urlretrieve(config.data_url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')

    # Load data and move to device
    y_full = torch.Tensor(np.loadtxt(fname='DataTrn.txt')).to(config.device)
    labels_full = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt')).to(config.device)
    labels_full = (
                labels_full @ torch.diag(torch.tensor([1, 2, 3], device=config.device, dtype=labels_full.dtype))).sum(
        axis=1)

    # Create train/test split if metrics enabled
    if config.enable_metrics:
        print(f"Creating train/test split (test_size={config.test_size})...")
        y_train, y_test, labels_train, labels_test, train_indices, test_indices = create_train_test_split(
            y_full, labels_full,
            test_size=config.test_size,
            random_seed=config.split_random_seed
        )
        print(f"Train set: {y_train.shape[0]} samples")
        print(f"Test set: {y_test.shape[0]} samples")

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
    else:
        # Use full dataset
        y = y_full
        labels = labels_full
        split_data = None

    # Initialize inducing points and latent variables (following original)
    Vx = torch.randn(config.n_inducing_points, config.latent_dim, device=config.device)
    mu_x, _, _ = torch.pca_lowrank(y, config.latent_dim)
    mu_x = mu_x / mu_x.std(dim=0)

    print(f"Data loaded: {y.shape[0]} samples, {y.shape[1]} dimensions")
    print(f"Latent initialization: {mu_x.shape}")
    print(f"Inducing points: {Vx.shape}")
    config.total_size = y.shape[0]

    return y, labels, Vx, mu_x, split_data


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
        mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(Y_batch, U_samples)

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


def train_oil_flow_ssvi(model: OilFlow_SSVI_GPLVM, y, config: OilFlowSSVIConfig,
                        logger: SSVIOriginalGPLVMResultsLogger = None, split_data=None, labels=None):
    """Training function for Oil Flow SSVI GP-LVM with optional logging and metrics"""
    losses = []

    print(f"Training Oil Flow SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")

    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(f"Metrics computation enabled (frequency: every {config.checkpoint_frequency} iterations)")
        metrics_computer = SSVIMetricsComputer(device=config.device)

    start_time = time.time()

    for i in range(config.training_iterations):
        # Sample batch indices
        N = y.shape[0]
        batch_indices = torch.randperm(N, device=y.device)[:config.batch_size]
        y_batch = y[batch_indices]

        # Training step - U sampling happens inside automatically each step
        loss = model.train_step(y_batch, batch_indices)
        losses.append(loss)

        # Metrics computation (periodic)
        if (metrics_computer is not None and
                config.enable_metrics and
                i % config.checkpoint_frequency == 0):

            print(f"  Computing metrics at iteration {i}...")
            try:
                # Use full datasets for proper evaluation
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data['y_train'],
                    y_test=split_data['y_test'],
                    config=config,  
                    iteration=i,
                    elbo_loss=loss,
                    n_samples=config.P  # Use config P parameter for consistency
                )

                print(
                    f"    Train RMSE: {metrics.train_rmse:.4f}, Test RMSE: {metrics.test_rmse:.4f} (Ratio: {metrics.test_rmse / metrics.train_rmse:.3f})")
                print(f"    Train NLPD: {metrics.train_nlpd:.3f}, Test NLPD: {metrics.test_nlpd:.3f}")
                print(f"    Latent dims: {metrics.total_latent_dims}")
                print(
                    f"    Full datasets: Train={split_data['y_train'].shape[0]}, Test={split_data['y_test'].shape[0]} samples")

                # Log metrics
                if logger:
                    logger.log_training(f"Metrics at iteration {i}: {metrics.to_dict()}")

            except Exception as e:
                print(f"    Warning: Error computing metrics at iteration {i}: {e}")

        # Training plot saving (periodic)
        if (logger and config.save_training_plots and
                i > 0 and i % config.checkpoint_frequency == 0):

            print(f"  Saving training plots at iteration {i}...")
            try:
                # Generate filename with iteration number
                plot_filename = f"oilflow_ssvi_iter_{i}_seed_{config.random_seed}.png"

                # Create plots/checkpoints subdirectory for organized checkpoint storage
                plots_checkpoints_dir = os.path.join(logger.get_results_dir(), 'plots', 'checkpoints')
                os.makedirs(plots_checkpoints_dir, exist_ok=True)

                # Use the standardized plotting function for checkpoints
                from src.ssvi_alignment_utils import create_standardized_scattered_plot
                create_standardized_scattered_plot(
                    model, losses, labels, config, plots_checkpoints_dir, 
                    "ssvi", config.random_seed, y, model_type="ssvi", 
                    custom_filename=plot_filename
                )

                print(f"    Training plots saved: plots/checkpoints/{plot_filename}")

            except Exception as e:
                print(f"    Warning: Error saving training plots at iteration {i}: {e}")

        # Training model saving (periodic) - save iteration models like original GP-LVM
        if (logger and logger.get_results_dir() and
                i > 0 and i % config.checkpoint_frequency == 0):

            print(f"  Saving iteration {i} model...")
            try:
                from src.ssvi_alignment_utils import save_iteration_model
                save_iteration_model(model, "ssvi", i, config.random_seed, logger.get_results_dir())
                print(f"    Iteration {i} model saved")

            except Exception as e:
                print(f"    Warning: Error saving iteration {i} model: {e}")

        # Gradient logging (only if logger provided and enabled)
        if logger and config.enable_gradient_logging and i % config.gradient_log_frequency == 0:
            gradient_info = model.get_gradient_info()
            logger.log_gradients(i, loss, gradient_info)

        # Progress reporting
        if i % config.checkpoint_frequency == 0:
            elapsed = time.time() - start_time
            message = f"Iteration {i}/{config.training_iterations}: Loss={loss:.4f}, Time={elapsed:.1f}s"
            print(f"Iteration {i}/{config.training_iterations}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            # Log to file
            if logger:
                logger.log_training(message)

    total_time = time.time() - start_time
    final_message = f"Training completed! Final Loss: {losses[-1]:.4f}, Total time: {total_time:.1f}s ({config.training_iterations / total_time:.1f} iter/s)"
    print(f"Training completed!")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Total time: {total_time:.1f}s ({config.training_iterations / total_time:.1f} iter/s)")
    # Log to file
    if logger:
        logger.log_training(final_message)

    return losses, metrics_computer


# === Pre-defined Configurations ===

def get_fast_config():
    """Fast configuration for testing"""
    return OilFlowSSVIConfig(
        training_iterations=200,
        batch_size=50,
        n_inducing_points=8,
        encoder_hidden_dims=[128, 64],
        local_lr=2e-3,
        global_lr=1e-2,
        P=4,
        checkpoint_frequency=50,
        enable_gradient_logging=True,
        gradient_log_frequency=25,
        enable_metrics=True,
        test_size_int=100,  # 900 train, 100 test
        save_training_plots=True,
        plot_training=True,
        plot_latent_space=True,
        plot_hyperparams=True
    )


def main(config: OilFlowSSVIConfig = None, config_name: str = "fast"):
    """Main function for Oil Flow SSVI GP-LVM example"""

    # Use default config if none provided
    if config is None:
        config = get_fast_config()

    # Set device from config
    torch.set_default_device(config.device)
    
    # Set random seed (same as original GPLVM code)
    torch.manual_seed(config.random_seed)

    # Print configuration
    config.print_config()

    # Load oil flow data using original GP LVM method
    print("Loading oil flow data using original GP LVM method...")
    original_data = load_oilflow_data_for_ssvi(config)
    
    # Extract data in original format
    Y_train = original_data['Y_train']
    Y_test = original_data['Y_test']
    labels_train = original_data['labels_train']
    labels_test = original_data['labels_test']
    
    # Use training data for SSVI model (same as original)
    y = Y_train
    labels = labels_train
    
    # Set total_size for SSVI model
    config.total_size = y.shape[0]
    
    # Create Vx and mu_x for SSVI model
    Vx = torch.randn(config.n_inducing, config.latent_dim, device=config.device)
    mu_x = torch.randn(y.shape[0], config.latent_dim, device=config.device)
    
    # Create split_data for compatibility
    split_data = {
        'y_train': Y_train,
        'y_test': Y_test,
        'labels_train': labels_train,
        'labels_test': labels_test
    }

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

    # Enable SSVI logger for plots and logging
    logger = SSVIOriginalGPLVMResultsLogger(config, config_name) if config.save_results else None

    # Train model
    print(f"\nTraining model...")
    losses, metrics_computer = train_oil_flow_ssvi(model, y, config, logger, split_data, labels)

    # Test single step functionality
    print(f"\nTesting single step...")
    test_indices = torch.randperm(y.shape[0], device=y.device)[:10]
    test_y = y[test_indices]
    test_loss = model.train_step(test_y, test_indices)
    print(f"Test step loss: {test_loss:.4f}")

    # SSVI visualization is now enabled via logger
    print(f"\nSSVI visualization enabled - plots will be saved during training...")

    # Use the same logger that was used during training (no duplicate logger creation)
    if config.save_results and logger:
        ssvi_results_dir = logger.get_results_dir()
        
        # Log training completion
        logger.log_training("=== SSVI Training Completed ===")
        logger.log_training(f"Final Loss: {losses[-1]:.4f}")
        
        # Save gradient logs if available (additional SSVI data)
        if hasattr(model, 'gradient_logs') and model.gradient_logs:
            gradient_logs_path = os.path.join(ssvi_results_dir, 'gradient_logs.jsonl')
            with open(gradient_logs_path, 'w') as f:
                for log_entry in model.gradient_logs:
                    f.write(json.dumps(log_entry) + '\n')
            logger.log_training(f"Gradient logs saved: {gradient_logs_path}")
                
        # Create standardized scattered plots (same as original generalised_gplvm_code but scattered)
        from src.ssvi_alignment_utils import create_standardized_scattered_plot
        create_standardized_scattered_plot(model, losses, labels, config, ssvi_results_dir, "ssvi", config.random_seed, y, model_type="ssvi")
        
        # Create initial experiment summary (will be updated by integrate_original_gplvm_outputs)
        # This creates a placeholder that will be overwritten with complete metrics
        initial_summary = {
            'ssvi': {
                'train_rmse': 0.0,
                'test_rmse': 0.0,
                'train_nlpd': 0.0,
                'test_nlpd': 0.0,
                'final_loss': float(losses[-1]),
                'final_noise': float(model.sigma2) if hasattr(model, 'sigma2') else 0.0,
                'model_name': 'ssvi',
                'random_seed': config.random_seed,
                'loss_history': [float(loss) for loss in losses]
            }
        }
        
        # Save initial summary (will be updated later with complete metrics)
        summary_path = os.path.join(ssvi_results_dir, "experiment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(initial_summary, f, indent=2)
        logger.log_training(f"Initial experiment summary saved: {summary_path}")
        
        # Save checkpoint metrics and training progress plot (SSVI only)
        if metrics_computer is not None and metrics_computer.metrics_history:
            print(f"\n=== Saving Checkpoint Metrics and Training Progress Plot ===")
            try:
                # Save checkpoint metrics to JSON
                checkpoint_json_path = logger.save_checkpoint_metrics(metrics_computer, "ssvi", config.random_seed)
                print(f"Checkpoint metrics saved: {checkpoint_json_path}")
                
                # Save training progress plot
                checkpoint_plot_path = logger.save_checkpoint_plot(metrics_computer, "ssvi", config.random_seed)
                print(f"Training progress plot saved: {checkpoint_plot_path}")
                
            except Exception as e:
                print(f"Warning: Could not save checkpoint data: {e}")
        
        print(f"SSVI results saved to: {ssvi_results_dir}")
        print(f"Files created: config.json, training.log, experiment_summary.json")

    # Print final model parameters
    print(f"\nFinal Model Parameters:")
    print(f"  Observation noise (sigma2): {model.sigma2:.4f}")
    print(f"  Signal variances (sigma2_s): {model.gp.sigma2_s}")
    print(f"  Lengthscales (1/beta): {(1 / model.gp.beta).flatten()}")

    # Integrate original GP LVM outputs and create final experiment summary
    # This will overwrite the initial summary with complete metrics including NLPD
    print(f"\n=== Creating Final Experiment Summary ===")
    try:
        original_logger, original_results = integrate_original_gplvm_outputs(
            model, config, config_name, losses, Y_train, Y_test, labels_train, labels_test, logger, metrics_computer
        )
        print("Final experiment summary with complete metrics successfully created!")
        logger.log_training("=== SSVI Experiment Completed ===")
    except Exception as e:
        print(f"Warning: Could not create final experiment summary: {e}")

    # Return final_metrics if computed, otherwise None
    final_metrics = None
    if metrics_computer is not None and metrics_computer.metrics_history:
        final_metrics = metrics_computer.metrics_history[-1]

    return model, losses, config, metrics_computer, final_metrics


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
    try:
        config, config_name = load_config_with_overrides(
            config_path=args.config,
            config_class=OilFlowSSVIConfig,
            args=args
        )

        print(f"Using configuration: {config_name}")
        if args.config:
            print(f"Config file: {args.config}")

        # Run main function with loaded config
        main(config, config_name)

    except Exception as e:
        print(f"Error: {e}")
        print("\nAvailable configurations:")
        print_available_configs()
        sys.exit(1)