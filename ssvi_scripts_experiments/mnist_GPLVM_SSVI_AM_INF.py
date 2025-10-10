# %% 
"""
MNIST SSVI GP-LVM with Amortized Inference

Usage examples:
    # Run with default fast config
    python ssvi_scripts_experiments/mnist_GPLVM_SSVI_AM_INF.py
    
    # Run with base config
    python ssvi_scripts_experiments/mnist_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml
    
    # Run with base config and overrides
    python ssvi_scripts_experiments/mnist_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml --training-iterations 100 --device cuda
    
    # Run with custom config name for results
    python ssvi_scripts_experiments/mnist_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml --config-name my_experiment
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
from src.ssvi_utils import SSVIResultsLogger, visualize_ssvi_results, plot_metrics_history, plot_final_metrics_summary, plot_missing_nlpd_metrics_progression, plot_missing_rmse_metrics_progression
from src.ssvi_metrics import SSVIMetricsComputer, create_train_test_split
from src.missing_data_utils import preprocess_data_for_missing_training
from src.save_utils import print_final_model_parameters
from src.mnist_missing_reconstruction import visualize_mnist_missing_reconstruction, visualize_mnist_missing_reconstruction_detailed
from src.ssvi_config_utils import (
    load_config_with_overrides, create_argument_parser, 
    get_base_config_path, ensure_config_dir_exists, print_available_configs
)
from sklearn.model_selection import train_test_split

# Import OilFlowSSVIConfig from oil_flow_GPLVM_SSVI_AM_INF.py
from ssvi_scripts_experiments.oil_flow_GPLVM_SSVI_AM_INF import OilFlowSSVIConfig

# Import MNIST data loader
from src.data_utils import load_mnist_data, filter_mnist_by_digits

# Device will be set in main() from config

def load_oil_flow_data(config: OilFlowSSVIConfig):
    """Load and preprocess MNIST data with optional train/test split"""
    print("Loading MNIST data...")
    
    # Load data using PyTorch torchvision (no TensorFlow needed)
    N, d, q, X, Y, labels = load_mnist_data()
    
    # MNIST-specific preprocessing: normalize to [0, 1]
    Y = Y / 255.0
    
    # Apply digit filtering if configured
    Y, labels = filter_mnist_by_digits(
        Y, labels, 
        selected_digits=config.mnist_digits,
        samples_per_digit=config.samples_per_digit
    )
    
    N = len(Y)
    
    print(f"Using {N} MNIST samples for training/testing")
    
    # Debug: print labels type and shape (same as mnist_GPLVM.py)
    print(f"Labels type: {type(labels)}, Labels shape/length: {getattr(labels, 'shape', len(labels) if hasattr(labels, '__len__') else 'no length')}")
    
    # Ensure data is on CPU before converting to numpy for sklearn (same as mnist_GPLVM.py)
    y_full = Y.to(config.device)
    y_cpu = y_full.cpu() if y_full.is_cuda else y_full
    
    # Labels are already numpy array from load_mnist_data
    labels_cpu = labels
    
    y_np = y_cpu.numpy()
    
    # Create train/test split if metrics enabled (same as mnist_GPLVM.py)
    print(f"Creating train/test split (test_size={config.test_size})...")
    train_indices, test_indices, y_train_np, y_test_np, labels_train_np, labels_test_np = train_test_split(
        np.arange(len(y_full)), y_np, labels_cpu,
        test_size=config.test_size,
        random_state=config.random_seed
    )
    
    # Convert back to tensors on original device
    y_train = torch.tensor(y_train_np, dtype=torch.float64, device=config.device)
    y_test = torch.tensor(y_test_np, dtype=torch.float64, device=config.device)
    labels_train = labels_train_np  # Keep as numpy array for compatibility (same as mnist_GPLVM.py)
    labels_test = labels_test_np    # Keep as numpy array for compatibility (same as mnist_GPLVM.py)
    
    # Convert indices to tensors for consistency with oil flow version
    train_indices_tensor = torch.tensor(train_indices, dtype=torch.int64, device=config.device)
    test_indices_tensor = torch.tensor(test_indices, dtype=torch.int64, device=config.device)
    
    print(f"Data split shapes - Train: {y_train.shape}, Test: {y_test.shape}")
    print(f"Labels split info - Train: {type(labels_train)}, length: {len(labels_train) if hasattr(labels_train, '__len__') else 'no length'}")
    print(f"Labels split info - Test: {type(labels_test)}, length: {len(labels_test) if hasattr(labels_test, '__len__') else 'no length'}")

    # Use training data for model initialization
    y = y_train
    labels = labels_train
    split_data = {
        'y_train': y_train,
        'y_test': y_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'train_indices': train_indices_tensor,
        'test_indices': test_indices_tensor
    }
    
    # Initialize inducing points and latent variables (following original)
    Vx = torch.randn(config.n_inducing_points, config.latent_dim, dtype=torch.float64, device=config.device)
    mu_x, _, _ = torch.pca_lowrank(y, config.latent_dim)
    
    # No pseudotime for MNIST (unlike qPCR)

    print(f"Data loaded: {y.shape[0]} samples, {y.shape[1]} dimensions")
    print(f"Latent initialization: {mu_x.shape}")
    print(f"Inducing points: {Vx.shape}")
    config.total_size = y.shape[0]
    
    # Apply missing data preprocessing if enabled
    if config.enable_missing_data:
        print(f"\n=== Missing Data Preprocessing ===")
        
        # Store original training data before preprocessing
        y_original_train = y.clone()
        
        y_train_missing, train_missing_mask, train_missing_info = preprocess_data_for_missing_training(split_data['y_train'], config)
        y_test_missing, test_missing_mask, test_missing_info = preprocess_data_for_missing_training(split_data['y_test'], config)
        
        split_data.update({
            'y_train_original': split_data['y_train'].clone(),
            'y_test_original': split_data['y_test'].clone(),
            'y_train': y_train_missing,
            'y_test': y_test_missing,
            'train_missing_mask': train_missing_mask,
            'test_missing_mask': test_missing_mask
        })
        
        # Use training data with missing values for model training
        y = y_train_missing
        missing_mask = train_missing_mask
        missing_info = train_missing_info
        
        # Store missing data info for later use
        config.missing_info = missing_info
        config.y_original_train = y_original_train  # Original training data before missing data preprocessing
        config.missing_mask = missing_mask
        
        print(f"Missing data preprocessing completed.")
        print("=" * 50)

    return y, labels, Vx, mu_x, split_data

class OilFlow_SSVI_GPLVM(GPLVM_SSVI_AM_INF):
    """MNIST SSVI GP-LVM with Amortized Inference"""
    
    def __init__(self, config: OilFlowSSVIConfig, Vx, mu_x, M):
        # Use init_lengthscales to create beta values
        # beta = 1/lengthscales, so for init_lengthscales=1.0, beta=[1,1,1,1,1,1]
        beta = [[1.0/config.init_lengthscales] * config.latent_dim] * 1  # C=1 for GP-LVM
        
        # Use init_output_noise_scale for sigma2 initialization
        sigma2 = config.init_output_noise_scale
        
        # Initialize underlying SSVI model
        super().__init__(
            Vx=Vx,
            M=M,  # Observation dimension
            mu_x=mu_x,
            sigma2=sigma2,
            beta=beta,
            sigma2_s=config.sigma2_s,
            P=config.P,
            trainable_V=config.trainable_V,
            encoder_hidden_dims=config.encoder_hidden_dims,
            local_lr=config.local_lr, 
            global_lr=config.global_lr,
            device=config.device,
            enable_missing_data=config.enable_missing_data
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

        # Handle missing data preprocessing for encoder input
        if self.enable_missing_data:
            # Replace NaN values with -9.0 for encoder input (encoder expects clean data)
            Y_batch_clean = torch.where(torch.isnan(Y_batch), torch.full_like(Y_batch, -9.0, dtype=torch.float64), Y_batch)
            # Get local parameters via amortized inference (gradients only to encoder)
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(Y_batch_clean, U_samples)
        else:
            # Standard case: no missing data preprocessing needed
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(Y_batch, U_samples)
        
        # Now zero gradients for global optimizer before global computation
        self.optimizer_global.zero_grad()
        
        # Compute global ELBO loss (gradients flow to all parameters)
        if self.enable_missing_data:
            # Use missing data aware ELBO (coordinator handles the missing mask automatically)
            missing_mask = torch.isnan(Y_batch)
            # Use the same clean data for ELBO computation to maintain consistency with encoder
            loss = -self.ssvi_elbo(Y_batch_clean,
                                   mu_n_batch,
                                   sigma_n_batch,
                                   U_samples=U_samples,
                                   batch_size=self.config.batch_size,
                                   total_size=self.config.total_size,
                                   missing_mask=missing_mask)
        else:
            # Standard ELBO computation
            loss = -self.ssvi_elbo(Y_batch,
                                   mu_n_batch,
                                   sigma_n_batch,
                                   U_samples=U_samples,
                                   batch_size=self.config.batch_size,
                                   total_size=self.config.total_size,
                                   missing_mask=None)
        
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
    


def train_oil_flow_ssvi(model: OilFlow_SSVI_GPLVM, y_train, config: OilFlowSSVIConfig,
                        logger: SSVIResultsLogger = None, split_data=None, labels_train=None):
    """Training function for MNIST SSVI GP-LVM with optional logging and metrics"""
    losses = []
    
    print(f"Training MNIST SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")
    
    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(f"Metrics computation enabled (frequency: every {config.metrics_frequency} iterations)")
        metrics_computer = SSVIMetricsComputer(device=config.device)
    
    start_time = time.time()
    
    for i in range(config.training_iterations):
        # Sample batch indices
        N = y_train.shape[0]
        batch_indices = torch.randperm(N, device=y_train.device)[:config.batch_size]
        y_batch = y_train[batch_indices]
        
        # Training step - U sampling happens inside automatically each step
        loss = model.train_step(y_batch, batch_indices)
        losses.append(loss)
        
        # Metrics computation (periodic)
        if (metrics_computer is not None and 
            config.enable_metrics and 
            i % config.metrics_frequency == 0):
            
            print(f"  Computing metrics at iteration {i}...")

                # Use full datasets for proper evaluation
            # Compute metrics with missing data info if enabled
            if config.enable_missing_data:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data['y_train'],
                    y_test=split_data['y_test'],
                    iteration=i,
                    elbo_loss=loss,
                    n_samples=32,  # Fixed number of samples for metrics computation
                    y_train_original=split_data['y_train_original'],
                    y_test_original=split_data['y_test_original'],
                    train_missing_mask=split_data['train_missing_mask'],
                    test_missing_mask=split_data['test_missing_mask']
                )
            else:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data['y_train'],
                    y_test=split_data['y_test'],
                    iteration=i,
                    elbo_loss=loss,
                    n_samples=32  # Fixed number of samples for metrics computation
                )
            
            print(f"    Train RMSE: {metrics.train_rmse:.4f}, Test RMSE: {metrics.test_rmse:.4f} ")
            print(f"    Train NLPD: {metrics.train_nlpd:.3f}, Test NLPD: {metrics.test_nlpd:.3f}")
            if config.enable_missing_data and not np.isnan(metrics.train_missing_nlpd):
                print(f"    Train Missing Data NLPD: {metrics.train_missing_nlpd:.3f}")
                print(f"    Test Missing Data NLPD: {metrics.test_missing_nlpd:.3f}")
            if config.enable_missing_data and not np.isnan(metrics.train_missing_rmse):
                print(f"    Train Missing Data RMSE: {metrics.train_missing_rmse:.4f}")
                print(f"    Test Missing Data RMSE: {metrics.test_missing_rmse:.4f}")
            print(f"    Effective dims: {metrics.effective_latent_dims}/{metrics.total_latent_dims}")
            print(f"    Full datasets: Train={split_data['y_train'].shape[0]}, Test={split_data['y_test'].shape[0]} samples")
                
                # Log metrics
            if logger:
                logger.log_training(f"Metrics at iteration {i}: {metrics.to_dict()}")
                            
        # Checkpoint saving (models and plots together)
        if (logger and i > 0 and i % config.checkpoint_frequency == 0):
            print(f"  Saving checkpoint at iteration {i}...")
            try:
                # Save checkpoint model
                checkpoint_filename = f"mnist_ssvi_iter_{i}_seed_{config.random_seed}.pkl"
                logger.save_model(model, checkpoint_filename, is_checkpoint=True)
                print(f"    Checkpoint model saved: {checkpoint_filename}")
                
                # Save checkpoint plot (if enabled)
                if config.save_training_plots:
                    plot_filename = f"training_results_iter_{i}.png"
                    try:
                        from src.mnist_plot_helper import visualize_ssvi_results
                        visualize_ssvi_results(
                            model=model,
                            y_train=y_train,
                            labels_train=labels_train,
                            losses=losses,
                            config=config,
                            logger=logger,
                            save_filename=plot_filename,
                            show_plot=False,
                            is_checkpoint=True,
                            train_indices=split_data['train_indices']
                        )
                        print(f"    Checkpoint plot saved: {plot_filename}")
                    except Exception as e:
                        print(f"    Warning: Could not create checkpoint visualization at iteration {i}: {e}")
                
            except Exception as e:
                print(f"    Warning: Error saving checkpoint at iteration {i}: {e}")
        
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
    
    return losses, metrics_computer

# === Pre-defined Configurations ===

def get_fast_config():
    """Fast configuration for testing with missing data enabled"""
    return OilFlowSSVIConfig(
        training_iterations=100,
        batch_size=50,
        n_inducing_points=16,
        encoder_hidden_dims=[128, 64],
        local_lr=2e-3,
        global_lr=1e-2,
        P=16,
        log_frequency=50,
        checkpoint_frequency=100,  # Save checkpoint models every 100 iterations
        enable_gradient_logging=True,
        gradient_log_frequency=25,
        enable_metrics=True,
        metrics_frequency=50,
        save_training_plots=True,
        plot_training=True,
        plot_latent_space=True,
        plot_hyperparams=True,
        # === GP Hyperparameters ===
        init_lengthscales=1.0,          # Initial lengthscales value (1/beta) - default matches current standard
        init_output_noise_scale=1.0,    # Initial output noise scale (sigma2) - default matches current standard
        # === Missing Data Configuration ===
        enable_missing_data=True,      # Enable missing data for MNIST testing and visualization
        missing_data_ratio=0.3,        # Fraction of dimensions to artificially mask (for testing)
        reconstruction_samples=50,     # Number of samples for missing data reconstruction
        # === MNIST Data Filtering Configuration ===
        mnist_digits=[1, 2],           # Use only digits 1, 2 for fast testing
        samples_per_digit=100,         # 100 samples per digit = 200 total samples
        # === Random Seed Configuration ===
        random_seed=7,                 # Random seed for reproducibility (matches SSVI base config)
        # === Test Size Configuration ===
        test_size=0.2                  # Fast config uses 0.2 (20% for testing)
    )

def main(config: OilFlowSSVIConfig = None, config_name: str = "fast"):
    """Main function for MNIST SSVI GP-LVM example"""
    
    # Use default config if none provided
    if config is None:
        config = get_fast_config()
        # Override results directory for MNIST experiments when using fast config
        if config_name == "fast":
            config.results_dir = "mnist_results/ssvi_results"
    
    # Set device and dtype from config
    torch.set_default_device(config.device)
    torch.set_default_dtype(torch.float64)
    
    # Set random seed for reproducibility (same as original GPLVM code)
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)

    # Print configuration
    config.print_config()
    
    # Load MNIST data
    y_train, labels_train, Vx, mu_x, split_data = load_oil_flow_data(config)
    
    # Create model
    print("\nInitializing MNIST SSVI GP-LVM...")
    model = OilFlow_SSVI_GPLVM(
        config=config,
        Vx=Vx,
        mu_x=mu_x,
        M=y_train.shape[1]  # Observation dimension
    )
    
    print(f"Model initialized:")
    print(f"  Encoder: {model.encoder}")
    print(f"  Inducing points: {model.gp.Vx.shape}")
    print(f"  Latent variables: {model.mu_x.shape}")
    
    # Initialize logger with config-based naming
    experiment_name = f"ssvi_{config_name}"
    logger = SSVIResultsLogger(config, experiment_name=experiment_name)
    if logger.results_dir:
        print(f"Results will be saved to: {logger.results_dir}")
        # Log initial info
        logger.log_training("=== MNIST SSVI GP-LVM Training Session ===")
        logger.log_training(f"Model: {config.n_inducing_points} inducing points, latent_dim={config.latent_dim}")
        logger.log_training(f"Training: {config.training_iterations} iter, batch={config.batch_size}")
        logger.log_training(f"Learning rates: Local={config.local_lr}, Global={config.global_lr}")
    
    # Train model
    print(f"\nTraining model...")
    losses, metrics_computer = train_oil_flow_ssvi(model, y_train, config, logger, split_data, labels_train)
    
    # Visualize results
    print(f"\nVisualizing results...")
    try:
        from src.mnist_plot_helper import visualize_ssvi_results
        visualize_ssvi_results(model, y_train, labels_train, losses, config, logger, show_plot=False, train_indices=split_data['train_indices'])
    except Exception as e:
        print(f"Warning: Could not create latent space visualization: {e}")
    
    # Generate final metrics and plots if metrics were computed
    if metrics_computer is not None and config.enable_metrics and split_data is not None:
        print(f"\nGenerating final metrics and plots...")
        
        # Compute final comprehensive metrics on full test set
        print("Computing final metrics on full datasets...")
        
        # Compute final metrics with missing data info if enabled
        if config.enable_missing_data:
            final_metrics = metrics_computer.compute_full_metrics(
                model=model,
                y_train=split_data['y_train'],
                y_test=split_data['y_test'],
                iteration=config.training_iterations,
                elbo_loss=losses[-1] if losses else 0.0,
                n_samples=32,  # Fixed number of samples for final metrics computation
                y_train_original=split_data['y_train_original'],
                y_test_original=split_data['y_test_original'],
                train_missing_mask=split_data['train_missing_mask'],
                test_missing_mask=split_data['test_missing_mask']
            )
        else:
            final_metrics = metrics_computer.compute_full_metrics(
                model=model,
                y_train=split_data['y_train'],
                y_test=split_data['y_test'],
                iteration=config.training_iterations,
                elbo_loss=losses[-1] if losses else 0.0,
                n_samples=32  # Fixed number of samples for final metrics computation
            )
        
        # Print final metrics summary using the computed final_metrics
        summary = metrics_computer.get_metrics_summary()
        
        # Log the final metrics to the training log
        if logger:
            logger.log_training(f"Final comprehensive metrics computed: {final_metrics.to_dict()}")
        
        print(f"\n=== Final Model Performance ===")
        print(f"Final Train RMSE: {summary['final_train_rmse']:.4f}")
        print(f"Final Test RMSE: {summary['final_test_rmse']:.4f}")
        print(f"Final Train NLPD: {summary['final_train_nlpd']:.3f}")
        print(f"Final Test NLPD: {summary['final_test_nlpd']:.3f}")
        if config.enable_missing_data and not np.isnan(summary.get('final_train_missing_nlpd', float('nan'))):
            print(f"Final Train Missing Data NLPD: {summary['final_train_missing_nlpd']:.3f}")
            print(f"Final Test Missing Data NLPD: {summary['final_test_missing_nlpd']:.3f}")
        if config.enable_missing_data and not np.isnan(summary.get('final_train_missing_rmse', float('nan'))):
            print(f"Final Train Missing Data RMSE: {summary['final_train_missing_rmse']:.4f}")
            print(f"Final Test Missing Data RMSE: {summary['final_test_missing_rmse']:.4f}")
        print(f"Overfitting Ratio (Test/Train RMSE): {summary['overfitting_ratio']:.3f}")
        print(f"Effective Latent Dimensions: {summary['final_effective_dims']}/{summary['total_latent_dims']}")
        print("=" * 35)
        
        # Generate metrics plots
        print("Generating metrics progression plots...")
        plot_metrics_history(metrics_computer, logger, show_plot=False)
        
        print("Generating final metrics summary plots...")
        plot_final_metrics_summary(metrics_computer, model, logger, show_plot=False)
        
        # Generate missing NLPD metrics progression plot if missing data is enabled
        if config.enable_missing_data:
            print("Generating missing NLPD metrics progression plot...")
            plot_missing_nlpd_metrics_progression(metrics_computer, logger, show_plot=False)
            print("Generating missing RMSE metrics progression plot...")
            plot_missing_rmse_metrics_progression(metrics_computer, logger, show_plot=False)
        
        # Save checkpoint metrics and training losses in structured format
        if logger:
            print("Saving checkpoint metrics...")
            logger.save_checkpoint_metrics(metrics_computer, "mnist_ssvi", config.random_seed)
            print("Saving training losses...")
            logger.save_training_losses(losses, "mnist_ssvi", config.random_seed)

    
    
    # Print final model parameters
    print_final_model_parameters(model, "GP LVM SSVI")
    
    # Save final model (like original GP-LVM)
    if logger:
        print("Saving final model...")
        logger.save_model(model, f"mnist_ssvi_{config.random_seed}.pkl", is_checkpoint=False)
    
    # Return final_metrics if computed, otherwise None
    final_metrics = None
    if metrics_computer is not None and metrics_computer.metrics_history:
        final_metrics = metrics_computer.metrics_history[-1]
    
    
    # Save experiment summary
    if logger:
        # Create experiment summary with only the model-specific performance metrics
        experiment_summary = {}
        
        # Add model-specific performance metrics if available
        if summary:
            # Convert tensor to float for JSON serialization
            final_noise = getattr(model, 'sigma2', 1.0)
            if hasattr(final_noise, 'item'):
                final_noise = final_noise.item()
            
            experiment_summary['mnist_ssvi'] = {
                'train_rmse': summary.get('final_train_rmse', 0.0),
                'test_rmse': summary.get('final_test_rmse', 0.0),
                'train_nlpd': summary.get('final_train_nlpd', 0.0),
                'test_nlpd': summary.get('final_test_nlpd', 0.0),
                'train_missing_data_nlpd': summary.get('final_train_missing_nlpd', float('nan')),
                'test_missing_data_nlpd': summary.get('final_test_missing_nlpd', float('nan')),
                'train_missing_data_rmse': summary.get('final_train_missing_rmse', float('nan')),
                'test_missing_data_rmse': summary.get('final_test_missing_rmse', float('nan')),
                'final_loss': losses[-1] if losses else 0.0,
                'final_noise': float(final_noise)  # Ensure it's a Python float
            }
        
        logger.save_experiment_summary(experiment_summary)
    
    # Visualize missing data reconstruction if enabled
    if config.enable_missing_data and split_data is not None:
        print("\n=== Missing Data Reconstruction Visualization ===")
        
        # Simple visualization (3 columns: original, masked, samples)
        if logger:
            simple_viz_path = os.path.join(logger.plots_dir, f"missing_data_reconstruction_{config.random_seed}.png")
        else:
            simple_viz_path = f"missing_data_reconstruction_{config.random_seed}.png"
        
        visualize_mnist_missing_reconstruction(
            model=model,
            y_train=split_data['y_train'],
            missing_mask=split_data['train_missing_mask'],
            y_original=split_data['y_train_original'],
            n_images=10,
            n_samples=5,
            save_path=simple_viz_path,
            show_plot=False
        )
        
        # Detailed visualization (with mean and uncertainty)
        if logger:
            detailed_viz_path = os.path.join(logger.plots_dir, f"missing_data_reconstruction_detailed_{config.random_seed}.png")
        else:
            detailed_viz_path = f"missing_data_reconstruction_detailed_{config.random_seed}.png"
        
        visualize_mnist_missing_reconstruction_detailed(
            model=model,
            y_train=split_data['y_train'],
            missing_mask=split_data['train_missing_mask'],
            y_original=split_data['y_train_original'],
            n_images=10,
            n_samples=5,
            save_path=detailed_viz_path,
            show_plot=False
        )
        
        print("=" * 50)
    
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

    config, config_name = load_config_with_overrides(
        config_path=args.config,
        config_class=OilFlowSSVIConfig,
        args=args
    )
    
    # Override results directory for MNIST experiments when using fast config
    if config_name == "fast":
        config.results_dir = "mnist_results/ssvi_results"
    
    print(f"Using configuration: {config_name}")
    if args.config:
        print(f"Config file: {args.config}")
    
    # Run main function with loaded config
    main(config, config_name)
    
