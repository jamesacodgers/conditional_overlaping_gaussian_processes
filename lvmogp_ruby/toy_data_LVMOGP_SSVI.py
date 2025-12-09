"""
SSVI LVMOGP Toy Data Demonstration with Configuration and Results Management

Uses function-based batching: batch_fn_size specifies the number of functions per batch,
and all observations for each selected function are processed together.

Usage examples:
    # Run with default fast config (50 iterations, quick testing)
    python lvmogp_ruby/toy_data_LVMOGP_SSVI.py
    
    # Run with base config (from working notebook - 15000 iterations)
    python lvmogp_ruby/toy_data_LVMOGP_SSVI.py --config lvmogp_ruby/configs/base_config.yaml
    
    # Run with config and override specific parameters
    python lvmogp_ruby/toy_data_LVMOGP_SSVI.py --config lvmogp_ruby/configs/base_config.yaml --training-iterations 20000 --device cuda
    
    # Run with custom parameters (no config file, uses fast default + overrides)
    python lvmogp_ruby/toy_data_LVMOGP_SSVI.py --training-iterations 1000 --batch-fn-size 4 --random-seed 42
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
import argparse
from datetime import datetime
import time
import yaml

# Get the project root directory (parent of lvmogp_ruby)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add project root to path (contains src/ folder and lvmogp_ruby/)
sys.path.insert(0, project_root)

from lvmogp_ruby.ssvi_lvmogp import LVMOGP_SSVI
from lvmogp_ruby.lvmogp_test_functions import TestFun


# ============================================================================
# Configuration Class
# ============================================================================

@dataclass
class LVMOGPConfig:
    """Configuration for SSVI LVMOGP (defaults from working SSVI_LVMOGP_demonstration.ipynb)"""
    
    # === Device Configuration ===
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"
    
    # === Data Generation Configuration ===
    random_seed: int = 1
    surface_type: str = 'linear_relation'
    n_new_points: int = 30
    n_new_funs: int = 10
    n_observed_funs: int = 2  # Number of initially observed functions
    domain: list = None  # Will be set to [0, 2] if None
    noise: float = 0.05
    n_grid_points: int = 100
    observed_dims: int = 1
    latent_dims: int = 2
    latent_dims_lvmogp: int = 2
    max_points: int = 60
    lengthscales_x: list = None  # Will be set to [0.3, 0.3] if None
    lengthscales_h: list = None  # Will be set to [0.4, 0.4, 0.4, 0.4, 0.4, 0.4][:latent_dims] if None
    same_points: bool = False
    
    # === Model Architecture ===
    n_inducing_points: int = 32
    encoder_hidden_dims: list = None  # Will be set to [128, 64] if None
    H_var_init: float = 0.1  # Initial variance for H (not too small for SSVI)
    lik_variance_scale: float = 0.01  # Multiplier for data variance to set likelihood noise
    
    # === Training Configuration ===
    training_iterations: int = 15000
    batch_fn_size: int = 4  # Number of functions per batch (all observations for each function are used)
    
    # === Learning Rates ===
    local_lr: float = 1e-4   # Encoder parameters (0.0001)
    global_lr: float = 1e-4  # GP hyperparameters (0.0001)
    
    # === GP Hyperparameters ===
    init_lengthscales: float = 1.0  # Initial lengthscales value
    init_signal_variance: float = 1.0  # Initial signal variance
    P: int = 32  # Number of samples for SSVI
    trainable_V: bool = True
    
    # === Progress Reporting ===
    log_frequency: int = 500  # Log every N iterations
    checkpoint_frequency: int = 1000  # Save checkpoint plots every N iterations
    loss_save_frequency: int = 5  # Save loss every N iterations
    
    # === Results Configuration ===
    save_results: bool = True
    results_dir: str = "lvmogp_ruby/lvmogp_results/toy_data_results"  # Results directory (relative to current working directory)
    experiment_name: str = "toy_data_lvmogp_ssvi"  # Experiment name for folder structure
    
    # === Visualization Configuration ===
    plot_training: bool = True
    plot_predictions: bool = True
    plot_latent_space: bool = True
    save_final_plots: bool = True
    
    # === Missing Data Configuration ===
    enable_missing_data: bool = False
    
    def __post_init__(self):
        """Set defaults and resolve device if not specified"""
        if self.domain is None:
            self.domain = [0, 2]
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [128, 64]
        if self.lengthscales_x is None:
            self.lengthscales_x = [0.3, 0.3]
        if self.lengthscales_h is None:
            self.lengthscales_h = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4][:self.latent_dims]
        
        # Resolve device configuration
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def print_config(self):
        """Print configuration summary"""
        print("=" * 60)
        print("=== SSVI LVMOGP Configuration ===")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0) if self.device.startswith('cuda') else 'Not using CUDA'}")
        else:
            print("CUDA available: No")
        print(f"Random seed: {self.random_seed}")
        print(f"Data: {self.n_observed_funs + self.n_new_funs} functions total ({self.n_observed_funs} observed + {self.n_new_funs} new)")
        print(f"Model: {self.n_inducing_points} inducing points, latent_dim={self.latent_dims_lvmogp}")
        print(f"Encoder: {self.encoder_hidden_dims} hidden dims")
        print(f"Training: {self.training_iterations} iter, batch_fn_size={self.batch_fn_size}")
        print(f"Learning rates: Local={self.local_lr}, Global={self.global_lr}")
        print(f"SSVI: P={self.P} samples")
        print(f"Results dir: {self.results_dir}/{self.experiment_name}")
        print("=" * 60)
    
    def to_dict(self):
        """Convert config to dictionary for JSON serialization"""
        return {
            'device': self.device,
            'random_seed': self.random_seed,
            'surface_type': self.surface_type,
            'n_new_points': self.n_new_points,
            'n_new_funs': self.n_new_funs,
            'n_observed_funs': self.n_observed_funs,
            'domain': self.domain,
            'noise': self.noise,
            'n_grid_points': self.n_grid_points,
            'observed_dims': self.observed_dims,
            'latent_dims': self.latent_dims,
            'latent_dims_lvmogp': self.latent_dims_lvmogp,
            'max_points': self.max_points,
            'lengthscales_x': self.lengthscales_x,
            'lengthscales_h': self.lengthscales_h,
            'same_points': self.same_points,
            'n_inducing_points': self.n_inducing_points,
            'encoder_hidden_dims': self.encoder_hidden_dims,
            'H_var_init': self.H_var_init,
            'lik_variance_scale': self.lik_variance_scale,
            'training_iterations': self.training_iterations,
            'batch_fn_size': self.batch_fn_size,
            'local_lr': self.local_lr,
            'global_lr': self.global_lr,
            'init_lengthscales': self.init_lengthscales,
            'init_signal_variance': self.init_signal_variance,
            'P': self.P,
            'trainable_V': self.trainable_V,
            'log_frequency': self.log_frequency,
            'checkpoint_frequency': self.checkpoint_frequency,
            'loss_save_frequency': self.loss_save_frequency,
            'save_results': self.save_results,
            'results_dir': self.results_dir,
            'experiment_name': self.experiment_name,
            'plot_training': self.plot_training,
            'plot_predictions': self.plot_predictions,
            'plot_latent_space': self.plot_latent_space,
            'save_final_plots': self.save_final_plots,
            'enable_missing_data': self.enable_missing_data
        }


# ============================================================================
# Results Logger Class
# ============================================================================

class LVMOGPResultsLogger:
    """Independent results logger for LVMOGP experiments"""
    
    def __init__(self, config: LVMOGPConfig, experiment_name: str = None):
        self.config = config
        self.experiment_name = experiment_name if experiment_name is not None else config.experiment_name
        self.results_dir = None
        self.plots_dir = None
        self.checkpoints_dir = None
        self.training_results_dir = None
        self.log_file = None
        
        if config.save_results:
            self._setup_directories()
    
    def _setup_directories(self):
        """Setup results directory structure"""
        # Create timestamped experiment folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.experiment_name}_seed_{self.config.random_seed}_{timestamp}"
        
        # Main results directory
        self.results_dir = Path(self.config.results_dir) / exp_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories matching oil_flow structure
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.checkpoints_dir = self.plots_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.training_results_dir = self.results_dir / "training_results"
        self.training_results_dir.mkdir(exist_ok=True)
        
        # Create log file
        self.log_file = self.results_dir / "training_log.txt"
        
        print(f"Results directory created: {self.results_dir}")
        print(f"  Plots: {self.plots_dir}")
        print(f"  Checkpoints: {self.checkpoints_dir}")
        print(f"  Training results: {self.training_results_dir}")
    
    def log_training(self, message: str):
        """Log training message to file"""
        if self.log_file:
            with open(self.log_file, 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] {message}\n")
    
    def save_config(self):
        """Save configuration to JSON file in root results directory"""
        if self.results_dir:
            config_path = self.results_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            print(f"Configuration saved to: {config_path}")
    
    def save_losses(self, losses: list, iterations: list):
        """Save training losses to file"""
        if self.training_results_dir:
            losses_path = self.training_results_dir / f"training_losses_seed_{self.config.random_seed}.json"
            data = {
                'iterations': [int(i) for i in iterations],
                'losses': [float(loss) for loss in losses]
            }
            with open(losses_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Training losses saved to: {losses_path}")
    
    def save_plot(self, fig, filename: str, is_checkpoint: bool = False):
        """Save a plot to appropriate directory"""
        if is_checkpoint:
            save_dir = self.checkpoints_dir
        else:
            save_dir = self.plots_dir
        
        if save_dir:
            filepath = save_dir / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Plot saved: {filepath}")
            return filepath
        return None


# ============================================================================
# Data Generation
# ============================================================================

def generate_data(config: LVMOGPConfig):
    """Generate synthetic data using TestFun"""
    print("\n=== Generating Data ===")
    
    n_fun = config.n_observed_funs + config.n_new_funs
    n_points = [10] * config.n_observed_funs + [config.n_new_points] * config.n_new_funs
    
    test_fun = TestFun(
        domain=config.domain,
        seed=config.random_seed,
        n_fun=n_fun,
        observed_dims=config.observed_dims,
        test_type=config.surface_type,
        latent_dims=config.latent_dims,
        max_points=config.max_points,
        noise=config.noise,
        n_grid_points=config.n_grid_points,
        same_points=config.same_points,
        lengthscales_X=config.lengthscales_x,
        lengthscales_H=config.lengthscales_h
    )
    
    data_X, fun_nos, data_y = test_fun.create_data(n_points=n_points)
    
    print(f"Data generated: {len(data_X)} total observations")
    print(f"  Functions: {n_fun} ({config.n_observed_funs} observed + {config.n_new_funs} new)")
    print(f"  Input dimension: {data_X.shape[1]}")
    print(f"  Output dimension: {data_y.shape[1]}")
    
    return test_fun, data_X, fun_nos, data_y, n_points


# ============================================================================
# Model Creation
# ============================================================================

def create_model(config: LVMOGPConfig, data_X, data_y, fun_nos, n_fun):
    """Create LVMOGP model"""
    print("\n=== Creating Model ===")
    
    # Convert to tensors
    device = torch.device(config.device)
    Y_torch = torch.tensor(data_y, dtype=torch.float64, device=device)
    X_torch = torch.tensor(data_X, dtype=torch.float64, device=device)
    fn_torch = torch.tensor(fun_nos, dtype=torch.long, device=device)
    
    # Random H initialization
    H_mean = torch.tensor(
        np.random.uniform(-1, 1, (n_fun, config.latent_dims_lvmogp)),
        dtype=torch.float64,
        device=device
    )
    H_var = torch.ones_like(H_mean) * config.H_var_init
    
    # Likelihood noise
    lik_variance = np.var(data_y) * config.lik_variance_scale
    
    # Determine dimensions
    Q_x = X_torch.shape[1]  # Observed dimensions
    Q_h = config.latent_dims_lvmogp  # Function latent dimensions
    Q_total = Q_x + Q_h
    M = Y_torch.shape[1]  # Output dimension
    
    # Initialize inducing points in full space [Q_x + Q_h]
    Vx = torch.randn(config.n_inducing_points, Q_total, dtype=torch.float64, device=device) * 0.5
    
    # GP hyperparameters
    beta = [[config.init_lengthscales] * Q_total]  # Lengthscales
    sigma2_s = [config.init_signal_variance]  # Signal variance
    
    model = LVMOGP_SSVI(
        X_data=X_torch,
        X_data_fn=fn_torch,
        num_fns=n_fun,
        Q_h=Q_h,
        M=M,
        Vx=Vx,
        H_data_mean=H_mean,
        H_data_var=H_var,
        sigma2=lik_variance,
        beta=beta,
        sigma2_s=sigma2_s,
        P=config.P,
        trainable_V=config.trainable_V,
        encoder_hidden_dims=config.encoder_hidden_dims,
        local_lr=config.local_lr,
        global_lr=config.global_lr,
        device=device,
        enable_missing_data=config.enable_missing_data
    )
    
    print(f"Model created on device: {model.device}")
    print(f"  Encoder: {model.encoder}")
    print(f"  Inducing points: {model.gp.Vx.shape}")
    print(f"  Q_x (observed dims): {Q_x}, Q_h (latent dims): {Q_h}, Q_total: {Q_total}")
    print(f"  Likelihood variance: {lik_variance:.6f}")
    
    return model, Y_torch, X_torch, fn_torch


# ============================================================================
# Training
# ============================================================================

def train_model(model, Y_torch, config: LVMOGPConfig, logger: LVMOGPResultsLogger = None, 
                test_fun=None, n_points=None, n_fun=None):
    """Train LVMOGP model with function-based batching"""
    print("\n=== Training Model ===")
    
    device = torch.device(config.device)
    batch_fn_size = min(config.batch_fn_size, n_fun)
    
    # Store checkpoint data in model for access during training
    if test_fun is not None and n_points is not None and n_fun is not None:
        model._checkpoint_data = (test_fun, n_points, n_fun)
    
    # Storage for losses
    losses = []
    iterations_list = []
    
    model.train()
    
    print(f"Training on {config.device}: {config.training_iterations} iterations, batch_fn_size={batch_fn_size}")
    print(f"  (Processing all observations for {batch_fn_size} functions per batch)")
    if logger:
        logger.log_training(f"=== Training Started ===")
        logger.log_training(f"Iterations: {config.training_iterations}, Batch fn size: {batch_fn_size}")
        logger.log_training(f"Device: {config.device}")
    
    start_time = time.time()
    
    for iteration in range(config.training_iterations):
        # Random batch of function IDs (function-based batching)
        fn_ids_batch = torch.randperm(n_fun, device=device)[:batch_fn_size]
        
        # Training step: pass full Y and function IDs
        # Model internally gathers all observations for these functions
        loss = model.train_step(Y_torch, fn_ids_batch)
        
        # Store loss periodically
        if (iteration + 1) % config.loss_save_frequency == 0 or iteration == 0:
            losses.append(loss)
            iterations_list.append(iteration + 1)
        
        # Print progress
        if (iteration + 1) % config.log_frequency == 0 or iteration == 0:
            elapsed = time.time() - start_time
            message = f"Iteration {iteration + 1}/{config.training_iterations}: Loss = {loss:.4f}, Time = {elapsed:.1f}s"
            print(message)
            if logger:
                logger.log_training(message)
        
        # Checkpoint plots (predictions and latent space)
        if logger and (iteration + 1) % config.checkpoint_frequency == 0 and iteration > 0:
            print(f"  Saving checkpoint plots at iteration {iteration + 1}...")
            try:
                if hasattr(model, '_checkpoint_data'):
                    test_fun, n_points, n_fun = model._checkpoint_data
                    
                    # Switch to eval mode for predictions
                    model.eval()
                    
                    # Save predictions checkpoint
                    fig_pred = plot_predictions(model, test_fun, n_points, config)
                    filename_pred = f"predictions_iter_{iteration + 1}.png"
                    logger.save_plot(fig_pred, filename_pred, is_checkpoint=True)
                    plt.close(fig_pred)
                    
                    # Save latent space checkpoint
                    fig_latent = plot_latent_space(model, n_fun, config)
                    filename_latent = f"latent_space_iter_{iteration + 1}.png"
                    logger.save_plot(fig_latent, filename_latent, is_checkpoint=True)
                    plt.close(fig_latent)
                    
                    # Switch back to train mode
                    model.train()
                else:
                    print("  Warning: Checkpoint data not available, skipping checkpoint plots")
            except Exception as e:
                print(f"  Warning: Error saving checkpoint plots: {e}")
                # Make sure to switch back to train mode even if error occurred
                model.train()
    
    total_time = time.time() - start_time
    final_message = f"Training completed! Final Loss: {losses[-1]:.4f}, Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    print(final_message)
    if logger:
        logger.log_training(final_message)
    
    # Store results in model for compatibility
    results = {
        "loss": np.array(losses),
        "iterations": np.array(iterations_list),
        "H_mean": model.H_data_mean.mean(dim=0),
        "H_log_s2": torch.log(model.H_data_var.mean(dim=0))
    }
    model.results = results
    
    return losses, iterations_list


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_elbo_progression(losses, iterations, config: LVMOGPConfig):
    """Plot ELBO (loss) progression"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, losses, label='Loss (-ELBO)', color='blue', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (-ELBO)', fontsize=12)
    ax.set_title('LVMOGP SSVI Training Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_predictions(model, test_fun, n_points, config: LVMOGPConfig):
    """Plot predictions vs ground truth for all functions"""
    device = torch.device(config.device)
    n_fun = len(test_fun.functions)
    pts_per_fun = 100
    
    # Prepare test points (use get_default_dtype like original demo)
    x_new = torch.linspace(
        config.domain[0], config.domain[1], pts_per_fun,
        dtype=torch.get_default_dtype(), device=device
    ).view(-1, 1)
    
    xs_new = x_new.repeat(n_fun, 1)

    # Function IDs (use arange which should match 0, 1, 2, ..., n_fun-1)
    fun_ids = torch.arange(n_fun, dtype=torch.long, device=device)
    X_data_fun_new = torch.cat([
        torch.full((len(x_new),), fid, dtype=torch.long, device=device)
        for fid in fun_ids
    ])
    
    # Make predictions
    lvm_mu, lvm_sig2 = model.predict_f(xs_new, X_data_fun_new)
    
    # Get ground truth
    ys_new = []
    for f in test_fun.functions:
        y_np, _ = f.predict_y(x_new.cpu().numpy())
        ys_new.append(torch.from_numpy(y_np).to(dtype=x_new.dtype, device=x_new.device))
    
    # Create plot
    palette = (plt.rcParams['axes.prop_cycle'].by_key()['color']) * 10
    fig, axs = plt.subplots(1, ncols=n_fun, figsize=(3*n_fun, 3))
    if n_fun == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    x_np = x_new.cpu().numpy().flatten()
    lvm_mu_np = lvm_mu.cpu().numpy().squeeze()
    lvm_sig2_np = lvm_sig2.cpu().numpy().squeeze()
    
    expected_size = n_fun * pts_per_fun
    if lvm_mu_np.size != expected_size:
        pts_per_fun = lvm_mu_np.size // n_fun
    
    mu_np = lvm_mu_np.reshape(n_fun, pts_per_fun)
    s2_np = lvm_sig2_np.reshape(n_fun, pts_per_fun)
    
    for i in range(n_fun):
        mu_i, s2_i = mu_np[i], s2_np[i]
        
        axs[i].plot(x_np[:pts_per_fun], mu_i, color=palette[i], lw=1.5, label='Prediction')
        axs[i].fill_between(
            x_np[:pts_per_fun],
            mu_i + 2.0*np.sqrt(np.maximum(s2_i, 1e-8)),
            mu_i - 2.0*np.sqrt(np.maximum(s2_i, 1e-8)),
            alpha=0.20, color=palette[i]
        )
        
        axs[i].scatter(
            test_fun.xs[i][:n_points[i]],
            test_fun.ys[i][:n_points[i]],
            color='k', s=12, zorder=3, label='Training data'
        )
        
        axs[i].plot(
            x_np[:pts_per_fun],
            ys_new[i].cpu().numpy().flatten()[:pts_per_fun],
            linestyle=':', color='k', lw=1, label='Ground truth'
        )
        
        axs[i].set_title(f'Function {i+1}')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        if i == 0:
            axs[i].legend(fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_latent_space(model, n_fun, config: LVMOGPConfig):
    """Plot latent space with uncertainty ellipses"""
    palette = (plt.rcParams['axes.prop_cycle'].by_key()['color']) * 10
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Get H statistics from model (works during training or after)
    if hasattr(model, 'results') and model.results is not None:
        # Use stored results (H_log_s2 is log of variance, so exp to get variance)
        H_mean_trained = model.results["H_mean"].cpu().numpy()
        H_var_trained = model.results["H_log_s2"].exp().cpu().numpy()
    else:
        # Compute on-the-fly during training
        # Note: results["H_log_s2"] = log(mean(H_data_var)), so exp(log(mean)) = mean
        H_mean_trained = model.H_data_mean.mean(dim=0).cpu().numpy()  # Average over P samples
        H_var_trained = model.H_data_var.mean(dim=0).cpu().numpy()  # Average over P samples
    
    for i, (m, v) in enumerate(zip(H_mean_trained, H_var_trained)):
        ax.scatter(m[0], m[1], color=palette[i], s=60, zorder=3, edgecolors='black', linewidths=0.5)
        
        offset = np.array([0.02, 0.02])
        ax.annotate(str(i+1), m + offset, fontsize=10, fontweight='bold')
        
        width = 1.95 * np.sqrt(np.maximum(v[0], 1e-6))
        height = 1.95 * np.sqrt(np.maximum(v[1], 1e-6))
        
        ell = Ellipse(
            xy=m,
            width=width,
            height=height,
            angle=0,
            alpha=0.30,
            color=palette[i]
        )
        ax.add_patch(ell)
    
    ax.set_title('LVMOGP Latent Space', fontsize=14, fontweight='bold')
    ax.set_xlabel('Latent Dim 1', fontsize=12)
    ax.set_ylabel('Latent Dim 2', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================================
# Pre-defined Configurations
# ============================================================================

def get_fast_config():
    """Fast configuration for quick testing (default when no config specified)"""
    return LVMOGPConfig(
        # === Device Configuration ===
        device="auto",
        
        # === Data Generation Configuration ===
        random_seed=1,
        surface_type='linear_relation',
        n_new_points=20,  # Reduced from 30
        n_new_funs=3,  # Reduced from 10 (total 5 functions)
        n_observed_funs=2,
        noise=0.05,
        
        # === Model Architecture ===
        n_inducing_points=8,  # Very small for fast testing
        encoder_hidden_dims=[64, 32],  # Smaller network
        H_var_init=0.1,
        
        # === Training Configuration ===
        training_iterations=50,  # Very fast
        batch_fn_size=3,  # Process 3 functions per batch
        
        # === Learning Rates ===
        local_lr=1e-4,
        global_lr=1e-4,
        
        # === GP Hyperparameters ===
        P=16,  # Reduced from 32
        
        # === Progress Reporting ===
        log_frequency=25,  # Log every 25 iterations
        checkpoint_frequency=25,  # Checkpoint every 25 iterations
        
        # === Results Configuration ===
        results_dir="lvmogp_ruby/lvmogp_results/toy_data_results",
        experiment_name="fast_toy_data_lvmogp_ssvi"
    )


# ============================================================================
# Main Function
# ============================================================================

def main(config: LVMOGPConfig = None, config_name: str = "fast"):
    """Main function for SSVI LVMOGP demonstration"""
    
    # Use fast config by default if none provided
    if config is None:
        config = get_fast_config()
    
    # Set device and dtype from config
    torch.set_default_device(config.device)
    torch.set_default_dtype(torch.float64)
    
    # Set random seed for reproducibility (same as oil_flow GPLVM code)
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Print configuration
    config.print_config()
    
    # Initialize logger with config-based naming
    experiment_name = f"lvmogp_{config_name}"
    logger = LVMOGPResultsLogger(config, experiment_name=experiment_name) if config.save_results else None
    if logger:
        print(f"Results will be saved to: {logger.results_dir}")
        logger.save_config()
    
    # Generate data
    test_fun, data_X, fun_nos, data_y, n_points = generate_data(config)
    n_fun = config.n_observed_funs + config.n_new_funs
    
    # Create model
    model, Y_torch, X_torch, fn_torch = create_model(
        config, data_X, data_y, fun_nos, n_fun
    )
    
    # Train model
    losses, iterations_list = train_model(model, Y_torch, config, logger, test_fun, n_points, n_fun)
    
    # Save losses
    if logger:
        logger.save_losses(losses, iterations_list)
    
    # Generate final plots
    print("\n=== Generating Final Plots ===")
    
    # Switch to eval mode for final plots
    model.eval()
    
    # 1. ELBO progression
    if config.plot_training:
        print("Plotting ELBO progression...")
        fig_elbo = plot_elbo_progression(losses, iterations_list, config)
        if logger:
            logger.save_plot(fig_elbo, f"elbo_progression_final_seed_{config.random_seed}.png", is_checkpoint=False)
        if config.save_final_plots:
            plt.close(fig_elbo)
        else:
            plt.show()
    
    # 2. Predictions
    if config.plot_predictions:
        print("Plotting predictions...")
        fig_pred = plot_predictions(model, test_fun, n_points, config)
        if logger:
            logger.save_plot(fig_pred, f"predictions_final_seed_{config.random_seed}.png", is_checkpoint=False)
        if config.save_final_plots:
            plt.close(fig_pred)
        else:
            plt.show()
    
    # 3. Latent space
    if config.plot_latent_space:
        print("Plotting latent space...")
        fig_latent = plot_latent_space(model, n_fun, config)
        if logger:
            logger.save_plot(fig_latent, f"latent_space_final_seed_{config.random_seed}.png", is_checkpoint=False)
        if config.save_final_plots:
            plt.close(fig_latent)
        else:
            plt.show()
    
    print("\n=== Experiment Complete ===")
    if logger:
        print(f"All results saved to: {logger.results_dir}")
    
    return model, losses, test_fun, config


# ============================================================================
# Config Loading Utilities
# ============================================================================

def get_config_name_from_path(config_path: str) -> str:
    """Extract config name from file path for result naming."""
    # Remove extension and directory path
    config_name = Path(config_path).stem
    
    # Remove common prefixes/suffixes for cleaner naming
    if config_name.endswith('_config'):
        config_name = config_name[:-7]
    
    return config_name


def load_config_from_yaml(config_path: str) -> tuple:
    """
    Load configuration from YAML file.
    
    Returns:
        tuple: (config_object, config_name)
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object from dictionary
    config = LVMOGPConfig(**config_dict)
    
    # Extract config name from path
    config_name = get_config_name_from_path(config_path)
    
    # Automatically set results_dir to the same directory as the config file
    # This ensures results are saved in the same folder as the config for easy tracking
    config_dir = os.path.dirname(config_path)
    config.results_dir = os.path.join(config_dir, "toy_data_run_results")
    print(f"Results will be saved to config directory: {config.results_dir}")
    
    return config, config_name


def merge_config_with_args(config: LVMOGPConfig, args) -> LVMOGPConfig:
    """Merge config with command line arguments (args take precedence)"""
    config_dict = config.to_dict()
    
    # Update with command line args if they were explicitly provided
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and key != 'config' and key != 'no_save':
            # Convert arg names with dashes to underscores
            config_key = key.replace('-', '_')
            if config_key in config_dict:
                config_dict[config_key] = value
    
    # Handle special case for no_save flag
    if hasattr(args, 'no_save') and args.no_save:
        config_dict['save_results'] = False
    
    return LVMOGPConfig(**config_dict)


# ============================================================================
# Command Line Interface
# ============================================================================

def create_argument_parser():
    """Create argument parser for command line interface"""
    parser = argparse.ArgumentParser(
        description="SSVI LVMOGP Demonstration with Configuration"
    )
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (default: None, uses defaults)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use: auto, cuda, cpu (default: auto)')
    
    # Data configuration
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed (default: 1)')
    parser.add_argument('--n-new-funs', type=int, default=None,
                        help='Number of new functions (default: 10)')
    parser.add_argument('--n-new-points', type=int, default=None,
                        help='Number of points per new function (default: 30)')
    
    # Model configuration
    parser.add_argument('--n-inducing-points', type=int, default=None,
                        help='Number of inducing points (default: 32)')
    parser.add_argument('--latent-dims-lvmogp', type=int, default=None,
                        help='Latent dimensions for LVMOGP (default: 2)')
    
    # Training configuration
    parser.add_argument('--training-iterations', type=int, default=None,
                        help='Number of training iterations (default: 15000)')
    parser.add_argument('--batch-fn-size', type=int, default=None,
                        help='Number of functions per batch (default: 4)')
    parser.add_argument('--local-lr', type=float, default=None,
                        help='Local learning rate (default: 1e-4)')
    parser.add_argument('--global-lr', type=float, default=None,
                        help='Global learning rate (default: 1e-4)')
    
    # Logging
    parser.add_argument('--log-frequency', type=int, default=None,
                        help='Log frequency (default: 500)')
    parser.add_argument('--checkpoint-frequency', type=int, default=None,
                        help='Checkpoint frequency (default: 1000)')
    
    # Results
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory (default: lvmogp_results)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: toy_data_lvmogp_ssvi)')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving results')
    
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load config
    config_name = "fast"  # Default config name
    
    if args.config:
        # Load from YAML file
        print(f"Loading configuration from: {args.config}")
        config, config_name = load_config_from_yaml(args.config)
        # Merge with command line arguments (args take precedence)
        config = merge_config_with_args(config, args)
        print(f"Using configuration: {config_name}")
        print(f"Configuration loaded and merged with command line arguments")
    else:
        # Use fast config as default (quick testing)
        print("No config specified - using fast default configuration (50 iterations)")
        print("For full experiment, use: --config lvmogp_ruby/configs/base_config.yaml")
        config = get_fast_config()
        # Apply command line arguments if provided
        config = merge_config_with_args(config, args)
    
    # Run main function with config and config_name
    main(config, config_name)
