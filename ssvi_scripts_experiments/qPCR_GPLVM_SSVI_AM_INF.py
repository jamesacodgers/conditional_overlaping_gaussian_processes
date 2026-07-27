# %% 
"""
qPCR SSVI GP-LVM with Amortized Inference

Usage examples:
    # Run with default fast config
    python ssvi_scripts_experiments/qPCR_GPLVM_SSVI_AM_INF.py
    
    # Run with base config
    python ssvi_scripts_experiments/qPCR_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml
    
    # Run with base config and overrides
    python ssvi_scripts_experiments/qPCR_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml --training-iterations 100 --device cuda
    
    # Run with custom config name for results
    python ssvi_scripts_experiments/qPCR_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml --config-name my_experiment
"""

import torch 
import matplotlib.pyplot as plt
import pandas as pd

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
from src.ssvi_utils import SSVIResultsLogger, visualize_ssvi_results, plot_metrics_history, plot_final_metrics_summary, plot_missing_nlpd_metrics_progression, plot_missing_rmse_metrics_progression, plot_crps_metrics_progression, plot_crps_posterior_predictive_progression, plot_latent_slice_outer
from src.ssvi_metrics import SSVIMetricsComputer, create_train_test_split
from src.missing_data_utils import preprocess_data_for_missing_training
from src.save_utils import print_final_model_parameters
from src.ssvi_config_utils import (
    load_config_with_overrides, create_argument_parser, 
    get_base_config_path, ensure_config_dir_exists, print_available_configs
)
from src.ssvi_config import OilFlowSSVIConfig
from ssvi_scripts_experiments.mnist_ood_GPLVM_SSVI_AM_INF import OilFlow_SSVI_GPLVM
from sklearn.model_selection import train_test_split
from src.pytorch_scalers import MeanCenter, StandardScalerPytorch, MinMaxScalerPytorch, NoTransform

# Device will be set in main() from config

def get_pseudotime():
    """Get pseudotime for qPCR data (same as original qPCR_expt.py)"""
    # license: Copyright (c) 2014, the Open Data Science Initiative
    # license: https://www.elsevier.com/legal/elsevier-website-terms-and-conditions
    URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
    
    df = pd.read_csv(URL, index_col=0)
    data = torch.tensor(df.values, dtype=torch.float64)
    # we need to transpose data to correct its shape
    y = data.t()
    capture_time = y.new_tensor([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
    # we scale the time into the interval [0, 1]
    time = capture_time.log2() / 6
    return time

def load_oil_flow_data(config: OilFlowSSVIConfig):
    """Load and preprocess qPCR data with optional train/test split"""
    print("Loading qPCR data...")
    
    # Load data using the same function as Original GP-LVM
    from utils.data import load_real_data
    
    # Load data
    N, d, q, X, Y, labels = load_real_data('gene')
    
    # Get pseudotime for qPCR data
    pseudo_time = get_pseudotime()
    
    # Debug: print labels type and shape (same as qPCR_GPLVM.py)
    print(f"Labels type: {type(labels)}, Labels shape/length: {getattr(labels, 'shape', len(labels) if hasattr(labels, '__len__') else 'no length')}")
    
    # Ensure data is on CPU before converting to numpy for sklearn (same as qPCR_GPLVM.py)
    y_full = Y.to(config.device)
    y_cpu = y_full.cpu() if y_full.is_cuda else y_full
    
    # Handle labels - convert to numpy array if it's a tensor, otherwise keep as is (same as qPCR_GPLVM.py)
    if hasattr(labels, 'cpu'):
        labels_cpu = labels.cpu().numpy()
    elif hasattr(labels, 'numpy'):
        labels_cpu = labels.numpy()
    else:
        # If labels is already a list or numpy array, convert to numpy array
        labels_cpu = np.array(labels)
    
    y_np = y_cpu.numpy()
    pseudo_time_full = pseudo_time.to(config.device)
    
    # Create train/test split if metrics enabled (same as qPCR_GPLVM.py)
    print(f"Creating train/test split (test_size={config.test_size})...")
    train_indices, test_indices, y_train_np, y_test_np, labels_train_np, labels_test_np, pseudo_train_np, pseudo_test_np = train_test_split(
        np.arange(len(y_full)), y_np, labels_cpu, pseudo_time_full.cpu().numpy(),
        test_size=config.test_size,
        random_state=config.random_seed
    )
    
    # Convert back to tensors on original device
    y_train = torch.tensor(y_train_np, dtype=torch.float64, device=config.device)
    y_test = torch.tensor(y_test_np, dtype=torch.float64, device=config.device)
    labels_train = labels_train_np  # Keep as numpy array for compatibility (same as qPCR_GPLVM.py)
    labels_test = labels_test_np    # Keep as numpy array for compatibility (same as qPCR_GPLVM.py)
    pseudo_train = torch.tensor(pseudo_train_np, dtype=torch.float64, device=config.device)
    pseudo_test = torch.tensor(pseudo_test_np, dtype=torch.float64, device=config.device)
    
    # Convert indices to tensors for consistency with oil flow version
    train_indices_tensor = torch.tensor(train_indices, dtype=torch.int64, device=config.device)
    test_indices_tensor = torch.tensor(test_indices, dtype=torch.int64, device=config.device)
    
    print(f"Data split shapes - Train: {y_train.shape}, Test: {y_test.shape}")
    print(f"Labels split info - Train: {type(labels_train)}, length: {len(labels_train) if hasattr(labels_train, '__len__') else 'no length'}")
    print(f"Labels split info - Test: {type(labels_test)}, length: {len(labels_test) if hasattr(labels_test, '__len__') else 'no length'}")

    # === Data Standardization ===
    # Fit scaler on training data only (before missing data preprocessing)
    print(f"\n=== Data Standardization ===")
    print(f"Standardization method: {config.standardization}")

    if config.standardization == "standard":
        scaler = StandardScalerPytorch(device=config.device)
    elif config.standardization == "minmax":
        scaler = MinMaxScalerPytorch(device=config.device)
    elif config.standardization == "mean":
        scaler = MeanCenter(device=config.device)
    elif config.standardization == "none":
        scaler = NoTransform(device=config.device)
    else:
        raise ValueError(
            f"Unknown standardization method: {config.standardization}. Must be 'standard', 'minmax', 'mean', or 'none'"
        )

    # Fit scaler on training data only (critical: don't mix test data)
    print(f"Fitting scaler on training data only ({y_train.shape[0]} samples)...")
    scaler.fit(y_train)

    # Transform all data using the scaler fitted on training data
    print(f"Transforming train/test data...")
    print(
        f"  Before transform - Train shape: {y_train.shape}, Test shape: {y_test.shape}"
    )
    print(f"  Scaler mean shape: {scaler.mean.shape}")
    print(
        f"  Before transform - Train mean: {y_train.mean().item():.6f}, Test mean: {y_test.mean().item():.6f}"
    )
    print(
        f"  Before transform - Train std: {y_train.std().item():.6f}, Test std: {y_test.std().item():.6f}"
    )

    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    # Debug: Show statistics after transform
    print(
        f"  After transform - Train mean: {y_train.mean().item():.6f}, Test mean: {y_test.mean().item():.6f}"
    )
    print(
        f"  After transform - Train std: {y_train.std().item():.6f}, Test std: {y_test.std().item():.6f}"
    )
    print(
        f"  After transform - Train min: {y_train.min().item():.6f}, Test min: {y_test.min().item():.6f}"
    )
    print(
        f"  After transform - Train max: {y_train.max().item():.6f}, Test max: {y_test.max().item():.6f}"
    )

    print(f"Standardization completed.")
    print("=" * 50)

    # Use training data for model initialization
    y = y_train
    labels = labels_train
    split_data = {
        'y_train': y_train,
        'y_test': y_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'train_indices': train_indices_tensor,
        'test_indices': test_indices_tensor,
        'pseudo_train': pseudo_train,
        'pseudo_test': pseudo_test,
        # Scaler for inverse transforms (needed for visualization)
        'scaler': scaler,
    }
    # Initialize inducing points and latent variables (following original)
    Vx = torch.randn(config.n_inducing_points, config.latent_dim, dtype=torch.float64, device=config.device)
    mu_x, _, _ = torch.pca_lowrank(y, config.latent_dim)
    
    # Assign pseudotime to first latent dimension for qPCR data (consistent with original qPCR_expt.py)
    mu_x[:,0] = pseudo_train

    print(f"Data loaded: {y.shape[0]} samples, {y.shape[1]} dimensions")
    print(f"Latent initialization: {mu_x.shape}")
    print(f"Inducing points: {Vx.shape}")
    print(f"Pseudotime assigned to first latent dimension: {mu_x[:,0].min():.3f} to {mu_x[:,0].max():.3f}")
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

def train_oil_flow_ssvi(model: OilFlow_SSVI_GPLVM, y_train, config: OilFlowSSVIConfig,
                        logger: SSVIResultsLogger = None, split_data=None, labels_train=None):
    """Training function for qPCR SSVI GP-LVM with optional logging and metrics"""
    losses = []  # Batch ELBO losses (for training monitoring)
    full_elbo_losses = []  # Full dataset ELBO (for metrics and plots)
    full_elbo_iterations = []  # Iterations at which full ELBO was computed
    
    print(f"Training qPCR SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")
    
    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(f"Metrics computation enabled (frequency: every {config.metrics_frequency} iterations)")
        metrics_computer = SSVIMetricsComputer(device=config.device, metrics_batch_size=config.batch_size)
    
    start_time = time.time()
    init_prior_weight = 1
    for i in range(config.training_iterations):
        # Sample batch indices
        N = y_train.shape[0]
        batch_indices = torch.randperm(N, device=y_train.device)[:config.batch_size]
        y_batch = y_train[batch_indices]

        # Training step - U sampling happens inside automatically each step
        # prior_weight = init_prior_weight 
        # prior_weight = init_prior_weight + (i/config.training_iterations)*(1-init_prior_weight)
        prior_weight = (init_prior_weight-1)/2 + (init_prior_weight-1)/2*torch.cos(torch.tensor(torch.pi*i/config.training_iterations)) + 1
        loss = model.train_step(y_batch, batch_indices, prior_weight=prior_weight)
        losses.append(loss)
        
        # Metrics computation (periodic)
        if (metrics_computer is not None and 
            config.enable_metrics and 
            i % config.metrics_frequency == 0):
            
            print(f"  Computing metrics at iteration {i}...")

            # Compute full dataset loss (negative ELBO, for metrics and plots)
            if config.enable_missing_data:
                full_elbo = model.compute_full_elbo(
                    split_data['y_train'], missing_mask=split_data['train_missing_mask']
                )
            else:
                full_elbo = model.compute_full_elbo(split_data['y_train'])

            # Store as negative ELBO (loss) for consistency with batch losses
            full_elbo_losses.append(-full_elbo)
            full_elbo_iterations.append(i)
            print(f"    Full Dataset Loss: {-full_elbo:.4f} (batch loss: {loss:.4f})")

            # Use full datasets for proper evaluation
            # Compute metrics with missing data info if enabled
            if config.enable_missing_data:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data['y_train'],
                    y_test=split_data['y_test'],
                    iteration=i,
                    elbo_loss=full_elbo,  # Use full dataset ELBO
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
                    elbo_loss=full_elbo,  # Use full dataset ELBO
                    n_samples=32  # Fixed number of samples for metrics computation
                )
            
            print(f"    Train RMSE: {metrics.train_rmse:.4f}, Test RMSE: {metrics.test_rmse:.4f} ")
            print(f"    Train NLPD: {metrics.train_nlpd:.3f}, Test NLPD: {metrics.test_nlpd:.3f}")
            print(f"    Train CRPS: {metrics.train_crps:.4f}, Test CRPS (crps): {metrics.crps:.4f}")
            print(f"    crps_pp_sample: {metrics.crps_posterior_predictive_sample:.4f}, crps_pp_mixture: {metrics.crps_posterior_predictive_mixture:.4f}")
            print(f"    coverage_90: {metrics.coverage_90:.4f}, coverage_95: {metrics.coverage_95:.4f}")
            print(f"    interval_width_90: {metrics.interval_width_90:.4f}, interval_width_95: {metrics.interval_width_95:.4f}")
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
                if config.save_models:
                    # Save checkpoint model
                    checkpoint_filename = f"qpcr_ssvi_iter_{i}_seed_{config.random_seed}.pkl"
                    logger.save_model(model, checkpoint_filename, is_checkpoint=True)
                    print(f"    Checkpoint model saved: {checkpoint_filename}")
                
                # Save checkpoint plot (if enabled)
                if config.save_training_plots:
                    plot_filename = f"training_results_iter_{i}.png"
                    try:
                        from src.qpcr_plot_helper import visualize_ssvi_results as visualize_qpcr_ssvi_results
                        visualize_qpcr_ssvi_results(
                            model=model,
                            y_train=y_train,
                            labels_train=labels_train,
                            losses=full_elbo_losses,  # Use full dataset ELBO for plots
                            config=config,
                            logger=logger,
                            save_filename=plot_filename,
                            show_plot=False,
                            is_checkpoint=True,
                            train_indices=split_data['train_indices'],
                            full_elbo_iterations=full_elbo_iterations,
                            batch_losses=losses  # Batch losses up to current iteration
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

    # Compute final full dataset loss (negative ELBO)
    if config.enable_missing_data and split_data is not None:
        final_full_elbo = model.compute_full_elbo(
            split_data['y_train'], missing_mask=split_data['train_missing_mask']
        )
    elif split_data is not None:
        final_full_elbo = model.compute_full_elbo(split_data['y_train'])
    else:
        if config.enable_missing_data:
            missing_mask = torch.isnan(y_train)
        else:
            missing_mask = None
        final_full_elbo = model.compute_full_elbo(y_train, missing_mask=missing_mask)

    # Store as negative ELBO (loss) for consistency with batch losses
    full_elbo_losses.append(-final_full_elbo)
    full_elbo_iterations.append(config.training_iterations)

    final_message = f"Training completed! Final Full Loss: {-final_full_elbo:.4f} (batch loss: {losses[-1]:.4f}), Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    print(f"Training completed!")
    print(f"Final Full Dataset Loss: {-final_full_elbo:.4f}")
    print(f"Final Batch Loss: {losses[-1]:.4f}")
    print(f"Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)")
    # Log to file
    if logger:
        logger.log_training(final_message)
    
    # Return full ELBO losses for plots (with iterations for x-axis)
    return {
        "batch_losses": losses,
        "full_elbo_losses": full_elbo_losses,
        "full_elbo_iterations": full_elbo_iterations,
    }, metrics_computer

# === Pre-defined Configurations ===

def get_fast_config():
    """Fast configuration for testing with missing data enabled"""
    return OilFlowSSVIConfig(
        training_iterations=200,
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
        enable_missing_data=False,     # Disabled for fast config
        missing_data_ratio=0.3,        # Fraction of dimensions to artificially mask (for testing)
        reconstruction_samples=50,     # Number of samples for missing data reconstruction
        # === Random Seed Configuration ===
        random_seed=7,                 # Random seed for reproducibility (matches SSVI base config)
        # === Test Size Configuration ===
        test_size=0.2,                 # Fast config uses 0.2 (20% for testing)
        # === Data Standardization Configuration ===
        standardization="mean"         # "standard" (mean=0, std=1), "minmax" ([0,1]), "mean" (mean=0), or "none" (no transform)
    )

def main(config: OilFlowSSVIConfig = None, config_name: str = "fast"):
    """Main function for qPCR SSVI GP-LVM example"""
    
    # Use default config if none provided
    if config is None:
        config = get_fast_config()
        # Override results directory for qPCR experiments when using fast config
        if config_name == "fast":
            config.results_dir = "qpcr_results/ssvi_results"

    config.save_models = False

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
    
    # Load qPCR data
    y_train, labels_train, Vx, mu_x, split_data = load_oil_flow_data(config)
    
    # Create model
    print("\nInitializing qPCR SSVI GP-LVM...")
    model = OilFlow_SSVI_GPLVM(
        config=config,
        Vx=Vx,
        mu_x=mu_x,
        M=y_train.shape[1]  # Observation dimension
    )
    
    print(f"Model initialized:")
    print(f"  Mu Encoder: {model.mu_encoder}")
    print(f"  Sigma Encoder: {model.sigma_encoder}")
    print(f"  Inducing points: {model.gp.Vx.shape}")
    print(f"  Latent variables: {model.mu_x.shape}")
    
    # Initialize logger with config-based naming
    experiment_name = f"ssvi_{config_name}"
    logger = SSVIResultsLogger(config, experiment_name=experiment_name)
    if logger.results_dir:
        print(f"Results will be saved to: {logger.results_dir}")
        # Log initial info
        logger.log_training("=== qPCR SSVI GP-LVM Training Session ===")
        logger.log_training(f"Model: {config.n_inducing_points} inducing points, latent_dim={config.latent_dim}")
        logger.log_training(f"Training: {config.training_iterations} iter, batch={config.batch_size}")
        logger.log_training(f"Learning rates: Local={config.local_lr}, Global={config.global_lr}")
    
    # Train model
    print(f"\nTraining model...")
    training_result, metrics_computer = train_oil_flow_ssvi(model, y_train, config, logger, split_data, labels_train)
    
    # Extract losses from training result
    batch_losses = training_result["batch_losses"]
    full_elbo_losses = training_result["full_elbo_losses"]
    full_elbo_iterations = training_result["full_elbo_iterations"]
    
    # Visualize results (use full ELBO losses for plots)
    print(f"\nVisualizing results...")
    try:
        from src.qpcr_plot_helper import visualize_ssvi_results as visualize_qpcr_ssvi_results
        visualize_qpcr_ssvi_results(
            model, 
            y_train, 
            labels_train, 
            full_elbo_losses, 
            config, 
            logger, 
            show_plot=False, 
            train_indices=split_data['train_indices'],
            full_elbo_iterations=full_elbo_iterations,
            batch_losses=batch_losses
        )
    except Exception as e:
        print(f"Warning: Could not create latent space visualization: {e}")
    
    # Generate final metrics and plots if metrics were computed
    if metrics_computer is not None and config.enable_metrics and split_data is not None:
        print(f"\nGenerating final metrics and plots...")
        
        # Compute final comprehensive metrics on full test set
        print("Computing final metrics on full datasets...")
        
        # Compute final metrics with missing data info if enabled
        # Use final full ELBO for metrics (not batch ELBO)
        # Note: full_elbo_losses stores -elbo (loss), so negate to get ELBO
        final_full_elbo = -full_elbo_losses[-1] if full_elbo_losses else 0.0
        
        if config.enable_missing_data:
            final_metrics = metrics_computer.compute_full_metrics(
                model=model,
                y_train=split_data['y_train'],
                y_test=split_data['y_test'],
                iteration=config.training_iterations,
                elbo_loss=final_full_elbo,
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
                elbo_loss=final_full_elbo,
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
        print(f"Final Train CRPS: {summary['final_train_crps']:.4f}")
        print(f"Final Test CRPS (crps): {summary['crps']:.4f}")
        print(f"Final crps_pp_sample: {summary['crps_posterior_predictive_sample']:.4f}")
        print(f"Final crps_pp_mixture: {summary['crps_posterior_predictive_mixture']:.4f}")
        print(f"Final coverage_90: {summary['coverage_90']:.4f}")
        print(f"Final coverage_95: {summary['coverage_95']:.4f}")
        print(f"Final interval_width_90: {summary['interval_width_90']:.4f}")
        print(f"Final interval_width_95: {summary['interval_width_95']:.4f}")
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
        plot_metrics_history(metrics_computer, logger, show_plot=False, batch_losses=batch_losses)

        print("Generating CRPS & coverage metrics progression plot...")
        plot_crps_metrics_progression(metrics_computer, logger, show_plot=False)

        print("Generating posterior-predictive CRPS progression plot...")
        plot_crps_posterior_predictive_progression(metrics_computer, logger, show_plot=False)
        
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
            logger.save_checkpoint_metrics(metrics_computer, "qpcr_ssvi", config.random_seed)
            print("Saving training losses (full dataset loss)...")
            # Save full dataset losses (negative ELBO, for metrics/plots)
            logger.save_training_losses(full_elbo_losses, "qpcr_ssvi_full_loss", config.random_seed)
            # Also save batch losses for reference
            logger.save_training_losses(batch_losses, "qpcr_ssvi_batch", config.random_seed)

    
    # Generate latent slice visualization (after all metrics and plots)
    try:
        print(f"\nGenerating latent slice visualizations (adaptive grid from data)...")
        
        # Plot slice centered at origin
        latent_slice_path_origin = os.path.join(logger.plots_dir, f"latent_slice_origin_seed_{config.random_seed}.png")
        plot_latent_slice_outer(
            model=model,
            y_train=y_train,
            center="zeros",
            save_path=latent_slice_path_origin,
            scaler=split_data.get('scaler')
        )
        print(f"  Saved latent slice (origin) to: {latent_slice_path_origin}")
        
        # Plot slices centered at a few interesting data points
        N = y_train.shape[0]  # Number of training points from actual data
        # Select 3 representative points (beginning, middle, end)
        representative_indices = [0, N // 2, N - 1]
        
        for idx in representative_indices:
            latent_slice_path = os.path.join(logger.plots_dir, f"latent_slice_point_{idx}_seed_{config.random_seed}.png")
            plot_latent_slice_outer(
                model=model,
                y_train=y_train,
                center="data_point",
                n_idx=idx,
                save_path=latent_slice_path,
                scaler=split_data.get('scaler')
            )
            print(f"  Saved latent slice (point {idx}) to: {latent_slice_path}")
    except Exception as e:
        print(f"  Warning: Failed to generate latent slice visualizations: {e}")
    
    # Print final model parameters
    print_final_model_parameters(model, "GP LVM SSVI")
    
    # Save final model (like original GP-LVM)
    if logger and config.save_models:
        print("Saving final model...")
        logger.save_model(model, f"qpcr_ssvi_{config.random_seed}.pkl", is_checkpoint=False)
    
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
            
            experiment_summary['qpcr_ssvi'] = {
                'train_rmse': summary.get('final_train_rmse', 0.0),
                'test_rmse': summary.get('final_test_rmse', 0.0),
                'train_nlpd': summary.get('final_train_nlpd', 0.0),
                'test_nlpd': summary.get('final_test_nlpd', 0.0),
                'train_crps': summary.get('final_train_crps', 0.0),
                'test_crps': summary.get('final_test_crps', 0.0),
                'crps': summary.get('crps', 0.0),  # held-out CRPS (same as test_crps)
                'train_crps_posterior_predictive_sample': summary.get('final_train_crps_posterior_predictive_sample', float('nan')),
                'test_crps_posterior_predictive_sample': summary.get('final_test_crps_posterior_predictive_sample', float('nan')),
                'crps_posterior_predictive_sample': summary.get('crps_posterior_predictive_sample', float('nan')),
                'train_crps_posterior_predictive_mixture': summary.get('final_train_crps_posterior_predictive_mixture', float('nan')),
                'test_crps_posterior_predictive_mixture': summary.get('final_test_crps_posterior_predictive_mixture', float('nan')),
                'crps_posterior_predictive_mixture': summary.get('crps_posterior_predictive_mixture', float('nan')),
                'train_coverage_90': summary.get('final_train_coverage_90', float('nan')),
                'train_coverage_95': summary.get('final_train_coverage_95', float('nan')),
                'test_coverage_90': summary.get('final_test_coverage_90', float('nan')),
                'test_coverage_95': summary.get('final_test_coverage_95', float('nan')),
                'coverage_90': summary.get('coverage_90', float('nan')),
                'coverage_95': summary.get('coverage_95', float('nan')),
                'train_interval_width_90': summary.get('final_train_interval_width_90', float('nan')),
                'train_interval_width_95': summary.get('final_train_interval_width_95', float('nan')),
                'test_interval_width_90': summary.get('final_test_interval_width_90', float('nan')),
                'test_interval_width_95': summary.get('final_test_interval_width_95', float('nan')),
                'interval_width_90': summary.get('interval_width_90', float('nan')),
                'interval_width_95': summary.get('interval_width_95', float('nan')),
                'train_missing_data_nlpd': summary.get('final_train_missing_nlpd', float('nan')),
                'test_missing_data_nlpd': summary.get('final_test_missing_nlpd', float('nan')),
                'train_missing_data_rmse': summary.get('final_train_missing_rmse', float('nan')),
                'test_missing_data_rmse': summary.get('final_test_missing_rmse', float('nan')),
                'final_full_loss': full_elbo_losses[-1] if full_elbo_losses else 0.0,
                'final_batch_loss': batch_losses[-1] if batch_losses else 0.0,
                'final_noise': float(final_noise)  # Ensure it's a Python float
            }
        
        logger.save_experiment_summary(experiment_summary)
    
    return model, training_result, config, metrics_computer, final_metrics

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
    
    # Override results directory for qPCR experiments when using fast config
    if config_name == "fast":
        config.results_dir = "qpcr_results/ssvi_results"
    
    print(f"Using configuration: {config_name}")
    if args.config:
        print(f"Config file: {args.config}")
    
    # Run main function with loaded config
    main(config, config_name)
    
