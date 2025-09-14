#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to align SSVI GP-LVM outputs with original GP-LVM structure
"""

import sys
import os
import torch
import numpy as np
import pickle as pkl
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add generalised_gplvm_code to path for original utilities
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

from utils.data import load_real_data
from utils.metrics import rmse
from utils.visualisation import plot_report
from src.data_utils import download_oilflow_data_if_needed


def load_oilflow_data_for_ssvi(config):
    """
    Load oilflow data using the same method as original GP LVM
    Returns data in the same format as original GP LVM
    """
    # Download data if needed (same as original)
    download_oilflow_data_if_needed()
    
    # Load data using original method
    N, d, q, X, Y, labels = load_real_data(config.dataset)
    
    # Create train/test split (same as original)
    Y_train, Y_test = train_test_split(Y.numpy(), test_size=config.test_size_int, random_state=config.random_seed)
    lb_train, lb_test = train_test_split(labels, test_size=config.test_size_int, random_state=config.random_seed)
    
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    
    # Return in same format as original GP LVM
    return {
        'Y_train': Y_train,
        'Y_test': Y_test,
        'labels_train': lb_train,
        'labels_test': lb_test,
        'N': len(Y_train),
        'data_dim': Y_train.shape[1],
        'latent_dim': config.latent_dim,
        'n_inducing': config.n_inducing
    }


def compute_original_gplvm_metrics(model, Y_train, Y_test, config):
    """
    Compute metrics in the same way as original GP LVM using SSVI model
    Returns train and test RMSE
    """
    from src.ssvi_metrics import SSVIMetricsComputer
    
    # Create metrics computer
    metrics_computer = SSVIMetricsComputer()
    
    # Compute RMSE using SSVI method
    train_rmse = metrics_computer.compute_rmse(model, Y_train, n_samples=10)
    test_rmse = metrics_computer.compute_rmse(model, Y_test, n_samples=10)
    
    return train_rmse, test_rmse


def save_ssvi_model_like_original(model, model_name, random_seed, results_dir):
    """
    Save SSVI model in the same format as original GP LVM
    """
    # Save to models folder (exactly like original GP-LVM logger.save_model)
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    filename = f'{model_name}_{random_seed}.pkl'
    models_filepath = os.path.join(models_dir, filename)
    
    # Save model state dict (same as original)
    with open(models_filepath, 'wb') as file:
        pkl.dump(model.state_dict(), file)
    
    # Also save to pre_trained_models folder (exactly like original GP-LVM direct save)
    pre_trained_dir = os.path.join(results_dir, 'pre_trained_models')
    os.makedirs(pre_trained_dir, exist_ok=True)
    
    pre_trained_filename = f'oilflow_{model_name}_{random_seed}.pkl'
    pre_trained_filepath = os.path.join(pre_trained_dir, pre_trained_filename)
    
    with open(pre_trained_filepath, 'wb') as file:
        pkl.dump(model.state_dict(), file)
    
    return models_filepath


def create_standardized_scattered_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, model_type="original", custom_filename=None):
    """
    Create standardized scattered plots that match the original generalised_gplvm_code plot_report function exactly
    but use scattered points instead of error bars. Works for both original GP-LVM and SSVI models.
    
    This function replicates the exact structure and styling of the original plot_report function from
    generalised_gplvm_code/utils/visualisation.py but removes error bars for scattered visualization.
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    import pathlib
    
    # Use exact same styling as original plot_report function
    plt.style.use('ggplot')
    plt.figure(figsize=(12,4))
    
    # Get model parameters exactly like original GP-LVM
    if model_type == "ssvi":
        # For SSVI models - convert to same format as original GP-LVM
        inv_lengthscale = model.gp.beta.detach().cpu().numpy().flatten()
        lengthscales = 1.0 / model.gp.beta
        latent_dim = config.latent_dim
    else:
        # For original GP-LVM models - use exact same approach as original
        inv_lengthscale = 1/model.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
        lengthscales = model.covar_module.base_kernel.lengthscale
        latent_dim = model.X.latent_dim
    
    # Print lengthscales exactly like original plot_report
    print(inv_lengthscale)
    
    # Get top 2 smallest lengthscales (exactly like original GP-LVM)
    values, indices = torch.topk(lengthscales, k=2, largest=False)
    l1 = indices.numpy().flatten()[0]
    l2 = indices.numpy().flatten()[1]
    
    # Subplot 1: 2D latent subspace (exactly like original plot_report)
    plt.subplot(131)
    plt.title(f'2d latent subspace [{model_name}]', fontsize='small')
    plt.xlabel('Latent dim 1', fontsize='small')
    plt.ylabel('Latent dim 2', fontsize='small')
    
    # Get latent representations - exactly like original GP-LVM
    if model_type == "ssvi":
        # For SSVI models - use full dataset and sample to get scattered representation
        with torch.no_grad():
            U_samples = model.get_u_samples()
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                y_encoder_input = torch.where(torch.isnan(Y_train), torch.zeros_like(Y_train), Y_train)
            else:
                y_encoder_input = Y_train
            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U_samples)
            # Take mean across U samples to get scattered representation
            X_mean = mu_x.mean(dim=0).detach().cpu().numpy()
    else:
        # For original GP-LVM models - use exact same approach as original plot_report
        with torch.no_grad():
            model_name_class = model.X.__class__.__name__
            if model_name_class == 'VariationalLatentVariable':  # 'gauss' model
                # Use variational parameters directly (like original)
                X_mean = model.X.q_mu.detach().numpy()
            elif model_name_class == 'NNEncoder':  # 'nn_gauss' model
                # Use encoder methods (like original)
                model.X.jitter = 1e-5
                X_mean = model.X.mu(torch.tensor(Y_train, dtype=torch.float32)).detach().numpy()
            else:
                # Fallback: use direct latent values (like original)
                X_mean = model.X.X.detach().numpy()
    
    # Plot points for each label (exactly like original plot_report but scattered only)
    # Use exact same colors as original oilflow experiment: ['r', 'b', 'g']
    colors = ['r', 'b', 'g']
    # Handle labels - convert to numpy if it's a tensor
    if hasattr(labels_train, 'cpu'):
        labels_train_np = labels_train.cpu().numpy()
    else:
        labels_train_np = labels_train
    
    # Use exact same loop structure as original plot_report
    for i, label in enumerate(np.unique(labels_train_np)):
        X_i = X_mean[labels_train_np == label]
        plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
        # Note: No error bars (X_scales=None) as requested - scattered only
    
    # Subplot 2: Inverse lengthscales (exactly like original plot_report)
    plt.subplot(132)
    plt.bar(np.arange(latent_dim), height=inv_lengthscale, color='teal')
    plt.title('Inverse Lengthscale with SE-ARD kernel', fontsize='small')
    plt.xlabel('Latent dims', fontsize='small')
    
    # Subplot 3: Loss curve (exactly like original plot_report)
    plt.subplot(133)
    plt.plot(losses, label='batch_size=100', color='orange')
    plt.xlabel('Steps', fontsize='small')
    plt.title('Neg. ELBO Loss', fontsize='small')
    
    plt.tight_layout()
    
    # Use model name (gauss/nn_gauss) instead of class name for consistency
    if model_name == 'VariationalLatentVariable':
        model_short_name = 'gauss'
    elif model_name == 'NNEncoder':
        model_short_name = 'nn_gauss'
    elif model_name == 'ssvi':
        model_short_name = 'ssvi'
    else:
        model_short_name = model_name.lower()
    
    # Use custom filename if provided, otherwise use default naming convention
    if custom_filename:
        plot_filename = custom_filename
    else:
        plot_filename = f"oilflow_{model_short_name}_{random_seed}.png"
    
    # Create plots subdirectory only if we're not already in a plots directory
    if os.path.basename(results_dir) == 'plots':
        # We're already in a plots directory, save directly here
        plot_path = os.path.join(results_dir, plot_filename)
    else:
        # Create plots subdirectory for better organization
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        plot_path = os.path.join(plots_subdir, plot_filename)
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Standardized scattered plots saved to: {plot_path}")

# Removed old plotting functions - now using create_standardized_scattered_plot for consistency


def print_original_gplvm_summary(model_name, mse_train, mse_test, final_loss, final_noise):
    """
    Print summary in the exact same format as original GP LVM
    """
    print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
    print(f'Test Reconstruction error {model_name} = ' + str(mse_test))


def create_ssvi_results_summary(model_name, random_seed, losses, mse_train, mse_test, final_loss, final_noise):
    """
    Create results summary in the same format as original GP LVM
    """
    return {
        'model_name': model_name,
        'random_seed': random_seed,
        'train_rmse': float(mse_train.item()) if hasattr(mse_train, 'item') else float(mse_train),
        'test_rmse': float(mse_test.item()) if hasattr(mse_test, 'item') else float(mse_test),
        'final_loss': float(final_loss),
        'final_noise': float(final_noise),
        'loss_history': [float(loss) for loss in losses]
    }


class SSVIOriginalGPLVMResultsLogger:
    """
    Results logger that matches the original GP LVM output structure
    """
    def __init__(self, config, experiment_name="ssvi"):
        self.config = config
        self.experiment_name = experiment_name
        self.results_dir = None
        
        if config.save_results:
            timestamp = datetime.now().strftime("%m_%d_%H_%M")
            self.results_dir = os.path.join(
                config.results_dir,
                f"oil_flow_ssvi_{experiment_name}_res_{timestamp}"
            )
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Create plots subdirectory (like SSVIResultsLogger)
            self.plots_dir = os.path.join(self.results_dir, "plots")
            os.makedirs(self.plots_dir, exist_ok=True)
            
            # Initialize log files (like SSVIResultsLogger)
            self.training_log_path = os.path.join(self.results_dir, "training.log")
            self.gradient_log_path = os.path.join(self.results_dir, "gradient_log.jsonl")
            
            # Save config
            self.save_config()
            
            # Initialize training log file (same as original GP LVM)
            self.log_training(f"=== SSVI GP LVM Experiment Started ===")
            self.log_training(f"Experiment: {experiment_name}")
            self.log_training(f"Timestamp: {timestamp}")
    
    def save_config(self):
        """Save configuration to results directory"""
        if self.results_dir:
            config_dict = {
                'experiment_name': self.experiment_name,
                'dataset': self.config.dataset,
                'latent_dim': self.config.latent_dim,
                'n_inducing': self.config.n_inducing,
                'training_iterations': self.config.training_iterations,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'random_seed': self.config.random_seed,
                'test_size': self.config.test_size_int,
                'device': str(self.config.device)
            }
            
            config_path = os.path.join(self.results_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    def save_model(self, model, model_name, random_seed):
        """Save model in same format as original GP LVM"""
        if self.results_dir:
            # Save using original format only (exactly like original GP-LVM)
            save_ssvi_model_like_original(model, model_name, random_seed, self.results_dir)
    
    def save_training_results(self, model_name, random_seed, losses, noise_trace, mse_train, mse_test):
        """Save training results in same format as original GP LVM"""
        if self.results_dir:
            # Create training_results folder (exactly like original GP-LVM)
            training_results_dir = os.path.join(self.results_dir, "training_results")
            os.makedirs(training_results_dir, exist_ok=True)
            
            # Save loss history (exactly like original GP-LVM)
            loss_file = os.path.join(training_results_dir, f"{model_name}_{random_seed}_losses.json")
            with open(loss_file, 'w') as f:
                json.dump({
                    'losses': [float(loss) for loss in losses],
                    'noise_trace': [float(noise) for noise in noise_trace] if noise_trace else [],
                    'train_rmse': float(mse_train),
                    'test_rmse': float(mse_test),
                    'model_name': model_name,
                    'seed': random_seed
                }, f, indent=2)
    
    def log_training(self, message):
        """Log training message in same format as original GP LVM"""
        if self.results_dir:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.training_log_path, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
    
    def log_gradients(self, iteration: int, loss: float, gradient_info: dict):
        """Log gradient information to file (like SSVIResultsLogger)"""
        if (self.results_dir and 
            hasattr(self.config, 'enable_gradient_logging') and 
            self.config.enable_gradient_logging):
            log_entry = {
                'iteration': iteration,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                **gradient_info
            }
            with open(self.gradient_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def save_plots(self, figure, filename: str):
        """Save matplotlib figure to plots subdirectory (like SSVIResultsLogger)"""
        if self.results_dir:
            plot_path = os.path.join(self.plots_dir, filename)
            figure.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.log_training(f"Plot saved: plots/{filename}")
    
    def save_final_summary(self, results_summary):
        """Save final experiment summary in same format as original GP LVM"""
        if self.results_dir:
            summary_file = os.path.join(self.results_dir, "experiment_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            self.log_training(f"Experiment summary saved: {summary_file}")
    
    def get_results_dir(self):
        """Get results directory path"""
        return self.results_dir


def save_iteration_model(model, model_name, iteration, random_seed, results_dir):
    """
    Save model at specific iteration (like SSVI does)
    """
    # Save to models folder with iteration number
    models_dir = os.path.join(results_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    filename = f'{model_name}_iter_{iteration}_seed_{random_seed}.pkl'
    filepath = os.path.join(models_dir, filename)
    
    with open(filepath, 'wb') as file:
        pkl.dump(model.state_dict(), file)
    
    return filepath


def create_iteration_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, iteration, model_type="original"):
    """
    Create iteration-based plot (like SSVI does)
    """
    plot_filename = f"oilflow_{model_name}_iter_{iteration}_seed_{random_seed}.png"
    
    # Create the standardized plot with custom filename
    create_standardized_scattered_plot(
        model, losses, labels_train, config, results_dir, 
        model_name, random_seed, Y_train, model_type=model_type, 
        custom_filename=plot_filename
    )
    
    return plot_filename


def integrate_original_gplvm_outputs(model, config, config_name, losses, Y_train, Y_test, labels_train, labels_test, logger=None):
    """
    Integrate original GP LVM outputs into SSVI training
    This function should be called after SSVI training is complete
    """
    # Use existing logger if provided, otherwise create new one
    if logger is None:
        logger = SSVIOriginalGPLVMResultsLogger(config, config_name)
        if logger.get_results_dir():
            print(f"Results will be saved to: {logger.get_results_dir()}")
    else:
        print(f"Using existing logger, results will be saved to: {logger.get_results_dir()}")
    
    # Compute metrics using original GP LVM method
    print("Computing metrics using original GP LVM method...")
    mse_train, mse_test = compute_original_gplvm_metrics(model, Y_train, Y_test, config)
    
    # Print summary in original format (exactly like original GP LVM)
    print_original_gplvm_summary("ssvi", mse_train, mse_test, losses[-1] if losses else 0.0, 0.0)
    
    # Save model (same as original)
    logger.save_model(model, "ssvi", config.random_seed)
    
    # Create plots (same as original)
    create_standardized_scattered_plot(model, losses, labels_train, config, logger.get_results_dir(), "ssvi", config.random_seed, Y_train, model_type="ssvi")
    
    # Save training results
    logger.save_training_results("ssvi", config.random_seed, losses, [], mse_train, mse_test)
    
    # Create final summary
    results_summary = create_ssvi_results_summary("ssvi", config.random_seed, losses, mse_train, mse_test, losses[-1] if losses else 0.0, 0.0)
    
    return logger, results_summary
