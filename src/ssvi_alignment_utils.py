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
    # Ensure tensors are on CPU before converting to numpy
    Y_cpu = Y.cpu() if Y.is_cuda else Y
    labels_cpu = labels.cpu() if hasattr(labels, 'is_cuda') and labels.is_cuda else labels
    
    Y_train, Y_test = train_test_split(Y_cpu.numpy(), test_size=config.test_size_int, random_state=config.random_seed)
    lb_train, lb_test = train_test_split(labels_cpu, test_size=config.test_size_int, random_state=config.random_seed)
    
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=config.device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32, device=config.device)
    
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
    Save SSVI model in the same format as original GP LVM (only pre_trained_models)
    """
    # Save to pre_trained_models folder (exactly like original generalised_gplvm_code)
    pre_trained_dir = os.path.join(results_dir, 'pre_trained_models')
    os.makedirs(pre_trained_dir, exist_ok=True)
    
    filename = f'oilflow_{model_name}_{random_seed}.pkl'
    filepath = os.path.join(pre_trained_dir, filename)
    
    # Save model state dict (same as original)
    with open(filepath, 'wb') as file:
        pkl.dump(model.state_dict(), file)
    
    return filepath


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
    
    print(f"[DEBUG] Starting create_standardized_scattered_plot for model_type={model_type}, model_name={model_name}")
    print(f"[DEBUG] Y_train device in plotting function: {Y_train.device}")
    print(f"[DEBUG] labels_train type in plotting function: {type(labels_train)}")
    
    # Use exact same styling as original plot_report function
    plt.style.use('ggplot')
    plt.figure(figsize=(12,4))
    
    # Get model parameters exactly like original GP-LVM
    if model_type == "ssvi":
        print("[DEBUG] Processing SSVI model parameters...")
        # For SSVI models - convert to same format as original GP-LVM
        inv_lengthscale = model.gp.beta.detach().cpu().numpy().flatten()
        lengthscales = (1.0 / model.gp.beta).detach().cpu()
        latent_dim = config.latent_dim
        print(f"[DEBUG] SSVI inv_lengthscale shape: {inv_lengthscale.shape}, device: CPU")
    else:
        print("[DEBUG] Processing original GP-LVM model parameters...")
        # For original GP-LVM models - use exact same approach as original
        # Ensure lengthscale is moved to CPU before numpy conversion
        print(f"[DEBUG] lengthscale device before detach: {model.covar_module.base_kernel.lengthscale.device}")
        lengthscales_cpu = model.covar_module.base_kernel.lengthscale.detach().cpu()
        print(f"[DEBUG] lengthscales_cpu device after detach.cpu(): {lengthscales_cpu.device}")
        inv_lengthscale = (1/lengthscales_cpu).numpy().flatten()
        lengthscales = lengthscales_cpu
        latent_dim = model.X.latent_dim
        print(f"[DEBUG] Original GP-LVM inv_lengthscale shape: {inv_lengthscale.shape}, device: CPU")
    
    # Print lengthscales exactly like original plot_report
    print(inv_lengthscale)
    
    # Get top 2 smallest lengthscales (exactly like original GP-LVM)
    values, indices = torch.topk(lengthscales, k=2, largest=False)
    l1 = indices.cpu().numpy().flatten()[0]
    l2 = indices.cpu().numpy().flatten()[1]
    
    # Subplot 1: 2D latent subspace (exactly like original plot_report)
    plt.subplot(131)
    plt.title(f'2d latent subspace [{model_name}]', fontsize='small')
    plt.xlabel('Latent dim 1', fontsize='small')
    plt.ylabel('Latent dim 2', fontsize='small')
    
    # Get latent representations - exactly like original GP-LVM
    print("[DEBUG] Getting latent representations...")
    if model_type == "ssvi":
        print("[DEBUG] Processing SSVI latent representations...")
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
            print(f"[DEBUG] SSVI X_mean shape: {X_mean.shape}, device: CPU")
    else:
        print("[DEBUG] Processing original GP-LVM latent representations...")
        # For original GP-LVM models - use exact same approach as original plot_report
        with torch.no_grad():
            model_name_class = model.X.__class__.__name__
            print(f"[DEBUG] Model class: {model_name_class}")
            if model_name_class == 'VariationalLatentVariable':  # 'gauss' model
                print("[DEBUG] Using VariationalLatentVariable (gauss model)...")
                # Use variational parameters directly (like original)
                print(f"[DEBUG] q_mu device before detach: {model.X.q_mu.device}")
                X_mean = model.X.q_mu.detach().cpu().numpy()
                print(f"[DEBUG] gauss X_mean shape: {X_mean.shape}, device: CPU")
            elif model_name_class == 'NNEncoder':  # 'nn_gauss' model
                print("[DEBUG] Using NNEncoder (nn_gauss model)...")
                # Use Y_train on the same device as the model to avoid device mismatch
                # Ensure Y_train is on the same device as the model before calling mu()
                print(f"[DEBUG] Y_train device: {Y_train.device}")
                print(f"[DEBUG] model.X.mu_layers[0].weight device: {model.X.mu_layers[0].weight.device}")
                Y_train_device = Y_train.to(model.X.mu_layers[0].weight.device) if Y_train.device != model.X.mu_layers[0].weight.device else Y_train
                print(f"[DEBUG] Y_train_device device: {Y_train_device.device}")
                print(f"[DEBUG] About to call model.X.mu(Y_train_device)")
                X_mean_result = model.X.mu(Y_train_device)
                print(f"[DEBUG] X_mean_result device before detach: {X_mean_result.device}")
                print(f"[DEBUG] X_mean_result requires_grad: {X_mean_result.requires_grad}")
                print(f"[DEBUG] About to detach X_mean_result")
                X_mean_detached = X_mean_result.detach()
                print(f"[DEBUG] X_mean_detached device: {X_mean_detached.device}")
                print(f"[DEBUG] About to move X_mean_detached to CPU")
                X_mean_cpu = X_mean_detached.cpu()
                print(f"[DEBUG] X_mean_cpu device: {X_mean_cpu.device}")
                print(f"[DEBUG] About to convert X_mean_cpu to numpy")
                X_mean = X_mean_cpu.numpy()
                print(f"[DEBUG] nn_gauss X_mean shape: {X_mean.shape}, device: CPU")
            else:
                print("[DEBUG] Using fallback (direct latent values)...")
                # Fallback: use direct latent values (like original)
                X_mean = model.X.X.detach().cpu().numpy()
                print(f"[DEBUG] fallback X_mean shape: {X_mean.shape}, device: CPU")
    
    # Plot points for each label (exactly like original plot_report but scattered only)
    # Use exact same colors as original oilflow experiment: ['r', 'b', 'g']
    colors = ['r', 'b', 'g']
    # Handle labels - convert to numpy if it's a tensor
    print("[DEBUG] Processing labels...")
    if hasattr(labels_train, 'cpu'):
        print(f"[DEBUG] labels_train device before cpu(): {labels_train.device}")
        labels_train_np = labels_train.cpu().numpy()
        print(f"[DEBUG] labels_train_np shape: {labels_train_np.shape}, device: CPU")
    else:
        print("[DEBUG] labels_train is already numpy array")
        labels_train_np = labels_train
    
    # Use exact same loop structure as original plot_report
    print("[DEBUG] Creating scatter plot...")
    for i, label in enumerate(np.unique(labels_train_np)):
        X_i = X_mean[labels_train_np == label]
        plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
        # Note: No error bars (X_scales=None) as requested - scattered only
    print("[DEBUG] Scatter plot completed")
    
    # Subplot 2: Inverse lengthscales (exactly like original plot_report)
    print("[DEBUG] Creating lengthscale bar plot...")
    plt.subplot(132)
    plt.bar(np.arange(latent_dim), height=inv_lengthscale, color='teal')
    plt.title('Inverse Lengthscale with SE-ARD kernel', fontsize='small')
    plt.xlabel('Latent dims', fontsize='small')
    print("[DEBUG] Lengthscale bar plot completed")
    
    # Subplot 3: Loss curve (exactly like original plot_report)
    print("[DEBUG] Creating loss curve plot...")
    plt.subplot(133)
    plt.plot(losses, label='batch_size=100', color='orange')
    plt.xlabel('Steps', fontsize='small')
    plt.title('Neg. ELBO Loss', fontsize='small')
    print("[DEBUG] Loss curve plot completed")
    
    plt.tight_layout()
    print("[DEBUG] Layout tightened")
    
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
    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        # We're already in a plots directory or checkpoints subdirectory, save directly here
        plot_path = os.path.join(results_dir, plot_filename)
    else:
        # Create plots subdirectory for better organization
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        plot_path = os.path.join(plots_subdir, plot_filename)
    
    print("[DEBUG] Saving plot to file...")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print("[DEBUG] Plot saved successfully")
    plt.close()
    print("[DEBUG] Plot closed")
    
    print(f"Standardized scattered plots saved to: {plot_path}")
    print("[DEBUG] create_standardized_scattered_plot completed successfully")

# Removed old plotting functions - now using create_standardized_scattered_plot for consistency


def print_original_gplvm_summary(model_name, mse_train, mse_test, final_loss, final_noise):
    """
    Print summary in the exact same format as original GP LVM
    """
    print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
    print(f'Test Reconstruction error {model_name} = ' + str(mse_test))


def create_unified_experiment_summary(model_name, random_seed, losses, mse_train, mse_test, final_loss, final_noise, train_nlpd, test_nlpd, model_type="ssvi"):
    """
    Create unified experiment summary in consistent format
    
    Args:
        model_name: Name of the model (e.g., "ssvi", "gauss")
        random_seed: Random seed used
        losses: Training loss history
        mse_train: Training RMSE
        mse_test: Test RMSE  
        final_loss: Final training loss
        final_noise: Final noise parameter
        train_nlpd: Training NLPD (always provided)
        test_nlpd: Test NLPD (always provided)
        model_type: Type of model for formatting ("ssvi" or "original")
    
    Returns:
        Dictionary with unified experiment summary format
    """
    # Summary data in original GP-LVM format order with NLPD values always included
    base_summary = {
        'train_rmse': float(mse_train.item()) if hasattr(mse_train, 'item') else float(mse_train),
        'test_rmse': float(mse_test.item()) if hasattr(mse_test, 'item') else float(mse_test),
        'train_nlpd': float(train_nlpd),
        'test_nlpd': float(test_nlpd),
        'final_loss': float(final_loss),
        'final_noise': float(final_noise),
        'model_name': model_name,
        'random_seed': random_seed,
        'loss_history': [float(loss) for loss in losses]
    }
    
    # Return in appropriate format based on model type
    if model_type == "ssvi":
        # For SSVI, wrap in model name for consistency with original format
        return {model_name: base_summary}
    else:
        # For original GP-LVM, return flat format
        return base_summary


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
                f"ssvi_{experiment_name}_seed_{config.random_seed}_{timestamp}"
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
    
    def save_checkpoint_metrics(self, metrics_computer, model_name: str, random_seed: int):
        """Save checkpoint metrics to JSON file in training_results folder"""
        if self.results_dir and metrics_computer and metrics_computer.metrics_history:
            # Create training_results folder
            training_results_dir = os.path.join(self.results_dir, "training_results")
            os.makedirs(training_results_dir, exist_ok=True)
            
            # Prepare checkpoint data
            checkpoint_data = {
                'model_name': model_name,
                'seed': random_seed,
                'checkpoint_frequency': self.config.checkpoint_frequency,
                'total_iterations': self.config.training_iterations,
                'checkpoints': []
            }
            
            # Extract metrics from each checkpoint
            for metrics in metrics_computer.metrics_history:
                checkpoint_entry = {
                    'iteration': metrics.iteration,
                    'train_rmse': float(metrics.train_rmse),
                    'test_rmse': float(metrics.test_rmse),
                    'train_nlpd': float(metrics.train_nlpd),
                    'test_nlpd': float(metrics.test_nlpd),
                    'elbo_loss': float(metrics.elbo_loss)
                }
                checkpoint_data['checkpoints'].append(checkpoint_entry)
            
            # Save to JSON file
            checkpoint_file = os.path.join(training_results_dir, f"{model_name}_{random_seed}_checkpoint_metrics.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.log_training(f"Checkpoint metrics saved: training_results/{os.path.basename(checkpoint_file)}")
            return checkpoint_file
    
    def save_checkpoint_plot(self, metrics_computer, model_name: str, random_seed: int):
        """Create and save training progress plot with RMSE and NLPD subplots"""
        if self.results_dir and metrics_computer and metrics_computer.metrics_history:
            import matplotlib.pyplot as plt
            
            # Extract data for plotting
            iterations = [m.iteration for m in metrics_computer.metrics_history]
            train_rmse = [m.train_rmse for m in metrics_computer.metrics_history]
            test_rmse = [m.test_rmse for m in metrics_computer.metrics_history]
            train_nlpd = [m.train_nlpd for m in metrics_computer.metrics_history]
            test_nlpd = [m.test_nlpd for m in metrics_computer.metrics_history]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot RMSE
            ax1.plot(iterations, train_rmse, 'b-', label='Train RMSE', linewidth=2)
            ax1.plot(iterations, test_rmse, 'r-', label='Test RMSE', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('RMSE')
            ax1.set_title(f'Training Progress - RMSE (Seed: {random_seed})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot NLPD
            ax2.plot(iterations, train_nlpd, 'b-', label='Train NLPD', linewidth=2)
            ax2.plot(iterations, test_nlpd, 'r-', label='Test NLPD', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('NLPD')
            ax2.set_title(f'Training Progress - NLPD (Seed: {random_seed})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{model_name}_{random_seed}_training_progress.png"
            plot_path = os.path.join(self.plots_dir, plot_filename)
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.log_training(f"Training progress plot saved: plots/{plot_filename}")
            return plot_path


def save_iteration_model(model, model_name, iteration, random_seed, results_dir):
    """
    Save model at specific iteration to pre_trained_models/checkpoints folder
    """
    # Save to pre_trained_models/checkpoints folder for organized checkpoint storage
    pre_trained_dir = os.path.join(results_dir, 'pre_trained_models', 'checkpoints')
    os.makedirs(pre_trained_dir, exist_ok=True)
    
    filename = f'oilflow_{model_name}_iter_{iteration}_seed_{random_seed}.pkl'
    filepath = os.path.join(pre_trained_dir, filename)
    
    with open(filepath, 'wb') as file:
        pkl.dump(model.state_dict(), file)
    
    return filepath


def create_iteration_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, iteration, model_type="original"):
    """
    Create iteration-based plot saved to plots/checkpoints folder
    """
    plot_filename = f"oilflow_{model_name}_iter_{iteration}_seed_{random_seed}.png"
    
    # Create plots/checkpoints subdirectory for organized checkpoint storage
    plots_checkpoints_dir = os.path.join(results_dir, 'plots', 'checkpoints')
    os.makedirs(plots_checkpoints_dir, exist_ok=True)
    
    # Create the standardized plot with custom filename in checkpoints folder
    create_standardized_scattered_plot(
        model, losses, labels_train, config, plots_checkpoints_dir, 
        model_name, random_seed, Y_train, model_type=model_type, 
        custom_filename=plot_filename
    )
    
    return plot_filename


def integrate_original_gplvm_outputs(model, config, config_name, losses, Y_train, Y_test, labels_train, labels_test, logger=None, metrics_computer=None):
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
    
    # Get NLPD values from metrics_computer if available
    train_nlpd = None
    test_nlpd = None
    if metrics_computer is not None and metrics_computer.metrics_history:
        final_metrics = metrics_computer.metrics_history[-1]
        train_nlpd = final_metrics.train_nlpd
        test_nlpd = final_metrics.test_nlpd
    
    # Print summary in original format (exactly like original GP LVM)
    print_original_gplvm_summary("ssvi", mse_train, mse_test, losses[-1] if losses else 0.0, 0.0)
    
    # Save model (same as original)
    logger.save_model(model, "ssvi", config.random_seed)
    
    # Create plots (same as original)
    create_standardized_scattered_plot(model, losses, labels_train, config, logger.get_results_dir(), "ssvi", config.random_seed, Y_train, model_type="ssvi")
    
    # Save training results
    logger.save_training_results("ssvi", config.random_seed, losses, [], mse_train, mse_test)
    
    # Create final summary with NLPD values using unified function
    results_summary = create_unified_experiment_summary(
        "ssvi", config.random_seed, losses, mse_train, mse_test, 
        losses[-1] if losses else 0.0, 0.0, train_nlpd, test_nlpd, model_type="ssvi"
    )
    
    # Save the final summary with NLPD values
    logger.save_final_summary(results_summary)
    
    return logger, results_summary
