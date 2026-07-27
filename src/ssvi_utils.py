"""
Utilities for SSVI GP-LVM models including logging, data loading, and visualization
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import tarfile
import time
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Any, List
from pathlib import Path


class SSVIResultsLogger:
    """Generic logger for SSVI GP-LVM results and gradients"""
    
    def __init__(self, config, experiment_name: str = "ssvi_gplvm"):
        self.config = config
        self.experiment_name = experiment_name
        
        if hasattr(config, 'save_results') and config.save_results:
            # Create results directory with timestamp
            timestamp = datetime.now().strftime("%d_%m_%H_%M")
            
            # Use results_dir from config (should always be available)
            if hasattr(config, 'results_dir') and config.results_dir:
                # Use results_dir as specified in config (relative to current working directory)
                # This allows the config to control whether paths are relative to project root or current directory
                results_folder = config.results_dir
            else:
                raise ValueError("config.results_dir must be set. Please ensure your config file has a results_dir field.")

            # Create results directory with seed
            seed = getattr(config, 'random_seed', 7)  # Default to 7 if not available
            self.results_dir = f"{results_folder}/{experiment_name}_seed_{seed}_{timestamp}"
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Create plots subdirectory
            self.plots_dir = os.path.join(self.results_dir, "plots")
            os.makedirs(self.plots_dir, exist_ok=True)
            
            # Create plots/checkpoints subdirectory for iteration plots
            self.plots_checkpoints_dir = os.path.join(self.plots_dir, "checkpoints")
            os.makedirs(self.plots_checkpoints_dir, exist_ok=True)
            
            # Create pre_trained_models subdirectory
            self.models_dir = os.path.join(self.results_dir, "pre_trained_models")
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Create pre_trained_models/checkpoints subdirectory for iteration models
            self.models_checkpoints_dir = os.path.join(self.models_dir, "checkpoints")
            os.makedirs(self.models_checkpoints_dir, exist_ok=True)
            
            # Create training_results subdirectory
            self.training_results_dir = os.path.join(self.results_dir, "training_results")
            os.makedirs(self.training_results_dir, exist_ok=True)

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
    
    def save_plots(self, figure, filename: str, is_checkpoint: bool = False):
        """Save matplotlib figure to plots subdirectory or checkpoints subdirectory"""
        if self.results_dir:
            if is_checkpoint:
                plot_path = os.path.join(self.plots_checkpoints_dir, filename)
                self.log_training(f"Checkpoint plot saved: plots/checkpoints/{filename}")
            else:
                plot_path = os.path.join(self.plots_dir, filename)
                self.log_training(f"Plot saved: plots/{filename}")
            figure.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    def save_model(self, model, filename: str, is_checkpoint: bool = False):
        """Save model to pre_trained_models subdirectory or checkpoints subdirectory"""
        if self.results_dir:
            import pickle
            if is_checkpoint:
                model_path = os.path.join(self.models_checkpoints_dir, filename)
                self.log_training(f"Checkpoint model saved: pre_trained_models/checkpoints/{filename}")
            else:
                model_path = os.path.join(self.models_dir, filename)
                self.log_training(f"Model saved: pre_trained_models/{filename}")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    
    def save_checkpoint_metrics(self, metrics_computer, model_name: str, random_seed: int):
        """Save checkpoint metrics to JSON file in training_results folder"""
        if self.results_dir and metrics_computer and metrics_computer.metrics_history:
            # Prepare checkpoint data
            checkpoint_data = {
                'model_name': model_name,
                'seed': random_seed,
                'total_iterations': getattr(self.config, 'training_iterations', 0),
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
                    'train_crps': float(metrics.train_crps),
                    'test_crps': float(metrics.test_crps),
                    'crps': float(metrics.crps),
                    'train_crps_posterior_predictive_sample': float(metrics.train_crps_posterior_predictive_sample),
                    'test_crps_posterior_predictive_sample': float(metrics.test_crps_posterior_predictive_sample),
                    'crps_posterior_predictive_sample': float(metrics.crps_posterior_predictive_sample),
                    'train_crps_posterior_predictive_mixture': float(metrics.train_crps_posterior_predictive_mixture),
                    'test_crps_posterior_predictive_mixture': float(metrics.test_crps_posterior_predictive_mixture),
                    'crps_posterior_predictive_mixture': float(metrics.crps_posterior_predictive_mixture),
                    'train_coverage_90': float(metrics.train_coverage_90),
                    'train_coverage_95': float(metrics.train_coverage_95),
                    'test_coverage_90': float(metrics.test_coverage_90),
                    'test_coverage_95': float(metrics.test_coverage_95),
                    'coverage_90': float(metrics.coverage_90),
                    'coverage_95': float(metrics.coverage_95),
                    'train_interval_width_90': float(metrics.train_interval_width_90),
                    'train_interval_width_95': float(metrics.train_interval_width_95),
                    'test_interval_width_90': float(metrics.test_interval_width_90),
                    'test_interval_width_95': float(metrics.test_interval_width_95),
                    'interval_width_90': float(metrics.interval_width_90),
                    'interval_width_95': float(metrics.interval_width_95),
                    'train_missing_nlpd': float(metrics.train_missing_nlpd),
                    'test_missing_nlpd': float(metrics.test_missing_nlpd),
                    'full_dataset_elbo_loss': float(-metrics.elbo_loss)  # Convert positive ELBO to negative (loss) for consistency
                }
                checkpoint_data['checkpoints'].append(checkpoint_entry)
            
            # Save to JSON file
            checkpoint_file = os.path.join(self.training_results_dir, f"{model_name}_{random_seed}_checkpoint_metrics.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.log_training(f"Checkpoint metrics saved: training_results/{os.path.basename(checkpoint_file)}")
    
    def save_training_losses(self, losses, model_name: str, random_seed: int):
        """Save training losses to JSON file in training_results folder"""
        if self.results_dir:
            losses_data = {
                'model_name': model_name,
                'seed': random_seed,
                'losses': [float(loss) for loss in losses],
                'total_iterations': len(losses)
            }
            
            losses_file = os.path.join(self.training_results_dir, f"{model_name}_{random_seed}_losses.json")
            with open(losses_file, 'w') as f:
                json.dump(losses_data, f, indent=2)
            
            self.log_training(f"Training losses saved: training_results/{os.path.basename(losses_file)}")
    
    def save_experiment_summary(self, summary_data):
        """Save final experiment summary in same format as original GP LVM"""
        if self.results_dir:
            summary_file = os.path.join(self.results_dir, "experiment_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            self.log_training(f"Experiment summary saved: {os.path.basename(summary_file)}")
    
    def get_results_dir(self):
        """Get results directory path"""
        return self.results_dir





def visualize_ssvi_results(model, y_train, labels_train, losses, config, logger: SSVIResultsLogger, save_filename: str = "training_results.png", show_plot: bool = True, n_optimization_steps: int = None, is_test_data: bool = False, is_checkpoint: bool = False, train_indices: torch.Tensor = None, full_elbo_iterations: list = None, batch_losses: list = None):
    """
    Visualize SSVI training results and latent space
    
    Parameters
    ----------
    losses : list
        Full ELBO losses (negative ELBO values)
    full_elbo_iterations : list, optional
        Iteration numbers corresponding to full_elbo_losses. If provided, used for x-axis of full ELBO plot.
    batch_losses : list, optional
        Batch ELBO losses (one per training iteration). If provided, creates an additional plot showing batch ELBO progression.
    """
    
    if not (hasattr(config, 'plot_training') and config.plot_training or
            hasattr(config, 'plot_latent_space') and config.plot_latent_space or
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams):
        return

    
    with torch.no_grad():
        # Color mapping for labels (assuming 3 classes)
        c = ["r", "g", "b"]
        colors = [c[int(i)-1] for i in labels_train]
        
        plot_flags = [
            hasattr(config, 'plot_training') and config.plot_training,
            hasattr(config, 'plot_latent_space') and config.plot_latent_space,
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams
        ]
        plot_batch_elbo = batch_losses is not None and len(batch_losses) > 0
        n_plots = sum(plot_flags) + (1 if plot_batch_elbo else 0)
        fig_width = 4 * n_plots
        fig = plt.figure(figsize=(fig_width, 4))
        
        plot_idx = 1
        
        # Plot 1: Full ELBO Training curve
        if plot_flags[0]:
            plt.subplot(1, n_plots, plot_idx)
            if full_elbo_iterations is not None and len(full_elbo_iterations) == len(losses):
                # Use iteration numbers on x-axis
                plt.plot(full_elbo_iterations, losses, 'b-', alpha=0.7)
            else:
                # Fallback: use indices if iterations not provided
                plt.plot(losses, 'b-', alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('SSVI Training Curve (Full ELBO)')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 2: Batch ELBO progression (if batch_losses provided)
        if plot_batch_elbo:
            plt.subplot(1, n_plots, plot_idx)
            # Batch losses are computed at every iteration, so x-axis is just iteration indices
            batch_iterations = list(range(len(batch_losses)))
            plt.plot(batch_iterations, batch_losses, 'g-', alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Batch ELBO Progression')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 2: Latent space visualization
        if plot_flags[1]:
            plt.subplot(1, n_plots, plot_idx)
            
            # Find the most informative dimensions (largest lengthscales = smallest beta)
            lengthscales = 1 / model.gp.beta.flatten()
            top_2_dims = torch.argsort(lengthscales, descending=True)[:2]
            dim1, dim2 = top_2_dims[0].item(), top_2_dims[1].item()
            lengthscales_cpu = lengthscales.cpu()  # Move to CPU for string formatting
            # Get latent variables using new explicit training method
            epsilon = model.get_epsilon()
            U = model.get_u_samples(epsilon)  # [P, M, C, Lx]
            
            # Handle missing data if enabled
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                # Preprocess encoder input to handle missing data (replace NaN with -9.0)
                y_encoder_input = torch.where(torch.isnan(y_train), torch.full_like(y_train, -9.0), y_train)
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)  # [P, N, A]
            else:
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, epsilon)  # [P, N, A]
            
            # Keep as torch tensors for sampling, convert to numpy after sampling
            mu_x_cpu = mu_x.cpu()  # Move to CPU but keep as tensor
            sigma_x_cpu = sigma_x.cpu()  # Move to CPU but keep as tensor
            # (see plot_latent_slice_outer usage). Convert to standard deviation for sampling/plotting.
            sigma_x_std_cpu = torch.sqrt(torch.clamp(sigma_x_cpu, min=1e-12))
            
            # Plot samples from all P samples with transparency
            for p in range(model.P):
                # Use top 2 most informative latent dimensions
                sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
                sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
                plt.scatter(sample_1.detach().cpu().numpy(),
                            sample_2.detach().cpu().numpy(), alpha=0.6, color=colors)
            
            plt.xlabel(f'Latent Dimension {dim1} (lengthscale={lengthscales_cpu[dim1]:.2f})')
            plt.ylabel(f'Latent Dimension {dim2} (lengthscale={lengthscales_cpu[dim2]:.2f})') 
            plt.title(f'Latent Space - Top 2 Informative Dims (P={model.P} samples)')
            plot_idx += 1
        
        # Generate fractions plot using the same mu_x and top dimensions
        try:
            # Load fractions data - check data directory first, then current directory
            frctn_path = Path("data") / "oil_fractions.txt"
            
            if not frctn_path.exists():
                # Fallback to current directory (SSVI approach)
                frctn_path = Path(".") / "DataTrnFrctns.txt"

            if not frctn_path.exists():
                raise FileNotFoundError(f"Fractions file not found. Checked: data/oil_fractions.txt and DataTrnFrctns.txt")

            w = np.loadtxt(frctn_path)  # (N, 2)
            w3 = 1.0 - w.sum(axis=1, keepdims=True)  # (N, 1)
            fractions = np.concatenate((w, w3), axis=1)  # (N, 3)
            
            # Debug: print fractions shapes
            print(f"Debug - fractions shape before subsetting: {fractions.shape}")
            print(f"Debug - y_train shape: {y_train.shape}")
            
            # Subset fractions data to match training data using train_indices
            train_indices_cpu = train_indices.cpu().numpy()
            fractions = fractions[train_indices_cpu]
            print(f"Debug - fractions shape after subsetting with train_indices: {fractions.shape}")
            
            # Create fractions plot using the same mu_x and top dimensions from above
            # Make figure wider to account for colorbar, but keep plot area square
            fig_frac, ax_frac = plt.subplots(1, 1, figsize=(7.5, 6))
            
            # Use the same top 2 dimensions that were calculated above
            # Plot samples from all P samples with transparency (consistent with latent dims plot)
            for p in range(model.P):
                # Sample from the latent distributions (consistent with main latent space plot)
                sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_std_cpu[p, :, dim1]).sample()
                sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_std_cpu[p, :, dim2]).sample()
                scatter = ax_frac.scatter(sample_1.detach().cpu().numpy(),
                                        sample_2.detach().cpu().numpy(), 
                                        c=fractions, s=20, alpha=0.6)
            ax_frac.set_title("Latent Space Colored by Fractions")
            ax_frac.set_xlabel(f'mu_{dim1}')
            ax_frac.set_ylabel(f'mu_{dim2}')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax_frac)
            cbar.set_label('Fractions')
            
            # Ensure equal aspect ratio for square plot
            ax_frac.set_aspect("equal", adjustable='box')
            fig_frac.tight_layout()
            
            # Save fractions plot
            fractions_filename = save_filename.replace('training_results', 'latent_space_by_fractions')
            logger.save_plots(fig_frac, fractions_filename, is_checkpoint=is_checkpoint)
            plt.close(fig_frac)

        except FileNotFoundError as e:
            print(f"Warning: Could not create fractions plot: {e}")
        
        # Plot 3: Hyperparameters (lengthscales)
        if plot_flags[2]:
            plt.subplot(1, n_plots, plot_idx)
            lengthscales = (1 / model.gp.beta.flatten()).cpu()
            plt.bar(range(len(lengthscales)), lengthscales, 
                   color="blue", alpha=0.7)
            plt.xlabel('Latent Dimension')
            plt.ylabel('Lengthscale')
            plt.title('GP Lengthscales')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        logger.save_plots(fig, save_filename, is_checkpoint=is_checkpoint)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # Close figure to free memory
        
        # Additional plot: 1D histogram
        print("Creating additional plot: 1D histogram...")
        
        # Ensure labels_train is on CPU for numpy operations
        labels_train_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train
        
        # Check if we can reuse existing computations from latent space plot
        if plot_flags[1] and 'mu_x_cpu' in locals() and 'sigma_x_cpu' in locals():
            # Reuse existing computations from latent space plot
            # We already have mu_x_cpu, sigma_x_cpu, and lengthscales_cpu from the latent space plot
            lengthscales = lengthscales_cpu  # Use existing lengthscales
        else:
            # Need to compute the data (latent space plot wasn't computed)
            # Find the most important dimensions (largest lengthscales = smallest beta)
            lengthscales = (1 / model.gp.beta.flatten()).cpu()  # Move to CPU
            
            # Get latent variables
            epsilon = model.get_epsilon()
            U = model.get_u_samples(epsilon)  # [P, M, C, Lx]
            
            # Handle missing data if enabled
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                y_encoder_input = torch.where(torch.isnan(y_train), torch.full_like(y_train, -9.0), y_train)
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)  # [P, N, A]
            else:
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, epsilon)  # [P, N, A]
            
            # Move to CPU for plotting
            mu_x_cpu = mu_x.cpu()
            sigma_x_cpu = sigma_x.cpu()
            sigma_x_std_cpu = torch.sqrt(torch.clamp(sigma_x_cpu, min=1e-12))
        
        # Use dimensions already computed in latent space plot section
        most_important_dim = dim1  # Use the first (most important) dimension for 1D histogram
        
        # 1. Create 1D histogram for the most important dimension
        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 6))
        
        # Collect samples from all P samples for the most important dimension
        all_samples = []
        all_labels = []
        for p in range(model.P):
            samples = torch.distributions.Normal(mu_x_cpu[p, :, most_important_dim], 
                                               sigma_x_std_cpu[p, :, most_important_dim]).sample()
            # Ensure samples are on CPU before converting to numpy
            all_samples.append(samples.detach().cpu().numpy())
            # Repeat labels for each P sample
            all_labels.append(labels_train_cpu)
        
        all_samples = np.concatenate(all_samples)
        all_labels = np.concatenate(all_labels)
        
        # Create histogram with different colors for different classes
        for class_idx in range(3):  # Assuming 3 classes
            class_mask = all_labels == (class_idx + 1)
            if np.any(class_mask):
                class_samples = all_samples[class_mask]
                ax_hist.hist(class_samples, bins=30, alpha=0.7, 
                           label=f'Class {class_idx + 1}', 
                           color=['red', 'green', 'blue'][class_idx])
        
        ax_hist.set_xlabel(f'Latent Dimension {most_important_dim} (Most Important)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'1D Histogram - Most Important Dimension {most_important_dim}')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Save histogram plot
        hist_filename = save_filename.replace('training_results', '1d_histogram_most_important_dim')
        logger.save_plots(fig_hist, hist_filename, is_checkpoint=is_checkpoint)
        if show_plot:
            plt.show()
        else:
            plt.close(fig_hist)
        
        print("Additional plots created successfully!")


def plot_metrics_history(metrics_computer, 
                        logger: SSVIResultsLogger = None,
                        show_plot: bool = True,
                        batch_losses: list = None) -> plt.Figure:
    """
    Plot metrics progression over training iterations
    
    Args:
        metrics_computer: SSVIMetricsComputer with metrics history
        logger: Optional logger for saving plots
        show_plot: Whether to display the plot
        batch_losses: Optional list of batch ELBO losses (one per iteration) for additional plot
        
    Returns:
        matplotlib Figure object
    """
    if not hasattr(metrics_computer, 'metrics_history') or not metrics_computer.metrics_history:
        print("No metrics history available for plotting")
        return None
    
    # Extract data from metrics history
    iterations = [m.iteration for m in metrics_computer.metrics_history]
    train_rmse = [m.train_rmse for m in metrics_computer.metrics_history]
    test_rmse = [m.test_rmse for m in metrics_computer.metrics_history]
    train_nlpd = [m.train_nlpd for m in metrics_computer.metrics_history]
    test_nlpd = [m.test_nlpd for m in metrics_computer.metrics_history]
    elbo_loss = [m.elbo_loss for m in metrics_computer.metrics_history]
    effective_dims = [m.effective_latent_dims for m in metrics_computer.metrics_history]
    
    # Determine if we need to add batch ELBO plot
    plot_batch_elbo = batch_losses is not None and len(batch_losses) > 0
    
    # Create subplots: 3x2 if batch ELBO (5 plots), otherwise 2x2 (4 plots)
    if plot_batch_elbo:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]
        ax5 = axes[2, 0]
        # Hide the last subplot (bottom right)
        axes[2, 1].axis('off')
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: RMSE comparison (train vs test)
    ax1.plot(iterations, train_rmse, 'b-', label='Train RMSE', linewidth=2)
    ax1.plot(iterations, test_rmse, 'r-', label='Test RMSE', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Reconstruction Error (RMSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: NLPD comparison (train vs test)
    ax2.plot(iterations, train_nlpd, 'b-', label='Train NLPD', linewidth=2)
    ax2.plot(iterations, test_nlpd, 'r-', label='Test NLPD', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('NLPD (lower is better)')
    ax2.set_title('Predictive Density (NLPD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Full ELBO Loss progression (negate to be consistent with training results plot)
    # elbo_loss stores positive ELBO, but training results plot shows negative ELBO as "Loss"
    elbo_loss_negated = [-x for x in elbo_loss]
    ax3.plot(iterations, elbo_loss_negated, 'g-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss (Full ELBO)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effective latent dimensions
    ax4.plot(iterations, effective_dims, 'purple', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Effective Dimensions')
    ax4.set_title('Model Complexity (Effective Latent Dims)')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Batch ELBO progression (if provided)
    if plot_batch_elbo:
        batch_iterations = list(range(len(batch_losses)))
        ax5.plot(batch_iterations, batch_losses, 'orange', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Loss')
        ax5.set_title('Batch ELBO Progression')
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if logger provided
    if logger:
        logger.save_plots(fig, "metrics_progression.png")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_crps_metrics_progression(metrics_computer,
                                  logger: SSVIResultsLogger = None,
                                  show_plot: bool = True) -> plt.Figure:
    """
    Plot CRPS, predictive interval coverage, and mean interval width (MPIW).

    Panel 1: train/test CRPS
    Panel 2: train/test coverage_90 and coverage_95
    Panel 3: train/test interval_width_90 and interval_width_95
    """
    if not hasattr(metrics_computer, 'metrics_history') or not metrics_computer.metrics_history:
        print("No metrics history available for CRPS/coverage plotting")
        return None

    iterations = [m.iteration for m in metrics_computer.metrics_history]
    train_crps = [m.train_crps for m in metrics_computer.metrics_history]
    test_crps = [m.test_crps for m in metrics_computer.metrics_history]
    train_c90 = [m.train_coverage_90 for m in metrics_computer.metrics_history]
    test_c90 = [m.test_coverage_90 for m in metrics_computer.metrics_history]
    train_c95 = [m.train_coverage_95 for m in metrics_computer.metrics_history]
    test_c95 = [m.test_coverage_95 for m in metrics_computer.metrics_history]
    train_w90 = [m.train_interval_width_90 for m in metrics_computer.metrics_history]
    test_w90 = [m.test_interval_width_90 for m in metrics_computer.metrics_history]
    train_w95 = [m.train_interval_width_95 for m in metrics_computer.metrics_history]
    test_w95 = [m.test_interval_width_95 for m in metrics_computer.metrics_history]

    valid_train = [not np.isnan(x) for x in train_crps]
    valid_test = [not np.isnan(x) for x in test_crps]

    if not any(valid_train) and not any(valid_test):
        print("No valid CRPS data available for plotting")
        return None

    fig, (ax_crps, ax_cov, ax_width) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    if any(valid_train):
        train_iterations = [iterations[i] for i in range(len(iterations)) if valid_train[i]]
        ax_crps.plot(train_iterations, [train_crps[i] for i in range(len(train_crps)) if valid_train[i]],
                     'b-', label='Train CRPS', linewidth=2, marker='o', markersize=4)
    if any(valid_test):
        test_iterations = [iterations[i] for i in range(len(iterations)) if valid_test[i]]
        ax_crps.plot(test_iterations, [test_crps[i] for i in range(len(test_crps)) if valid_test[i]],
                     'r-', label='Test CRPS', linewidth=2, marker='s', markersize=4)

    ax_crps.set_ylabel('CRPS (lower is better)')
    ax_crps.set_title('CRPS, Coverage & Interval Width Progression')
    ax_crps.legend()
    ax_crps.grid(True, alpha=0.3)

    def _plot_series(ax, values, iters, style, label):
        valid = [not np.isnan(x) for x in values]
        if any(valid):
            ax.plot([iters[i] for i in range(len(iters)) if valid[i]],
                    [values[i] for i in range(len(values)) if valid[i]],
                    style, label=label, linewidth=2, marker='o', markersize=4)

    _plot_series(ax_cov, train_c90, iterations, 'b-', 'Train coverage_90')
    _plot_series(ax_cov, test_c90, iterations, 'r-', 'Test coverage_90')
    _plot_series(ax_cov, train_c95, iterations, 'b--', 'Train coverage_95')
    _plot_series(ax_cov, test_c95, iterations, 'r--', 'Test coverage_95')
    ax_cov.axhline(0.90, color='gray', linestyle=':', alpha=0.7, label='Nominal 0.90')
    ax_cov.axhline(0.95, color='black', linestyle=':', alpha=0.5, label='Nominal 0.95')
    ax_cov.set_ylabel('Coverage')
    ax_cov.set_ylim(0.0, 1.05)
    ax_cov.legend(fontsize=9)
    ax_cov.grid(True, alpha=0.3)

    _plot_series(ax_width, train_w90, iterations, 'b-', 'Train interval_width_90')
    _plot_series(ax_width, test_w90, iterations, 'r-', 'Test interval_width_90')
    _plot_series(ax_width, train_w95, iterations, 'b--', 'Train interval_width_95')
    _plot_series(ax_width, test_w95, iterations, 'r--', 'Test interval_width_95')
    ax_width.set_xlabel('Iteration')
    ax_width.set_ylabel('Mean interval width (lower is sharper)')
    ax_width.legend(fontsize=9)
    ax_width.grid(True, alpha=0.3)

    plt.tight_layout()

    if logger:
        logger.save_plots(fig, "crps_metrics_progression.png")

    if show_plot:
        plt.show()

    return fig


def plot_crps_posterior_predictive_progression(metrics_computer,
                                               logger: SSVIResultsLogger = None,
                                               show_plot: bool = True) -> plt.Figure:
    """
    Separate progression plot for posterior-predictive CRPS variants.

    Panel 1: crps_posterior_predictive_sample (train/test)
    Panel 2: crps_posterior_predictive_mixture (train/test)
    """
    if not hasattr(metrics_computer, 'metrics_history') or not metrics_computer.metrics_history:
        print("No metrics history available for posterior-predictive CRPS plotting")
        return None

    iterations = [m.iteration for m in metrics_computer.metrics_history]
    train_sample = [m.train_crps_posterior_predictive_sample for m in metrics_computer.metrics_history]
    test_sample = [m.test_crps_posterior_predictive_sample for m in metrics_computer.metrics_history]
    train_mix = [m.train_crps_posterior_predictive_mixture for m in metrics_computer.metrics_history]
    test_mix = [m.test_crps_posterior_predictive_mixture for m in metrics_computer.metrics_history]

    def _has_valid(values):
        return any(not np.isnan(x) for x in values)

    if not (_has_valid(train_sample) or _has_valid(test_sample) or _has_valid(train_mix) or _has_valid(test_mix)):
        print("No valid posterior-predictive CRPS data available for plotting")
        return None

    fig, (ax_sample, ax_mix) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    def _plot_series(ax, values, iters, style, label):
        valid = [not np.isnan(x) for x in values]
        if any(valid):
            ax.plot([iters[i] for i in range(len(iters)) if valid[i]],
                    [values[i] for i in range(len(values)) if valid[i]],
                    style, label=label, linewidth=2, marker='o', markersize=4)

    _plot_series(ax_sample, train_sample, iterations, 'b-', 'Train sample CRPS')
    _plot_series(ax_sample, test_sample, iterations, 'r-', 'Test sample CRPS')
    ax_sample.set_ylabel('CRPS (lower is better)')
    ax_sample.set_title('Posterior-Predictive CRPS Variants')
    ax_sample.legend()
    ax_sample.grid(True, alpha=0.3)

    _plot_series(ax_mix, train_mix, iterations, 'b-', 'Train mixture CRPS')
    _plot_series(ax_mix, test_mix, iterations, 'r-', 'Test mixture CRPS')
    ax_mix.set_xlabel('Iteration')
    ax_mix.set_ylabel('CRPS (lower is better)')
    ax_mix.legend()
    ax_mix.grid(True, alpha=0.3)

    plt.tight_layout()

    if logger:
        logger.save_plots(fig, "crps_posterior_predictive_progression.png")

    if show_plot:
        plt.show()

    return fig


def plot_missing_nlpd_metrics_progression(metrics_computer,
                                         logger: SSVIResultsLogger = None,
                                         show_plot: bool = True) -> plt.Figure:
    """
    Plot missing NLPD metrics progression over training iterations
    
    Args:
        metrics_computer: SSVIMetricsComputer with metrics history
        logger: Optional logger for saving plots
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if not hasattr(metrics_computer, 'metrics_history') or not metrics_computer.metrics_history:
        print("No metrics history available for missing NLPD plotting")
        return None
    
    # Extract missing NLPD data from metrics history
    iterations = [m.iteration for m in metrics_computer.metrics_history]
    train_missing_nlpd = [m.train_missing_nlpd for m in metrics_computer.metrics_history]
    test_missing_nlpd = [m.test_missing_nlpd for m in metrics_computer.metrics_history]
    
    # Check if we have any valid missing NLPD data
    valid_train = [not np.isnan(x) for x in train_missing_nlpd]
    valid_test = [not np.isnan(x) for x in test_missing_nlpd]
    
    if not any(valid_train) and not any(valid_test):
        print("No valid missing NLPD data available for plotting")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot train missing NLPD if available
    if any(valid_train):
        train_iterations = [iterations[i] for i in range(len(iterations)) if valid_train[i]]
        train_values = [train_missing_nlpd[i] for i in range(len(train_missing_nlpd)) if valid_train[i]]
        ax.plot(train_iterations, train_values, 'b-', label='Train Missing NLPD', linewidth=2, marker='o', markersize=4)
    
    # Plot test missing NLPD if available
    if any(valid_test):
        test_iterations = [iterations[i] for i in range(len(iterations)) if valid_test[i]]
        test_values = [test_missing_nlpd[i] for i in range(len(test_missing_nlpd)) if valid_test[i]]
        ax.plot(test_iterations, test_values, 'r-', label='Test Missing NLPD', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Missing Data NLPD (lower is better)')
    ax.set_title('Missing Data Predictive Density Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if logger provided
    if logger:
        logger.save_plots(fig, "missing_nlpd_metrics_progression.png")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_missing_rmse_metrics_progression(metrics_computer,
                                         logger: SSVIResultsLogger = None,
                                         show_plot: bool = True) -> plt.Figure:
    """
    Plot missing RMSE metrics progression over training iterations
    
    Args:
        metrics_computer: SSVIMetricsComputer with metrics history
        logger: Optional logger for saving plots
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if not hasattr(metrics_computer, 'metrics_history') or not metrics_computer.metrics_history:
        print("No metrics history available for missing RMSE plotting")
        return None
    
    # Extract missing RMSE data from metrics history
    iterations = [m.iteration for m in metrics_computer.metrics_history]
    train_missing_rmse = [m.train_missing_rmse for m in metrics_computer.metrics_history]
    test_missing_rmse = [m.test_missing_rmse for m in metrics_computer.metrics_history]
    
    # Check if we have any valid missing RMSE data
    valid_train = [not np.isnan(x) for x in train_missing_rmse]
    valid_test = [not np.isnan(x) for x in test_missing_rmse]
    
    if not any(valid_train) and not any(valid_test):
        print("No valid missing RMSE data available for plotting")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot train missing RMSE if available
    if any(valid_train):
        train_iterations = [iterations[i] for i in range(len(iterations)) if valid_train[i]]
        train_values = [train_missing_rmse[i] for i in range(len(train_missing_rmse)) if valid_train[i]]
        ax.plot(train_iterations, train_values, 'b-', label='Train Missing RMSE', linewidth=2, marker='o', markersize=4)
    
    # Plot test missing RMSE if available
    if any(valid_test):
        test_iterations = [iterations[i] for i in range(len(iterations)) if valid_test[i]]
        test_values = [test_missing_rmse[i] for i in range(len(test_missing_rmse)) if valid_test[i]]
        ax.plot(test_iterations, test_values, 'r-', label='Test Missing RMSE', linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Missing Data RMSE (lower is better)')
    ax.set_title('Missing Data Reconstruction Error Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if logger provided
    if logger:
        logger.save_plots(fig, "missing_rmse_metrics_progression.png")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_final_metrics_summary(metrics_computer,
                              model,
                              logger: SSVIResultsLogger = None,
                              show_plot: bool = True) -> plt.Figure:
    """
    Create summary plots of final model performance metrics
    
    Args:
        metrics_computer: SSVIMetricsComputer with computed metrics
        model: Trained SSVI GP-LVM model
        logger: Optional logger for saving plots
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if not hasattr(metrics_computer, 'metrics_history') or not metrics_computer.metrics_history:
        print("No metrics available for summary")
        return None
    
    # Get final metrics
    final_metrics = metrics_computer.metrics_history[-1]
    metrics_summary = metrics_computer.get_metrics_summary()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Train vs Test RMSE comparison
    categories = ['Train RMSE', 'Test RMSE']
    values = [final_metrics.train_rmse, final_metrics.test_rmse]
    colors = ['blue', 'red']
    
    bars1 = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylabel('RMSE')
    ax1.set_title('Final Reconstruction Performance')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Plot 2: Train vs Test NLPD comparison  
    categories = ['Train NLPD', 'Test NLPD']
    values = [final_metrics.train_nlpd, final_metrics.test_nlpd]
    
    bars2 = ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.set_ylabel('NLPD (lower is better)')
    ax2.set_title('Final Uncertainty Quantification')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Latent space utilization
    with torch.no_grad():
        if hasattr(model, 'gp') and hasattr(model.gp, 'beta'):
            lengthscales = (1.0 / model.gp.beta.flatten()).cpu()
            dims = range(len(lengthscales))
            
            ax3.bar(dims, lengthscales, color='green', alpha=0.7)
            ax3.set_xlabel('Latent Dimension')
            ax3.set_ylabel('Lengthscale')
            ax3.set_title(f'Lengthscales ({final_metrics.effective_latent_dims}/{final_metrics.total_latent_dims} effective)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Lengthscales not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Lengthscales (Not Available)')
    
    # Plot 4: Summary metrics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Train RMSE', f'{final_metrics.train_rmse:.4f}'],
        ['Test RMSE', f'{final_metrics.test_rmse:.4f}'],
        ['Train NLPD', f'{final_metrics.train_nlpd:.3f}'],
        ['Test NLPD', f'{final_metrics.test_nlpd:.3f}'],
        ['Train CRPS', f'{final_metrics.train_crps:.4f}'],
        ['Test CRPS (crps)', f'{final_metrics.crps:.4f}'],
        ['crps_pp_sample', f'{final_metrics.crps_posterior_predictive_sample:.4f}'],
        ['crps_pp_mixture', f'{final_metrics.crps_posterior_predictive_mixture:.4f}'],
        ['coverage_90', f'{final_metrics.coverage_90:.4f}'],
        ['coverage_95', f'{final_metrics.coverage_95:.4f}'],
        ['interval_width_90', f'{final_metrics.interval_width_90:.4f}'],
        ['interval_width_95', f'{final_metrics.interval_width_95:.4f}'],
        ['Overfitting Ratio', f'{metrics_summary.get("overfitting_ratio", 0):.3f}'],
        ['Effective Dims', f'{final_metrics.effective_latent_dims}/{final_metrics.total_latent_dims}'],
        ['Final Loss', f'{final_metrics.elbo_loss:.4f}']
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Model Performance Summary')
    
    plt.tight_layout()
    
    # Save plot if logger provided
    if logger:
        logger.save_plots(fig, "final_metrics_summary.png")
    
    if show_plot:
        plt.show()
    
    return fig


@torch.no_grad()
def plot_latent_slice_outer(model, y_train, center="zeros", n_idx=None, save_path=None, scaler=None):
    """
    Plot 1D latent slice with outer-expectation uncertainty bands.
    
    This function creates a visualization of the GP-LVM's predictive behavior along
    the most informative latent dimension (smallest lengthscale), showing both the
    mean prediction and ELBO-consistent uncertainty bands.
    
    Args:
        model: GPLVM_SSVI_AM_INF model instance
        y_train: Training data [N, M] - used to encode latent representations
        center: str, either "zeros" (default) or "data_point". If "data_point", 
                must provide n_idx to specify which data point to use as center.
        n_idx: int, index of data point to use as center when center="data_point"
        save_path: str, optional path to save the plot. If None, displays the plot.
        scaler: optional scaler for inverse transform (to visualize in original scale)
    """
    device = model.device
    Q, M, P = model.A, model.M, model.P
    N = y_train.shape[0]  # Number of training points from actual data

    # Encode training data to get latent representations
    epsilon = model.get_epsilon()
    U_samples = model.get_u_samples(epsilon)
    y_encoder_input = torch.where(torch.isnan(y_train), 
                                   torch.full_like(y_train, -9.0, dtype=torch.float64), 
                                   y_train)
    mu_x_encoded, sigma_x_encoded = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
    # Take first P sample instead of averaging
    mu_x_mean = mu_x_encoded[0]  # [N, Q] - first sample from P samples

    # 1) pick most-informative latent dim (smallest lengthscale)
    betas = model.gp.beta[0].detach().cpu().numpy()  # (Q,)
    ells = 1.0 / np.maximum(betas, 1e-12)
    j = int(np.argmax(ells))

    # 2) center
    x0 = torch.zeros(Q, dtype=torch.float64, device=device)
    if center == "data_point":
        assert n_idx is not None, "n_idx must be provided when center='data_point'"
        x0 = mu_x_mean[n_idx].to(device)

    # 3) Use fixed range for consistent comparison across models
    # Use a standard range of [-3, 3] which covers typical latent space
    lo, hi = -3.0, 3.0
    G = 200  # Grid resolution
    
    xs = torch.linspace(lo, hi, G, dtype=torch.float64, device=device)
    Xgrid = x0.repeat(G, 1)
    Xgrid[:, j] = xs

    # Get GP expectation - returns [G, M, C]
    f_full = model.gp.get_gp_expectation(Xgrid)  # [G, M, C]

    # ELBO-consistent variance using GP covariance method
    gp_cov = model.gp.get_gp_covariance(Xgrid)        # [G, M, C, C]
    
    # Handle C dimension (channels/components)
    C = f_full.shape[-1]
    if C == 1:
        # C=1 case (standard GP-LVM) - squeeze the channel dimension
        f = f_full[..., 0]                                        # [G, M]
        gp_var = gp_cov[..., 0, 0]                                # [G, M]
        pred_var = gp_var + model.sigma2                          # [G, M]
    else:
        # Multi-output case - average over channels for visualization
        f = f_full.mean(dim=-1)                                   # [G, M]
        # For variance, use diagonal and average over channels
        pred_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1).mean(dim=-1) + model.sigma2  # [G, M]
    
    pred_sd = torch.sqrt(torch.clamp(pred_var, min=1e-12))        # [G, M]

    # Inverse transform predictions if scaler provided (for visualization in original scale)
    if scaler is not None:
        f = scaler.inverse_transform(f)  # [G, M]
        # For standard deviation, scale by scaler's std if available
        if hasattr(scaler, 'std'):
            pred_sd = pred_sd * scaler.std  # [G, M]
        # For variance-based scalers, we need to handle differently
        # For now, just scale by std if available, otherwise keep as is

    # Compute Y-axis limits from actual data for consistent scaling across models
    # Get data in original scale (inverse transform if scaler provided)
    y_data_for_limits = y_train.clone()
    if scaler is not None:
        y_data_for_limits = scaler.inverse_transform(y_data_for_limits)
    
    # Compute global Y-axis limits from data range (with padding for uncertainty bands)
    y_min_global = y_data_for_limits.min().item()
    y_max_global = y_data_for_limits.max().item()
    y_range = y_max_global - y_min_global
    y_padding = 1.0 * y_range  # 100% padding (fixed proportion for consistency across scripts, extra space for histograms)
    y_lim_min = y_min_global - y_padding
    y_lim_max = y_max_global + y_padding

    # 9) plot - show first 12 dimensions (or M if M < 12)
    xnp = xs.detach().cpu().numpy()
    
    # Set X-axis limits for consistency (already using [-3, 3] range)
    x_lim_min, x_lim_max = -3.0, 3.0
    
    # Plot only first 12 dimensions (or all if M < 12)
    M_plot = min(M, 12)
    n_cols = 4
    n_rows = int(np.ceil(M_plot / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), constrained_layout=True)
    axes = axes.ravel()
    
    # Sample from q(X) distribution along dimension j for histogram
    # mu_x_encoded is [P, N, Q], sigma_x_encoded is [P, N, Q] (variance)
    # Use first P sample instead of averaging
    sigma_x_first = sigma_x_encoded[0]  # [N, Q] - first sample variance from P samples
    # Sample from N(mu_x_mean, sigma_x_first) for each data point
    x_samples_j = torch.normal(mu_x_mean[:, j], torch.sqrt(torch.clamp(sigma_x_first[:, j], min=1e-8)))  # [N]
    x_j_values = x_samples_j.detach().cpu().numpy()
    
    for m in range(M_plot):
        mu = f[:, m].cpu().numpy()
        sd = pred_sd[:, m].cpu().numpy()
        ax = axes[m]
        ax.plot(xnp, mu, lw=2)
        ax.fill_between(xnp, mu - 2*sd, mu + 2*sd, alpha=0.25)
        ax.set_title(f"y[{m}] vs x_{j} (l={ells[j]:.2g})", fontsize=9)
        ax.set_xlabel(f"x_{j}")
        ax.set_ylabel(f"y[{m}]")
        # Set consistent axis limits
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(y_lim_min, y_lim_max)
        
        # Add histogram directly on the same axes, aligned with x-axis at bottom
        # Compute histogram and scale to make it bigger (use 25% of y-axis range)
        # Use fixed number of bins for consistent granularity across all models
        counts, bins = np.histogram(x_j_values, bins=15, density=True)
        y_range_hist = (y_lim_max - y_lim_min) * 0.25
        max_count = counts.max() if len(counts) > 0 and counts.max() > 0 else 1.0
        counts_scaled = (counts / max_count) * y_range_hist if max_count > 0 else counts * 0
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, counts_scaled, width=(bins[1] - bins[0]), 
               bottom=y_lim_min, alpha=0.6, color='lightblue', edgecolor='black')
    
    for k in range(M_plot, len(axes)):
        axes[k].axis("off")
    
    if save_path:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        print(f"Saved latent slice plot to: {save_path}")
    else:
        plt.show()








 