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
import plotly.graph_objects as go
import plotly.express as px


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
                    'train_missing_nlpd': float(metrics.train_missing_nlpd),
                    'test_missing_nlpd': float(metrics.test_missing_nlpd),
                    'elbo_loss': float(metrics.elbo_loss)
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
            top_3_dims = torch.argsort(lengthscales, descending=True)[:3]  # Also compute top 3 for 3D plots
            dim1, dim2 = top_2_dims[0].item(), top_2_dims[1].item()
            dim3 = top_3_dims[2].item()  # Third dimension for 3D plots
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
            
            # Plot samples from all P samples with transparency
            for p in range(model.P):
                # Use top 2 most informative latent dimensions
                sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_cpu[p, :, dim1]).sample()
                sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_cpu[p, :, dim2]).sample()
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
                sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_cpu[p, :, dim1]).sample()
                sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_cpu[p, :, dim2]).sample()
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
            
            # Create 3D fractions plot using the same fractions data
            print("Creating 3D fractions plot...")
            
            # Create plotly 3D scatter plot colored by fractions
            fig_3d_frac = go.Figure()
            
            # Plot samples from all P samples with fractions as colors (consistent with 2D fractions plot)
            for p in range(model.P):
                # Sample from the latent distributions (consistent with 2D fractions plot)
                sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_cpu[p, :, dim1]).sample()
                sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_cpu[p, :, dim2]).sample()
                sample_3 = torch.distributions.Normal(mu_x_cpu[p, :, dim3], sigma_x_cpu[p, :, dim3]).sample()
                
                fig_3d_frac.add_trace(go.Scatter3d(
                    x=sample_1.detach().cpu().numpy(),
                    y=sample_2.detach().cpu().numpy(),
                    z=sample_3.detach().cpu().numpy(),
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=fractions[:, 0],  # Use first fraction component (consistent with 2D plot)
                        colorscale='Viridis',
                        opacity=0.6,
                        colorbar=dict(title="Fraction 1")
                    ),
                    name='Latent Points',
                    showlegend=False  # Hide individual P sample traces in legend
                ))
            
            # Update layout
            fig_3d_frac.update_layout(
                title=f'3D Latent Space Colored by Fractions - Top 3 Dimensions<br>Dim {dim1}, {dim2}, {dim3}',
                scene=dict(
                    xaxis_title=f'Dimension {dim1} (lengthscale={lengthscales[dim1].item():.2f})',
                    yaxis_title=f'Dimension {dim2} (lengthscale={lengthscales[dim2].item():.2f})',
                    zaxis_title=f'Dimension {dim3} (lengthscale={lengthscales[dim3].item():.2f})'
                ),
                width=800,
                height=600
            )
            
            # Save 3D fractions plot as HTML
            plot_3d_frac_filename = save_filename.replace('training_results', '3d_latent_space_by_fractions')
            plot_3d_frac_filename = plot_3d_frac_filename.replace('.png', '.html')
            
            # Save to appropriate directory (checkpoints or main plots)
            if is_checkpoint:
                plot_3d_frac_path = os.path.join(logger.plots_checkpoints_dir, plot_3d_frac_filename)
            else:
                plot_3d_frac_path = os.path.join(logger.plots_dir, plot_3d_frac_filename)
            
            fig_3d_frac.write_html(plot_3d_frac_path)
            print(f"3D fractions plot saved to: {plot_3d_frac_path}")
            
            if show_plot:
                fig_3d_frac.show()
            
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
        
        # Additional plots: 1D histogram and 3D scatter plot
        print("Creating additional plots: 1D histogram and 3D scatter plot...")
        
        # Ensure labels_train is on CPU for numpy operations
        labels_train_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train
        
        # Check if we can reuse existing computations from latent space plot
        if plot_flags[1] and 'mu_x_cpu' in locals() and 'sigma_x_cpu' in locals():
            # Reuse existing computations from latent space plot
            # We already have mu_x_cpu, sigma_x_cpu, and lengthscales_cpu from the latent space plot
            # Just need to compute top_3_dims for the 3D plot
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
        
        # Use dimensions already computed in latent space plot section
        most_important_dim = dim1  # Use the first (most important) dimension for 1D histogram
        
        # 1. Create 1D histogram for the most important dimension
        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 6))
        
        # Collect samples from all P samples for the most important dimension
        all_samples = []
        all_labels = []
        for p in range(model.P):
            samples = torch.distributions.Normal(mu_x_cpu[p, :, most_important_dim], 
                                               sigma_x_cpu[p, :, most_important_dim]).sample()
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
        
        # 2. Create 3D scatter plot for top 3 most important dimensions using plotly
        dim1, dim2, dim3 = top_3_dims[0].item(), top_3_dims[1].item(), top_3_dims[2].item()
        
        # Collect samples for all 3 dimensions
        samples_3d = []
        for p in range(model.P):
            sample_1 = torch.distributions.Normal(mu_x_cpu[p, :, dim1], sigma_x_cpu[p, :, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_cpu[p, :, dim2], sigma_x_cpu[p, :, dim2]).sample()
            sample_3 = torch.distributions.Normal(mu_x_cpu[p, :, dim3], sigma_x_cpu[p, :, dim3]).sample()
            
            # Ensure all samples are on CPU before converting to numpy
            samples_3d.append({
                'dim1': sample_1.detach().cpu().numpy(),
                'dim2': sample_2.detach().cpu().numpy(),
                'dim3': sample_3.detach().cpu().numpy(),
                'labels': labels_train_cpu
            })
        
        # Create plotly 3D scatter plot
        fig_3d = go.Figure()
        
        # Color mapping for labels
        colors = ['red', 'green', 'blue']
        color_names = ['Class 1', 'Class 2', 'Class 3']
        
        # Add traces for each class
        for class_idx in range(3):
            class_mask = labels_train_cpu == (class_idx + 1)
            if np.any(class_mask):
                # Combine samples from all P samples for this class
                x_vals = np.concatenate([samples_3d[p]['dim1'][class_mask] for p in range(model.P)])
                y_vals = np.concatenate([samples_3d[p]['dim2'][class_mask] for p in range(model.P)])
                z_vals = np.concatenate([samples_3d[p]['dim3'][class_mask] for p in range(model.P)])
                
                fig_3d.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors[class_idx],
                        opacity=0.6
                    ),
                    name=color_names[class_idx]
                ))
        
        # Update layout
        fig_3d.update_layout(
            title=f'3D Scatter Plot - Top 3 Most Important Dimensions<br>Dim {dim1}, {dim2}, {dim3}',
            scene=dict(
                xaxis_title=f'Dimension {dim1} (lengthscale={lengthscales[dim1].item():.2f})',
                yaxis_title=f'Dimension {dim2} (lengthscale={lengthscales[dim2].item():.2f})',
                zaxis_title=f'Dimension {dim3} (lengthscale={lengthscales[dim3].item():.2f})'
            ),
            width=800,
            height=600
        )
        
        # Save 3D plot as HTML
        plot_3d_filename = save_filename.replace('training_results', '3d_scatter_top3_dims')
        plot_3d_filename = plot_3d_filename.replace('.png', '.html')
        
        # Save to appropriate directory (checkpoints or main plots)
        if is_checkpoint:
            plot_3d_path = os.path.join(logger.plots_checkpoints_dir, plot_3d_filename)
        else:
            plot_3d_path = os.path.join(logger.plots_dir, plot_3d_filename)
        
        fig_3d.write_html(plot_3d_path)
        print(f"3D scatter plot saved to: {plot_3d_path}")
        
        if show_plot:
            fig_3d.show()
        
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
def plot_latent_slice_outer(model, y_train, center="zeros", n_idx=None, save_path=None):
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
    mu_x_mean = mu_x_encoded.mean(0)  # [N, Q] - average over P samples

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

    # 9) plot - show first 12 dimensions (or M if M < 12)
    xnp = xs.detach().cpu().numpy()
    
    # Plot only first 12 dimensions (or all if M < 12)
    M_plot = min(M, 12)
    n_cols = 4
    n_rows = int(np.ceil(M_plot / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), constrained_layout=True)
    axes = axes.ravel()
    
    for m in range(M_plot):
        mu = f[:, m].cpu().numpy()
        sd = pred_sd[:, m].cpu().numpy()
        ax = axes[m]
        ax.plot(xnp, mu, lw=2)
        ax.fill_between(xnp, mu - 2*sd, mu + 2*sd, alpha=0.25)
        ax.set_title(f"y[{m}] vs x_{j} (l={ells[j]:.2g})", fontsize=9)
        ax.set_xlabel(f"x_{j}")
        ax.set_ylabel(f"y[{m}]")
    
    for k in range(M_plot, len(axes)):
        axes[k].axis("off")
    
    if save_path:
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        print(f"Saved latent slice plot to: {save_path}")
    else:
        plt.show()








 