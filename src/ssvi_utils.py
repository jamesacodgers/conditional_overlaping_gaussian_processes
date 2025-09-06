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


class SSVIResultsLogger:
    """Generic logger for SSVI GP-LVM results and gradients"""
    
    def __init__(self, config, experiment_name: str = "ssvi_gplvm"):
        self.config = config
        self.experiment_name = experiment_name
        
        if hasattr(config, 'save_results') and config.save_results:
            # Create results directory with timestamp
            timestamp = datetime.now().strftime("%d_%m_%H_%M")
            
            # Determine correct results path based on current working directory
            # Organize results by experiment type: SSVI vs Original GP-LVM
            if "gplvm" in experiment_name.lower():
                results_folder = "oil_flow_original_gplvm_results"
            else:
                results_folder = "oil_flow_ssvi_results"
                
            if os.path.basename(os.getcwd()) == "ssvi_scripts_experiments":
                # Running from within ssvi_scripts_experiments directory
                self.results_dir = f"results/{results_folder}/{experiment_name}_res_{timestamp}"
            else:
                # Running from project root
                self.results_dir = f"ssvi_scripts_experiments/results/{results_folder}/{experiment_name}_res_{timestamp}"
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
        """Save matplotlib figure to results directory"""
        if self.results_dir:
            plot_path = os.path.join(self.results_dir, filename)
            figure.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.log_training(f"Plot saved: {filename}")
    
    def get_results_dir(self):
        """Get results directory path"""
        return self.results_dir





def visualize_ssvi_results(model, y, labels, losses, config, logger: SSVIResultsLogger, save_filename: str = "training_results.png", show_plot: bool = True):
    """Visualize SSVI training results and latent space"""
    
    if not (hasattr(config, 'plot_training') and config.plot_training or
            hasattr(config, 'plot_latent_space') and config.plot_latent_space or
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams):
        return
    
    with torch.no_grad():
        # Color mapping for labels (assuming 3 classes)
        c = ["r", "g", "b"]
        colors = [c[int(i)-1] for i in labels]
        
        plot_flags = [
            hasattr(config, 'plot_training') and config.plot_training,
            hasattr(config, 'plot_latent_space') and config.plot_latent_space,
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams
        ]
        n_plots = sum(plot_flags)
        fig_width = 4 * n_plots
        fig = plt.figure(figsize=(fig_width, 4))
        
        plot_idx = 1
        
        # Plot 1: Training curve
        if plot_flags[0]:
            plt.subplot(1, n_plots, plot_idx)
            plt.plot(losses, 'b-', alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('SSVI Training Curve')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 2: Latent space visualization
        if plot_flags[1]:
            plt.subplot(1, n_plots, plot_idx)
            
            # Find the two most informative dimensions (largest lengthscales = smallest beta)
            lengthscales = 1 / model.gp.beta.flatten()
            top_2_dims = torch.argsort(lengthscales, descending=True)[:2]
            dim1, dim2 = top_2_dims[0].item(), top_2_dims[1].item()
            lengthscales_cpu = lengthscales.cpu()  # Move to CPU for string formatting
            U = model.get_u_samples()  # [P, M, C, Lx]
            mu_x, sigma_x = model.get_mu_sigma_x_samples(y, U)  # [N, A]
            mu_x =  mu_x.cpu().numpy()  # Move to CPU for plotting
            # Plot samples from all P samples with transparency
            for p in range(model.P):
                # Use top 2 most informative latent dimensions
                plt.scatter(mu_x[p,:, 0], mu_x[p,:, 1], 
                           c=colors, alpha=0.15, s=10)
            
            plt.xlabel(f'Latent Dimension {dim1} (lengthscale={lengthscales_cpu[dim1]:.2f})')
            plt.ylabel(f'Latent Dimension {dim2} (lengthscale={lengthscales_cpu[dim2]:.2f})') 
            plt.title(f'Latent Space - Top 2 Informative Dims (P={model.P} samples)')
            plot_idx += 1
        
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
        logger.save_plots(fig, save_filename)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # Close figure to free memory


def plot_metrics_history(metrics_computer, 
                        logger: SSVIResultsLogger = None,
                        show_plot: bool = True) -> plt.Figure:
    """
    Plot metrics progression over training iterations
    
    Args:
        metrics_computer: SSVIMetricsComputer with metrics history
        logger: Optional logger for saving plots
        show_plot: Whether to display the plot
        
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
    
    # Create subplots
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
    
    # Plot 3: ELBO Loss progression
    ax3.plot(iterations, elbo_loss, 'g-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('ELBO Loss')
    ax3.set_title('Training Loss (ELBO)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effective latent dimensions
    ax4.plot(iterations, effective_dims, 'purple', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Effective Dimensions')
    ax4.set_title('Model Complexity (Effective Latent Dims)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if logger provided
    if logger:
        logger.save_plots(fig, "metrics_progression.png")
    
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


def save_metrics_to_json(metrics_computer,
                        logger: SSVIResultsLogger,
                        filename: str = "metrics_summary.json"):
    """
    Save complete metrics summary to JSON file
    
    Args:
        metrics_computer: SSVIMetricsComputer with computed metrics
        logger: SSVIResultsLogger for file saving
        filename: Name of JSON file
    """
    if not logger.results_dir:
        print("No results directory available for saving metrics")
        return
    
    # Prepare complete metrics data
    metrics_data = {
        'summary': metrics_computer.get_metrics_summary(),
        'history': [metrics.to_dict() for metrics in metrics_computer.metrics_history],
        'computed_at': datetime.now().isoformat()
    }
    
    # Save to JSON file
    filepath = os.path.join(logger.results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.log_training(f"Metrics saved to: {filename}")
    print(f"Complete metrics saved to: {filepath}")


def save_metrics_to_text(metrics_computer,
                        logger: SSVIResultsLogger,
                        model=None,
                        filename: str = "metrics_summary.txt"):
    """
    Save metrics summary to easy-to-read text file for comparison
    
    Args:
        metrics_computer: SSVIMetricsComputer with computed metrics
        logger: SSVIResultsLogger for file saving
        model: Optional model to extract lengthscales from
        filename: Name of text file
    """
    if not logger.results_dir:
        print("No results directory available for saving metrics")
        return
    
    if not metrics_computer.metrics_history:
        print("No metrics history available for text export")
        return
    
    # Get final metrics and summary
    final_metrics = metrics_computer.metrics_history[-1]
    summary = metrics_computer.get_metrics_summary()
    
    # Get lengthscales if model is provided
    lengthscales_info = ""
    if model is not None:
        try:
            # Get lengthscales (1/beta)
            beta_values = model.Abeta.exp()  # Convert from log space
            lengthscales = 1.0 / beta_values
            lengthscales_list = lengthscales.detach().cpu().numpy().tolist()
            
            lengthscales_info = f"""
LENGTHSCALES (1/beta):
{', '.join([f'{ls:.4f}' for ls in lengthscales_list])}"""
            
        except Exception as e:
            lengthscales_info = f"\nLENGTHSCALES: Could not extract ({str(e)})\n"
    
    # Create text content
    text_content = f"""=== SSVI GP-LVM Metrics Summary ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FINAL PERFORMANCE METRICS:
Train RMSE:        {final_metrics.train_rmse:.6f}
Test RMSE:         {final_metrics.test_rmse:.6f}
Train NLPD:        {final_metrics.train_nlpd:.6f}
Test NLPD:         {final_metrics.test_nlpd:.6f}

MODEL ANALYSIS:
Overfitting Ratio: {summary.get('overfitting_ratio', float('inf')):.6f}
Effective Dims:    {final_metrics.effective_latent_dims}/{final_metrics.total_latent_dims}
Final Loss:        {final_metrics.elbo_loss:.6f}
Total Iterations:  {summary.get('total_iterations', 0)}{lengthscales_info}

PAPER-STYLE COMPARISON FORMAT:
Method     | Train RMSE | Test RMSE | Train NLPD | Test NLPD | Effective Dims
GP-LVM SSVI| {final_metrics.train_rmse:.4f}     | {final_metrics.test_rmse:.4f}    | {final_metrics.train_nlpd:.3f}     | {final_metrics.test_nlpd:.3f}     | {final_metrics.effective_latent_dims}/{final_metrics.total_latent_dims}

TRAINING PROGRESSION:
"""
    
    # Add training progression
    for i, metrics in enumerate(metrics_computer.metrics_history):
        text_content += f"Iter {metrics.iteration:4d}: Train RMSE={metrics.train_rmse:.4f}, Test RMSE={metrics.test_rmse:.4f}, "
        text_content += f"Train NLPD={metrics.train_nlpd:.3f}, Test NLPD={metrics.test_nlpd:.3f}, "
        text_content += f"Effective Dims={metrics.effective_latent_dims}/{metrics.total_latent_dims}\n"
    
    text_content += f"\n=== End of Summary ===\n"
    
    # Save to text file
    filepath = os.path.join(logger.results_dir, filename)
    with open(filepath, 'w') as f:
        f.write(text_content)
    
    logger.log_training(f"Text metrics saved to: {filename}")
    print(f"Text metrics saved to: {filepath}")
    
    return filepath







 