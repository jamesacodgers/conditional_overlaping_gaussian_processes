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
                results_folder = "orig_gplvm_results"
            else:
                results_folder = "ssvi_results"
                
            # Create results directory
            self.results_dir = f"results/{results_folder}/{experiment_name}_{timestamp}"
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Create plots subdirectory
            self.plots_dir = os.path.join(self.results_dir, "plots")
            os.makedirs(self.plots_dir, exist_ok=True)
            
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
        """Save matplotlib figure to plots subdirectory"""
        if self.results_dir:
            plot_path = os.path.join(self.plots_dir, filename)
            figure.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.log_training(f"Plot saved: plots/{filename}")
    
    def get_results_dir(self):
        """Get results directory path"""
        return self.results_dir





def visualize_ssvi_results(model, y, labels, losses, config, logger, save_filename: str = "training_results.png", show_plot: bool = True):
    """Visualize SSVI training results and latent space using standardized plotting"""
    
    # Use the standardized plotting function for consistency
    from src.ssvi_alignment_utils import create_standardized_scattered_plot
    
    # Create the standardized plot directly with custom filename
    create_standardized_scattered_plot(
        model, losses, labels, config, logger.plots_dir, 
        "ssvi", config.random_seed, y, model_type="ssvi", 
        custom_filename=save_filename
    )
    
    # Log the saved plot
    logger.log_training(f"Plot saved: plots/{save_filename}")
    
    if show_plot:
        # Display the plot
        import matplotlib.pyplot as plt
        final_plot_path = os.path.join(logger.plots_dir, save_filename)
        fig = plt.figure(figsize=(12, 4))
        img = plt.imread(final_plot_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        plt.close(fig)


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
    # Remove effective dimensions computation
    
    # Create subplots - only 3 plots, no plot 4
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
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
    
    
    plt.tight_layout()
    
    # Save plot if logger provided
    if logger:
        logger.save_plots(fig, "metrics_progression.png")
    
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









 