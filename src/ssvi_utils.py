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
from typing import Any


class SSVIResultsLogger:
    """Generic logger for SSVI GP-LVM results and gradients"""
    
    def __init__(self, config, experiment_name: str = "ssvi_gplvm"):
        self.config = config
        self.experiment_name = experiment_name
        
        if hasattr(config, 'save_results') and config.save_results:
            # Create results directory with timestamp
            timestamp = datetime.now().strftime("%d_%m_%H_%M")
            self.results_dir = f"results/{experiment_name}_results/{experiment_name}_res_{timestamp}"
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





def visualize_ssvi_results(model, y, labels, losses, config, logger: SSVIResultsLogger):
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
            
            # Plot samples from all P samples with transparency
            for p in range(model.P):
                # Use top 2 most informative latent dimensions
                U = model.get_u_samples()  # [P, M, C, Lx]
                mu_x_p, sigma_x_p = model.get_mu_sigma_x_samples(y, U)  # [N, A]
                mu_x_p = model.mu_x[p, :, [dim1, dim2]].cpu()  # [N, 2] - move to CPU for plotting
                plt.scatter(mu_x_p[:, 0], mu_x_p[:, 1], 
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
        logger.save_plots(fig, "training_results.png")
        plt.show()








 