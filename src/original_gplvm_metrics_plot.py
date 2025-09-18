#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function to create metrics progression plot for original GP-LVM
"""

import matplotlib.pyplot as plt
import os

def plot_metrics_progression(train_rmse_list, train_nlpd_list, metrics_iterations, metrics_losses, 
                           results_dir, model_name, random_seed):
    """
    Plot metrics progression over training iterations (train metrics only)
    Exactly matches SSVI plot structure and styling
    
    Args:
        train_rmse_list: List of training RMSE values
        train_nlpd_list: List of training NLPD values  
        metrics_iterations: List of iteration numbers when metrics were computed
        metrics_losses: List of losses at the same iterations as metrics
        results_dir: Directory to save the plot
        model_name: Name of the model
        random_seed: Random seed for filename
    """
    if not train_rmse_list or not train_nlpd_list:
        print("No metrics data available for progression plot")
        return
    
    # Create subplots (exactly like SSVI)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: RMSE comparison (train only - exactly like SSVI but only train line)
    ax1.plot(metrics_iterations, train_rmse_list, 'b-', label='Train RMSE', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Reconstruction Error (RMSE)')  # Exact same title as SSVI
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: NLPD comparison (train only - exactly like SSVI but only train line)
    ax2.plot(metrics_iterations, train_nlpd_list, 'b-', label='Train NLPD', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('NLPD (lower is better)')
    ax2.set_title('Predictive Density (NLPD)')  # Exact same title as SSVI
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ELBO Loss progression (exactly like SSVI)
    ax3.plot(metrics_iterations, metrics_losses, 'g-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('ELBO Loss')
    ax3.set_title('Training Loss (ELBO)')  # Exact same title as SSVI
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'metrics_progression.png')
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics progression plot saved to: {plot_path}")
