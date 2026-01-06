#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k-NN Clustering Metrics Progression Plotting

This module provides utilities for plotting k-NN classification accuracy
over training iterations, similar to ELBO/RMSE/NLPD progression plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_knn_metrics_progression(knn_k1, knn_k3, knn_k7, knn_k12,
                                   iterations, save_dir, model_name, random_seed):
    """
    Plot k-NN accuracy progression over training iterations.
    
    Creates a single plot with 4 lines showing accuracy for k=1, 3, 7, 12.
    
    Parameters
    ----------
    knn_k1, knn_k3, knn_k7, knn_k12 : list of float
        Accuracies for k=1, 3, 7, 12
    iterations : list of int
        Iteration numbers corresponding to each metric
    save_dir : str
        Directory to save the plot
    model_name : str
        Model name for filename (e.g., 'gauss', 'nn_gauss', 'ood_ssvi')
    random_seed : int
        Random seed for filename
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define k values and their data
    k_values = [1, 3, 7, 12]
    accuracies = [knn_k1, knn_k3, knn_k7, knn_k12]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']
    
    # Plot each k value as a separate line
    for k, acc, color, marker in zip(k_values, accuracies, colors, markers):
        ax.plot(iterations, acc, color=color, marker=marker, 
                label=f'k={k}', linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('k-NN Classification Accuracy', fontsize=12)
    ax.set_title('k-NN Clustering Quality Progression', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show percentage range nicely
    all_acc = [a for acc_list in accuracies for a in acc_list if acc_list]
    if all_acc:
        min_acc = max(0.0, min(all_acc) - 0.05)
        max_acc = min(1.0, max(all_acc) + 0.05)
        ax.set_ylim([min_acc, max_acc])
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f'knn_metrics_progression_{model_name}_{random_seed}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"k-NN metrics progression plot saved to: {save_path}")


def plot_knn_metrics_progression_ssvi(knn_metrics_history, iterations, 
                                       plots_dir, random_seed):
    """
    Plot k-NN accuracy progression for SSVI model.
    
    Convenience wrapper that extracts metrics from history dictionary.
    
    Parameters
    ----------
    knn_metrics_history : list of dict
        List of k-NN metrics dictionaries at each checkpoint
    iterations : list of int
        Iteration numbers corresponding to each metric
    plots_dir : str
        Directory to save the plot (plots will be saved to plots/ subdirectory)
    random_seed : int
        Random seed for filename
    """
    # Extract metrics from history
    knn_k1 = [m.get('knn_1', 0.0) for m in knn_metrics_history]
    knn_k3 = [m.get('knn_3', 0.0) for m in knn_metrics_history]
    knn_k7 = [m.get('knn_7', 0.0) for m in knn_metrics_history]
    knn_k12 = [m.get('knn_12', 0.0) for m in knn_metrics_history]
    
    plot_knn_metrics_progression(
        knn_k1, knn_k3, knn_k7, knn_k12,
        iterations, plots_dir, 'ood_ssvi', random_seed
    )

