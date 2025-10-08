"""
VAE-specific Plotting and Visualization

Tailored plots for VAE models that preserve the same output structure as SSVI.
No NLPD or missing data plots - those are for GP models.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, List
from src.vae_metrics import VAEMetricsComputer


def visualize_vae_results(
    vae_model,
    y_train: torch.Tensor,
    labels_train,
    losses: List[float],
    config,
    logger,
    save_filename: str = "training_results.png",
    show_plot: bool = True,
    is_checkpoint: bool = False,
    train_indices: Optional[torch.Tensor] = None
):
    """
    Visualize VAE training results.
    Preserves same structure as SSVI visualization.
    
    Args:
        vae_model: VAE model instance
        y_train: Training observations
        labels_train: Training labels
        losses: List of training losses
        config: VAEConfig instance
        logger: Results logger
        save_filename: Filename for saving plot
        show_plot: Whether to display plot
        is_checkpoint: Whether this is a checkpoint plot
        train_indices: Training indices (for consistent ordering)
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Loss curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Latent space visualization (first 2 dimensions)
    if config.latent_dim >= 2:
        ax2 = plt.subplot(2, 3, 2)
        
        # Encode training data to latent space
        with torch.no_grad():
            # MLP VAE accepts flat vectors directly
            q = vae_model.encode(y_train)
            z_mean = q.mean
            z_mean_np = z_mean.cpu().numpy()
        
        # Handle labels (could be tensor or numpy)
        if hasattr(labels_train, 'cpu'):
            labels_np = labels_train.cpu().numpy()
        else:
            labels_np = np.array(labels_train)
        
        # Plot first 2 latent dimensions
        scatter = ax2.scatter(
            z_mean_np[:, 0],
            z_mean_np[:, 1],
            c=labels_np,
            cmap='viridis',
            alpha=0.6,
            s=30
        )
        plt.colorbar(scatter, ax=ax2, label='Class')
        ax2.set_xlabel('Latent Dim 1', fontsize=12)
        ax2.set_ylabel('Latent Dim 2', fontsize=12)
        ax2.set_title('Latent Space (2D)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. Reconstruction quality (random samples)
    ax3 = plt.subplot(2, 3, 3)
    
    # Get reconstructions for a batch
    with torch.no_grad():
        batch_size = min(100, y_train.shape[0])
        y_batch = y_train[:batch_size]
        # MLP VAE accepts flat vectors directly
        _, _, l = vae_model(y_batch, num_samples=1)
        y_pred = l.loc.mean(dim=1)  # (batch, num_samples, data_dim) → (batch, data_dim)
        
        # Compute residuals
        residuals = (y_batch - y_pred).cpu().numpy()
        
        # Plot residuals distribution
        ax3.hist(residuals.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('Reconstruction Error', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Latent dimensions variance (uncertainty)
    ax4 = plt.subplot(2, 3, 4)
    
    with torch.no_grad():
        # Get latent representations with uncertainties
        q = vae_model.encode(y_train)
        z_std = torch.sqrt(torch.diagonal(q.covariance_matrix, dim1=-2, dim2=-1))
        z_std_np = z_std.cpu().numpy()
        
        # Average std per latent dimension
        avg_std_per_dim = z_std_np.mean(axis=0)
        
        ax4.bar(range(config.latent_dim), avg_std_per_dim, color='steelblue', alpha=0.7)
        ax4.set_xlabel('Latent Dimension', fontsize=12)
        ax4.set_ylabel('Average Std Dev', fontsize=12)
        ax4.set_title('Latent Uncertainty per Dimension', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Sample predictions vs actual (first few dimensions)
    ax5 = plt.subplot(2, 3, 5)
    
    n_samples = min(5, y_train.shape[0])
    n_dims_to_plot = min(5, y_train.shape[1])
    
    with torch.no_grad():
        y_sample = y_train[:n_samples]
        # MLP VAE accepts flat vectors directly
        _, _, l = vae_model(y_sample, num_samples=1)
        y_pred_sample = l.loc.mean(dim=1)  # (batch, num_samples, data_dim) → (batch, data_dim)
        
        y_sample_np = y_sample.cpu().numpy()
        y_pred_np = y_pred_sample.cpu().numpy()
        
        x_pos = np.arange(n_dims_to_plot)
        width = 0.35
        
        for i in range(n_samples):
            offset = (i - n_samples/2) * width / n_samples
            ax5.plot(x_pos + offset, y_sample_np[i, :n_dims_to_plot], 'o-', 
                    label=f'True {i+1}', alpha=0.6, markersize=4)
            ax5.plot(x_pos + offset, y_pred_np[i, :n_dims_to_plot], 's--', 
                    label=f'Pred {i+1}', alpha=0.6, markersize=4)
        
        ax5.set_xlabel('Dimension', fontsize=12)
        ax5.set_ylabel('Value', fontsize=12)
        ax5.set_title(f'Predictions vs Actual (first {n_dims_to_plot} dims)', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=8, ncol=2)
        ax5.grid(True, alpha=0.3)
    
    # 6. Training info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    info_text = [
        f"VAE-MLP Training Results",
        f"",
        f"Configuration:",
        f"  Latent Dim: {config.latent_dim}",
        f"  Data Dim: {config.data_dim}",
        f"  Batch Size: {config.batch_size}",
        f"  Learning Rate: {config.lr:.4f}",
        f"",
        f"Architecture:",
        f"  MLP Encoder: {config.enc_mlp_hidden_dims}",
        f"  MLP Decoder: {config.dec_mlp_hidden_dims}",
        f"",
        f"Training:",
        f"  Iterations: {len(losses)}",
        f"  Final Loss: {losses[-1]:.4f}",
        f"",
        f"Data:",
        f"  Train Samples: {y_train.shape[0]}",
    ]
    
    ax6.text(0.1, 0.95, '\n'.join(info_text), 
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    if logger and logger.results_dir:
        if is_checkpoint:
            save_path = os.path.join(logger.plots_checkpoints_dir, save_filename)
        else:
            save_path = os.path.join(logger.plots_dir, save_filename)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_vae_metrics_history(
    metrics_computer: VAEMetricsComputer,
    logger,
    show_plot: bool = False
):
    """
    Plot metrics progression over training.
    Preserves same structure as SSVI metrics plots.
    
    Args:
        metrics_computer: VAEMetricsComputer instance
        logger: Results logger
        show_plot: Whether to display plot
    """
    if not metrics_computer.metrics_history:
        print("No metrics history to plot")
        return
    
    # Extract metrics
    iterations = [m.iteration for m in metrics_computer.metrics_history]
    train_rmse = [m.train_rmse for m in metrics_computer.metrics_history]
    test_rmse = [m.test_rmse for m in metrics_computer.metrics_history]
    train_elbo = [m.train_elbo for m in metrics_computer.metrics_history]
    test_elbo = [m.test_elbo for m in metrics_computer.metrics_history]
    train_kl = [m.train_kl for m in metrics_computer.metrics_history]
    test_kl = [m.test_kl for m in metrics_computer.metrics_history]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE
    axes[0, 0].plot(iterations, train_rmse, 'b-', label='Train RMSE', linewidth=2)
    axes[0, 0].plot(iterations, test_rmse, 'r-', label='Test RMSE', linewidth=2)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('RMSE', fontsize=12)
    axes[0, 0].set_title('RMSE Progression', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # ELBO
    axes[0, 1].plot(iterations, train_elbo, 'b-', label='Train ELBO', linewidth=2)
    axes[0, 1].plot(iterations, test_elbo, 'r-', label='Test ELBO', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('ELBO', fontsize=12)
    axes[0, 1].set_title('ELBO Progression', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[1, 0].plot(iterations, train_kl, 'b-', label='Train KL', linewidth=2)
    axes[1, 0].plot(iterations, test_kl, 'r-', label='Test KL', linewidth=2)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('KL Divergence', fontsize=12)
    axes[1, 0].set_title('KL Divergence Progression', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting ratio
    overfitting_ratio = [test / train if train > 0 else 1.0 for train, test in zip(train_rmse, test_rmse)]
    axes[1, 1].plot(iterations, overfitting_ratio, 'g-', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect fit')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Test/Train RMSE Ratio', fontsize=12)
    axes[1, 1].set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if logger and logger.results_dir:
        save_path = os.path.join(logger.plots_dir, 'metrics_progression.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_vae_final_metrics_summary(
    metrics_computer: VAEMetricsComputer,
    vae_model,
    logger,
    show_plot: bool = False
):
    """
    Plot final metrics summary.
    Preserves same structure as SSVI final metrics plots.
    
    Args:
        metrics_computer: VAEMetricsComputer instance
        vae_model: VAE model instance
        logger: Results logger
        show_plot: Whether to display plot
    """
    summary = metrics_computer.get_metrics_summary()
    
    if not summary:
        print("No metrics summary to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Final metrics comparison
    ax1 = axes[0]
    metrics_names = ['RMSE', 'ELBO', 'KL']
    train_values = [
        summary['final_train_rmse'],
        summary['final_train_elbo'],
        summary['final_train_kl']
    ]
    test_values = [
        summary['final_test_rmse'],
        summary['final_test_elbo'],
        summary['final_test_kl']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    # Normalize for visualization (so different scales can be compared)
    train_norm = np.array(train_values) / np.max(np.abs(train_values))
    test_norm = np.array(test_values) / np.max(np.abs(test_values))
    
    ax1.bar(x - width/2, train_norm, width, label='Train (normalized)', color='steelblue', alpha=0.7)
    ax1.bar(x + width/2, test_norm, width, label='Test (normalized)', color='coral', alpha=0.7)
    
    ax1.set_ylabel('Normalized Value', fontsize=12)
    ax1.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add actual values as text
    for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
        ax1.text(i - width/2, train_norm[i] + 0.05, f'{train_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, test_norm[i] + 0.05, f'{test_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Summary statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    summary_text = [
        "Final VAE Performance Summary",
        "",
        "Reconstruction Quality:",
        f"  Train RMSE: {summary['final_train_rmse']:.4f}",
        f"  Test RMSE:  {summary['final_test_rmse']:.4f}",
        f"  Overfitting Ratio: {summary['overfitting_ratio']:.3f}",
        "",
        "ELBO (Evidence Lower Bound):",
        f"  Train ELBO: {summary['final_train_elbo']:.4f}",
        f"  Test ELBO:  {summary['final_test_elbo']:.4f}",
        "",
        "KL Divergence (Regularization):",
        f"  Train KL: {summary['final_train_kl']:.4f}",
        f"  Test KL:  {summary['final_test_kl']:.4f}",
        "",
        "Log Likelihood:",
        f"  Train LL: {summary['final_train_ll']:.4f}",
        f"  Test LL:  {summary['final_test_ll']:.4f}",
    ]
    
    if 'best_test_rmse' in summary:
        summary_text.extend([
            "",
            "Training Progression:",
            f"  Best Test RMSE: {summary['best_test_rmse']:.4f}",
            f"  Worst Test RMSE: {summary['worst_test_rmse']:.4f}",
        ])
    
    ax2.text(0.1, 0.95, '\n'.join(summary_text),
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    if logger and logger.results_dir:
        save_path = os.path.join(logger.plots_dir, 'final_metrics_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()

