"""
VAE Utilities for training, checkpointing, and visualization

Preserves SSVI output structure for easy comparison between models.
Uses VAE-specific metrics and plots (no NLPD, no missing data).
"""

import torch
import os
import json
import time
from typing import Dict, Any, Optional
from src.ssvi_utils import SSVIResultsLogger
from src.vae_metrics import VAEMetricsComputer
from src.vae_plots import visualize_vae_results, plot_vae_metrics_history, plot_vae_final_metrics_summary


def train_vae_with_metrics(
    vae_model,
    y_train: torch.Tensor,
    config,
    logger: Optional[SSVIResultsLogger] = None,
    split_data: Optional[Dict] = None,
    labels_train: Optional[torch.Tensor] = None
):
    """
    Training function for VAE with metrics, checkpoints, and plots.
    Preserves same structure as SSVI for easy comparison.
    
    Args:
        vae_model: VAE model instance
        y_train: (N, M) training observations
        config: VAEConfig instance
        logger: Results logger
        split_data: Dictionary with train/test split data
        labels_train: Training labels for visualization
        
    Returns:
        training_history: Dictionary with training metrics
        metrics_computer: SSVIMetricsComputer instance with evaluation metrics
        model_wrapper: VAEWrapper instance for further use
    """
    print(f"Training VAE...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")
    
    # Prepare optimizer
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=config.lr)
    
    # Tracking
    losses = []
    elbo_history = []
    kl_history = []
    ll_history = []
    rmse_history = []
    
    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(f"Metrics computation enabled (frequency: every {config.metrics_frequency} iterations)")
        metrics_computer = VAEMetricsComputer(device=config.device)
    
    start_time = time.time()
    
    for i in range(config.training_iterations):
        # Sample batch indices
        N = y_train.shape[0]
        batch_indices = torch.randperm(N, device=y_train.device)[:config.batch_size]
        y_batch = y_train[batch_indices]  # Flat vectors (batch, data_dim)
        
        # Training step
        optimizer.zero_grad()
        
        # Compute ELBO
        elbo, metrics = vae_model.elbo(
            y_batch, 
            num_samples=config.num_samples,
            beta_nll=config.beta_nll,
            beta=config.beta
        )
        
        # Loss is negative ELBO
        loss = -elbo.mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        elbo_val = metrics['elbo'] if 'elbo' in metrics else metrics.get('beta_nll_elbo', 0.0)
        elbo_history.append(elbo_val.item() if hasattr(elbo_val, 'item') else elbo_val)
        kl_history.append(metrics['kl'].item())
        ll_history.append(metrics['ll'].item())
        rmse_history.append(metrics['RMSE'].item())
        
        # Metrics computation (periodic) - same as SSVI
        if (metrics_computer is not None and 
            config.enable_metrics and 
            i % config.metrics_frequency == 0):
            
            print(f"  Computing metrics at iteration {i}...")
            
            # Use VAE-specific metrics (RMSE, ELBO, KL, LL - no NLPD)
            eval_metrics = metrics_computer.compute_full_metrics(
                vae_model=vae_model,
                y_train=split_data['y_train'],
                y_test=split_data['y_test'],
                iteration=i,
                elbo_loss=loss.item(),
                n_samples=32  # Fixed number of samples for metrics computation
            )
            
            print(f"    Train RMSE: {eval_metrics.train_rmse:.4f}, Test RMSE: {eval_metrics.test_rmse:.4f} (Ratio: {eval_metrics.test_rmse/eval_metrics.train_rmse:.3f})")
            print(f"    Train ELBO: {eval_metrics.train_elbo:.3f}, Test ELBO: {eval_metrics.test_elbo:.3f}")
            print(f"    Train KL: {eval_metrics.train_kl:.3f}, Test KL: {eval_metrics.test_kl:.3f}")
            print(f"    Full datasets: Train={split_data['y_train'].shape[0]}, Test={split_data['y_test'].shape[0]} samples")
            
            # Log metrics
            if logger:
                logger.log_training(f"Metrics at iteration {i}: {eval_metrics.to_dict()}")
        
        # Checkpoint saving (models and plots together) - same as SSVI
        if (logger and i > 0 and i % config.checkpoint_frequency == 0):
            print(f"  Saving checkpoint at iteration {i}...")
            try:
                # Save checkpoint model
                save_vae_checkpoint(
                    vae_model=vae_model,
                    optimizer=optimizer,
                    iteration=i,
                    loss=loss.item(),
                    config=config,
                    logger=logger
                )
                print(f"    Checkpoint model saved")
                
                # Save checkpoint plot (if enabled) - same as SSVI
                if config.save_training_plots:
                    save_vae_training_plot(
                        vae_model=vae_model,
                        y_train=y_train,
                        labels_train=labels_train,
                        losses=losses,
                        config=config,
                        logger=logger,
                        iteration=i,
                        split_data=split_data
                    )
                    print(f"    Checkpoint plot saved")
                
            except Exception as e:
                print(f"    Warning: Error saving checkpoint at iteration {i}: {e}")
        
        # Progress reporting
        if i % config.log_frequency == 0:
            elapsed = time.time() - start_time
            message = f"Iteration {i}/{config.training_iterations}: Loss={loss.item():.4f}, ELBO={elbo_history[-1]:.4f}, KL={kl_history[-1]:.4f}, LL={ll_history[-1]:.4f}, RMSE={rmse_history[-1]:.4f}, Time={elapsed:.1f}s"
            print(f"Iteration {i}/{config.training_iterations}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  ELBO: {elbo_history[-1]:.4f}, KL: {kl_history[-1]:.4f}, LL: {ll_history[-1]:.4f}")
            print(f"  RMSE: {rmse_history[-1]:.4f}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            # Log to file
            if logger:
                logger.log_training(message)
    
    total_time = time.time() - start_time
    final_message = f"Training completed! Final Loss: {losses[-1]:.4f}, Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    print(f"Training completed!")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Final ELBO: {elbo_history[-1]:.4f}")
    print(f"Final KL: {kl_history[-1]:.4f}, Final LL: {ll_history[-1]:.4f}")
    print(f"Final RMSE: {rmse_history[-1]:.4f}")
    print(f"Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)")
    # Log to file
    if logger:
        logger.log_training(final_message)
    
    # Package training history
    training_history = {
        'losses': losses,
        'elbo': elbo_history,
        'kl': kl_history,
        'll': ll_history,
        'rmse': rmse_history
    }
    
    return training_history, metrics_computer


def save_vae_checkpoint(
    vae_model,
    optimizer,
    iteration: int,
    loss: float,
    config,
    logger: SSVIResultsLogger
):
    """
    Save VAE checkpoint model.
    Preserves same structure as SSVI checkpoints.
    
    Args:
        vae_model: VAE model instance
        optimizer: Optimizer instance
        iteration: Current iteration
        loss: Current loss value
        config: VAEConfig instance
        logger: Results logger
    """
    checkpoint_filename = f"oilflow_vae_iter_{iteration}_seed_{config.random_seed}.pt"
    checkpoint_path = os.path.join(logger.models_checkpoints_dir, checkpoint_filename)
    
    torch.save({
        'iteration': iteration,
        'model_state_dict': vae_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }, checkpoint_path)


def save_vae_training_plot(
    vae_model,
    y_train: torch.Tensor,
    labels_train,
    losses: list,
    config,
    logger: SSVIResultsLogger,
    iteration: int,
    split_data: Dict
):
    """
    Save VAE training visualization plot.
    Uses VAE-specific visualization.
    
    Args:
        vae_model: VAE model instance
        y_train: Training observations
        labels_train: Training labels
        losses: List of training losses
        config: VAEConfig instance
        logger: Results logger
        iteration: Current iteration
        split_data: Dictionary with train/test split data
    """
    plot_filename = f"training_results_iter_{iteration}.png"
    visualize_vae_results(
        vae_model=vae_model,
        y_train=y_train,
        labels_train=labels_train,
        losses=losses,
        config=config,
        logger=logger,
        save_filename=plot_filename,
        show_plot=False,
        is_checkpoint=True,
        train_indices=split_data['train_indices']
    )


def save_vae_final_results(
    vae_model,
    training_history: Dict,
    config,
    logger: SSVIResultsLogger,
    metrics_computer: Optional[VAEMetricsComputer] = None,
    split_data: Optional[Dict] = None
):
    """
    Save VAE final results: model, training history, metrics, and plots.
    Preserves same structure as SSVI for easy comparison.
    Uses VAE-specific metrics and plots (no NLPD, no missing data).
    
    Args:
        vae_model: VAE model instance
        training_history: Dictionary with training metrics
        config: VAEConfig instance
        logger: Results logger
        metrics_computer: VAEMetricsComputer instance (optional)
        split_data: Dictionary with train/test split data (optional)
    """
    if logger is None:
        return
    
    # Save final model
    print("\nSaving final model...")
    final_model_filename = f"oilflow_vae_seed_{config.random_seed}.pt"
    final_model_path = os.path.join(logger.models_dir, final_model_filename)
    torch.save({
        'model_state_dict': vae_model.state_dict(),
        'config': config,
        'training_history': training_history,
    }, final_model_path)
    print(f"  Final model saved: {final_model_filename}")
    
    # Save training history
    print("Saving training history...")
    history_filename = f"training_history_seed_{config.random_seed}.json"
    history_path = os.path.join(logger.training_results_dir, history_filename)
    # Convert tensors to lists for JSON serialization
    history_dict = {
        'losses': training_history['losses'],
        'elbo': [float(e) if hasattr(e, 'item') else e for e in training_history['elbo']],
        'kl': training_history['kl'],
        'll': training_history['ll'],
        'rmse': training_history['rmse'],
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"  Training history saved: {history_filename}")
    
    # Save final metrics if available
    if metrics_computer is not None and split_data is not None:
        print("\nGenerating final metrics and plots...")
        
        # Compute final comprehensive metrics
        print("Computing final metrics on full datasets...")
        final_metrics = metrics_computer.compute_full_metrics(
            vae_model=vae_model,
            y_train=split_data['y_train'],
            y_test=split_data['y_test'],
            iteration=config.training_iterations,
            elbo_loss=training_history['losses'][-1] if training_history['losses'] else 0.0,
            n_samples=32
        )
        
        # Print final metrics summary
        summary = metrics_computer.get_metrics_summary()
        
        # Log the final metrics
        logger.log_training(f"Final comprehensive metrics computed: {final_metrics.to_dict()}")
        
        print(f"\n=== Final VAE Performance ===")
        print(f"Final Train RMSE: {summary['final_train_rmse']:.4f}")
        print(f"Final Test RMSE: {summary['final_test_rmse']:.4f}")
        print(f"Final Train ELBO: {summary['final_train_elbo']:.3f}")
        print(f"Final Test ELBO: {summary['final_test_elbo']:.3f}")
        print(f"Final Train KL: {summary['final_train_kl']:.3f}")
        print(f"Final Test KL: {summary['final_test_kl']:.3f}")
        print(f"Overfitting Ratio (Test/Train RMSE): {summary['overfitting_ratio']:.3f}")
        print("=" * 35)
        
        # Generate VAE-specific metrics plots
        print("Generating metrics progression plots...")
        plot_vae_metrics_history(metrics_computer, logger, show_plot=False)
        
        print("Generating final metrics summary plots...")
        plot_vae_final_metrics_summary(metrics_computer, vae_model, logger, show_plot=False)
        
        # Save checkpoint metrics and training losses
        print("Saving checkpoint metrics...")
        logger.save_checkpoint_metrics(metrics_computer, "oilflow_vae", config.random_seed)
        print("Saving training losses...")
        logger.save_training_losses(training_history['losses'], "oilflow_vae", config.random_seed)
    
    # Save experiment summary
    experiment_summary = {
        'oilflow_vae': {
            'final_loss': training_history['losses'][-1],
            'final_elbo': float(training_history['elbo'][-1]) if hasattr(training_history['elbo'][-1], 'item') else training_history['elbo'][-1],
            'final_kl': training_history['kl'][-1],
            'final_ll': training_history['ll'][-1],
            'final_rmse': training_history['rmse'][-1],
            'latent_dim': config.latent_dim,
            'architecture': "mlp_tabular",
            }
        }
    
    # Add metrics if available
    if metrics_computer is not None and metrics_computer.metrics_history:
        summary = metrics_computer.get_metrics_summary()
        experiment_summary['oilflow_vae'].update({
            'train_rmse': summary.get('final_train_rmse', 0.0),
            'test_rmse': summary.get('final_test_rmse', 0.0),
            'train_nlpd': summary.get('final_train_nlpd', 0.0),
            'test_nlpd': summary.get('final_test_nlpd', 0.0),
        })
    
    logger.save_experiment_summary(experiment_summary)
    print("Experiment summary saved")

