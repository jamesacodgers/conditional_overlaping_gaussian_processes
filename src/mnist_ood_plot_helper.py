#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for MNIST/FashionMNIST OOD GP-LVM visualization

DUPLICATE OF mnist_plot_helper.py with ONLY legend changes for OOD detection.
Legend shows dataset name (MNIST/FashionMNIST) and ID/OOD status.
"""

import sys
import os
import torch
import numpy as np
import pickle as pkl
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# Add generalised_gplvm_code to path for original utilities
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

# Import SSVI utilities
from src.ssvi_utils import SSVIResultsLogger

# FashionMNIST class names for legend
FASHION_MNIST_CLASSES = {
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def get_label_name(label_val, dataset_name, is_ood=False):
    """
    Get legend label name - just the class name without dataset prefix or ID/OOD status.
    
    Parameters
    ----------
    label_val : int
        The class label (0-9)
    dataset_name : str
        'MNIST' or 'FashionMNIST'
    is_ood : bool
        Whether this is OOD data (ignored, kept for compatibility)
        
    Returns
    -------
    str
        Just the class name like "Trouser" or "1"
    """
    if dataset_name == "MNIST":
        return str(int(label_val))
    elif dataset_name == "FashionMNIST":
        return FASHION_MNIST_CLASSES.get(int(label_val), f'Class {int(label_val)}')
    else:
        return f'Class {int(label_val)}'


def get_consistent_mnist_colors(labels_train):
    """
    Get consistent color mapping for MNIST digit labels (0-9) across SSVI and original GP LVM.
    Ensures the same digit always gets the same color regardless of which plotting function is used.
    """
    # Use tab10 colormap for 10 digit classes (0-9)
    cmap_colors = plt.get_cmap("tab10").colors
    
    # Create consistent color mapping based on unique labels
    unique_labels = np.unique(labels_train)
    color_map = {}
    
    # Assign colors in order of unique labels to ensure consistency
    for i, label in enumerate(sorted(unique_labels)):
        color_map[label] = cmap_colors[i % len(cmap_colors)]
    
    # Return colors in the same order as labels_train
    colors = [color_map[label] for label in labels_train]
    return colors, color_map


def matplotlib_to_plotly_color(matplotlib_color):
    """Convert matplotlib RGB tuple to Plotly format"""
    if isinstance(matplotlib_color, tuple) and len(matplotlib_color) == 3:
        r, g, b = matplotlib_color
        return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
    return matplotlib_color


def create_standardized_scattered_plot(model, losses, labels_train, config, results_dir, model_name, random_seed,
                                       Y_train, model_type="original", custom_filename=None, train_indices=None,
                                       dataset_name="MNIST", is_ood=False, full_loss_iterations: list = None, batch_losses: list = None):
    """
    Create standardized scattered plots that match the SSVI plotting exactly.

    This function replicates the exact structure and styling of the SSVI visualization
    from src/ssvi_utils.py with the same subplot order and styling.
    
    Parameters
    ----------
    losses : list
        Full ELBO losses (negative ELBO values)
    full_loss_iterations : list, optional
        Iteration numbers corresponding to losses. If provided, used for x-axis of full ELBO plot.
    batch_losses : list, optional
        Batch ELBO losses (one per training iteration). If provided, creates a plot showing batch ELBO progression.
    dataset_name : str
        'MNIST' or 'FashionMNIST' - used for legend labels
    is_ood : bool
        Whether this is OOD data - used for legend labels
    """

    # Use default matplotlib style to match SSVI (override any global ggplot style)
    plt.style.use('default')

    with torch.no_grad():
        # Color mapping for MNIST labels (digits 0-9)
        # MNIST labels are integers 0, 1, 2, ..., 9 from load_real_data('mnist')
        # Use consistent color mapping across SSVI and original GP LVM
        colors, color_map = get_consistent_mnist_colors(labels_train)

        # Determine which plots to show
        plot_batch_elbo = batch_losses is not None and len(batch_losses) > 0
        n_plots = 1 + (1 if plot_batch_elbo else 0) + 1 + 1  # Full ELBO, Batch ELBO (optional), Latent space, Hyperparams
        fig_width = 4 * n_plots
        fig = plt.figure(figsize=(fig_width, 4))

        plot_idx = 1

        # Plot 1: Full ELBO Training curve (always shown)
        plt.subplot(1, n_plots, plot_idx)
        if full_loss_iterations is not None and len(full_loss_iterations) == len(losses):
            # Use iteration numbers on x-axis
            plt.plot(full_loss_iterations, losses, 'b-', alpha=0.7)
        else:
            # Fallback: use indices if iterations not provided
            plt.plot(losses, 'b-', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('ELBO Training Curve (Full ELBO)')
        plt.grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 2: Batch ELBO progression (if batch_losses provided) - second position
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

        # Plot 3: Latent space visualization (exactly like SSVI)
        plt.subplot(1, n_plots, plot_idx)

        # Find the two most informative dimensions (copy original approach exactly)
        inv_lengthscale = 1 / model.covar_module.base_kernel.lengthscale
        values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=2, largest=False)
        dim1 = indices.cpu().numpy().flatten()[0]  # Move to CPU first
        dim2 = indices.cpu().numpy().flatten()[1]  # Move to CPU first
        # For consistency with SSVI, use sqrt(lengthscale) for axis labels too
        lengthscales_cpu = torch.sqrt(inv_lengthscale.flatten()).cpu()  # Move to CPU for string formatting

        # Get latent variables (use mu_x directly, no sampling)
        model_name_class = model.X.__class__.__name__
        if model_name_class == 'VariationalLatentVariable':  # 'gauss' model
            # Variational GP-LVM (gauss)
            mu_x = model.X.q_mu.detach().cpu().numpy()
            sigma_x = torch.nn.functional.softplus(model.X.q_log_sigma).detach().cpu().numpy()
        elif model_name_class == 'NNEncoder':  # 'nn_gauss' model
            Y_train_device = Y_train.to(model.X.mu_layers[0].weight.device) if Y_train.device != model.X.mu_layers[
                0].weight.device else Y_train

            # Handle missing data for nn_gauss plotting (same as training)
            if torch.isnan(Y_train_device).any():
                # Replace NaN values with zeros for encoder input (consistent with training)
                Y_train_for_plotting = Y_train_device.clone()
                Y_train_for_plotting[torch.isnan(Y_train_for_plotting)] = -999.0
            else:
                Y_train_for_plotting = Y_train_device

            # Process data in minibatches to reduce memory usage
            batch_size = getattr(config, 'batch_size', 128)  # Use config batch_size if available, otherwise default to 128
            N = Y_train_for_plotting.shape[0]
            mu_x_batches = []
            sigma_x_batches = []
            
            # Store original jitter to restore later (same pattern as test prediction)
            original_jitter = model.X.jitter
            latent_dim = model.X.latent_dim
            device = original_jitter.device
            dtype = original_jitter.dtype
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = Y_train_for_plotting[start_idx:end_idx]
                batch_size_actual = end_idx - start_idx
                
                # Create jitter for batch (same pattern as test prediction)
                batch_jitter = torch.eye(latent_dim, dtype=dtype, device=device).unsqueeze(0) * 1e-5
                model.X.jitter = torch.cat([batch_jitter for _ in range(batch_size_actual)], axis=0)
                
                mu_x_batch = model.X.mu(y_batch)
                sigma_covariances_batch = model.X.sigma(y_batch)
                
                # Extract standard deviations from covariance matrices (same as original code)
                # sigma_covariances_batch is [batch_size, latent_dim, latent_dim]
                # Extract diagonal for each covariance matrix and take sqrt
                sigma_x_batch = torch.stack([torch.sqrt(cov.diag()) for cov in sigma_covariances_batch])
                
                mu_x_batches.append(mu_x_batch.detach().cpu())
                sigma_x_batches.append(sigma_x_batch.detach().cpu())
            
            # Restore original jitter
            model.X.jitter = original_jitter
            
            # Concatenate all batches
            mu_x_result = torch.cat(mu_x_batches, dim=0)
            sigma_x_all = torch.cat(sigma_x_batches, dim=0)
            
            # Convert to numpy (matching original code format)
            mu_x = mu_x_result.numpy()
            sigma_x = sigma_x_all.numpy()

        # Plot mu_x directly (no sampling) - exactly like SSVI
        mu_x_torch = torch.from_numpy(mu_x).to(dtype=torch.float64)
        sigma_x_torch = torch.from_numpy(sigma_x).to(dtype=torch.float64)

        # Sample P_samples times using torch.distributions.Normal (same as SSVI)
        for p in range(64):
            sample_1 = torch.distributions.Normal(mu_x_torch[:, dim1], sigma_x_torch[:, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_torch[:, dim2], sigma_x_torch[:, dim2]).sample()
            plt.scatter(sample_1.detach().cpu().numpy(),
                        sample_2.detach().cpu().numpy(), alpha=0.6, color=colors)

        plt.xlabel(f'Latent Dimension {dim1} (lengthscale={lengthscales_cpu[dim1]:.2f})')
        plt.ylabel(f'Latent Dimension {dim2} (lengthscale={lengthscales_cpu[dim2]:.2f})')
        plt.title(f'Latent Space - Top 2 Informative Dims')
        # Add legend with class names below the plot in a horizontal line
        labels_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train
        from matplotlib.patches import Patch
        legend_handles = [Patch(color=color_map[l], label=get_label_name(l, dataset_name, is_ood)) 
                          for l in sorted(np.unique(labels_cpu))]
        plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=len(legend_handles), fontsize='small', frameon=True)
        plot_idx += 1

        # Plot 4: Hyperparameters (lengthscales) - consistent with SSVI
        plt.subplot(1, n_plots, plot_idx)
        # For mathematical consistency with SSVI version:
        # SSVI: K = exp(-0.5 * d^2 / init_lengthscales)
        # GP-LVM: K = exp(-0.5 * d^2 / lengthscale^2)
        # For equivalence: lengthscale = sqrt(init_lengthscales)
        lengthscales_plot = 1 / model.covar_module.base_kernel.lengthscale.flatten().cpu()
        lengthscales_plot = lengthscales_plot**2
        plt.bar(range(len(lengthscales_plot)), lengthscales_plot,
                color="blue", alpha=0.7)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Lengthscale')
        plt.title('GP Lengthscales')
        plt.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Use model name (gauss/nn_gauss) instead of class name for consistency
    if model_name == 'VariationalLatentVariable':
        model_short_name = 'gauss'
    elif model_name == 'NNEncoder':
        model_short_name = 'nn_gauss'
    else:
        model_short_name = model_name.lower()

    # Use custom filename if provided, otherwise use default naming convention
    if custom_filename:
        plot_filename = custom_filename
    else:
        plot_filename = f"mnist_{model_short_name}_{random_seed}.png"

    # Create plots subdirectory only if we're not already in a plots directory
    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        # We're already in a plots directory or checkpoints subdirectory, save directly here
        plot_path = os.path.join(results_dir, plot_filename)
    else:
        # Create plots subdirectory for better organization
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        plot_path = os.path.join(plots_subdir, plot_filename)

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Standardized scattered plots saved to: {plot_path}")

    # Additional plots: 1D histogram and 3D scatter plots
    print("Creating additional plots: 1D histogram and 3D scatter plots...")

    # Ensure labels_train is on CPU for numpy operations
    labels_train_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train

    # Reuse dimensions from main plot (consistent with existing approach)
    # dim1, dim2, and lengthscales_cpu are already computed in the main plot section
    # Just need to compute dim3 for 3D plots
    values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=3, largest=False)
    dim3 = indices.cpu().numpy().flatten()[2]  # Third dimension for 3D plots

    # 1. Create 1D histogram for the most important dimension
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 6))

    # Collect samples from all P samples for the most important dimension
    all_samples = []
    all_labels = []
    for p in range(64):  # Same number of samples as main plot
        samples = torch.distributions.Normal(mu_x_torch[:, dim1], sigma_x_torch[:, dim1]).sample()
        # Ensure samples are on CPU before converting to numpy
        all_samples.append(samples.detach().cpu().numpy())
        # Repeat labels for each P sample
        all_labels.append(labels_train_cpu)

    all_samples = np.concatenate(all_samples)
    all_labels = np.concatenate(all_labels)

    # Create histogram with different colors for different classes (using consistent color mapping)
    _, color_map = get_consistent_mnist_colors(labels_train_cpu)
    unique_labels = np.unique(all_labels)
    
    for label_val in unique_labels:
        class_mask = all_labels == label_val
        if np.any(class_mask):
            class_samples = all_samples[class_mask]
            # CHANGED: Use get_label_name for OOD-aware legend
            ax_hist.hist(class_samples, bins=30, alpha=0.7,
                         label=get_label_name(label_val, dataset_name, is_ood),
                         color=color_map[label_val])

    ax_hist.set_xlabel(f'Latent Dimension {dim1} (Most Important)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'1D Histogram - Most Important Dimension {dim1}')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # Save histogram plot
    hist_filename = plot_filename.replace('mnist_', '1d_histogram_most_important_dim_')
    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        hist_plot_path = os.path.join(results_dir, hist_filename)
    else:
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        hist_plot_path = os.path.join(plots_subdir, hist_filename)

    fig_hist.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved 1D histogram to: {hist_plot_path}")
    plt.close(fig_hist)

    # 2. Create 3D scatter plot for top 3 most important dimensions using plotly
    # Collect samples for all 3 dimensions
    samples_3d = []
    for p in range(64):  # Same number of samples as main plot
        sample_1 = torch.distributions.Normal(mu_x_torch[:, dim1], sigma_x_torch[:, dim1]).sample()
        sample_2 = torch.distributions.Normal(mu_x_torch[:, dim2], sigma_x_torch[:, dim2]).sample()
        sample_3 = torch.distributions.Normal(mu_x_torch[:, dim3], sigma_x_torch[:, dim3]).sample()

        # Ensure all samples are on CPU before converting to numpy
        samples_3d.append({
            'dim1': sample_1.detach().cpu().numpy(),
            'dim2': sample_2.detach().cpu().numpy(),
            'dim3': sample_3.detach().cpu().numpy(),
            'labels': labels_train_cpu
        })

    # Create plotly 3D scatter plot
    fig_3d = go.Figure()

    # Color mapping for labels (using consistent color mapping)
    _, color_map = get_consistent_mnist_colors(labels_train_cpu)
    unique_labels = np.unique(labels_train_cpu)

    # Add traces for each class
    for label_val in unique_labels:
        class_mask = labels_train_cpu == label_val
        if np.any(class_mask):
            # Combine samples from all P samples for this class
            x_vals = np.concatenate([samples_3d[p]['dim1'][class_mask] for p in range(64)])
            y_vals = np.concatenate([samples_3d[p]['dim2'][class_mask] for p in range(64)])
            z_vals = np.concatenate([samples_3d[p]['dim3'][class_mask] for p in range(64)])

            fig_3d.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=4,
                    color=matplotlib_to_plotly_color(color_map[label_val]),
                    opacity=0.6
                ),
                # CHANGED: Use get_label_name for OOD-aware legend
                name=get_label_name(label_val, dataset_name, is_ood)
            ))

    # Update layout
    fig_3d.update_layout(
        title=f'3D Scatter Plot - Top 3 Most Important Dimensions<br>Dim {dim1}, {dim2}, {dim3}',
        scene=dict(
            xaxis_title=f'Dimension {dim1} (lengthscale={lengthscales_cpu[dim1].item():.2f})',
            yaxis_title=f'Dimension {dim2} (lengthscale={lengthscales_cpu[dim2].item():.2f})',
            zaxis_title=f'Dimension {dim3} (lengthscale={lengthscales_cpu[dim3].item():.2f})'
        ),
        width=800,
        height=600
    )

    # Save 3D plot as HTML
    plot_3d_filename = plot_filename.replace('mnist_', '3d_scatter_top3_dims_')
    plot_3d_filename = plot_3d_filename.replace('.png', '.html')

    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        plot_3d_path = os.path.join(results_dir, plot_3d_filename)
    else:
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        plot_3d_path = os.path.join(plots_subdir, plot_3d_filename)

    fig_3d.write_html(plot_3d_path)
    print(f"Saved 3D scatter plot to: {plot_3d_path}")

    print("Additional plots created successfully!")


def save_iteration_model(model, model_name, iteration, random_seed, results_dir):
    """
    Save model at specific iteration to pre_trained_models/checkpoints folder
    """
    # Save to pre_trained_models/checkpoints folder for organized checkpoint storage
    pre_trained_dir = os.path.join(results_dir, 'pre_trained_models', 'checkpoints')
    os.makedirs(pre_trained_dir, exist_ok=True)

    filename = f'mnist_{model_name}_iter_{iteration}_seed_{random_seed}.pkl'
    filepath = os.path.join(pre_trained_dir, filename)

    with open(filepath, 'wb') as file:
        pkl.dump(model.state_dict(), file)

    return filepath


def create_iteration_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, iteration,
                          model_type="original", train_indices=None, dataset_name="MNIST", is_ood=False,
                          full_loss_iterations: list = None, batch_losses: list = None):
    """
    Create iteration-based plot saved to plots/checkpoints folder
    
    Parameters
    ----------
    losses : list
        Full ELBO losses (negative ELBO values)
    full_loss_iterations : list, optional
        Iteration numbers corresponding to losses. If provided, used for x-axis of full ELBO plot.
    batch_losses : list, optional
        Batch ELBO losses (one per training iteration). If provided, creates a plot showing batch ELBO progression.
    dataset_name : str
        'MNIST' or 'FashionMNIST' - used for legend labels
    is_ood : bool
        Whether this is OOD data - used for legend labels
        
    Returns
    -------
    str
        plot_filename - filename of saved plot
    """
    plot_filename = f"mnist_{model_name}_iter_{iteration}_seed_{random_seed}.png"

    # Create plots/checkpoints subdirectory for organized checkpoint storage
    plots_checkpoints_dir = os.path.join(results_dir, 'plots', 'checkpoints')
    os.makedirs(plots_checkpoints_dir, exist_ok=True)

    # Create the standardized plot with custom filename in checkpoints folder
    create_standardized_scattered_plot(
        model, losses, labels_train, config, plots_checkpoints_dir,
        model_name, random_seed, Y_train, model_type=model_type,
        custom_filename=plot_filename, train_indices=train_indices,
        dataset_name=dataset_name, is_ood=is_ood,
        full_loss_iterations=full_loss_iterations, batch_losses=batch_losses
    )

    return plot_filename


def visualize_ssvi_results(model, y_train, labels_train, losses, config, logger: SSVIResultsLogger,
                           save_filename: str = "training_results.png", show_plot: bool = True,
                           n_optimization_steps: int = None, is_test_data: bool = False, is_checkpoint: bool = False,
                           train_indices: torch.Tensor = None, dataset_name: str = "MNIST", is_ood: bool = False,
                           full_elbo_iterations: list = None, batch_losses: list = None):
    """
    Visualize SSVI training results and latent space
    
    Parameters
    ----------
    losses : list
        Full ELBO losses (negative ELBO values)
    full_elbo_iterations : list, optional
        Iteration numbers corresponding to full_elbo_losses. If provided, used for x-axis of full ELBO plot.
    batch_losses : list, optional
        Batch ELBO losses (one per training iteration). If provided, creates a 4th plot showing batch ELBO progression.
    dataset_name : str
        'MNIST' or 'FashionMNIST' - used for legend labels
    is_ood : bool
        Whether this is OOD data - used for legend labels
    """

    if not (hasattr(config, 'plot_training') and config.plot_training or
            hasattr(config, 'plot_latent_space') and config.plot_latent_space or
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams):
        return

    with torch.no_grad():
        # Color mapping for MNIST labels (digits 0-9)
        # MNIST labels are integers 0, 1, 2, ..., 9 from load_real_data('mnist')
        # Use consistent color mapping across SSVI and original GP LVM
        colors, color_map = get_consistent_mnist_colors(labels_train)

        # Determine which plots to show
        plot_latent_space = hasattr(config, 'plot_latent_space') and config.plot_latent_space
        plot_hyperparams = hasattr(config, 'plot_hyperparams') and config.plot_hyperparams
        plot_batch_elbo = batch_losses is not None and len(batch_losses) > 0
        n_plots = 1 + (1 if plot_latent_space else 0) + (1 if plot_hyperparams else 0) + (1 if plot_batch_elbo else 0)
        fig_width = 4 * n_plots
        fig = plt.figure(figsize=(fig_width, 4))

        plot_idx = 1

        # Plot 1: Full ELBO Training curve (always shown)
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

        # Plot 2: Batch ELBO progression (if batch_losses provided) - second position
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

        # Plot 3: Latent space visualization
        if plot_latent_space:
            plt.subplot(1, n_plots, plot_idx)

            # Find the most informative dimensions (largest lengthscales = smallest beta)
            lengthscales = 1 / model.gp.beta.flatten()
            top_2_dims = torch.argsort(lengthscales, descending=True)[:2]
            top_3_dims = torch.argsort(lengthscales, descending=True)[:3]  # Also compute top 3 for 3D plots
            dim1, dim2 = top_2_dims[0].item(), top_2_dims[1].item()
            dim3 = top_3_dims[2].item()  # Third dimension for 3D plots
            lengthscales_cpu = lengthscales.cpu()  # Move to CPU for string formatting
            
            # Get latent variables using batched processing to reduce memory usage
            batch_size = getattr(config, 'batch_size', 128)  # Use config batch_size if available, otherwise default to 128
            epsilon = model.get_epsilon()
            U_samples = model.get_u_samples(epsilon)  # [P, M, C, Lx] - Sample U once (shared across all batches)
            
            # Process data in minibatches
            N = y_train.shape[0]
            mu_x_batches = []
            sigma_x_batches = []
            
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                y_batch = y_train[start_idx:end_idx]
                
                # Handle missing data if enabled
                if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                    y_batch_clean = torch.where(
                        torch.isnan(y_batch),
                        torch.full_like(y_batch, -9.0, dtype=torch.float64),
                        y_batch
                    )
                    mu_x_batch, sigma_x_batch = model.get_mu_sigma_x_train_samples(y_batch_clean, epsilon)
                else:
                    mu_x_batch, sigma_x_batch = model.get_mu_sigma_x_train_samples(y_batch, epsilon)
                
                mu_x_batches.append(mu_x_batch)
                sigma_x_batches.append(sigma_x_batch)
            
            # Concatenate all batches along the N dimension: [P, batch_size, A] -> [P, N, A]
            mu_x = torch.cat(mu_x_batches, dim=1)
            sigma_x = torch.cat(sigma_x_batches, dim=1)

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
            # Add legend with class names below the plot in a horizontal line
            labels_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train
            from matplotlib.patches import Patch
            legend_handles = [Patch(color=color_map[l], label=get_label_name(l, dataset_name, is_ood)) 
                              for l in sorted(np.unique(labels_cpu))]
            plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                      ncol=len(legend_handles), fontsize='small', frameon=True)
            plot_idx += 1

        # Plot 4: Hyperparameters (lengthscales)
        if plot_hyperparams:
            plt.subplot(1, n_plots, plot_idx)
            lengthscales = (1 / model.gp.beta.flatten()).cpu()
            plt.bar(range(len(lengthscales)), lengthscales,
                    color="blue", alpha=0.7)
            plt.xlabel('Latent Dimension')
            plt.ylabel('Lengthscale')
            plt.title('GP Lengthscales')
            plt.grid(True, alpha=0.3)
            plot_idx += 1

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # Save plot
        logger.save_plots(fig, save_filename, is_checkpoint=is_checkpoint)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # Close figure to free memory

        # Additional plots: 1D histogram and 3D scatter plot
        if plot_latent_space:
            print("Creating additional plots: 1D histogram and 3D scatter plot...")

            # Ensure labels_train is on CPU for numpy operations
            labels_train_cpu = labels_train.cpu().numpy() if torch.is_tensor(labels_train) else labels_train

            # We already have mu_x_cpu, sigma_x_cpu, lengthscales_cpu, dim1, dim2, dim3, and top_3_dims from the latent space plot
            lengthscales = lengthscales_cpu  # Use existing lengthscales

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

        # Create histogram with different colors for different classes (using consistent color mapping)
        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            class_mask = all_labels == label
            if np.any(class_mask):
                class_samples = all_samples[class_mask]
                # CHANGED: Use get_label_name for OOD-aware legend
                ax_hist.hist(class_samples, bins=30, alpha=0.7,
                             label=get_label_name(label, dataset_name, is_ood),
                             color=color_map[label])

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

        # Color mapping for labels (using consistent color mapping)
        unique_labels = np.unique(labels_train_cpu)
        
        # Add traces for each class
        for label in unique_labels:
            class_mask = labels_train_cpu == label
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
                        color=matplotlib_to_plotly_color(color_map[label]),
                        opacity=0.6
                    ),
                    # CHANGED: Use get_label_name for OOD-aware legend
                    name=get_label_name(label, dataset_name, is_ood)
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

