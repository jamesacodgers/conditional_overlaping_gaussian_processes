#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to align SSVI GP-LVM outputs with original GP-LVM structure
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


def get_swiss_roll_colors(labels_train):
    """
    Get continuous color mapping for Swiss roll labels.
    Swiss roll labels are continuous values (position parameter t), so we use a gradient colormap.
    """
    # Use viridis colormap for continuous values (popular for scientific visualization)
    cmap = plt.get_cmap("viridis")
    
    # Normalize labels to [0, 1] range for colormap
    labels_array = np.array(labels_train)
    labels_min = labels_array.min()
    labels_max = labels_array.max()
    
    if labels_max - labels_min > 0:
        normalized_labels = (labels_array - labels_min) / (labels_max - labels_min)
    else:
        normalized_labels = np.zeros_like(labels_array)
    
    # Map normalized values to colors
    colors = [cmap(val) for val in normalized_labels]
    
    return colors, cmap, labels_min, labels_max


def matplotlib_to_plotly_color(matplotlib_color):
    """Convert matplotlib RGB tuple to Plotly format"""
    if isinstance(matplotlib_color, tuple) and len(matplotlib_color) == 3:
        r, g, b = matplotlib_color
        return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'
    return matplotlib_color


def create_standardized_scattered_plot(model, losses, labels_train, config, results_dir, model_name, random_seed,
                                       Y_train, model_type="original", custom_filename=None, train_indices=None):
    """
    Create standardized scattered plots that match the SSVI plotting exactly.

    This function replicates the exact structure and styling of the SSVI visualization
    from src/ssvi_utils.py with the same subplot order and styling.
    """

    # Use default matplotlib style to match SSVI (override any global ggplot style)
    plt.style.use('default')

    with torch.no_grad():
        # Color mapping for Swiss roll labels (continuous position parameter)
        # Swiss roll labels are continuous values from make_swiss_roll
        colors, cmap, labels_min, labels_max = get_swiss_roll_colors(labels_train)

        # Create figure with same layout as SSVI (3 subplots)
        n_plots = 3
        fig_width = 4 * n_plots
        fig = plt.figure(figsize=(fig_width, 4))

        plot_idx = 1

        # Plot 1: Training curve (exactly like SSVI)
        plt.subplot(1, n_plots, plot_idx)
        plt.plot(losses, 'b-', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('ELBO Training Curve')
        plt.grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 2: Latent space visualization (exactly like SSVI)
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

            mu_x_result = model.X.mu(Y_train_for_plotting)
            sigma_covariances = model.X.sigma(Y_train_for_plotting)
            mu_x = mu_x_result.detach().cpu().numpy()
            # Apply sqrt to convert covariance matrices to standard deviations (matching original code)
            sigma_x = np.array([torch.sqrt(x.diag()).cpu().numpy() for x in sigma_covariances.detach()])

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
        plot_idx += 1

        # Plot 3: Hyperparameters (lengthscales) - consistent with SSVI
        plt.subplot(1, n_plots, plot_idx)
        # For mathematical consistency with SSVI version:
        # SSVI: K = exp(-0.5 * d^2 / init_lengthscales)
        # GP-LVM: K = exp(-0.5 * d^2 / lengthscale^2)
        # For equivalence: lengthscale = sqrt(init_lengthscales)
        lengthscales_plot = torch.sqrt(1 / model.covar_module.base_kernel.lengthscale.flatten()).cpu()
        plt.bar(range(len(lengthscales_plot)), lengthscales_plot,
                color="blue", alpha=0.7)
        plt.xlabel('Latent Dimension')
        plt.ylabel('Lengthscale')
        plt.title('GP Lengthscales')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

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
        plot_filename = f"swiss_roll_{model_short_name}_{random_seed}.png"

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
    # Try to get dim3 for 3D plots (will fail if less than 3 latent dimensions)
    try:
        values, indices = torch.topk(model.covar_module.base_kernel.lengthscale, k=3, largest=False)
        dim3 = indices.cpu().numpy().flatten()[2]  # Third dimension for 3D plots
    except:
        dim3 = None  # Not enough dimensions for 3D plot

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

    # Create histogram with continuous color mapping based on label values
    # Create bins and color by average label value in each bin
    n_bins = 30
    hist_values, bin_edges = np.histogram(all_samples, bins=n_bins)
    
    # Get colormap for continuous labels
    _, cmap_gradient, labels_min, labels_max = get_swiss_roll_colors(labels_train_cpu)
    
    # For each bin, assign color based on average label value of points in that bin
    for i in range(len(bin_edges) - 1):
        bin_mask = (all_samples >= bin_edges[i]) & (all_samples < bin_edges[i + 1])
        if np.any(bin_mask):
            # Get average label value for points in this bin
            avg_label = np.mean(all_labels[bin_mask])
            # Normalize to [0, 1] for colormap
            if labels_max - labels_min > 0:
                normalized_val = (avg_label - labels_min) / (labels_max - labels_min)
            else:
                normalized_val = 0.5
            color = cmap_gradient(normalized_val)
            
            # Plot this bin
            bin_samples = all_samples[bin_mask]
            ax_hist.hist(bin_samples, bins=[bin_edges[i], bin_edges[i + 1]], 
                        alpha=0.7, color=color, edgecolor='black', linewidth=0.5)

    ax_hist.set_xlabel(f'Latent Dimension {dim1} (Most Important)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'1D Histogram - Most Important Dimension {dim1}')
    ax_hist.grid(True, alpha=0.3)
    
    # Add colorbar to show label values
    sm = plt.cm.ScalarMappable(cmap=cmap_gradient, norm=plt.Normalize(vmin=labels_min, vmax=labels_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_hist)
    cbar.set_label('Swiss Roll Position (t)', rotation=270, labelpad=15)

    # Save histogram plot
    hist_filename = plot_filename.replace('swiss_roll_', '1d_histogram_most_important_dim_')
    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        hist_plot_path = os.path.join(results_dir, hist_filename)
    else:
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        hist_plot_path = os.path.join(plots_subdir, hist_filename)

    fig_hist.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved 1D histogram to: {hist_plot_path}")
    plt.close(fig_hist)

    # 2. Create 3D scatter plot for top 3 most important dimensions using plotly (only if dim3 exists)
    if dim3 is not None:
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

        # Combine samples from all P samples
        x_vals = np.concatenate([samples_3d[p]['dim1'] for p in range(64)])
        y_vals = np.concatenate([samples_3d[p]['dim2'] for p in range(64)])
        z_vals = np.concatenate([samples_3d[p]['dim3'] for p in range(64)])
        # Repeat labels for all P samples
        color_vals = np.tile(labels_train_cpu, 64)

        # Create single trace with continuous color mapping
        fig_3d.add_trace(go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers',
            marker=dict(
                size=3,
                color=color_vals,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Position (t)")
            ),
            name='Swiss Roll'
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
        plot_3d_filename = plot_filename.replace('swiss_roll_', '3d_scatter_top3_dims_')
        plot_3d_filename = plot_3d_filename.replace('.png', '.html')

        if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
            plot_3d_path = os.path.join(results_dir, plot_3d_filename)
        else:
            plots_subdir = os.path.join(results_dir, 'plots')
            os.makedirs(plots_subdir, exist_ok=True)
            plot_3d_path = os.path.join(plots_subdir, plot_3d_filename)

        fig_3d.write_html(plot_3d_path)
        print(f"Saved 3D scatter plot to: {plot_3d_path}")
    else:
        print(f"Skipped 3D scatter plot (less than 3 latent dimensions)")

    print("Additional plots created successfully!")


def save_iteration_model(model, model_name, iteration, random_seed, results_dir):
    """
    Save model at specific iteration to pre_trained_models/checkpoints folder
    """
    # Save to pre_trained_models/checkpoints folder for organized checkpoint storage
    pre_trained_dir = os.path.join(results_dir, 'pre_trained_models', 'checkpoints')
    os.makedirs(pre_trained_dir, exist_ok=True)

    filename = f'swiss_roll_{model_name}_iter_{iteration}_seed_{random_seed}.pkl'
    filepath = os.path.join(pre_trained_dir, filename)

    with open(filepath, 'wb') as file:
        pkl.dump(model.state_dict(), file)

    return filepath


def create_iteration_plot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, iteration,
                          model_type="original", train_indices=None):
    """
    Create iteration-based plot saved to plots/checkpoints folder
    """
    plot_filename = f"swiss_roll_{model_name}_iter_{iteration}_seed_{random_seed}.png"

    # Create plots/checkpoints subdirectory for organized checkpoint storage
    plots_checkpoints_dir = os.path.join(results_dir, 'plots', 'checkpoints')
    os.makedirs(plots_checkpoints_dir, exist_ok=True)

    # Create the standardized plot with custom filename in checkpoints folder
    create_standardized_scattered_plot(
        model, losses, labels_train, config, plots_checkpoints_dir,
        model_name, random_seed, Y_train, model_type=model_type,
        custom_filename=plot_filename, train_indices=train_indices
    )

    return plot_filename


def visualize_ssvi_results(model, y_train, labels_train, losses, config, logger: SSVIResultsLogger,
                           save_filename: str = "training_results.png", show_plot: bool = True,
                           n_optimization_steps: int = None, is_test_data: bool = False, is_checkpoint: bool = False,
                           train_indices: torch.Tensor = None):
    """Visualize SSVI training results and latent space"""

    if not (hasattr(config, 'plot_training') and config.plot_training or
            hasattr(config, 'plot_latent_space') and config.plot_latent_space or
            hasattr(config, 'plot_hyperparams') and config.plot_hyperparams):
        return

    with torch.no_grad():
        # Color mapping for Swiss roll labels (continuous position parameter)
        # Swiss roll labels are continuous values from make_swiss_roll
        colors, cmap, labels_min, labels_max = get_swiss_roll_colors(labels_train)

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

            # Find the most informative dimensions (largest lengthscales = smallest beta)
            lengthscales = 1 / model.gp.beta.flatten()
            top_2_dims = torch.argsort(lengthscales, descending=True)[:2]
            dim1, dim2 = top_2_dims[0].item(), top_2_dims[1].item()
            # Try to get dim3 for 3D plots (will fail if less than 3 latent dimensions)
            try:
                top_3_dims = torch.argsort(lengthscales, descending=True)[:3]
                dim3 = top_3_dims[2].item()  # Third dimension for 3D plots
            except:
                dim3 = None  # Not enough dimensions for 3D plot
            lengthscales_cpu = lengthscales.cpu()  # Move to CPU for string formatting
            # Get latent variables using new explicit training method
            U = model.get_u_samples()  # [P, M, C, Lx]

            # Handle missing data if enabled
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                # Preprocess encoder input to handle missing data (replace NaN with -9.0)
                y_encoder_input = torch.where(torch.isnan(y_train), torch.full_like(y_train, -9.0, dtype=torch.float64), y_train)
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U)  # [P, N, A]
            else:
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, U)  # [P, N, A]

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
            U = model.get_u_samples()  # [P, M, C, Lx]

            # Handle missing data if enabled
            if hasattr(model, 'enable_missing_data') and model.enable_missing_data:
                y_encoder_input = torch.where(torch.isnan(y_train), torch.full_like(y_train, -9.0, dtype=torch.float64), y_train)
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, U)  # [P, N, A]
            else:
                mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_train, U)  # [P, N, A]

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

        # Create histogram with continuous color mapping
        n_bins = 30
        hist_values, bin_edges = np.histogram(all_samples, bins=n_bins)
        
        # Get colormap
        _, cmap_gradient, labels_min, labels_max = get_swiss_roll_colors(labels_train_cpu)
        
        # For each bin, assign color based on average label value
        for i in range(len(bin_edges) - 1):
            bin_mask = (all_samples >= bin_edges[i]) & (all_samples < bin_edges[i + 1])
            if np.any(bin_mask):
                avg_label = np.mean(all_labels[bin_mask])
                if labels_max - labels_min > 0:
                    normalized_val = (avg_label - labels_min) / (labels_max - labels_min)
                else:
                    normalized_val = 0.5
                color = cmap_gradient(normalized_val)
                
                bin_samples = all_samples[bin_mask]
                ax_hist.hist(bin_samples, bins=[bin_edges[i], bin_edges[i + 1]], 
                            alpha=0.7, color=color, edgecolor='black', linewidth=0.5)

        ax_hist.set_xlabel(f'Latent Dimension {most_important_dim} (Most Important)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'1D Histogram - Most Important Dimension {most_important_dim}')
        ax_hist.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_gradient, norm=plt.Normalize(vmin=labels_min, vmax=labels_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_hist)
        cbar.set_label('Swiss Roll Position (t)', rotation=270, labelpad=15)

        # Save histogram plot
        hist_filename = save_filename.replace('training_results', '1d_histogram_most_important_dim')
        logger.save_plots(fig_hist, hist_filename, is_checkpoint=is_checkpoint)
        if show_plot:
            plt.show()
        else:
            plt.close(fig_hist)

        # 2. Create 3D scatter plot for top 3 most important dimensions using plotly (only if dim3 exists)
        if dim3 is not None:
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

            # Combine samples from all P samples
            x_vals = np.concatenate([samples_3d[p]['dim1'] for p in range(model.P)])
            y_vals = np.concatenate([samples_3d[p]['dim2'] for p in range(model.P)])
            z_vals = np.concatenate([samples_3d[p]['dim3'] for p in range(model.P)])
            # Repeat labels for all P samples
            color_vals = np.tile(labels_train_cpu, model.P)

            # Create single trace with continuous color mapping
            fig_3d.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=3,
                    color=color_vals,
                    colorscale='Viridis',
                    opacity=0.6,
                    colorbar=dict(title="Position (t)")
                ),
                name='Swiss Roll'
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
        else:
            print(f"Skipped 3D scatter plot for SSVI (less than 3 latent dimensions)")

        print("Additional plots created successfully!")
