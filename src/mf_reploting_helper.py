"""
Replotting helper functions for Original GP-LVM (Mean Field) models.

This module contains functions specifically designed for replotting saved models,
with improved device handling and PDF output support.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import no_grad
import os
import pathlib
import plotly.graph_objects as go


@no_grad()
def plot_latent_slice_original_gplvm_replot(model, y_train, model_name, center="zeros", n_idx=None, save_path=None, scaler=None):
    """
    Plot 1D latent slice for original GP-LVM models (gauss and nn_gauss) - for replotting.
    
    Modified version for replotting that:
    - Ensures all tensors are on the correct device
    - Saves plots as PDF instead of PNG
    
    Args:
        model: Original BayesianGPLVM model instance (gauss or nn_gauss)
        y_train: Training data [N, M]
        model_name: Type of model ('gauss' or 'nn_gauss')
        center: str, either "zeros" (default) or "data_point". If "data_point", 
                must provide n_idx to specify which data point to use as center.
        n_idx: int, index of data point to use as center when center="data_point"
        save_path: str, optional path to save the plot. If None, displays the plot.
        scaler: optional scaler for inverse transform (to visualize in original scale)
    """
    # Determine device
    device = next(model.parameters()).device
    
    # Get latent dimension
    if hasattr(model.X, 'latent_dim'):
        Q = model.X.latent_dim
    else:
        raise ValueError("Model does not have latent_dim attribute")
    
    M = y_train.shape[1]  # Observation dimension
    N = y_train.shape[0]  # Number of training points
    
    # Ensure y_train is on the correct device
    y_train = y_train.to(device)
    
    model.eval()
    
    # Get latent representations based on model type
    if model_name == 'gauss':
        # For gauss: latent is variational parameters
        mu_x_mean = model.X.q_mu.detach()  # [N, Q]
        # Get variance from q_log_sigma
        sigma_x = torch.nn.functional.softplus(model.X.q_log_sigma).detach()  # [N, Q] - standard deviation
    elif model_name == 'nn_gauss':
        # For nn_gauss: encode the training data
        # Handle missing data if present
        y_encoder_input = torch.where(torch.isnan(y_train), 
                                     torch.full_like(y_train, -999.0, dtype=torch.float64), 
                                     y_train)
        mu_x_mean = model.X.mu(y_encoder_input).detach()  # [N, Q]
        # Get variance from encoder
        sigma_x_tensors = model.X.sigma(y_encoder_input).detach()  # List of covariance matrices or tensor
        # Extract diagonal standard deviations
        if isinstance(sigma_x_tensors, list):
            sigma_x = torch.stack([torch.sqrt(x.diag()) for x in sigma_x_tensors])  # [N, Q]
        else:
            sigma_x = torch.sqrt(torch.diagonal(sigma_x_tensors, dim1=-2, dim2=-1))  # [N, Q]
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Use 'gauss' or 'nn_gauss'")
    
    # 1) Pick most-informative latent dim (smallest lengthscale)
    # For original GP-LVM: lengthscale from kernel
    kernel_lengthscales = model.covar_module.base_kernel.lengthscale.flatten().detach().cpu().numpy()
    # Convert to same scale as SSVI (init_lengthscales scale)
    ells = kernel_lengthscales
    j = int(np.argmin(ells))
    
    # 2) Center
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
    
    # Get GP predictions using original GP-LVM's forward method
    # model(x) returns MultivariateNormal distribution
    Y_pred = model(Xgrid)  # Returns MultivariateNormal
    
    # Extract predictive mean and variance
    # Y_pred.loc has shape [M, G] (transposed)
    # Y_pred.covariance_matrix has shape [M] list of [G, G] matrices
    f_mean = Y_pred.loc.T  # [G, M] - transpose to match our convention
    
    # Extract diagonal variances from covariance matrices
    # Y_pred.covariance_matrix is a batch of covariance matrices
    f_var = torch.stack([torch.diag(Y_pred.covariance_matrix[m]) for m in range(M)]).T  # [G, M]
    
    # Add observation noise to get full predictive variance
    obs_noise = model.likelihood.noise.item()
    pred_var = f_var + obs_noise  # [G, M]
    pred_sd = torch.sqrt(torch.clamp(pred_var, min=1e-12))  # [G, M]
    
    # Inverse transform predictions if scaler provided (for visualization in original scale)
    if scaler is not None:
        f_mean = scaler.inverse_transform(f_mean)  # [G, M]
        # For standard deviation, scale by scaler's std if available
        if hasattr(scaler, 'std'):
            pred_sd = pred_sd * scaler.std  # [G, M]
    
    # Compute Y-axis limits from actual data for consistent scaling across models
    # Get data in original scale (inverse transform if scaler provided)
    y_data_for_limits = y_train.clone()
    if scaler is not None:
        y_data_for_limits = scaler.inverse_transform(y_data_for_limits)
    
    # Compute global Y-axis limits from data range (with padding for uncertainty bands)
    y_min_global = y_data_for_limits.min().item()
    y_max_global = y_data_for_limits.max().item()
    y_range = y_max_global - y_min_global
    y_padding = 1.0 * y_range  # 100% padding (fixed proportion for consistency across scripts, extra space for histograms)
    y_lim_min = y_min_global - y_padding
    y_lim_max = y_max_global + y_padding
    
    # 4) Plot - show first 12 dimensions (or M if M < 12)
    xnp = xs.detach().cpu().numpy()
    
    # Set X-axis limits for consistency (already using [-3, 3] range)
    x_lim_min, x_lim_max = -3.0, 3.0
    
    # Plot only first 12 dimensions (or all if M < 12)
    M_plot = min(M, 12)
    n_cols = 4
    n_rows = int(np.ceil(M_plot / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows), constrained_layout=True)
    axes = axes.ravel()
    
    # Sample from q(X) distribution along dimension j for histogram
    # Sample from N(mu_x_mean, sigma_x^2) for each data point
    x_samples_j = torch.normal(mu_x_mean[:, j], torch.clamp(sigma_x[:, j], min=1e-8))  # [N]
    x_j_values = x_samples_j.detach().cpu().numpy()
    
    for m in range(M_plot):
        mu = f_mean[:, m].cpu().numpy()
        sd = pred_sd[:, m].cpu().numpy()
        ax = axes[m]
        ax.plot(xnp, mu, lw=2)
        ax.fill_between(xnp, mu - 2*sd, mu + 2*sd, alpha=0.25)
        ax.set_title(f"y[{m}] vs x_{j} (l={ells[j]:.2g})", fontsize=9)
        ax.set_xlabel(f"x_{j}")
        ax.set_ylabel(f"y[{m}]")
        # Set consistent axis limits
        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(y_lim_min, y_lim_max)
        
        # Add histogram directly on the same axes, aligned with x-axis at bottom
        # Compute histogram and scale to make it bigger (use 25% of y-axis range)
        # Use fixed number of bins for consistent granularity across all models
        counts, bins = np.histogram(x_j_values, bins=15, density=True)
        y_range_hist = (y_lim_max - y_lim_min) * 0.25
        max_count = counts.max() if len(counts) > 0 and counts.max() > 0 else 1.0
        counts_scaled = (counts / max_count) * y_range_hist if max_count > 0 else counts * 0
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, counts_scaled, width=(bins[1] - bins[0]), 
               bottom=y_lim_min, alpha=0.6, color='lightblue', edgecolor='black')
    
    for k in range(M_plot, len(axes)):
        axes[k].axis("off")
    
    if save_path:
        # Ensure PDF extension
        if not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved latent slice plot to: {save_path}")
    else:
        plt.show()
    
    model.train()


@no_grad()
def create_standardized_scattered_plot_replot(model, losses, labels_train, config, results_dir, model_name, random_seed, Y_train, model_type="original", custom_filename=None, train_indices=None, full_loss_iterations: list = None, batch_losses: list = None):
    """
    Create standardized scattered plots that match the SSVI plotting exactly - for replotting.
    
    Modified version for replotting that:
    - Saves all matplotlib plots as PDF instead of PNG
    - HTML plots (3D plots) are still saved as HTML
    
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
    """
    # Use default matplotlib style to match SSVI (override any global ggplot style)
    plt.style.use('default')
    
    with torch.no_grad():
        # Color mapping for labels (exactly like SSVI)
        c = ["r", "g", "b"]
        colors = [c[int(i)-1] for i in labels_train]
        
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
            Y_train_device = Y_train.to(model.X.mu_layers[0].weight.device) if Y_train.device != model.X.mu_layers[0].weight.device else Y_train
            
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
        lengthscales_plot = 1 / model.covar_module.base_kernel.lengthscale.flatten().cpu()
        lengthscales_plot = lengthscales_plot**2
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
        plot_filename = f"oilflow_{model_short_name}_{random_seed}.pdf"  # Default to PDF
    
    # Ensure PDF extension
    if not plot_filename.endswith('.pdf'):
        plot_filename = plot_filename.rsplit('.', 1)[0] + '.pdf'
    
    # Create plots subdirectory only if we're not already in a plots directory
    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        # We're already in a plots directory or checkpoints subdirectory, save directly here
        plot_path = os.path.join(results_dir, plot_filename)
    else:
        # Create plots subdirectory for better organization
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        plot_path = os.path.join(plots_subdir, plot_filename)
    
    # Save as PDF
    plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Standardized scattered plots saved to: {plot_path}")
    
    # Generate fractions plot using the same mu_x and top dimensions
    try:
        # Load fractions data - check data directory first, then current directory
        frctn_path = pathlib.Path("data") / "oil_fractions.txt"
        
        if not frctn_path.exists():
            # Fallback to current directory (SSVI approach)
            frctn_path = pathlib.Path(".") / "DataTrnFrctns.txt"

        if not frctn_path.exists():
            raise FileNotFoundError(f"Fractions file not found. Checked: data/oil_fractions.txt and DataTrnFrctns.txt")

        w = np.loadtxt(frctn_path)  # (N, 2)
        w3 = 1.0 - w.sum(axis=1, keepdims=True)  # (N, 1)
        fractions = np.concatenate((w, w3), axis=1)  # (N, 3)
        
        # Debug: print fractions shapes
        print(f"Debug - fractions shape before subsetting: {fractions.shape}")
        print(f"Debug - Y_train shape: {Y_train.shape}")
        
        # Subset fractions data to match training data using train_indices
        train_indices_cpu = train_indices.cpu().numpy()
        fractions = fractions[train_indices_cpu]
        print(f"Debug - fractions shape after subsetting with train_indices: {fractions.shape}")
        
        # Create fractions plot using sampling from the distribution (same as main plot)
        fig_frac, ax_frac = plt.subplots(1, 1, figsize=(7.5, 6))
        
        # Plot samples from all P samples with transparency (consistent with latent dims plot)
        for p in range(64):
            # Sample from the latent distributions (consistent with main latent space plot)
            sample_1 = torch.distributions.Normal(mu_x_torch[:, dim1], sigma_x_torch[:, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_torch[:, dim2], sigma_x_torch[:, dim2]).sample()
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
        
        # Save fractions plot with same naming pattern as main plot - as PDF
        fractions_filename = plot_filename.replace('oilflow_', 'latent_space_by_fractions_')
        if not fractions_filename.endswith('.pdf'):
            fractions_filename = fractions_filename.rsplit('.', 1)[0] + '.pdf'
        
        if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
            frac_plot_path = os.path.join(results_dir, fractions_filename)
        else:
            plots_subdir = os.path.join(results_dir, 'plots')
            os.makedirs(plots_subdir, exist_ok=True)
            frac_plot_path = os.path.join(plots_subdir, fractions_filename)
        
        fig_frac.savefig(frac_plot_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Saved fractions plot to: {frac_plot_path}")
        plt.close(fig_frac)
        
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
        
        # Create histogram with different colors for different classes
        for class_idx in range(3):  # Assuming 3 classes
            class_mask = all_labels == (class_idx + 1)
            if np.any(class_mask):
                class_samples = all_samples[class_mask]
                ax_hist.hist(class_samples, bins=30, alpha=0.7, 
                           label=f'Class {class_idx + 1}', 
                           color=['red', 'green', 'blue'][class_idx])
        
        ax_hist.set_xlabel(f'Latent Dimension {dim1} (Most Important)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'1D Histogram - Most Important Dimension {dim1}')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Save histogram plot - as PDF
        hist_filename = plot_filename.replace('oilflow_', '1d_histogram_most_important_dim_')
        if not hist_filename.endswith('.pdf'):
            hist_filename = hist_filename.rsplit('.', 1)[0] + '.pdf'
        if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
            hist_plot_path = os.path.join(results_dir, hist_filename)
        else:
            plots_subdir = os.path.join(results_dir, 'plots')
            os.makedirs(plots_subdir, exist_ok=True)
            hist_plot_path = os.path.join(plots_subdir, hist_filename)
        
        fig_hist.savefig(hist_plot_path, format='pdf', bbox_inches='tight', dpi=300)
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
        
        # Color mapping for labels
        colors_3d = ['red', 'green', 'blue']
        color_names = ['Class 1', 'Class 2', 'Class 3']
        
        # Add traces for each class
        for class_idx in range(3):
            class_mask = labels_train_cpu == (class_idx + 1)
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
                        color=colors_3d[class_idx],
                        opacity=0.6
                    ),
                    name=color_names[class_idx]
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
        
        # Save 3D plot as HTML (keep HTML for interactive plots)
        plot_3d_filename = plot_filename.replace('oilflow_', '3d_scatter_top3_dims_')
        plot_3d_filename = plot_3d_filename.replace('.pdf', '.html')
        
        if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
            plot_3d_path = os.path.join(results_dir, plot_3d_filename)
        else:
            plots_subdir = os.path.join(results_dir, 'plots')
            os.makedirs(plots_subdir, exist_ok=True)
            plot_3d_path = os.path.join(plots_subdir, plot_3d_filename)
        
        fig_3d.write_html(plot_3d_path)
        print(f"Saved 3D scatter plot to: {plot_3d_path}")
        
        # 3. Create 3D fractions plot using the same fractions data
        print("Creating 3D fractions plot...")
        
        # Create plotly 3D scatter plot colored by fractions
        fig_3d_frac = go.Figure()
        
        # Plot samples from all P samples with fractions as colors (consistent with 2D fractions plot)
        for p in range(64):  # Same number of samples as main plot
            # Sample from the latent distributions (consistent with 2D fractions plot)
            sample_1 = torch.distributions.Normal(mu_x_torch[:, dim1], sigma_x_torch[:, dim1]).sample()
            sample_2 = torch.distributions.Normal(mu_x_torch[:, dim2], sigma_x_torch[:, dim2]).sample()
            sample_3 = torch.distributions.Normal(mu_x_torch[:, dim3], sigma_x_torch[:, dim3]).sample()
            
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
                xaxis_title=f'Dimension {dim1} (lengthscale={lengthscales_cpu[dim1].item():.2f})',
                yaxis_title=f'Dimension {dim2} (lengthscale={lengthscales_cpu[dim2].item():.2f})',
                zaxis_title=f'Dimension {dim3} (lengthscale={lengthscales_cpu[dim3].item():.2f})'
            ),
            width=800,
            height=600
        )
        
        # Save 3D fractions plot as HTML (keep HTML for interactive plots)
        plot_3d_frac_filename = plot_filename.replace('oilflow_', '3d_latent_space_by_fractions_')
        plot_3d_frac_filename = plot_3d_frac_filename.replace('.pdf', '.html')
        
        if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
            plot_3d_frac_path = os.path.join(results_dir, plot_3d_frac_filename)
        else:
            plots_subdir = os.path.join(results_dir, 'plots')
            os.makedirs(plots_subdir, exist_ok=True)
            plot_3d_frac_path = os.path.join(plots_subdir, plot_3d_frac_filename)
        
        fig_3d_frac.write_html(plot_3d_frac_path)
        print(f"Saved 3D fractions plot to: {plot_3d_frac_path}")
        
        print("Additional plots created successfully!")
        
    except FileNotFoundError as e:
        print(f"Warning: Could not create fractions plot: {e}")

