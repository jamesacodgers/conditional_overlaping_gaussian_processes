# %%
"""
Replot Original GP-LVM qPCR Results from Checkpoint

This script loads a trained Original GP-LVM model from a checkpoint and regenerates all plots
without retraining the model.

Usage:
    python ssvi_scripts_experiments/qpcr_original_gplvm_replot_results.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import pickle
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Get the project root directory (parent of ssvi_scripts_experiments)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add both project root and generalised_gplvm_code to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

from utils.data import load_real_data

# Handle missing prettytable dependency (only used for parameter printing, not needed for replotting)
import sys
try:
    from prettytable import PrettyTable
except ImportError:
    # Create a dummy PrettyTable class if not available
    class PrettyTable:
        def __init__(self, *args, **kwargs):
            self.rows = []
        def add_row(self, *args, **kwargs):
            self.rows.append(args[0] if args else [])
        def __str__(self):
            return "PrettyTable not available (install with: pip install prettytable)"
    
    # Monkey-patch the module before importing bayesianGPLVM
    class DummyPrettyTableModule:
        PrettyTable = PrettyTable
    
    sys.modules['prettytable'] = DummyPrettyTableModule()

from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import PointLatentVariable, MAPLatentVariable, VariationalLatentVariable, NNEncoder
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from models.likelihoods import GaussianLikelihoodWithMissingObs
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import Positive
from torch.distributions import transforms

from src.original_gplvm_config import OriginalGPLVMConfig
from src.original_gplvm_config_utils import dict_to_dataclass
from src.original_gplvm_results_logger import OriginalGPLVMResultsLogger
from src.original_gplvm_metrics_plot import (
    plot_metrics_progression, 
    plot_missing_nlpd_metrics_progression, 
    plot_missing_rmse_metrics_progression
)
from src.mf_reploting_helper import (
    plot_latent_slice_original_gplvm_replot
)
from src.missing_data_utils import preprocess_data_for_missing_training
from src.save_utils import print_final_model_parameters
from src.pytorch_scalers import MeanCenter, StandardScalerPytorch, MinMaxScalerPytorch, NoTransform
import plotly.graph_objects as go
import pathlib


def get_pseudotime():
    """Get pseudotime for qPCR data (same as qPCR_GPLVM.py)"""
    # license: Copyright (c) 2014, the Open Data Science Initiative
    # license: https://www.elsevier.com/legal/elsevier-website-terms-and-conditions
    URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
    
    df = pd.read_csv(URL, index_col=0)
    data = torch.tensor(df.values, dtype=torch.float64)
    # we need to transpose data to correct its shape
    y = data.t()
    capture_time = y.new_tensor([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
    # we scale the time into the interval [0, 1]
    time = capture_time.log2() / 6
    return time


class qPCRModel(BayesianGPLVM):
    """qPCR Model class - same as in training script"""
    def __init__(self, n, data_dim, latent_dim, n_inducing, X, nn_layers=None, device=None, config=None):
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        
        # Locations Z corresponding to u_{d}, they can be randomly initialized or 
        # regularly placed with shape (n_inducing x latent_dim).
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inducing_inputs = torch.randn(n_inducing, latent_dim, dtype=torch.float64, device=device)
    
        # Sparse Variational Formulation
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        super(qPCRModel, self).__init__(X, q_f)
        
        # Kernel 
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
        # Initialize kernel parameters from config if provided
        if config is not None:
            with torch.no_grad():
                # Set lengthscales to match SSVI formula mathematically
                # SSVI: K = exp(-0.5 * d^2 / init_lengthscales)
                # GP-LVM: K = exp(-0.5 * d^2 / lengthscale^2)
                # For equivalence: lengthscale = sqrt(init_lengthscales)
                lengthscale_device = torch.sqrt(torch.tensor(config.init_lengthscales, dtype=torch.float64, device=device))
                self.covar_module.base_kernel.initialize(lengthscale=lengthscale_device)

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


def get_consistent_qpcr_colors(labels_train):
    """
    Get consistent color mapping for qPCR labels across SSVI and original GP LVM.
    Ensures the same class always gets the same color regardless of which plotting function is used.
    """
    # Use tab10 colormap with reversed order (same as original qPCR_expt.py)
    cmap_colors = plt.get_cmap("tab10").colors[::-1]
    
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


def create_standardized_scattered_plot_qpcr_replot(model, losses, labels_train, config, results_dir, model_name, random_seed,
                                       Y_train, model_type="qPCR", custom_filename=None, train_indices=None,
                                       full_loss_iterations: list = None, batch_losses: list = None):
    """
    Create standardized scattered plots for qPCR - copied from qpcr_plot_helper.py and modified to save as PDF.
    
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
        # Color mapping for qPCR labels (exactly like original qPCR_expt.py)
        # qPCR labels are integers 0, 1, 2, ... from load_real_data('gene')
        # Use consistent color mapping across SSVI and original GP LVM
        colors, color_map = get_consistent_qpcr_colors(labels_train)

        # Only plot latent space and lengthscales (remove ELBO plots)
        n_plots = 2  # Latent space, Hyperparams
        fig_width = 4 * n_plots
        fig = plt.figure(figsize=(fig_width, 4))

        plot_idx = 1

        # Plot 1: Latent space visualization (exactly like SSVI)
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

        plt.xlabel(f'Latent Dimension {dim1}')
        plt.ylabel(f'Latent Dimension {dim2}')
        plt.title(f'Latent Space - Top 2 Informative Dims')
        plot_idx += 1

        # Plot 2: Hyperparameters (lengthscales) - consistent with SSVI
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
        plot_filename = f"qpcr_{model_short_name}_{random_seed}.pdf"

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
    plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
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
    _, color_map = get_consistent_qpcr_colors(labels_train_cpu)
    unique_labels = np.unique(all_labels)
    
    for label_val in unique_labels:
        class_mask = all_labels == label_val
        if np.any(class_mask):
            class_samples = all_samples[class_mask]
            ax_hist.hist(class_samples, bins=30, alpha=0.7,
                         label=f'Class {label_val}',
                         color=color_map[label_val])

    ax_hist.set_xlabel(f'Latent Dimension {dim1} (Most Important)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title(f'1D Histogram - Most Important Dimension {dim1}')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # Save histogram plot as PDF
    hist_filename = plot_filename.replace('qpcr_', '1d_histogram_most_important_dim_')
    if not hist_filename.endswith('.pdf'):
        hist_filename = hist_filename.rsplit('.', 1)[0] + '.pdf'
    
    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        hist_plot_path = os.path.join(results_dir, hist_filename)
    else:
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        hist_plot_path = os.path.join(plots_subdir, hist_filename)

    fig_hist.savefig(hist_plot_path, format='pdf', dpi=300, bbox_inches='tight')
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
    _, color_map = get_consistent_qpcr_colors(labels_train_cpu)
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
                name=f'Class {label_val}'
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

    # Save 3D plot as HTML (plotly plots remain HTML)
    plot_3d_filename = plot_filename.replace('qpcr_', '3d_scatter_top3_dims_')
    plot_3d_filename = plot_3d_filename.replace('.pdf', '.html')

    if os.path.basename(results_dir) == 'plots' or os.path.basename(results_dir) == 'checkpoints':
        plot_3d_path = os.path.join(results_dir, plot_3d_filename)
    else:
        plots_subdir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_subdir, exist_ok=True)
        plot_3d_path = os.path.join(plots_subdir, plot_3d_filename)

    fig_3d.write_html(plot_3d_path)
    print(f"Saved 3D scatter plot to: {plot_3d_path}")

    print("Additional plots created successfully!")


def load_config_from_results_dir(results_dir: str) -> OriginalGPLVMConfig:
    """Load configuration from results directory config.json"""
    config_path = os.path.join(results_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dict to OriginalGPLVMConfig dataclass
    config = dict_to_dataclass(config_dict, OriginalGPLVMConfig)
    
    return config


def load_model_from_checkpoint(results_dir: str, model_name: str, model_filename: str = None):
    """Load model from checkpoint pickle file (state_dict)"""
    models_dir = os.path.join(results_dir, "pre_trained_models")
    
    if model_filename is None:
        # Look for files like qpcr_gauss_7.pkl or qpcr_nn_gauss_7.pkl
        # Also check for oilflow_ prefix (in case logger was used with different prefix)
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and model_name in f]
        if not model_files:
            raise FileNotFoundError(f"No model files found for {model_name} in {models_dir}")
        model_filename = model_files[0]
        print(f"Using model file: {model_filename}")
    
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model state_dict from: {model_path}")
    
    # Load state_dict
    with open(model_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    return state_dict


def reconstruct_model(model_name: str, config: OriginalGPLVMConfig, Y_train: torch.Tensor, state_dict: dict):
    """Reconstruct the model architecture and load state_dict"""
    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = config.latent_dim
    n_inducing = config.n_inducing
    
    # Determine device
    device = config.device if config.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # PCA initialization (same as training)
    if config.enable_missing_data:
        # For missing data, we need to handle this differently
        # Use mean imputation for PCA
        Y_train_for_pca = Y_train.clone()
        Y_train_for_pca[torch.isnan(Y_train_for_pca)] = 0.0
        mu_x_pca, _, _ = torch.pca_lowrank(Y_train_for_pca, latent_dim)
    else:
        mu_x_pca, _, _ = torch.pca_lowrank(Y_train, latent_dim)
    
    X_init = torch.nn.Parameter(mu_x_pca)
    
    # Define prior for X
    X_prior_mean = mu_x_pca
    nn_layers = None
    prior_x = None
    
    if model_name == 'gauss':
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean, dtype=torch.float64))
        X = VariationalLatentVariable(X_init, prior_x, data_dim)
    elif model_name == 'nn_gauss':
        nn_layers = config.nn_layers
        prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1], dtype=torch.float64, device=device))
        X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    # Create model
    model = qPCRModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers, device=device, config=config)
    
    # Create likelihood (needed for model structure)
    if config.enable_missing_data:
        likelihood = GaussianLikelihoodWithMissingObs(noise_constraint=Positive(transform=transforms.ExpTransform()))
    else:
        likelihood = GaussianLikelihood(noise_constraint=Positive(transform=transforms.ExpTransform()))
    
    # Move to device (same as training script)
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        # Also move the latent variable (X) to CUDA using proper to() method
        if hasattr(model, 'X'):
            model.X = model.X.to(device)
    
    # Filter out likelihood keys from state_dict (likelihood is a separate object, not a model submodule)
    # The state_dict may contain keys like "likelihood.noise_covar.raw_noise" which are not part of the model
    model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('likelihood.')}
    
    # Load state_dict (use strict=False to handle any remaining mismatches gracefully)
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys in state_dict: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Warning: Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys in state_dict: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Warning: Unexpected keys in state_dict: {unexpected_keys}")
    
    # Try to load likelihood state if available (though it's a separate object)
    likelihood_state_dict = {k.replace('likelihood.', ''): v for k, v in state_dict.items() if k.startswith('likelihood.')}
    if likelihood_state_dict:
        try:
            # Filter out constraint-related keys that can't be loaded
            loadable_likelihood_keys = {k: v for k, v in likelihood_state_dict.items() 
                                      if not k.endswith('_constraint.lower_bound') and not k.endswith('_constraint.upper_bound')}
            if loadable_likelihood_keys:
                likelihood.load_state_dict(loadable_likelihood_keys, strict=False)
        except Exception as e:
            print(f"  Warning: Could not load likelihood state: {e}")
    
    # Attach likelihood to model (some functions expect model.likelihood)
    model.likelihood = likelihood
    
    # Ensure model and likelihood are in float64 (same as training)
    model = model.to(dtype=torch.float64)
    likelihood = likelihood.to(dtype=torch.float64)
    
    # Fix device placement for NNEncoder tensors (created on CPU in original code)
    # This matches the training script exactly - critical for nn_gauss models
    if model_name == 'nn_gauss' and hasattr(model.X, 'jitter'):
        # Use the proper to() method which handles jitter tensor
        model.X = model.X.to(device)
        # Explicitly move jitter tensor to device (it's not a registered parameter, so .to() doesn't move it)
        if hasattr(model.X, 'jitter') and isinstance(model.X.jitter, torch.Tensor):
            model.X.jitter = model.X.jitter.to(device)
        # Also ensure prior_x tensors are on correct device
        if hasattr(model.X, 'prior_x'):
            if hasattr(model.X.prior_x, 'loc'):
                model.X.prior_x.loc = model.X.prior_x.loc.to(device)
            if hasattr(model.X.prior_x, 'covariance_matrix'):
                model.X.prior_x.covariance_matrix = model.X.prior_x.covariance_matrix.to(device)
    
    model.eval()
    likelihood.eval()
    
    return model, likelihood


def load_data_for_plotting(config: OriginalGPLVMConfig):
    """Load and preprocess qPCR data (same as training script)"""
    # Load data
    N, d, q, X, Y, labels = load_real_data('gene')
    
    # Ensure data is on CPU before converting to numpy for sklearn
    Y_cpu = Y.cpu() if Y.is_cuda else Y
    # Handle labels - convert to numpy array if it's a tensor, otherwise keep as is
    if hasattr(labels, 'cpu'):
        labels_cpu = labels.cpu().numpy()
    elif hasattr(labels, 'numpy'):
        labels_cpu = labels.numpy()
    else:
        # If labels is already a list or numpy array, convert to numpy array
        labels_cpu = np.array(labels)
    Y_np = Y_cpu.numpy()
    indices = np.arange(len(Y))
    
    # Get pseudotime for qPCR data
    pseudo_time = get_pseudotime()
    
    # Split data and labels together to maintain correspondence
    train_indices, test_indices, Y_train_np, Y_test_np, lb_train_np, lb_test_np, pseudo_train_np, pseudo_test_np = train_test_split(
        indices, Y_np, labels_cpu, pseudo_time.cpu().numpy(),
        test_size=config.test_size,
        random_state=config.random_seed
    )
    
    # Convert back to tensors on original device
    device = config.device if config.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    Y_train = torch.tensor(Y_train_np, dtype=torch.float64, device=device)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float64, device=device)
    lb_train = lb_train_np  # Keep as numpy array for compatibility
    lb_test = lb_test_np    # Keep as numpy array for compatibility
    
    # === Data Standardization ===
    # Fit scaler on training data only (before missing data preprocessing)
    if config.standardization == "standard":
        scaler = StandardScalerPytorch(device=device)
    elif config.standardization == "minmax":
        scaler = MinMaxScalerPytorch(device=device)
    elif config.standardization == "mean":
        scaler = MeanCenter(device=device)
    elif config.standardization == "none":
        scaler = NoTransform(device=device)
    else:
        raise ValueError(
            f"Unknown standardization method: {config.standardization}. Must be 'standard', 'minmax', 'mean', or 'none'"
        )
    
    # Fit scaler on training data only
    scaler.fit(Y_train)
    
    # Transform all data using the scaler fitted on training data
    Y_train = scaler.transform(Y_train)
    Y_test = scaler.transform(Y_test)
    
    # Apply missing data preprocessing if enabled
    Y_train_original = None
    Y_test_original = None
    train_missing_mask = None
    test_missing_mask = None
    
    if config.enable_missing_data:
        Y_train_original = Y_train.clone()
        Y_test_original = Y_test.clone()
        Y_train_missing, train_missing_mask, _ = preprocess_data_for_missing_training(Y_train, config)
        Y_test_missing, test_missing_mask, _ = preprocess_data_for_missing_training(Y_test, config)
        Y_train = Y_train_missing
        Y_test = Y_test_missing
    
    return {
        'y_train': Y_train,
        'y_test': Y_test,
        'labels_train': lb_train,
        'labels_test': lb_test,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'y_train_original': Y_train_original,
        'y_test_original': Y_test_original,
        'train_missing_mask': train_missing_mask,
        'test_missing_mask': test_missing_mask,
        'scaler': scaler,
    }


def load_metrics_from_checkpoint(results_dir: str, model_name: str, config: OriginalGPLVMConfig):
    """Load checkpoint metrics from JSON file"""
    metrics_path = os.path.join(results_dir, "training_results", f"{model_name}_{config.random_seed}_checkpoint_metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"Warning: Metrics file not found: {metrics_path}")
        return None
    
    print(f"Loading metrics from: {metrics_path}")
    
    # Read JSON file as text to handle NaN values
    with open(metrics_path, 'r') as f:
        json_text = f.read()
    
    # Replace unquoted NaN with null
    json_text = re.sub(r'\bNaN\b', 'null', json_text)
    json_text = re.sub(r'\bnan\b', 'null', json_text)
    
    # Parse JSON
    metrics_data = json.loads(json_text)
    
    # Extract metrics
    train_rmse_list = [c['train_rmse'] for c in metrics_data['checkpoints']]
    train_nlpd_list = [c['train_nlpd'] for c in metrics_data['checkpoints']]
    metrics_iterations = [c['iteration'] for c in metrics_data['checkpoints']]
    metrics_losses = [c['full_dataset_elbo_loss'] for c in metrics_data['checkpoints']]
    
    # Handle optional fields
    effective_dims_list = [c.get('effective_dims', None) for c in metrics_data['checkpoints']]
    train_missing_nlpd_list = [c.get('train_missing_nlpd', None) for c in metrics_data['checkpoints']]
    train_missing_rmse_list = [c.get('train_missing_rmse', None) for c in metrics_data['checkpoints']]
    
    # Convert None to NaN for consistency
    effective_dims_list = [d if d is not None else float('nan') for d in effective_dims_list]
    train_missing_nlpd_list = [d if d is not None else float('nan') for d in train_missing_nlpd_list]
    train_missing_rmse_list = [d if d is not None else float('nan') for d in train_missing_rmse_list]
    
    print(f"Loaded {len(train_rmse_list)} metric checkpoints")
    
    return {
        'train_rmse_list': train_rmse_list,
        'train_nlpd_list': train_nlpd_list,
        'metrics_iterations': metrics_iterations,
        'metrics_losses': metrics_losses,
        'effective_dims_list': effective_dims_list,
        'train_missing_nlpd_list': train_missing_nlpd_list,
        'train_missing_rmse_list': train_missing_rmse_list,
    }


def load_losses_from_checkpoint(results_dir: str, model_name: str, config: OriginalGPLVMConfig):
    """Load training losses from JSON files"""
    # Load full dataset losses
    full_loss_path = os.path.join(results_dir, "training_results", f"{model_name}_{config.random_seed}_full_loss_{config.random_seed}_losses.json")
    batch_loss_path = os.path.join(results_dir, "training_results", f"{model_name}_{config.random_seed}_batch_{config.random_seed}_losses.json")
    
    full_losses = None
    batch_losses = None
    
    if os.path.exists(full_loss_path):
        print(f"Loading full ELBO losses from: {full_loss_path}")
        with open(full_loss_path, 'r') as f:
            full_loss_data = json.load(f)
        full_losses = full_loss_data['losses']
    else:
        print(f"Warning: Full loss file not found: {full_loss_path}")
    
    if os.path.exists(batch_loss_path):
        print(f"Loading batch losses from: {batch_loss_path}")
        with open(batch_loss_path, 'r') as f:
            batch_loss_data = json.load(f)
        batch_losses = batch_loss_data['losses']
    else:
        print(f"Warning: Batch loss file not found: {batch_loss_path}")
    
    return full_losses, batch_losses


def main(results_dir: str, model_name: str = None, model_filename: str = None):
    """
    Main function to replot Original GP-LVM qPCR results
    
    Args:
        results_dir: Path to results directory containing config.json and pre_trained_models/
        model_name: Model name ('gauss' or 'nn_gauss'). If None, will try to detect from available models.
        model_filename: Specific model filename. If None, will auto-detect.
    """
    print("=" * 70)
    print("Original GP-LVM qPCR Results Replotting")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    
    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FOLDER_NAME = "qpcr_original_gplvm_replotting_results"
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER_NAME)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 1. Load configuration
    print("\n1. Loading configuration...")
    config = load_config_from_results_dir(results_dir)
    
    # Set device and dtype (same as training script)
    device = config.device if config.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("  Warning: CUDA requested but not available, using CPU")
        device = 'cpu'
    config.device = device
    
    # Set default dtype to float64 (same as training script)
    torch.set_default_dtype(torch.float64)
    
    print(f"  Device: {device}")
    print(f"  Models: {config.models}")
    print(f"  Random seed: {config.random_seed}")
    
    # Auto-detect all available models if model_name not provided
    models_dir = os.path.join(results_dir, "pre_trained_models")
    if not os.path.exists(models_dir):
        raise ValueError("Could not find pre_trained_models directory.")
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    # Extract model names from filenames (e.g., qpcr_gauss_7.pkl or oilflow_gauss_7.pkl -> gauss)
    available_models = []
    for f in model_files:
        if 'nn_gauss' in f:
            if 'nn_gauss' not in available_models:
                available_models.append('nn_gauss')
        elif 'gauss' in f:
            if 'gauss' not in available_models:
                available_models.append('gauss')
    
    if not available_models:
        raise ValueError("Could not auto-detect any models. Please specify model_name.")
    
    # If model_name is specified, use only that one; otherwise process all available models
    if model_name is not None:
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not found in available models: {available_models}")
        models_to_process = [model_name]
        print(f"  Processing specified model: {model_name}")
    else:
        models_to_process = available_models
        print(f"  Auto-detected models: {models_to_process}")
        print(f"  Will process all {len(models_to_process)} model(s)")
    
    # Store models_to_process for final summary
    num_models = len(models_to_process)
    
    # Process each model
    for model_name in models_to_process:
        print("\n" + "=" * 70)
        print(f"Processing model: {model_name}")
        print("=" * 70)
        
        # 2. Load model
        print(f"\n2. Loading model checkpoint ({model_name})...")
        state_dict = load_model_from_checkpoint(results_dir, model_name, model_filename)
        
        # 3. Load data (only once, reuse for all models)
        if 'split_data' not in locals():
            print("\n3. Loading data...")
            split_data = load_data_for_plotting(config)
        y_train = split_data['y_train']
        labels_train = split_data['labels_train']
        train_indices = split_data['train_indices']
        
        # 4. Reconstruct model
        print(f"\n4. Reconstructing model ({model_name})...")
        model, likelihood = reconstruct_model(model_name, config, y_train, state_dict)
        print(f"  Model reconstructed successfully")
        
        # 5. Load metrics
        print("\n5. Loading metrics...")
        metrics = load_metrics_from_checkpoint(results_dir, model_name, config)
        
        # 6. Load losses
        print("\n6. Loading losses...")
        full_losses, batch_losses = load_losses_from_checkpoint(results_dir, model_name, config)
        
        # 7. Initialize logger for output
        print("\n7. Initializing logger...")
        # Create a minimal logger for output directory structure
        class MinimalLogger:
            def __init__(self, output_dir):
                self.results_dir = output_dir
                self.plots_dir = os.path.join(output_dir, "plots")
                os.makedirs(self.plots_dir, exist_ok=True)
                self.training_results_dir = os.path.join(output_dir, "training_results")
                os.makedirs(self.training_results_dir, exist_ok=True)
            
            def get_results_dir(self):
                return self.results_dir
        
        logger = MinimalLogger(output_dir)
        print(f"Results will be saved to: {logger.get_results_dir()}")
        
        # 8. Regenerate plots
        print("\n8. Regenerating plots...")
    
        # 8.1 Main visualization plot (training curve, latent space, hyperparameters)
        try:
            print("  Generating main visualization plot...")
            # Use full_losses if available, otherwise fall back to metrics_losses from checkpoint metrics
            losses_for_plot = full_losses
            if losses_for_plot is None or len(losses_for_plot) == 0:
                if metrics and metrics.get('metrics_losses'):
                    losses_for_plot = metrics['metrics_losses']
                    print(f"    Using losses from checkpoint metrics ({len(losses_for_plot)} points)")
                else:
                    losses_for_plot = []
                    print(f"    Warning: No losses available for ELBO plot")
            
            # Create full_loss_iterations if we have metrics
            full_loss_iterations = None
            if metrics and metrics['metrics_iterations']:
                # Use metrics iterations for full losses if available
                if losses_for_plot and len(losses_for_plot) == len(metrics['metrics_iterations']):
                    full_loss_iterations = metrics['metrics_iterations']
                else:
                    # Fallback: create iterations based on length
                    full_loss_iterations = list(range(len(losses_for_plot))) if losses_for_plot else None
            
            # Use create_standardized_scattered_plot but save as PDF
            plot_filename = f"qpcr_{model_name}_{config.random_seed}.pdf"
            plot_path = os.path.join(logger.plots_dir, plot_filename)
            
            # Create plot (saves as PDF)
            create_standardized_scattered_plot_qpcr_replot(
                model=model,
                losses=losses_for_plot if losses_for_plot else [],
                labels_train=labels_train,
                config=config,
                results_dir=logger.plots_dir,  # Pass plots_dir directly
                model_name=model_name,
                random_seed=config.random_seed,
                Y_train=y_train,
                model_type="qPCR",
                train_indices=torch.from_numpy(train_indices).to(dtype=torch.int64, device=y_train.device),
                full_loss_iterations=full_loss_iterations,
                batch_losses=batch_losses
            )
            print(f"    Main visualization plot saved (as PDF)")
        except Exception as e:
            import traceback
            print(f"    ✗ Error: Failed to generate main visualization: {e}")
            traceback.print_exc()
    
        # 8.2 Metrics progression plots
        if metrics:
            try:
                print("  Generating metrics progression plots...")
                plot_metrics_progression(
                    train_rmse_list=metrics['train_rmse_list'],
                    train_nlpd_list=metrics['train_nlpd_list'],
                    metrics_iterations=metrics['metrics_iterations'],
                    metrics_losses=metrics['metrics_losses'],
                    effective_dims_list=metrics['effective_dims_list'],
                    results_dir=logger.plots_dir,
                    model_name=model_name,
                    random_seed=config.random_seed,
                    batch_losses=batch_losses
                )
                print("    Metrics progression plots saved")
            except Exception as e:
                import traceback
                print(f"    ✗ Error: Failed to generate metrics progression plots: {e}")
                traceback.print_exc()
        
        # Missing data metrics plots
        if config.enable_missing_data:
            try:
                if any(not np.isnan(x) for x in metrics['train_missing_nlpd_list']):
                    print("  Generating missing NLPD metrics progression plot...")
                    plot_missing_nlpd_metrics_progression(
                        train_missing_nlpd_list=metrics['train_missing_nlpd_list'],
                        metrics_iterations=metrics['metrics_iterations'],
                        results_dir=logger.plots_dir,
                        model_name=model_name,
                        random_seed=config.random_seed
                    )
                    print("    Missing NLPD metrics progression plot saved")
            except Exception as e:
                import traceback
                print(f"    ✗ Error: Failed to generate missing NLPD plot: {e}")
                traceback.print_exc()
            
            try:
                if any(not np.isnan(x) for x in metrics['train_missing_rmse_list']):
                    print("  Generating missing RMSE metrics progression plot...")
                    plot_missing_rmse_metrics_progression(
                        train_missing_rmse_list=metrics['train_missing_rmse_list'],
                        metrics_iterations=metrics['metrics_iterations'],
                        results_dir=logger.plots_dir,
                        model_name=model_name,
                        random_seed=config.random_seed
                    )
                    print("    Missing RMSE metrics progression plot saved")
            except Exception as e:
                import traceback
                print(f"    ✗ Error: Failed to generate missing RMSE plot: {e}")
                traceback.print_exc()
        
        # 8.3 Latent slice visualizations
        try:
            print("  Generating latent slice visualizations...")
            plots_dir = logger.plots_dir
            
            # Plot slice centered at origin
            latent_slice_path_origin = os.path.join(plots_dir, f"latent_slice_origin_{model_name}_seed_{config.random_seed}.pdf")
            plot_latent_slice_original_gplvm_replot(
                model=model,
                y_train=y_train,
                model_name=model_name,
                center="zeros",
                save_path=latent_slice_path_origin,
                scaler=split_data.get('scaler')
            )
            if os.path.exists(latent_slice_path_origin):
                print(f"    ✓ Saved latent slice (origin) to: {latent_slice_path_origin}")
            
            # Plot slices centered at a few interesting data points
            N_train = y_train.shape[0]
            representative_indices = [0, N_train // 2, N_train - 1]
            
            for idx in representative_indices:
                latent_slice_path = os.path.join(plots_dir, f"latent_slice_point_{idx}_{model_name}_seed_{config.random_seed}.pdf")
                plot_latent_slice_original_gplvm_replot(
                    model=model,
                    y_train=y_train,
                    model_name=model_name,
                    center="data_point",
                    n_idx=idx,
                    save_path=latent_slice_path,
                    scaler=split_data.get('scaler')
                )
                if os.path.exists(latent_slice_path):
                    print(f"    ✓ Saved latent slice (point {idx}) to: {latent_slice_path}")
        except Exception as e:
            import traceback
            print(f"    ✗ Error: Failed to generate latent slice visualizations: {e}")
            traceback.print_exc()
    
        # Print final model parameters
        print("\n9. Final model parameters:")
        print(f"\nFinal Original GP-LVM ({model_name}) Model Parameters:")
        if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
            print(f"  Observation noise (sigma2): {model.likelihood.noise.item():.4f}")
        elif hasattr(model, 'sigma2'):
            print(f"  Observation noise (sigma2): {model.sigma2:.4f}")
        else:
            print(f"  Observation noise (sigma2): Not available")
        
        # Print kernel hyperparameters
        if hasattr(model, 'covar_module'):
            lengthscales = model.covar_module.base_kernel.lengthscale.flatten()
            print(f"  Lengthscales: {lengthscales.detach().cpu().numpy()}")
            if hasattr(model.covar_module, 'outputscale'):
                print(f"  Output scale: {model.covar_module.outputscale.item():.4f}")
        
        print(f"\n{'=' * 70}")
        print(f"All plots regenerated successfully for {model_name}!")
        print(f"{'=' * 70}")
    
    print("\n" + "=" * 70)
    print(f"All models processed successfully! ({num_models} model(s))")
    print("=" * 70)


if __name__ == "__main__":
    # ============================================================================
    # HARDCODED CONFIGURATION - Modify these as needed
    # ============================================================================
    # Hardcoded results folder path
    RESULTS_FOLDER_PATH = "ssvi_scripts_experiments/qpcr_results/original_gplvm_results/orig_gplvm_fast_seed_42_01_24_17_58"
    # Model name ('gauss' or 'nn_gauss'). If None, will auto-detect.
    MODEL_NAME = None  # Set to 'gauss' or 'nn_gauss' to specify, or None to auto-detect
    # ============================================================================
    
    # Convert to absolute path if relative
    results_dir = RESULTS_FOLDER_PATH
    if not os.path.isabs(results_dir):
        # Try relative to current working directory first
        if not os.path.exists(results_dir):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(project_root, RESULTS_FOLDER_PATH)
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"Using results directory: {results_dir}")
    
    # Auto-detect model filename
    model_filename = None
    
    main(results_dir, model_name=MODEL_NAME, model_filename=model_filename)

