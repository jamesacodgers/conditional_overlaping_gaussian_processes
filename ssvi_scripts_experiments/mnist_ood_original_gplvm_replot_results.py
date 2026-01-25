# %%
"""
Replot Original GP-LVM MNIST OOD Results from Checkpoint

This script loads a trained Original GP-LVM model from a checkpoint and regenerates all plots
without retraining the model.

Usage:
    python ssvi_scripts_experiments/mnist_ood_original_gplvm_replot_results.py
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
from sklearn.model_selection import train_test_split

# Get the project root directory (parent of ssvi_scripts_experiments)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add both project root and generalised_gplvm_code to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

# Handle missing prettytable dependency (only used for parameter printing, not needed for replotting)
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

from ssvi_scripts_experiments.mnist_ood_GPLVM import OODOriginalGPLVMConfig, load_ood_data
from src.original_gplvm_config_utils import dict_to_dataclass
from src.original_gplvm_results_logger import OriginalGPLVMResultsLogger
from src.original_gplvm_metrics_plot import (
    plot_metrics_progression, 
    plot_missing_nlpd_metrics_progression, 
    plot_missing_rmse_metrics_progression
)
from src.mf_reploting_helper import (
    plot_latent_slice_original_gplvm_replot,
    create_standardized_scattered_plot_mnist_ood_replot
)
from src.missing_data_utils import preprocess_data_for_missing_training
from src.save_utils import print_final_model_parameters
from src.pytorch_scalers import MeanCenter, StandardScalerPytorch, MinMaxScalerPytorch, NoTransform
from src.data_utils import load_image_dataset, filter_mnist_by_digits
from src.mnist_missing_reconstruction import (
    visualize_prior_samples_original_gplvm,
    visualize_prior_samples_original_gplvm_averaged
)
from src.mf_reploting_helper import visualize_test_image_predictions_original_gplvm_replot
import plotly.graph_objects as go
import pathlib


class MNISTModel(BayesianGPLVM):
    """MNIST Model class - same as in training script"""
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
        super(MNISTModel, self).__init__(X, q_f)
        
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


def load_config_from_results_dir(results_dir: str) -> OODOriginalGPLVMConfig:
    """Load configuration from results directory config.json"""
    config_path = os.path.join(results_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dict to OODOriginalGPLVMConfig dataclass
    config = dict_to_dataclass(config_dict, OODOriginalGPLVMConfig)
    
    return config


def load_model_state_dict_from_checkpoint(results_dir: str, model_name: str, model_filename: str = None):
    """Load model state_dict from checkpoint pickle file"""
    models_dir = os.path.join(results_dir, "pre_trained_models")
    
    if model_filename is None:
        # Look for files like mnist_gauss_7.pkl or mnist_nn_gauss_7.pkl
        # Also check for oilflow_ prefix (in case logger was used with different prefix)
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        # Filter by model_name (gauss or nn_gauss)
        model_files = [f for f in model_files if model_name in f or ('gauss' in f and model_name == 'gauss') or ('nn_gauss' in f and model_name == 'nn_gauss')]
        if not model_files:
            raise FileNotFoundError(f"No model files found for {model_name} in {models_dir}")
        model_filename = model_files[0]
        print(f"Using model file: {model_filename}")
    
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model state_dict from: {model_path}")
    with open(model_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    return state_dict


def reconstruct_model(model_name: str, config: OODOriginalGPLVMConfig, Y_train: torch.Tensor, state_dict: dict):
    """Reconstruct the model and likelihood from state_dict"""
    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = config.latent_dim
    n_inducing = config.n_inducing
    
    # Apply PCA initialization (same as training)
    if config.enable_missing_data:
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
        nn_layers = (512, 256)  # Hardcoded architecture
        prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1], dtype=torch.float64, device=config.device))
        X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
    
    # Initialize model
    model = MNISTModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers, device=config.device, config=config)
    
    # Use missing data aware likelihood if enabled
    if config.enable_missing_data:
        likelihood = GaussianLikelihoodWithMissingObs(noise_constraint=Positive(transform=transforms.ExpTransform()))
    else:
        likelihood = GaussianLikelihood(noise_constraint=Positive(transform=transforms.ExpTransform()))
    
    # Initialize likelihood noise from config
    if config is not None:
        with torch.no_grad():
            noise_scale = torch.tensor(config.init_output_noise_scale, dtype=torch.float64, device=config.device)
            likelihood.initialize(noise=noise_scale)
    
    # Move model and likelihood to device
    if config.device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        if hasattr(model, 'X'):
            model.X = model.X.to(config.device)
    
    # Fix device placement for NNEncoder tensors
    if model_name == 'nn_gauss' and hasattr(model.X, 'jitter'):
        model.X = model.X.to(config.device)
        if hasattr(model.X, 'prior_x'):
            if hasattr(model.X.prior_x, 'loc'):
                model.X.prior_x.loc = model.X.prior_x.loc.to(config.device)
            if hasattr(model.X.prior_x, 'covariance_matrix'):
                model.X.prior_x.covariance_matrix = model.X.prior_x.covariance_matrix.to(config.device)
    
    # Convert to float64
    model = model.to(torch.float64)
    likelihood = likelihood.to(torch.float64)
    
    # Filter likelihood keys from state_dict (likelihood is separate object)
    model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('likelihood.')}
    likelihood_state_dict = {k.replace('likelihood.', ''): v for k, v in state_dict.items() if k.startswith('likelihood.')}
    
    # Load model state_dict
    model.load_state_dict(model_state_dict, strict=False)
    
    # Load likelihood state_dict if available
    if likelihood_state_dict:
        likelihood.load_state_dict(likelihood_state_dict, strict=False)
    
    # Attach likelihood to model for consistency with plotting functions
    model.likelihood = likelihood
    
    # Fix jitter device placement for nn_gauss
    if model_name == 'nn_gauss' and hasattr(model.X, 'jitter'):
        model.X.jitter = model.X.jitter.to(config.device)
    
    print("  Model reconstructed successfully")
    return model, likelihood


def load_data_for_plotting(config: OODOriginalGPLVMConfig):
    """Load and preprocess data for plotting (same as training)"""
    Y_train, Y_test, lb_train, lb_test, split_data = load_ood_data(config)
    
    # Apply missing data preprocessing if enabled
    if config.enable_missing_data:
        print(f"\n=== Missing Data Preprocessing ===")
        Y_train_original = Y_train.clone()
        Y_test_original = Y_test.clone()
        
        Y_train_missing, train_missing_mask, missing_info = preprocess_data_for_missing_training(Y_train, config)
        Y_test_missing, test_missing_mask, _ = preprocess_data_for_missing_training(Y_test, config)
        
        Y_train = Y_train_missing
        Y_test = Y_test_missing
        split_data['y_train'] = Y_train
        split_data['y_test'] = Y_test
        split_data['y_train_original'] = Y_train_original
        split_data['y_test_original'] = Y_test_original
        split_data['train_missing_mask'] = train_missing_mask
        split_data['test_missing_mask'] = test_missing_mask
        
        print(f"Missing data preprocessing completed.")
        print("=" * 50)
    
    return Y_train, Y_test, lb_train, lb_test, split_data


def load_metrics_from_checkpoint(results_dir: str, model_name: str, config: OODOriginalGPLVMConfig):
    """Load metrics from checkpoint JSON file"""
    metrics_path = os.path.join(results_dir, "training_results", f"{model_name}_{config.random_seed}_checkpoint_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"Warning: Metrics file not found: {metrics_path}")
        return None
    
    print(f"Loading metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        # Handle NaN values in JSON
        content = f.read()
        content = content.replace('NaN', 'null')
        metrics_data = json.loads(content)
    
    print(f"Loaded {len(metrics_data.get('checkpoints', []))} metric checkpoints")
    return metrics_data


def load_losses_from_checkpoint(results_dir: str, model_name: str, config: OODOriginalGPLVMConfig):
    """Load full and batch losses from checkpoint JSON files"""
    # Pattern matches save_training_losses: f"{model_name}_{random_seed}_losses.json"
    # When model_name is "gauss_full_loss", filename becomes "gauss_full_loss_42_losses.json"
    full_loss_path = os.path.join(results_dir, "training_results", f"{model_name}_full_loss_{config.random_seed}_losses.json")
    batch_loss_path = os.path.join(results_dir, "training_results", f"{model_name}_batch_{config.random_seed}_losses.json")
    
    full_losses = None
    batch_losses = None
    full_loss_iterations = None
    
    if os.path.exists(full_loss_path):
        print(f"Loading full losses from: {full_loss_path}")
        with open(full_loss_path, 'r') as f:
            content = f.read()
            content = content.replace('NaN', 'null')
            full_loss_data = json.loads(content)
            full_losses = full_loss_data.get('losses', [])
            full_loss_iterations = full_loss_data.get('iterations', list(range(len(full_losses))))
    else:
        print(f"Warning: Full loss file not found: {full_loss_path}")
    
    if os.path.exists(batch_loss_path):
        print(f"Loading batch losses from: {batch_loss_path}")
        with open(batch_loss_path, 'r') as f:
            content = f.read()
            content = content.replace('NaN', 'null')
            batch_loss_data = json.loads(content)
            batch_losses = batch_loss_data.get('losses', [])
    else:
        print(f"Warning: Batch loss file not found: {batch_loss_path}")
    
    return full_losses, batch_losses, full_loss_iterations


def main(results_dir: str, model_name: str = None, model_filename: str = None, output_folder_name: str = None):
    """Main function to reload checkpoint and regenerate plots"""
    
    # ============================================================================
    # HARDCODED CONFIGURATION - Modify these as needed
    # ============================================================================
    # Output folder name - will be created in ssvi_scripts_experiments directory
    if output_folder_name is None:
        OUTPUT_FOLDER_NAME = "mnist_ood_original_gplvm_replotting_results"
    else:
        OUTPUT_FOLDER_NAME = output_folder_name
    # ============================================================================
    
    # Hardcode output dir to create output folder in ssvi_scripts_experiments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER_NAME)
    
    print("=" * 70)
    print("Original GP-LVM MNIST OOD Results Replotting")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Set default dtype to float64
    torch.set_default_dtype(torch.float64)
    
    # 1. Load configuration
    print("\n1. Loading configuration...")
    config = load_config_from_results_dir(results_dir)
    print(f"  Device: {config.device}")
    print(f"  Random seed: {config.random_seed}")
    print(f"  ID Dataset: {config.id_dataset}")
    print(f"  OOD Dataset: {config.get_ood_dataset()}")
    
    # Auto-detect available models if model_name not specified
    if model_name is None:
        models_dir = os.path.join(results_dir, "pre_trained_models")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            # Extract model names (gauss or nn_gauss)
            models_to_process = []
            for f in model_files:
                if 'gauss' in f and 'nn_gauss' not in f:
                    if 'gauss' not in models_to_process:
                        models_to_process.append('gauss')
                elif 'nn_gauss' in f:
                    if 'nn_gauss' not in models_to_process:
                        models_to_process.append('nn_gauss')
            
            if not models_to_process:
                raise ValueError(f"No model files found in {models_dir}")
            print(f"  Auto-detected models: {models_to_process}")
        else:
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
    else:
        models_to_process = [model_name]
    
    # 2. Load data (only once, reuse for all models)
    print("\n2. Loading data...")
    y_train, y_test, labels_train, labels_test, split_data = load_data_for_plotting(config)
    print(f"  Train shape: {y_train.shape}, Test shape: {y_test.shape}")
    
    # Process each model
    for model_name in models_to_process:
        print("\n" + "=" * 70)
        print(f"Processing model: {model_name}")
        print("=" * 70)
        
        try:
            # 3. Load model checkpoint
            print(f"\n3. Loading model checkpoint ({model_name})...")
            state_dict = load_model_state_dict_from_checkpoint(results_dir, model_name, model_filename)
            
            # 4. Reconstruct model
            print(f"\n4. Reconstructing model ({model_name})...")
            model, likelihood = reconstruct_model(model_name, config, y_train, state_dict)
            
            # 5. Load metrics
            print(f"\n5. Loading metrics...")
            metrics = load_metrics_from_checkpoint(results_dir, model_name, config)
            
            # 6. Load losses
            print(f"\n6. Loading losses...")
            full_losses, batch_losses, full_loss_iterations = load_losses_from_checkpoint(results_dir, model_name, config)
            
            # Use metrics losses as fallback if full_losses is None
            if full_losses is None and metrics is not None:
                if 'checkpoints' in metrics and len(metrics['checkpoints']) > 0:
                    # Extract losses and iterations from checkpoints
                    # Check both 'full_loss' and 'full_dataset_elbo_loss' (used in checkpoint metrics)
                    full_losses = []
                    full_loss_iterations = []
                    for cp in metrics['checkpoints']:
                        loss = cp.get('full_dataset_elbo_loss', cp.get('full_loss', float('nan')))
                        if np.isfinite(loss):
                            full_losses.append(loss)
                            full_loss_iterations.append(cp.get('iteration', len(full_losses) - 1))
                    if full_losses:
                        print(f"  Using losses from checkpoint metrics ({len(full_losses)} points)")
                elif 'metrics_losses' in metrics:
                    full_losses = metrics['metrics_losses']
                    print(f"  Using losses from checkpoint metrics ({len(full_losses)} points)")
            
            # 7. Initialize logger
            print(f"\n7. Initializing logger...")
            logger = OriginalGPLVMResultsLogger(config, experiment_name="mnist_ood_replot")
            logger.results_dir = output_dir
            logger.plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(logger.plots_dir, exist_ok=True)
            print(f"Results will be saved to: {output_dir}")
            
            # 8. Regenerate plots
            print(f"\n8. Regenerating plots...")
            
            # 8.1 Main visualization plot
            print("  Generating main visualization plot...")
            try:
                # Use full_losses if available, otherwise extract from metrics checkpoints
                if full_losses is None and metrics is not None:
                    if 'checkpoints' in metrics and len(metrics['checkpoints']) > 0:
                        # Check both 'full_loss' and 'full_dataset_elbo_loss' (used in checkpoint metrics)
                        losses_for_plot = []
                        for cp in metrics['checkpoints']:
                            loss = cp.get('full_dataset_elbo_loss', cp.get('full_loss', float('nan')))
                            if np.isfinite(loss):
                                losses_for_plot.append(loss)
                        # Update full_loss_iterations if not already set
                        if full_loss_iterations is None:
                            full_loss_iterations = [cp.get('iteration', i) for i, cp in enumerate(metrics['checkpoints']) if np.isfinite(cp.get('full_dataset_elbo_loss', cp.get('full_loss', float('nan'))))]
                    elif 'metrics_losses' in metrics:
                        losses_for_plot = metrics['metrics_losses']
                    else:
                        losses_for_plot = []
                else:
                    losses_for_plot = full_losses if full_losses else []
                
                if losses_for_plot:
                    print(f"    Using losses for plot ({len(losses_for_plot)} points)")
                else:
                    print(f"    Warning: No losses available for plotting")
                
                create_standardized_scattered_plot_mnist_ood_replot(
                    model=model,
                    losses=losses_for_plot if losses_for_plot else [],
                    labels_train=torch.from_numpy(labels_train).to(dtype=torch.int64, device=y_train.device) if isinstance(labels_train, np.ndarray) else labels_train,
                    config=config,
                    results_dir=logger.plots_dir,
                    model_name=model.X.__class__.__name__,
                    random_seed=config.random_seed,
                    Y_train=y_train,
                    model_type="original",
                    train_indices=torch.from_numpy(split_data['train_indices']).to(dtype=torch.int64, device=y_train.device) if 'train_indices' in split_data else None,
                    dataset_name=config.id_dataset,
                    is_ood=False,
                    full_loss_iterations=full_loss_iterations,
                    batch_losses=batch_losses
                )
                print("    Main visualization plot saved (as PDF)")
            except Exception as e:
                print(f"    ✗ Error: Failed to generate main visualization: {e}")
                import traceback
                traceback.print_exc()
            
            # 8.2 Metrics progression plots
            print("  Generating metrics progression plots...")
            try:
                if metrics and 'checkpoints' in metrics:
                    train_rmse_list = [cp.get('train_rmse', float('nan')) for cp in metrics['checkpoints']]
                    train_nlpd_list = [cp.get('train_nlpd', float('nan')) for cp in metrics['checkpoints']]
                    metrics_iterations = [cp.get('iteration', i) for i, cp in enumerate(metrics['checkpoints'])]
                    # Check both 'full_loss' and 'full_dataset_elbo_loss' (used in checkpoint metrics)
                    metrics_losses = [cp.get('full_dataset_elbo_loss', cp.get('full_loss', float('nan'))) for cp in metrics['checkpoints']]
                    effective_dims_list = [cp.get('effective_dims', 0) for cp in metrics['checkpoints']]
                    
                    plot_metrics_progression(
                        train_rmse_list, train_nlpd_list, metrics_iterations,
                        metrics_losses, effective_dims_list, logger.plots_dir, model_name, config.random_seed,
                        batch_losses=batch_losses
                    )
                    print("    Metrics progression plots saved")
                else:
                    print("    No metrics data available")
            except Exception as e:
                print(f"    ✗ Error: Failed to generate metrics progression plots: {e}")
            
            # 8.3 Latent slice visualizations
            print("  Generating latent slice visualizations...")
            try:
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
                print(f"    Saved latent slice (origin) to: {latent_slice_path_origin}")
                
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
                    print(f"    Saved latent slice (point {idx}) to: {latent_slice_path}")
            except Exception as e:
                print(f"    ✗ Error: Failed to generate latent slice visualizations: {e}")
                import traceback
                traceback.print_exc()
            
            # 8.4 Prior samples visualization
            print("  Generating prior samples visualization...")
            try:
                prior_samples_path = os.path.join(logger.plots_dir, f"prior_samples_{model_name}_{config.random_seed}.pdf")
                visualize_prior_samples_original_gplvm(
                    model=model,
                    likelihood=likelihood,
                    n_images=24,
                    save_path=prior_samples_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get('scaler')
                )
                print(f"    Prior samples saved to: {prior_samples_path}")
                
                # Prior samples averaged
                prior_samples_avg_path = os.path.join(logger.plots_dir, f"prior_samples_averaged_{model_name}_{config.random_seed}.pdf")
                visualize_prior_samples_original_gplvm_averaged(
                    model=model,
                    likelihood=likelihood,
                    n_images=8,
                    n_samples=8,
                    save_path=prior_samples_avg_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get('scaler')
                )
                print(f"    Prior samples averaged saved to: {prior_samples_avg_path}")
            except Exception as e:
                print(f"    ✗ Error: Failed to generate prior samples: {e}")
                import traceback
                traceback.print_exc()
            
            # 8.5 Test image predictions visualization
            print("  Generating test image predictions visualization...")
            try:
                test_image_predictions_path = os.path.join(logger.plots_dir, f"test_image_predictions_{model_name}_{config.random_seed}.pdf")
                visualize_test_image_predictions_original_gplvm_replot(
                    model=model,
                    likelihood=likelihood,
                    y_test=split_data['y_test'],
                    config=config,
                    model_name=model_name,
                    n_images=8,
                    save_path=test_image_predictions_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get('scaler')
                )
                print(f"    Test image predictions saved to: {test_image_predictions_path}")
            except Exception as e:
                print(f"    ✗ Error: Failed to generate test image predictions: {e}")
                import traceback
                traceback.print_exc()
            
            # 9. Final model parameters
            print(f"\n9. Final model parameters:")
            try:
                # Check if model has likelihood attached
                if hasattr(model, 'likelihood') and model.likelihood is not None:
                    obs_noise = model.likelihood.noise.item()
                    print(f"Final Original GP-LVM ({model_name}) Model Parameters:")
                    print(f"  Observation noise (sigma2): {obs_noise:.4f}")
                    
                    # Print lengthscales
                    lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
                    print(f"  Lengthscales: {lengthscales}")
                    
                    # Print output scale
                    output_scale = model.covar_module.outputscale.item()
                    print(f"  Output scale: {output_scale:.4f}")
                else:
                    print_final_model_parameters(model, f"Original GP-LVM ({model_name})")
            except Exception as e:
                print(f"  Warning: Could not print model parameters: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "=" * 70)
            print(f"All plots regenerated successfully for {model_name}!")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n✗ Error processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    # ============================================================================
    # HARDCODED CONFIGURATION - Modify these as needed
    # ============================================================================
    # Hardcoded results folder path
    # RESULTS_FOLDER_PATH = "/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/ssvi_scripts_experiments/ood_mnist_related/fashion_mnist_ood_experiment_final_v2_orig_gplvm/ood_run_results/orig_gplvm_ood_gplvm_FashionMNIST_ood_fmnist_id_mnist_ood_n_ind_64_arch_768_len_10_b_32_s_2_lr_00001_n_steps_200000_test_01_seed_2_01_20_01_53"
    RESULTS_FOLDER_PATH = "/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/ssvi_scripts_experiments/ood_mnist_related/reverse_mnist_ood_experiment_final_v3_orig_gplvm/ood_run_results/orig_gplvm_ood_gplvm_MNIST_ood_mnist_id_fmnist_ood_n_ind_64_arch_768_len_10_b_32_s_4_lr_00001_n_steps_200000_test_01_seed_4_01_21_16_05"
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
    
    print(f"Using results directory: {results_dir}")
    
    main(results_dir, model_name=MODEL_NAME)

