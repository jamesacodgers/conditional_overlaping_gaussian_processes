# %%
"""
Replot Original GP-LVM Oil Flow Results from Checkpoint

This script loads a trained Original GP-LVM model from a checkpoint and regenerates all plots
without retraining the model.

Usage:
    python ssvi_scripts_experiments/oil_flow_original_gplvm_replot_results.py
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
    plot_latent_slice_original_gplvm_replot,
    plot_latent_slice_original_gplvm_replot_single,
    create_standardized_scattered_plot_replot
)
from src.missing_data_utils import preprocess_data_for_missing_training
from src.data_utils import download_oilflow_data_if_needed
from src.save_utils import print_final_model_parameters


class OilFlowModel(BayesianGPLVM):
    """Oil Flow Model class - same as in training script"""
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
        super(OilFlowModel, self).__init__(X, q_f)
        
        # Kernel 
        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        
        # DO NOT initialize from config - all parameters should be loaded from checkpoint
        # The model structure is created here, but parameter values come from state_dict

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


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
        # Look for files like oilflow_gauss_7.pkl or oilflow_nn_gauss_7.pkl
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
        # Infer architecture from state_dict if available (in case config doesn't match checkpoint)
        nn_layers = config.nn_layers
        if state_dict:
            # Extract architecture from mu_layers weights in state_dict
            # Format: X.mu_layers.0.weight has shape [out_features, in_features]
            mu_layer_keys = [k for k in state_dict.keys() if k.startswith('X.mu_layers.') and k.endswith('.weight')]
            if mu_layer_keys:
                # Sort by layer index
                mu_layer_keys.sort(key=lambda x: int(x.split('.')[2]))
                inferred_layers = []
                for i, key in enumerate(mu_layer_keys):
                    weight_shape = state_dict[key].shape
                    out_features = weight_shape[0]
                    in_features = weight_shape[1]
                    
                    # First layer: in_features should be data_dim, out_features is first hidden layer
                    # Middle layers: out_features is next hidden layer
                    # Last layer: out_features should be latent_dim (don't add to layers tuple)
                    if i == 0:
                        # First layer: verify input is data_dim
                        if in_features != data_dim:
                            print(f"  Warning: First layer input size ({in_features}) doesn't match data_dim ({data_dim})")
                        # Add output size as first hidden layer
                        inferred_layers.append(out_features)
                    elif i == len(mu_layer_keys) - 1:
                        # Last layer: verify output is latent_dim
                        if out_features != latent_dim:
                            print(f"  Warning: Last layer output size ({out_features}) doesn't match latent_dim ({latent_dim})")
                        # Don't add latent_dim to layers tuple (it's added automatically by NNEncoder)
                    else:
                        # Middle layers: add output size to inferred_layers
                        inferred_layers.append(out_features)
                
                if inferred_layers:
                    nn_layers = tuple(inferred_layers)
                    print(f"  Inferred nn_layers architecture from checkpoint: {nn_layers} (config had: {config.nn_layers})")
                else:
                    print(f"  Warning: Could not infer architecture, using config.nn_layers: {config.nn_layers}")
            else:
                print(f"  Warning: No mu_layers found in state_dict, using config.nn_layers: {config.nn_layers}")
        
        prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1], dtype=torch.float64, device=device))
        X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    # Create model
    # NOTE: Do NOT pass config here - we want to load everything from checkpoint, not initialize from config
    # The config is only used for architecture (n_inducing, etc.), not for parameter values
    model = OilFlowModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers, device=device, config=None)
    
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
    
    # Debug: Print all keys in state_dict to see what's available
    print(f"  DEBUG: Total keys in checkpoint state_dict: {len(state_dict)}")
    print(f"  DEBUG: Model keys (excluding likelihood): {len(model_state_dict)}")
    
    # Debug: Check for key model components
    key_components = {
        'X.q_mu': 'Latent variable mean (gauss)',
        'X.q_log_sigma': 'Latent variable log_sigma (gauss)',
        'X.mu_layers': 'Encoder mu layers (nn_gauss)',
        'X.sigma_layers': 'Encoder sigma layers (nn_gauss)',
        'covar_module.base_kernel.raw_lengthscale': 'Kernel lengthscales',
        'covar_module.raw_outputscale': 'Kernel outputscale',
        'inducing_inputs': 'Inducing points',
        'variational_strategy': 'Variational strategy (q_u)',
        'variational_distribution': 'Variational distribution (q_u)',
        'q_u': 'Variational distribution q_u'
    }
    
    # Also print ALL keys to see what's actually there
    print(f"  DEBUG: All keys in checkpoint state_dict:")
    for i, key in enumerate(sorted(state_dict.keys())):
        print(f"    {i+1}. {key}")
    
    print(f"  DEBUG: Checking for key model components in checkpoint:")
    for key_pattern, description in key_components.items():
        found = [k for k in model_state_dict.keys() if key_pattern in k]
        if found:
            print(f"    ✓ Found {description}: {len(found)} key(s) - {found[0] if found else 'N/A'}")
        else:
            print(f"    ✗ Missing {description}: {key_pattern}")
    
    # Debug: Print available likelihood keys in state_dict
    likelihood_keys_in_checkpoint = [k for k in state_dict.keys() if k.startswith('likelihood.')]
    if likelihood_keys_in_checkpoint:
        print(f"  Found {len(likelihood_keys_in_checkpoint)} likelihood keys in checkpoint:")
        for key in likelihood_keys_in_checkpoint[:10]:  # Show first 10 keys
            print(f"    - {key}")
        if len(likelihood_keys_in_checkpoint) > 10:
            print(f"    ... and {len(likelihood_keys_in_checkpoint) - 10} more")
    else:
        print(f"  Warning: No likelihood keys found in checkpoint state_dict")
        print(f"  Available state_dict keys (first 20): {list(state_dict.keys())[:20]}")
    
    # Load state_dict (use strict=False to handle any remaining mismatches gracefully)
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys in state_dict: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Warning: Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys in state_dict: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Warning: Unexpected keys in state_dict: {unexpected_keys}")
    
    # Debug: Verify loaded parameters after loading state_dict
    print(f"  DEBUG: Verifying loaded parameters:")
    if model_name == 'gauss':
        if hasattr(model.X, 'q_mu'):
            q_mu_mean = model.X.q_mu.mean().item()
            q_mu_std = model.X.q_mu.std().item()
            print(f"    ✓ X.q_mu: mean={q_mu_mean:.4f}, std={q_mu_std:.4f}")
        else:
            print(f"    ✗ X.q_mu: NOT FOUND")
        if hasattr(model.X, 'q_log_sigma'):
            q_log_sigma_mean = model.X.q_log_sigma.mean().item()
            q_log_sigma_std = model.X.q_log_sigma.std().item()
            print(f"    ✓ X.q_log_sigma: mean={q_log_sigma_mean:.4f}, std={q_log_sigma_std:.4f}")
            # Convert to actual sigma
            sigma_x = torch.nn.functional.softplus(model.X.q_log_sigma)
            print(f"    ✓ X.sigma (from q_log_sigma): mean={sigma_x.mean().item():.4f}, std={sigma_x.std().item():.4f}")
        else:
            print(f"    ✗ X.q_log_sigma: NOT FOUND")
    elif model_name == 'nn_gauss':
        if hasattr(model.X, 'mu_layers'):
            print(f"    ✓ X.mu_layers: {len(list(model.X.mu_layers))} layers")
        else:
            print(f"    ✗ X.mu_layers: NOT FOUND")
        if hasattr(model.X, 'sigma_layers'):
            print(f"    ✓ X.sigma_layers: {len(list(model.X.sigma_layers))} layers")
        else:
            print(f"    ✗ X.sigma_layers: NOT FOUND")
    
    # Check kernel parameters
    if hasattr(model, 'covar_module'):
        if hasattr(model.covar_module.base_kernel, 'lengthscale'):
            lengthscales = model.covar_module.base_kernel.lengthscale.flatten().detach().cpu().numpy()
            print(f"    ✓ Kernel lengthscales: {lengthscales}")
        if hasattr(model.covar_module, 'outputscale'):
            outputscale = model.covar_module.outputscale.item()
            print(f"    ✓ Kernel outputscale: {outputscale:.6f}")
    
    # Check inducing points
    if hasattr(model, 'inducing_inputs'):
        inducing_shape = model.inducing_inputs.shape
        inducing_mean = model.inducing_inputs.mean().item()
        print(f"    ✓ Inducing points: shape={inducing_shape}, mean={inducing_mean:.4f}")
    else:
        print(f"    ✗ Inducing points: NOT FOUND")
    
    # Check variational strategy parameters (critical for GP predictions!)
    if hasattr(model, 'variational_strategy'):
        vs = model.variational_strategy
        print(f"    ✓ Variational strategy exists")
        if hasattr(vs, 'variational_distribution'):
            vd = vs.variational_distribution
            print(f"    ✓ Variational distribution exists: {type(vd).__name__}")
            # Check if it has parameters
            if hasattr(vd, 'parameters'):
                vd_params = list(vd.parameters())
                print(f"    ✓ Variational distribution has {len(vd_params)} parameter(s)")
                for i, param in enumerate(vd_params):
                    print(f"      Param {i}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
            else:
                print(f"    ✗ Variational distribution has no parameters() method")
        else:
            print(f"    ✗ Variational strategy has no variational_distribution")
    else:
        print(f"    ✗ Model has no variational_strategy attribute")
    
    # Try to load likelihood state from checkpoint (this should contain the trained noise value)
    likelihood_state_dict = {k.replace('likelihood.', ''): v for k, v in state_dict.items() if k.startswith('likelihood.')}
    if likelihood_state_dict:
        try:
            # Filter out constraint-related keys that can't be loaded
            loadable_likelihood_keys = {k: v for k, v in likelihood_state_dict.items() 
                                      if not k.endswith('_constraint.lower_bound') and not k.endswith('_constraint.upper_bound')}
            if loadable_likelihood_keys:
                likelihood.load_state_dict(loadable_likelihood_keys, strict=False)
                # Verify likelihood noise was loaded correctly
                loaded_noise = likelihood.noise.item()
                print(f"  ✓ Loaded likelihood noise from checkpoint: {loaded_noise:.6f}")
            else:
                print(f"  Warning: No loadable likelihood keys found after filtering")
                # Fallback: initialize from config
                if config is not None:
                    with torch.no_grad():
                        noise_scale = torch.tensor(config.init_output_noise_scale, dtype=torch.float64, device=device)
                        likelihood.initialize(noise=noise_scale)
                    print(f"  Using config initialization (fallback): {likelihood.noise.item():.6f}")
        except Exception as e:
            print(f"  Warning: Could not load likelihood state: {e}")
            # Fallback: initialize from config
            if config is not None:
                with torch.no_grad():
                    noise_scale = torch.tensor(config.init_output_noise_scale, dtype=torch.float64, device=device)
                    likelihood.initialize(noise=noise_scale)
                print(f"  Using config initialization (fallback): {likelihood.noise.item():.6f}")
    else:
        print(f"  Warning: No likelihood state found in checkpoint")
        # Fallback: initialize from config (this is just the initial value, not the trained value)
        if config is not None:
            with torch.no_grad():
                noise_scale = torch.tensor(config.init_output_noise_scale, dtype=torch.float64, device=device)
                likelihood.initialize(noise=noise_scale)
            print(f"  Using config initialization (fallback - this is initial value, not trained): {likelihood.noise.item():.6f}")
    
    # Attach likelihood to model (some functions expect model.likelihood)
    model.likelihood = likelihood
    
    # Debug: Verify likelihood is attached and has correct noise value
    print(f"  DEBUG reconstruct_model: model.likelihood exists = {hasattr(model, 'likelihood')}")
    if hasattr(model, 'likelihood') and model.likelihood is not None:
        print(f"  DEBUG reconstruct_model: model.likelihood.noise = {model.likelihood.noise.item():.6f}")
    else:
        print(f"  DEBUG reconstruct_model: WARNING - model.likelihood is None!")
    
    # Debug: Print kernel parameters to verify they're loaded correctly
    if hasattr(model, 'covar_module'):
        lengthscales = model.covar_module.base_kernel.lengthscale.flatten().detach().cpu().numpy()
        outputscale = model.covar_module.outputscale.item()
        print(f"  DEBUG reconstruct_model: kernel lengthscales = {lengthscales}")
        print(f"  DEBUG reconstruct_model: kernel outputscale = {outputscale:.6f}")
    
    # Ensure model and likelihood are in float64 (same as training)
    model = model.to(dtype=torch.float64)
    likelihood = likelihood.to(dtype=torch.float64)
    
    # Debug: Verify likelihood noise after dtype conversion
    if hasattr(model, 'likelihood') and model.likelihood is not None:
        print(f"  DEBUG reconstruct_model: After dtype conversion, model.likelihood.noise = {model.likelihood.noise.item():.6f}")
    
    # Fix device placement for NNEncoder tensors (created on CPU in original code)
    # This matches the training script exactly (lines 297-306) - critical for nn_gauss models
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
    """Load and preprocess oil flow data (same as training script)"""
    # Download data if needed
    download_oilflow_data_if_needed()
    
    # Load data
    N, d, q, X, Y, labels = load_real_data(config.dataset)
    
    # Ensure data is on CPU before converting to numpy for sklearn
    Y_cpu = Y.cpu() if Y.is_cuda else Y
    labels_cpu = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    Y_np = Y_cpu.numpy()
    indices = np.arange(len(Y))
    
    # Split data and labels together to maintain correspondence
    train_indices, test_indices, Y_train_np, Y_test_np, lb_train_np, lb_test_np = train_test_split(
        indices, Y_np, labels_cpu,
        test_size=config.test_size,
        random_state=config.random_seed
    )
    
    # Convert back to tensors on original device
    device = config.device if config.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    Y_train = torch.tensor(Y_train_np, dtype=torch.float64, device=device)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float64, device=device)
    lb_train = lb_train_np  # Keep as numpy array for compatibility
    lb_test = lb_test_np    # Keep as numpy array for compatibility
    
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
    # Pattern matches save_training_losses: f"{model_name}_{random_seed}_losses.json"
    # When model_name is "gauss_full_loss", filename becomes "gauss_full_loss_42_losses.json"
    full_loss_path = os.path.join(results_dir, "training_results", f"{model_name}_full_loss_{config.random_seed}_losses.json")
    batch_loss_path = os.path.join(results_dir, "training_results", f"{model_name}_batch_{config.random_seed}_losses.json")
    
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
    Main function to replot Original GP-LVM results
    
    Args:
        results_dir: Path to results directory containing config.json and pre_trained_models/
        model_name: Model name ('gauss' or 'nn_gauss'). If None, will try to detect from available models.
        model_filename: Specific model filename. If None, will auto-detect.
    """
    print("=" * 70)
    print("Original GP-LVM Oil Flow Results Replotting")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    
    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_FOLDER_NAME = "oilflow_original_gplvm_replotting_results"
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
    # Extract model names from filenames (e.g., oilflow_gauss_7.pkl -> gauss)
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
            plot_filename = f"oilflow_{model_name}_{config.random_seed}.pdf"
            plot_path = os.path.join(logger.plots_dir, plot_filename)
            
            # Create plot (saves as PDF)
            create_standardized_scattered_plot_replot(
                model=model,
                losses=losses_for_plot if losses_for_plot else [],
                labels_train=labels_train,
                config=config,
                results_dir=logger.plots_dir,  # Pass plots_dir directly
                model_name=model_name,
                random_seed=config.random_seed,
                Y_train=y_train,
                model_type="original",
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
                save_path=latent_slice_path_origin
            )
            if os.path.exists(latent_slice_path_origin):
                print(f"    ✓ Saved latent slice (origin) to: {latent_slice_path_origin}")
            
            # Plot single slice centered at origin (y[0] vs x[j] only)
            latent_slice_single_path_origin = os.path.join(plots_dir, f"latent_slice_single_origin_{model_name}_seed_{config.random_seed}.pdf")
            plot_latent_slice_original_gplvm_replot_single(
                model=model,
                y_train=y_train,
                model_name=model_name,
                center="zeros",
                save_path=latent_slice_single_path_origin
            )
            if os.path.exists(latent_slice_single_path_origin):
                print(f"    ✓ Saved latent slice single (origin) to: {latent_slice_single_path_origin}")
            
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
                    save_path=latent_slice_path
                )
                if os.path.exists(latent_slice_path):
                    print(f"    ✓ Saved latent slice (point {idx}) to: {latent_slice_path}")
                
                # Plot single slice for this point (y[0] vs x[j] only)
                latent_slice_single_path = os.path.join(plots_dir, f"latent_slice_single_point_{idx}_{model_name}_seed_{config.random_seed}.pdf")
                plot_latent_slice_original_gplvm_replot_single(
                    model=model,
                    y_train=y_train,
                    model_name=model_name,
                    center="data_point",
                    n_idx=idx,
                    save_path=latent_slice_single_path
                )
                if os.path.exists(latent_slice_single_path):
                    print(f"    ✓ Saved latent slice single (point {idx}) to: {latent_slice_single_path}")
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
    RESULTS_FOLDER_PATH = "ssvi_scripts_experiments/oil_flow_related/final_v1_orig_gplvm/orig_gplvm_run_results/orig_gplvm_n_ind_32_arch_192_len_1_sig_1_b_32_s_1_lr_0001_n_steps_100000_P_64_miss_f_ratio_00_test_01_seed_1_01_24_18_51"
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

