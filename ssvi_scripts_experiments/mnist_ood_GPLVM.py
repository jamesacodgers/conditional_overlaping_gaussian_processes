#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST OOD Detection with Original GP-LVM (BayesianGPLVM)

Extended from mnist_GPLVM.py to add OOD detection capability.

Supports two OOD modes:
1. Cross-dataset OOD: ID and OOD from different datasets
   - FashionMNIST (in-distribution) vs MNIST (OOD)
   - MNIST (in-distribution) vs FashionMNIST (OOD)
2. Same-dataset OOD (class-based): ID and OOD from same dataset, different classes
   - Example: 8 FashionMNIST classes (ID) vs 2 held-out classes (OOD)
   - Supports splits: (0,1), (2,3), (4,5), (6,7), (8,9) as held-out OOD classes

OOD Metrics computed:
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- FPR@80 (False Positive Rate at 80% True Positive Rate)

2 different inference modes:
   models = ['gauss','nn_gauss']

Usage examples:
    # Run with default fast config (cross-dataset: FashionMNIST as ID, MNIST as OOD)
    python ssvi_scripts_experiments/mnist_ood_GPLVM.py
    
    # Run with MNIST as in-distribution (cross-dataset: MNIST as ID, FashionMNIST as OOD)
    python ssvi_scripts_experiments/mnist_ood_GPLVM.py --id-dataset MNIST
    
    # Same-dataset OOD: 8 FashionMNIST classes as ID, classes 8,9 as OOD
    python ssvi_scripts_experiments/mnist_ood_GPLVM.py \\
        --id-dataset FashionMNIST --id-classes 0 1 2 3 4 5 6 7 \\
        --ood-dataset FashionMNIST --ood-classes 8 9
    
    # Same-dataset OOD with convenience flag
    python ssvi_scripts_experiments/mnist_ood_GPLVM.py \\
        --id-dataset FashionMNIST --id-classes 0 1 2 3 4 5 6 7 \\
        --same-dataset-ood --ood-classes 8 9
    
    # Run with base config
    python ssvi_scripts_experiments/mnist_ood_GPLVM.py --config ssvi_scripts_experiments/original_gp_lvm_configs/base_lr_0001_s7.yaml
"""

import sys
import os
# Get the project root directory (parent of ssvi_scripts_experiments)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add both project root and generalised_gplvm_code to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

from dataclasses import dataclass
from typing import List
from utils.data import load_real_data 
import pickle as pkl
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import PointLatentVariable, MAPLatentVariable, VariationalLatentVariable, NNEncoder
from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior, MultivariateNormalPrior
from gpytorch.likelihoods import GaussianLikelihood
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'generalised_gplvm_code'))
from models.likelihoods import GaussianLikelihoodWithMissingObs
from utils.metrics import rmse_missing
from utils.visualisation import plot_y_reconstruction
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from sklearn.model_selection import train_test_split
from gpytorch.constraints import Positive
from torch.distributions import transforms
import json
from datetime import datetime

# Import metrics
from utils.metrics import rmse

# Import configuration system
from src.original_gplvm_config import OriginalGPLVMConfig
from src.original_gplvm_config_utils import (
    create_argument_parser, load_config_with_overrides, 
    ensure_config_dir_exists, print_available_configs
)
from src.original_gplvm_results_logger import OriginalGPLVMResultsLogger
from src.nlpd_utils import compute_nlpd_nll_based, compute_crps_nll_based, compute_all_crps_nll_based, compute_coverage_nll_based, compute_missing_data_nlpd_gauss
from src.original_gplvm_metrics_plot import plot_metrics_progression, compute_effective_latent_dims, plot_missing_nlpd_metrics_progression, plot_missing_rmse_metrics_progression, plot_crps_metrics_progression, plot_crps_posterior_predictive_progression
from src.missing_data_utils import preprocess_data_for_missing_training, rmse_missing_improved
from src.missing_data_experiment import run_missing_data_reconstruction_experiment
from src.unified_rmse_utils import compute_rmse_with_sampling, compute_missing_data_rmse_with_sampling
from src.data_utils import load_mnist_data, filter_mnist_by_digits, load_image_dataset
from src.mnist_missing_reconstruction import visualize_mnist_missing_reconstruction_gauss, visualize_mnist_missing_reconstruction_gauss_detailed, visualize_prior_samples_original_gplvm, visualize_prior_samples_original_gplvm_averaged, visualize_test_image_predictions_original_gplvm
from src.knn_clustering_metrics import compute_knn_accuracies_original_gplvm
from src.knn_progression_plot import plot_knn_metrics_progression

# Import OOD metrics
from src.ood_metrics import (
    evaluate_ood_detection_original_gplvm,
    compute_all_ood_metrics,
    OODMetrics
)
from src.pytorch_scalers import MeanCenter, StandardScalerPytorch, MinMaxScalerPytorch, NoTransform

plt.style.use('ggplot')


# === OOD-Specific Configuration Extension ===
@dataclass
class OODOriginalGPLVMConfig(OriginalGPLVMConfig):
    """Extended configuration for OOD detection experiments with Original GP-LVM.
    
    Supports two OOD modes:
    1. Cross-dataset OOD: ID and OOD from different datasets (MNIST vs FashionMNIST)
    2. Same-dataset OOD: ID and OOD from same dataset but different classes
       Example: 8 FashionMNIST classes as ID, 2 classes as OOD
    
    Configuration:
    - id_dataset: Name of in-distribution dataset ("MNIST" or "FashionMNIST")
    - id_classes: List of class indices for ID (None = all classes)
    - id_samples_per_class: Samples per ID class (None = use all)
    - ood_dataset: Name of OOD dataset (must be explicitly specified: "MNIST" or "FashionMNIST")
    - ood_classes: List of class indices for OOD (None = all classes)
    - ood_samples_per_class: Samples per OOD class (None = use all)
    
    Examples:
    ---------
    Cross-dataset OOD (FashionMNIST ID, MNIST OOD):
        id_dataset="FashionMNIST", id_classes=[0,1,2,3,4,5,6,7,8,9],
        ood_dataset="MNIST"
    
    Same-dataset OOD (8 FashionMNIST classes ID, 2 classes OOD):
        id_dataset="FashionMNIST", id_classes=[0,1,2,3,4,5,6,7],
        ood_dataset="FashionMNIST", ood_classes=[8,9]
    """
    # === In-Distribution (ID) Dataset Configuration ===
    id_dataset: str = "FashionMNIST"      # ID dataset name: "MNIST" or "FashionMNIST"
    id_classes: List[int] = None           # ID class indices (None = all classes)
    id_samples_per_class: int = None       # Samples per ID class (None = use all)
    
    # === Out-of-Distribution (OOD) Dataset Configuration ===
    ood_dataset: str = "MNIST"            # OOD dataset name (must be explicitly specified)
    ood_classes: List[int] = None          # OOD class indices (None = all classes)
    ood_samples_per_class: int = None      # Samples per OOD class (None = use all)
    
    # OOD Evaluation configuration
    ood_score_types: List[str] = None      # Will be set to ['nlpd', 'rmse', 'variance'] if None
    ood_n_samples: int = None              # Number of samples for OOD evaluation (None = use default 4)
    
    # === Data Standardization Configuration ===
    standardization: str = "mean"      # "standard" (mean=0, std=1), "minmax" ([0,1]), or "none" (no transform)
    
    def __post_init__(self):
        super().__post_init__()
        if self.ood_score_types is None:
            self.ood_score_types = ['nlpd', 'rmse', 'variance']
    
    def get_id_filter_params(self):
        """Get filtering parameters for in-distribution dataset"""
        return self.id_classes, self.id_samples_per_class
    
    def get_ood_filter_params(self):
        """Get filtering parameters for out-of-distribution dataset"""
        return self.ood_classes, self.ood_samples_per_class
    
    def get_ood_dataset(self):
        """Get the OOD dataset name (always explicitly specified)."""
        return self.ood_dataset
    
    def is_same_dataset_ood(self):
        """Check if ID and OOD are from the same dataset (class-based OOD)"""
        return self.ood_dataset == self.id_dataset
    
    def print_config(self):
        """Print configuration summary including OOD settings"""
        super().print_config()
        print("=== OOD Detection Configuration ===")
        print(f"In-Distribution Dataset: {self.id_dataset}")
        print(f"  ID Classes: {self.id_classes if self.id_classes else 'all'}")
        print(f"  ID Samples/Class: {self.id_samples_per_class if self.id_samples_per_class else 'all'}")
        print(f"OOD Dataset: {self.get_ood_dataset()}")
        print(f"  OOD Classes: {self.ood_classes if self.ood_classes else 'all'}")
        print(f"  OOD Samples/Class: {self.ood_samples_per_class if self.ood_samples_per_class else 'all'}")
        if self.is_same_dataset_ood():
            print(f"  Mode: Same-dataset OOD (class-based)")
        else:
            print(f"  Mode: Cross-dataset OOD")
        print(f"OOD Score Types: {self.ood_score_types}")
        print(f"OOD N Samples: {self.ood_n_samples if self.ood_n_samples else 'default (4)'}")
        print(f"Data Standardization: {self.standardization}")
        print("=" * 50)


def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class MNISTModel(BayesianGPLVM):
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
        #self.mean_module = ConstantMean(ard_num_dims=latent_dim)
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
                print(f"  DEBUG GP-LVM: Setting lengthscale to sqrt({config.init_lengthscales}) = {lengthscale_device.item():.6f}")
                self.covar_module.base_kernel.initialize(lengthscale=lengthscale_device)
                print(f"  DEBUG GP-LVM: After setting, lengthscale = {self.covar_module.base_kernel.lengthscale.flatten()[0]}")


     def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist
    
     def _get_batch_idx(self, batch_size):
            
         valid_indices = np.arange(self.n)
         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
         return np.sort(batch_indices)


def load_ood_data(config: OODOriginalGPLVMConfig):
    """Load and preprocess data with train/test split, plus OOD data.
    
    Supports two OOD modes:
    1. Cross-dataset OOD: ID and OOD from different datasets (e.g., FashionMNIST vs MNIST)
    2. Same-dataset OOD: ID and OOD from same dataset but different classes
       (e.g., 8 FashionMNIST classes as ID, 2 as OOD)
    """
    print(f"Loading {config.id_dataset} data as in-distribution...")
    
    # Load in-distribution data using load_image_dataset
    Y, labels = load_image_dataset(config.id_dataset)
    
    # Normalize to [0, 1]
    Y = Y / 255.0
    
    # Apply ID class filtering
    id_classes, id_samples_per_class = config.get_id_filter_params()
    Y, labels = filter_mnist_by_digits(
        Y, labels, 
        selected_digits=id_classes,
        samples_per_digit=id_samples_per_class
    )

    N = len(Y)
    
    print(f"Using {N} {config.id_dataset} samples for training/testing")
    if id_classes is not None:
        print(f"  ID classes: {id_classes}")
    
    # Ensure data is on CPU before converting to numpy for sklearn
    Y_cpu = Y.cpu() if Y.is_cuda else Y
    # Handle labels
    if hasattr(labels, 'cpu'):
        labels_cpu = labels.cpu().numpy()
    elif hasattr(labels, 'numpy'):
        labels_cpu = labels.numpy()
    else:
        labels_cpu = np.array(labels)
    Y_np = Y_cpu.numpy()
    indices = np.arange(len(Y))
    
    # Split data and labels together to maintain correspondence
    train_indices, test_indices, Y_train_np, Y_test_np, lb_train_np, lb_test_np = train_test_split(
        indices, Y_np, labels_cpu,
        test_size=config.test_size,
        random_state=config.random_seed
    )
    
    # Convert back to tensors on original device
    Y_train = torch.tensor(Y_train_np, dtype=torch.float64, device=config.device)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float64, device=config.device)
    lb_train = lb_train_np
    lb_test = lb_test_np
    
    print(f"Data split shapes - Train: {Y_train.shape}, Test: {Y_test.shape}")
    
    # === Load OOD data ===
    ood_dataset = config.get_ood_dataset()
    is_same_dataset = config.is_same_dataset_ood()
    
    if is_same_dataset:
        print(f"\nLoading {ood_dataset} data as OOD (same-dataset, different classes)...")
    else:
        print(f"\nLoading {ood_dataset} data as OOD (cross-dataset)...")
    
    Y_ood, labels_ood = load_image_dataset(ood_dataset, test_only=True)
    Y_ood = Y_ood / 255.0  # Normalize to [0, 1]
    
    # Apply OOD class filtering
    ood_classes, ood_samples_per_class = config.get_ood_filter_params()
    if ood_classes is not None or ood_samples_per_class is not None:
        Y_ood, labels_ood = filter_mnist_by_digits(
            Y_ood, labels_ood, 
            selected_digits=ood_classes,
            samples_per_digit=ood_samples_per_class
        )
    
    if ood_classes is not None:
        print(f"  OOD classes: {ood_classes}")
    
    # Use all OOD data (no additional sampling limit)
    y_ood = Y_ood.to(config.device)
    
    print(f"OOD data shape: {y_ood.shape}")

    # === Data Standardization ===
    # Fit scaler on training data only (before missing data preprocessing)
    print(f"\n=== Data Standardization ===")
    print(f"Standardization method: {config.standardization}")
    
    if config.standardization == "standard":
        scaler = StandardScalerPytorch(device=config.device)
    elif config.standardization == "minmax":
        scaler = MinMaxScalerPytorch(device=config.device)
    elif config.standardization == "mean":
        scaler = MeanCenter(device=config.device)
    elif config.standardization == "none":
        scaler = NoTransform(device=config.device)
    else:
        raise ValueError(f"Unknown standardization method: {config.standardization}. Must be 'standard', 'mean', 'minmax', or 'none'")
    
    # Fit scaler on training data only (critical: don't mix test data)
    print(f"Fitting scaler on training data only ({Y_train.shape[0]} samples)...")
    scaler.fit(Y_train)
    
    # Transform all data using the scaler fitted on training data
    print(f"Transforming train/test/OOD data...")
    print(f"  Before transform - Train shape: {Y_train.shape}, Test shape: {Y_test.shape}, OOD shape: {y_ood.shape}")
    print(f"  Scaler mean shape: {scaler.mean.shape}")
    print(f"  Before transform - Train mean: {Y_train.mean().item():.6f}, Test mean: {Y_test.mean().item():.6f}, OOD mean: {y_ood.mean().item():.6f}")
    print(f"  Before transform - Train std: {Y_train.std().item():.6f}, Test std: {Y_test.std().item():.6f}, OOD std: {y_ood.std().item():.6f}")
    
    Y_train = scaler.transform(Y_train)
    Y_test = scaler.transform(Y_test)
    y_ood = scaler.transform(y_ood)
    
    # Debug: Show statistics after transform
    print(f"  After transform - Train mean: {Y_train.mean().item():.6f}, Test mean: {Y_test.mean().item():.6f}, OOD mean: {y_ood.mean().item():.6f}")
    print(f"  After transform - Train std: {Y_train.std().item():.6f}, Test std: {Y_test.std().item():.6f}, OOD std: {y_ood.std().item():.6f}")
    print(f"  After transform - Train min: {Y_train.min().item():.6f}, Test min: {Y_test.min().item():.6f}, OOD min: {y_ood.min().item():.6f}")
    print(f"  After transform - Train max: {Y_train.max().item():.6f}, Test max: {Y_test.max().item():.6f}, OOD max: {y_ood.max().item():.6f}")
    
    print(f"Standardization completed.")
    print("=" * 50)

    split_data = {
        'y_train': Y_train,
        'y_test': Y_test,
        'labels_train': lb_train,
        'labels_test': lb_test,
        'train_indices': train_indices,
        'test_indices': test_indices,
        # OOD data
        'y_ood': y_ood,
        'labels_ood': labels_ood,
        # Scaler for inverse transforms (needed for visualization)
        'scaler': scaler
    }
    
    return Y_train, Y_test, lb_train, lb_test, split_data


def compute_full_elbo_minibatched(model, Y_train, elbo, model_name, config):
    """
    Compute full dataset ELBO using minibatches (same as training, just average batches).
    
    This processes data in minibatches to reduce memory usage, using the exact same
    computation as in the training loop, then averages the batch ELBOs.
    
    Args:
        model: The GP-LVM model
        Y_train: Full training data [N, M]
        elbo: VariationalELBO object
        model_name: Name of the model ('gauss', 'nn_gauss', etc.)
        config: Configuration object with batch_size attribute
        
    Returns:
        Full dataset ELBO (scalar float)
    """
    batch_size = config.batch_size
    N = len(Y_train)
    
    batch_elbos = []
    
    # Sample once for the full dataset
    if model_name in ['point', 'map', 'gauss']:
        # For 'point', 'map', 'gauss': sample once from full latent
        full_sample = model.sample_latent_variable()
    else:  # nn_gauss
        # For 'nn_gauss': encode full dataset once, then index for batches (same as training)
        if config.enable_missing_data:
            Y_train_for_encoder = Y_train.clone()
            Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -9.0
            full_sample = model.sample_latent_variable(Y_train_for_encoder)
        else:
            full_sample = model.sample_latent_variable(Y_train)
    
    # Process data in minibatches
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_indices = torch.arange(start_idx, end_idx, dtype=torch.int64, device=config.device)
        Y_batch = Y_train[batch_indices]
        
        # Index the full sample for this batch
        sample_batch = full_sample[batch_indices]
        
        # Compute output for this batch
        output_batch = model(sample_batch)
        
        # Compute ELBO for this batch
        batch_elbo = elbo(output_batch, Y_batch.T).sum()
        batch_elbos.append(batch_elbo)
    
    # Average the batch ELBOs
    # Stack tensors and take mean to handle device placement correctly
    if batch_elbos:
        mean_elbo = torch.stack(batch_elbos).mean()
        return mean_elbo.item()
    else:
        return 0.0


# === OOD Evaluation Functions ===

def evaluate_ood_detection_metrics(model, split_data, config: OODOriginalGPLVMConfig, 
                                    model_name: str, likelihood, logger=None):
    """Evaluate OOD detection performance for Original GP-LVM"""
    print(f"\n=== OOD Detection Evaluation ({model_name}) ===")
    print(f"ID Dataset: {config.id_dataset}")
    print(f"OOD Dataset: {config.get_ood_dataset()}")
    
    # Get test and OOD data
    Y_test = split_data['y_test']
    y_ood = split_data['y_ood']
    Y_train = split_data['y_train']
    
    print(f"ID Test samples: {Y_test.shape[0]}")
    print(f"OOD samples: {y_ood.shape[0]}")
    
    # Get latent representations
    if model_name == 'nn_gauss':
        # For nn_gauss, use encoder directly (no optimization needed)
        model.eval()
        with torch.no_grad():
            # Temporarily adjust jitter size for different batch sizes
            original_jitter = model.X.jitter
            latent_dim = config.latent_dim
            
            # Create jitter for test data
            test_jitter = torch.eye(latent_dim, dtype=torch.float64, device=config.device).unsqueeze(0) * 1e-5
            model.X.jitter = torch.cat([test_jitter for _ in range(len(Y_test))], axis=0)
            
            # For nn_gauss with missing data, clean the input
            if config.enable_missing_data:
                Y_test_clean = Y_test.clone()
                Y_test_clean[torch.isnan(Y_test_clean)] = -9.0
                test_X_mean = model.X.mu(Y_test_clean)
                test_X_covar = model.X.sigma(Y_test_clean)
            else:
                test_X_mean = model.X.mu(Y_test)
                test_X_covar = model.X.sigma(Y_test)
            
            # Create jitter for OOD data
            ood_jitter = torch.eye(latent_dim, dtype=torch.float64, device=config.device).unsqueeze(0) * 1e-5
            model.X.jitter = torch.cat([ood_jitter for _ in range(len(y_ood))], axis=0)
            
            if config.enable_missing_data:
                y_ood_clean = y_ood.clone()
                y_ood_clean[torch.isnan(y_ood_clean)] = -9.0
                ood_X_mean = model.X.mu(y_ood_clean)
                ood_X_covar = model.X.sigma(y_ood_clean)
            else:
                ood_X_mean = model.X.mu(y_ood)
                ood_X_covar = model.X.sigma(y_ood)
            
            # Restore original jitter
            model.X.jitter = original_jitter
    
    else:  # gauss model - need to optimize latent variables for new data
        # For test data, we already have optimized latent variables from training
        if 'test_X_mean' in split_data and 'test_X_covar' in split_data:
            test_X_mean = split_data['test_X_mean']
            test_X_covar = split_data['test_X_covar']
        else:
            # Fallback: use model's training latent - this won't work well for OOD
            print("  Warning: Using training latent for test - OOD detection may be inaccurate for gauss model")
            with torch.no_grad():
                test_X_mean = model.X.q_mu.detach()[:len(Y_test)]
                test_X_covar = torch.nn.functional.softplus(model.X.q_log_sigma).detach()[:len(Y_test)]
        
        # For OOD data, optimize new latent variables (same as test data optimization)
        print("  Optimizing latent representations for OOD data (gauss model)...")
        
        # Create prior for OOD data using PCA initialization
        mu_x_ood_pca, _, _ = torch.pca_lowrank(y_ood, config.latent_dim)
        prior_x_ood = NormalPrior(mu_x_ood_pca, torch.ones_like(mu_x_ood_pca, dtype=torch.float64))
        
        try:
            # Optimize latent variables for OOD data (same as test optimization)
            losses_ood, X_ood = model.predict_latent(
                Y_train, y_ood, 
                config.learning_rate, 
                likelihood, 
                config.random_seed,
                prior_x=prior_x_ood, 
                ae=False, 
                model_name='gauss',
                pca=config.pca_init, 
                steps=config.test_iterations
            )
            
            # Extract optimized latent variables
            ood_X_mean = X_ood.q_mu.detach()
            ood_X_covar = torch.nn.functional.softplus(X_ood.q_log_sigma).detach()
            
            print(f"  OOD latent optimization completed. Final loss: {losses_ood[-1]:.4f}")
            
        except Exception as e:
            print(f"  Warning: OOD latent optimization failed: {e}")
            print("  Falling back to PCA initialization...")
            ood_X_mean = mu_x_ood_pca
            ood_X_covar = torch.ones_like(ood_X_mean)
    
    obs_noise = likelihood.noise
    
    # Determine number of samples for OOD evaluation
    n_samples = config.ood_n_samples if config.ood_n_samples is not None else 4
    print(f"Using n_samples={n_samples} for OOD evaluation")
    
    ood_results = {}
    
    for score_type in config.ood_score_types:
        print(f"\nEvaluating with {score_type.upper()} scores...")
        
        try:
            metrics, id_scores, ood_scores = evaluate_ood_detection_original_gplvm(
                model=model,
                id_test_data=Y_test,
                ood_data=y_ood,
                id_X_mean=test_X_mean,
                id_X_covar=test_X_covar,
                ood_X_mean=ood_X_mean,
                ood_X_covar=ood_X_covar,
                model_name=model_name,
                obs_noise=obs_noise,
                score_type=score_type,
                n_samples=n_samples
            )
            
            print(f"  AUROC: {metrics.auroc:.4f}")
            print(f"  AUPRC: {metrics.auprc:.4f}")
            print(f"  FPR@80: {metrics.fpr_at_80_tpr:.4f}")
            
            ood_results[score_type] = {
                'metrics': metrics.to_dict(),
                'id_scores_mean': float(np.mean(id_scores)),
                'id_scores_std': float(np.std(id_scores)),
                'ood_scores_mean': float(np.mean(ood_scores)),
                'ood_scores_std': float(np.std(ood_scores)),
                'id_scores': id_scores.tolist(),
                'ood_scores': ood_scores.tolist()
            }
            
            # Log to file
            if logger:
                logger.log_training(f"OOD {score_type.upper()}: AUROC={metrics.auroc:.4f}, AUPRC={metrics.auprc:.4f}, FPR@80={metrics.fpr_at_80_tpr:.4f}")
        
        except Exception as e:
            print(f"  Warning: Failed to compute {score_type} scores: {e}")
            import traceback
            traceback.print_exc()
            ood_results[score_type] = {'error': str(e)}
    
    return ood_results


def plot_ood_score_histograms(ood_results, config: OODOriginalGPLVMConfig, 
                               model_name: str, logger=None):
    """Plot histograms of OOD scores"""
    if logger is None or logger.get_results_dir() is None:
        return
    
    plots_dir = os.path.join(logger.get_results_dir(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for score_type, result in ood_results.items():
        if 'error' in result:
            continue
        
        try:
            id_scores = np.array(result['id_scores'])
            ood_scores = np.array(result['ood_scores'])
            
            plt.figure(figsize=(10, 6))
            
            plt.hist(id_scores, bins=50, alpha=0.7, label=f'{config.id_dataset} (ID)', density=True)
            plt.hist(ood_scores, bins=50, alpha=0.7, label=f'{config.get_ood_dataset()} (OOD)', density=True)
            
            plt.xlabel(f'{score_type.upper()} Score')
            plt.ylabel('Density')
            plt.title(f'OOD Detection ({model_name}): {config.id_dataset} vs {config.get_ood_dataset()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(plots_dir, f"ood_hist_{score_type}_{model_name}_{config.random_seed}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  OOD histogram saved: {save_path}")
        
        except Exception as e:
            print(f"  Warning: Failed to plot {score_type} histogram: {e}")


def get_fast_config():
    """Fast configuration for OOD testing with Original GP-LVM.
    
    Default: Cross-dataset OOD with FashionMNIST as ID, MNIST as OOD.
    """
    return OODOriginalGPLVMConfig(
        device="cuda",
        training_steps={'gauss': 100, 'nn_gauss': 100},
        batch_size=50,
        n_inducing=15,
        learning_rate=0.002,
        random_seed=7,
        test_iterations=4000,
        test_size=0.2,
        checkpoint_frequency=4000,
        metrics_frequency=4000,
        # === GP Hyperparameters ===
        init_lengthscales=2.0,
        init_output_noise_scale=2.0,
        # === Model-specific Configuration ===
        nn_layers=(128, 64, 32),  # 3-layer neural network architecture for encoder
        # === Missing Data Configuration ===
        enable_missing_data=True,
        missing_data_ratio=0.3,
        reconstruction_samples=50,
        # === OOD Configuration (Cross-dataset mode) ===
        # FashionMNIST classes: 0=T-shirt, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,
        #                       5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot
        # MNIST classes: 0-9 = Digits 0-9
        id_dataset="FashionMNIST",
        id_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # All FashionMNIST classes
        id_samples_per_class=100,
        ood_dataset="MNIST",         # Explicitly specify OOD dataset
        ood_classes=[1, 2],          # Digits 1, 2
        ood_samples_per_class=100,
        # === OOD Evaluation Configuration ===
        ood_n_samples=16             # Number of samples for OOD evaluation
    )


def main(config: OODOriginalGPLVMConfig = None, config_name: str = "fast"):
    """Main function for OOD Detection Original GP LVM experiments"""
    
    # Use default config if none provided
    if config is None:
        config = get_fast_config()

    config.save_models = False
    
    # Override results directory for OOD experiments when using fast config
    if config_name == "fast":
        config.results_dir = "ood_results/original_gplvm_results"

    # Set device and dtype from config
    torch.set_default_device(config.device)
    torch.set_default_dtype(torch.float64)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Print configuration
    config.print_config()
    
    # Initialize results logger
    experiment_name = f"ood_gplvm_{config.id_dataset}_{config_name}"
    logger = OriginalGPLVMResultsLogger(config, experiment_name=experiment_name)
    if logger.get_results_dir():
        print(f"Results will be saved to: {logger.get_results_dir()}")
    
    # Load data (including OOD data)
    Y_train, Y_test, lb_train, lb_test, split_data = load_ood_data(config)
    
    # Apply missing data preprocessing if enabled
    if config.enable_missing_data:
        print(f"\n=== Missing Data Preprocessing ===")
        
        # Store original data for evaluation (already standardized at this point)
        Y_train_original = Y_train.clone()
        Y_test_original = Y_test.clone()
        
        # Use preprocessing function
        Y_train_missing, train_missing_mask, missing_info = preprocess_data_for_missing_training(Y_train, config)
        Y_test_missing, test_missing_mask, _ = preprocess_data_for_missing_training(Y_test, config)
        
        # Update data
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
    else:
        Y_train_original = None
        Y_test_original = None
        train_missing_mask = None
        test_missing_mask = None
    
    # Setting shapes
    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = config.latent_dim
    n_inducing = config.n_inducing
    pca = config.pca_init
    
    # Run models and store results
    models = config.models
    
    # Skip gauss model when missing data is disabled (only run nn_gauss)
    models = [m for m in models if m != 'gauss']
    print(f"Skipping 'gauss' model, running only: {models}")
    
    steps_per_model = config.training_steps
    
    model_dict = {}
    noise_trace_dict = {}
    results_summary = {}
    all_ood_results = {}
    
    for model_name in models:
        
        print(f"\n=== Training {model_name} model ===")
        logger.log_training(f"Starting training for model: {model_name}")
        
        # Always apply PCA initialization
        if config.enable_missing_data and Y_train_original is not None:
            mu_x_pca, _, _ = torch.pca_lowrank(Y_train_original, latent_dim)
        else:
            mu_x_pca, _, _ = torch.pca_lowrank(Y_train, latent_dim)
        X_init = torch.nn.Parameter(mu_x_pca)
        
        # Define prior for X using PCA-derived means
        X_prior_mean = mu_x_pca
        X_prior_mean_test = X_prior_mean[0:len(Y_test),:]
        
        nn_layers = None
        prior_x = None
        prior_x_test = None
        
        if model_name == 'gauss':
            ae = False
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean, dtype=torch.float64))
            prior_x_test = NormalPrior(X_prior_mean_test, torch.ones_like(X_prior_mean_test, dtype=torch.float64))
            X = VariationalLatentVariable(X_init, prior_x, data_dim)
        
        elif model_name == 'nn_gauss':
            ae = True
            nn_layers = config.nn_layers
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1], dtype=torch.float64, device=config.device))
            prior_x_test = MultivariateNormalPrior(X_prior_mean_test, torch.eye(X_prior_mean.shape[1], dtype=torch.float64, device=config.device))
            X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
            
        # Initialise model, likelihood, elbo and optimizer
        model = MNISTModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers, device=config.device, config=config)
        
        # Use missing data aware likelihood if enabled
        if config.enable_missing_data:
            likelihood = GaussianLikelihoodWithMissingObs(noise_constraint=Positive(transform=transforms.ExpTransform()))
        else:
            likelihood = GaussianLikelihood(noise_constraint=Positive(transform=transforms.ExpTransform()))

        model.covar_module.base_kernel.register_constraint(
            "raw_lengthscale", Positive(transform=transforms.ExpTransform())
        )
        model.covar_module.register_constraint(
            "raw_outputscale", Positive(transform=transforms.ExpTransform())
        )

        # Initialize likelihood noise from config
        if config is not None:
            with torch.no_grad():
                noise_scale = torch.tensor(config.init_output_noise_scale, dtype=torch.float64, device=config.device)
                likelihood.initialize(noise=noise_scale)
        elbo = VariationalELBO(likelihood, model, num_data=len(Y_train))

        # Move model and likelihood to CUDA if available
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

        optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
        ], lr=config.learning_rate)
    
        # Model params
        print(f'Training model params for model {model_name}')
        model.get_trainable_param_names()
    
        # Training loop
        loss_list = []  # Batch losses (negative ELBO)
        full_loss_list = []  # Full dataset losses (negative ELBO)
        full_loss_iterations = []  # Iterations at which full loss was computed
        noise_trace = []
        
        # Metrics tracking
        train_rmse_list = []
        train_nlpd_list = []
        train_crps_list = []
        train_crps_pp_sample_list = []
        train_crps_pp_mixture_list = []
        train_coverage_90_list = []
        train_coverage_95_list = []
        train_interval_width_90_list = []
        train_interval_width_95_list = []
        train_missing_nlpd_list = []
        train_missing_rmse_list = []
        effective_dims_list = []
        metrics_iterations = []
        metrics_losses = []  # Will store full dataset losses
        
        # k-NN metrics tracking
        knn_metrics_history = []
        
        iterator = trange(steps_per_model[model_name], leave=True)
        batch_size = config.batch_size
        for i in iterator: 
            batch_index = model._get_batch_idx(batch_size)
            batch_index_tensor = torch.tensor(batch_index, dtype=torch.int64, device=config.device)
            optimizer.zero_grad()
            if model_name in ['point','map', 'gauss']:
                sample = model.sample_latent_variable()
            else:
                if config.enable_missing_data:
                    Y_train_for_encoder = Y_train.clone()
                    Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -9.0
                    sample = model.sample_latent_variable(Y_train_for_encoder)
                else:
                    sample = model.sample_latent_variable(Y_train)
            sample_batch = sample[batch_index_tensor]
            output_batch = model(sample_batch)
            loss = -elbo(output_batch, Y_train[batch_index_tensor].T).sum()
            loss_list.append(float(loss.item()))
            noise_trace.append(float(np.round(likelihood.noise.item(),3)))
            iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
            
            # Compute training metrics periodically
            if i % config.metrics_frequency == 0 or i == steps_per_model[model_name] - 1:
                try:
                    with torch.no_grad():
                        model.eval()
                        
                        # Compute full dataset loss (negative ELBO) using minibatches
                        full_elbo = compute_full_elbo_minibatched(model, Y_train, elbo, model_name, config)
                        full_loss = -full_elbo  # Store as negative ELBO (loss)
                        full_loss_list.append(full_loss)
                        full_loss_iterations.append(i)
                        print(f"  Full Dataset Loss: {full_loss:.4f} (batch loss: {loss.item():.4f})")
                        
                        if config.enable_missing_data and model_name == 'nn_gauss':
                            Y_train_for_encoder = Y_train.clone()
                            Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -9.0
                            X_train_mean = model.get_X_mean(Y_train_for_encoder)
                            X_train_covar = model.X.sigma(Y_train_for_encoder).detach()
                        else:
                            X_train_mean = model.get_X_mean(Y_train)

                            if model_name == "nn_gauss":
                                X_train_covar = model.X.sigma(Y_train).detach()
                            else:
                                X_train_covar = torch.nn.functional.softplus(model.X.q_log_sigma).detach()

                        # Compute training RMSE
                        train_rmse = compute_rmse_with_sampling(
                            model=model,
                            y_true=Y_train,
                            X_mean=X_train_mean,
                            X_covar=X_train_covar,
                            model_name=model_name,
                            n_samples=4,
                            enable_missing_data=config.enable_missing_data,
                            batch_size=config.batch_size
                        )
                        # Compute training NLPD
                        try:
                            obs_noise = likelihood.noise
                            train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean, model_name, loss_list, X_train_covar, obs_noise, enable_missing_data=config.enable_missing_data, batch_size=config.batch_size)
                        except Exception as e:
                            print(f"  Warning: Could not get covariance for NLPD: {e}")
                            train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean, model_name, loss_list, enable_missing_data=config.enable_missing_data, batch_size=config.batch_size)

                        # Compute training CRPS family (same predictive path as NLPD)
                        try:
                            obs_noise = likelihood.noise
                            train_crps_all = compute_all_crps_nll_based(
                                model, Y_train, X_train_mean, model_name, X_train_covar, obs_noise,
                                enable_missing_data=config.enable_missing_data, batch_size=config.batch_size
                            )
                            train_crps = train_crps_all['crps']
                            train_crps_pp_sample = train_crps_all['crps_posterior_predictive_sample']
                            train_crps_pp_mixture = train_crps_all['crps_posterior_predictive_mixture']
                        except Exception as e:
                            print(f"  Warning: Could not compute CRPS: {e}")
                            train_crps = float('nan')
                            train_crps_pp_sample = float('nan')
                            train_crps_pp_mixture = float('nan')

                        # Predictive interval coverage (same predictive path as NLPD)
                        try:
                            obs_noise = likelihood.noise
                            train_cov = compute_coverage_nll_based(
                                model, Y_train, X_train_mean, model_name, X_train_covar, obs_noise,
                                enable_missing_data=config.enable_missing_data, batch_size=config.batch_size
                            )
                            train_coverage_90 = train_cov.get('coverage_90', float('nan'))
                            train_coverage_95 = train_cov.get('coverage_95', float('nan'))
                            train_interval_width_90 = train_cov.get('interval_width_90', float('nan'))
                            train_interval_width_95 = train_cov.get('interval_width_95', float('nan'))
                        except Exception as e:
                            print(f"  Warning: Could not compute coverage: {e}")
                            train_coverage_90 = float('nan')
                            train_coverage_95 = float('nan')
                            train_interval_width_90 = float('nan')
                            train_interval_width_95 = float('nan')
                        
                        # Compute missing data metrics if enabled
                        train_missing_nlpd = None
                        train_missing_rmse = None
                        if config.enable_missing_data and model_name in ['gauss', 'nn_gauss']:
                            try:
                                obs_noise = likelihood.noise
                                # Extract standard deviations from X_covar (handles both gauss and nn_gauss)
                                if model_name == 'nn_gauss':
                                    X_train_std = torch.stack([torch.sqrt(x.diag()) for x in X_train_covar])
                                else:  # gauss model
                                    X_train_std = X_train_covar
                                
                                train_missing_nlpd = compute_missing_data_nlpd_gauss(
                                    model=model,
                                    Y_original=Y_train_original,
                                    missing_mask=train_missing_mask,
                                    X_mean=X_train_mean,
                                    X_std=X_train_std,
                                    obs_noise=obs_noise,
                                    n_samples=config.reconstruction_samples,
                                    batch_size=config.batch_size
                                )
                                train_missing_rmse = compute_missing_data_rmse_with_sampling(
                                    model=model,
                                    Y_original=Y_train_original,
                                    missing_mask=train_missing_mask,
                                    X_mean=X_train_mean,
                                    X_covar=X_train_covar,
                                    model_name=model_name,
                                    n_samples=config.reconstruction_samples,
                                    batch_size=config.batch_size
                                )
                                print(f"  Train Missing Data NLPD at iter {i}: {train_missing_nlpd:.3f}")
                                print(f"  Train Missing Data RMSE at iter {i}: {train_missing_rmse:.3f}")
                            except Exception as e:
                                print(f"  Warning: Could not compute missing data metrics at iteration {i}: {e}")
                                train_missing_nlpd = float('nan')
                                train_missing_rmse = float('nan')
                        
                        # Compute effective dimensions
                        effective_dims, total_dims = compute_effective_latent_dims(model)
                        
                        # Store metrics
                        train_rmse_list.append(float(train_rmse.item()) if hasattr(train_rmse, 'item') else float(train_rmse))
                        train_nlpd_list.append(float(train_nlpd))
                        train_crps_list.append(float(train_crps))
                        train_crps_pp_sample_list.append(float(train_crps_pp_sample))
                        train_crps_pp_mixture_list.append(float(train_crps_pp_mixture))
                        train_coverage_90_list.append(float(train_coverage_90))
                        train_coverage_95_list.append(float(train_coverage_95))
                        train_interval_width_90_list.append(float(train_interval_width_90))
                        train_interval_width_95_list.append(float(train_interval_width_95))
                        train_missing_nlpd_list.append(float(train_missing_nlpd) if train_missing_nlpd is not None else float('nan'))
                        train_missing_rmse_list.append(float(train_missing_rmse) if train_missing_rmse is not None else float('nan'))
                        effective_dims_list.append(effective_dims)
                        metrics_iterations.append(i)
                        metrics_losses.append(full_loss)  # Use full dataset loss instead of batch loss
                        
                        # Compute k-NN clustering quality metrics
                        print(f"  Computing k-NN clustering metrics...")
                        knn_results = compute_knn_accuracies_original_gplvm(
                            model=model,
                            y_train=split_data['y_test'],
                            labels_train=split_data['labels_test'],
                            model_name=model_name,
                            k_values=[1, 3, 7, 12],
                            batch_size=config.batch_size
                        )
                        print(f"    k=1: {knn_results['knn_1']:.4f}")
                        print(f"    k=3: {knn_results['knn_3']:.4f}")
                        print(f"    k=7: {knn_results['knn_7']:.4f}")
                        print(f"    k=12: {knn_results['knn_12']:.4f}")
                        knn_metrics_history.append(knn_results)
                        
                        print(f"  Train metrics at iter {i}: RMSE={train_rmse:.4f}, NLPD={train_nlpd:.3f}, CRPS={train_crps:.4f}, pp_sample={train_crps_pp_sample:.4f}, pp_mix={train_crps_pp_mixture:.4f}, "
                              f"cov90={train_coverage_90:.3f}, cov95={train_coverage_95:.3f}, "
                              f"width90={train_interval_width_90:.4f}, width95={train_interval_width_95:.4f}, "
                              f"Effective dims: {effective_dims}/{total_dims}")
                        
                except Exception as e:
                    print(f"  Warning: Could not compute metrics at iteration {i}: {e}")
                finally:
                    model.train()

            # Save iteration-based models and plots
            if logger and logger.get_results_dir() and i > 0 and i % config.checkpoint_frequency == 0:
                if config.save_models:
                    print(f"  Saving iteration {i} model and plot...")
                else:
                    print(f"  Saving iteration {i} plot...")
                try:
                    from src.mnist_ood_plot_helper import save_iteration_model, create_iteration_plot
                    if config.save_models:
                        save_iteration_model(model, model_name, i, config.random_seed, logger.get_results_dir())
                    # Use full_loss_list for checkpoint plots (consistent with SSVI)
                    plot_filename = create_iteration_plot(model, full_loss_list, lb_train, config, logger.get_results_dir(), model_name, config.random_seed, Y_train, i, config.id_dataset, train_indices=torch.from_numpy(split_data['train_indices']).to(dtype=torch.int64, device=Y_train.device), is_ood=False, full_loss_iterations=full_loss_iterations, batch_losses=loss_list)
                except Exception as e:
                    print(f"  Warning: Could not create iteration plot at iteration {i}: {e}")
                
                # Save prior samples plots at checkpoint
                try:
                    plots_checkpoints_dir = os.path.join(logger.get_results_dir(), "plots", "checkpoints")
                    os.makedirs(plots_checkpoints_dir, exist_ok=True)
                    
                    prior_samples_path = os.path.join(plots_checkpoints_dir, f"prior_samples_{model_name}_iter_{i}_{config.random_seed}.png")
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
                    print(f"    Prior samples checkpoint saved: prior_samples_{model_name}_iter_{i}_{config.random_seed}.png")
                    
                    prior_samples_avg_path = os.path.join(plots_checkpoints_dir, f"prior_samples_averaged_{model_name}_iter_{i}_{config.random_seed}.png")
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
                    print(f"    Prior samples averaged checkpoint saved: prior_samples_averaged_{model_name}_iter_{i}_{config.random_seed}.png")
                    
                    test_image_predictions_checkpoint_path = os.path.join(plots_checkpoints_dir, f"test_image_predictions_{model_name}_iter_{i}_{config.random_seed}.png")
                    visualize_test_image_predictions_original_gplvm(
                        model=model,
                        likelihood=likelihood,
                        y_test=split_data['y_test'],
                        config=config,
                        model_name=model_name,
                        n_images=24,
                        save_path=test_image_predictions_checkpoint_path,
                        show_plot=False,
                        dataset_name=config.id_dataset,
                        image_shape=(28, 28),
                        scaler=split_data.get('scaler')
                    )
                    print(f"    Test image predictions checkpoint saved: test_image_predictions_{model_name}_iter_{i}_{config.random_seed}.png")
                except Exception as e:
                    print(f"  Warning: Could not create prior samples checkpoint at iteration {i}: {e}")
        model.store(loss_list, likelihood)
        
        # Compute final full dataset loss if not already computed at last iteration
        if not full_loss_list or full_loss_iterations[-1] != steps_per_model[model_name] - 1:
            with torch.no_grad():
                model.eval()
                final_full_elbo = compute_full_elbo_minibatched(model, Y_train, elbo, model_name, config)
                final_full_loss = -final_full_elbo
                full_loss_list.append(final_full_loss)
                full_loss_iterations.append(steps_per_model[model_name])
                model.train()
        
        final_full_loss = full_loss_list[-1] if full_loss_list else loss_list[-1]
        logger.log_training(f"Training completed for {model_name}. Final full loss: {final_full_loss:.4f} (batch loss: {loss_list[-1]:.4f})")
        
        # Save metrics progression plot
        if logger.get_results_dir() and train_rmse_list and train_nlpd_list:
            plot_metrics_progression(train_rmse_list, train_nlpd_list, metrics_iterations, 
                                   metrics_losses, effective_dims_list, logger.get_results_dir(), model_name, config.random_seed, batch_losses=loss_list)

            if train_crps_list:
                plot_crps_metrics_progression(
                    train_crps_list, metrics_iterations,
                    logger.get_results_dir(), model_name, config.random_seed,
                    train_coverage_90_list=train_coverage_90_list,
                    train_coverage_95_list=train_coverage_95_list,
                    train_interval_width_90_list=train_interval_width_90_list,
                    train_interval_width_95_list=train_interval_width_95_list,
                )
                plot_crps_posterior_predictive_progression(
                    train_crps_pp_sample_list, train_crps_pp_mixture_list,
                    metrics_iterations, logger.get_results_dir(), model_name, config.random_seed,
                )
            
            logger.save_checkpoint_metrics(train_rmse_list, train_nlpd_list, metrics_iterations, 
                                         metrics_losses, model_name, config.random_seed, effective_dims_list, train_missing_nlpd_list, train_missing_rmse_list,
                                         train_crps_list=train_crps_list,
                                         train_coverage_90_list=train_coverage_90_list,
                                         train_coverage_95_list=train_coverage_95_list,
                                         train_interval_width_90_list=train_interval_width_90_list,
                                         train_interval_width_95_list=train_interval_width_95_list,
                                         train_crps_pp_sample_list=train_crps_pp_sample_list,
                                         train_crps_pp_mixture_list=train_crps_pp_mixture_list)
            
            # Save training losses (full dataset loss and batch loss) - matching SSVI pattern
            print("Saving training losses (full dataset loss)...")
            # Save full dataset losses (negative ELBO, for metrics/plots)
            if full_loss_list:
                full_loss_list_floats = [float(x) for x in full_loss_list]
                logger.save_training_losses(
                    full_loss_list_floats, f"{model_name}_full_loss", config.random_seed
                )
            # Also save batch losses for reference
            loss_list_floats = [float(x) for x in loss_list]
            logger.save_training_losses(
                loss_list_floats, f"{model_name}_batch", config.random_seed
            )
            
            if config.enable_missing_data and train_missing_nlpd_list:
                plot_missing_nlpd_metrics_progression(train_missing_nlpd_list, metrics_iterations, 
                                                    logger.get_results_dir(), model_name, config.random_seed)
            
            if config.enable_missing_data and train_missing_rmse_list:
                plot_missing_rmse_metrics_progression(train_missing_rmse_list, metrics_iterations, 
                                                    logger.get_results_dir(), model_name, config.random_seed)
            
            # Save k-NN metrics and create progression plot
            if knn_metrics_history and metrics_iterations:
                print(f"Saving k-NN clustering metrics for {model_name}...")
                # Save k-NN metrics to JSON file
                knn_checkpoint_data = {
                    'model_name': model_name,
                    'seed': config.random_seed,
                    'total_iterations': steps_per_model[model_name],
                    'checkpoints': []
                }
                
                for iteration, knn_metrics in zip(metrics_iterations, knn_metrics_history):
                    knn_checkpoint_data['checkpoints'].append({
                        'iteration': iteration,
                        **knn_metrics  # Include all k-NN metrics (train_knn_1, test_knn_1, etc.)
                    })
                
                training_results_dir = os.path.join(logger.get_results_dir(), 'training_results')
                os.makedirs(training_results_dir, exist_ok=True)
                knn_checkpoint_file = os.path.join(training_results_dir, f"{model_name}_{config.random_seed}_knn_metrics.json")
                with open(knn_checkpoint_file, 'w') as f:
                    json.dump(knn_checkpoint_data, f, indent=2)
                logger.log_training(f"k-NN metrics saved: training_results/{os.path.basename(knn_checkpoint_file)}")
                
                print(f"Generating k-NN metrics progression plot for {model_name}...")
                # Extract metrics for plotting
                knn_k1 = [m['knn_1'] for m in knn_metrics_history]
                knn_k3 = [m['knn_3'] for m in knn_metrics_history]
                knn_k7 = [m['knn_7'] for m in knn_metrics_history]
                knn_k12 = [m['knn_12'] for m in knn_metrics_history]
                
                plot_knn_metrics_progression(
                    knn_k1, knn_k3, knn_k7, knn_k12,
                    metrics_iterations,
                    logger.get_results_dir(),
                    model_name,
                    config.random_seed
                )
            
        # Save models & training info
        print(model.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
        model_dict[model_name + '_' + str(config.random_seed)] = model
        noise_trace_dict[model_name + '_' + str(config.random_seed)] = noise_trace
        
        # Get training latent variables
        if config.enable_missing_data and model_name == 'nn_gauss':
            Y_train_for_encoder = Y_train.clone()
            Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -9.0
            X_train_mean = model.get_X_mean(Y_train_for_encoder)
            X_train_covar = model.X.sigma(Y_train_for_encoder).detach()
        else:
            X_train_mean = model.get_X_mean(Y_train)

            if model_name == "nn_gauss":
                X_train_covar = model.X.sigma(Y_train).detach()
            else:
                X_train_covar = torch.nn.functional.softplus(model.X.q_log_sigma).detach()

        
        # Save plots to results directory
        if logger.get_results_dir():
            try:
                from src.mnist_ood_plot_helper import create_standardized_scattered_plot
                # Use full_loss_list for final plots (consistent with SSVI)
                create_standardized_scattered_plot(model, full_loss_list, lb_train, config, logger.get_results_dir(), model.X.__class__.__name__, config.random_seed, Y_train, model_type=config.id_dataset, train_indices=torch.from_numpy(split_data['train_indices']).to(dtype=torch.int64, device=Y_train.device), dataset_name=config.id_dataset, is_ood=False, full_loss_iterations=full_loss_iterations, batch_losses=loss_list)
            except Exception as e:
                print(f"Warning: Could not create standardized scattered plot: {e}")
                print(f"Skipping plot creation.")
        
        if config.save_models:
            print(f'Saving {model_name} {config.random_seed}')
            logger.save_model(model, model_name, config.random_seed)

        ####################### Testing Framework ################################################
        if config.enable_testing:
            with torch.no_grad():
                model.eval()
                likelihood.eval()
                if hasattr(model.X, 'eval'):
                    model.X.eval()

            Y_test_for_encoder = Y_test.clone()
            Y_test_for_encoder[torch.isnan(Y_test_for_encoder)] = -9.0
            
            if ae is True:
                if model_name != 'iaf':
                    if model_name == 'nn_gauss':
                        original_jitter = model.X.jitter
                        test_jitter = torch.eye(latent_dim, dtype=torch.float64).unsqueeze(0) * 1e-5
                        model.X.jitter = torch.cat([test_jitter for i in range(len(Y_test))], axis=0)
                    
                    if config.enable_missing_data:
                        Y_train_for_encoder = Y_train.clone()
                        Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -9.0
                        X_test_mean, X_test_covar = model.predict_latent(Y_train_for_encoder, 
                                                                          Y_test_for_encoder,
                                                                          optimizer.defaults['lr'], 
                                                                          likelihood, 
                                                                          config.random_seed,
                                                                          prior_x=prior_x_test, 
                                                                          ae=True, 
                                                                          model_name='nn_gauss', 
                                                                          pca=pca)
                    else:
                        X_test_mean, X_test_covar = model.predict_latent(Y_train, 
                                                                          Y_test_for_encoder,
                                                                          optimizer.defaults['lr'], 
                                                                          likelihood, 
                                                                          config.random_seed,
                                                                          prior_x=prior_x_test, 
                                                                          ae=True, 
                                                                          model_name='nn_gauss', 
                                                                          pca=pca)
                    
                    if model_name == 'nn_gauss':
                        model.X.jitter = original_jitter
                else:
                    X_test_mean, X_flow_samples = model.predict_latent(Y_train, 
                                                                        Y_test_for_encoder,
                                                                        optimizer.defaults['lr'], 
                                                                        likelihood, 
                                                                        prior_x = prior_x,
                                                                        ae=True, 
                                                                        model_name='iaf',
                                                                        pca=pca)
            
            else: # either point, map or gauss
                try:
                    losses_test,  X_test = model.predict_latent(Y_train, Y_test_for_encoder, optimizer.defaults['lr'],
                                              likelihood, config.random_seed, prior_x=prior_x_test, ae=ae, 
                                              model_name=model_name,pca=pca, steps=config.test_iterations)
                except Exception as e:
                    print(f"Warning: predict_latent failed for {model_name}: {e}")
                    print("Skipping test evaluation for this model...")
                    continue
                    
            
            # Compute reconstructions
            if model_name in ('point', 'map'):
                    X_test_mean = X_test.X
            elif model_name == 'gauss':
                   X_test_mean = X_test.q_mu.detach()
                   X_test_covar = torch.nn.functional.softplus(X_test.q_log_sigma).detach()
       
            X_test_mean_device = X_test_mean.to(dtype=torch.float64, device=config.device)
            X_train_mean_device = X_train_mean.to(dtype=torch.float64, device=config.device)

            # Store test latent for OOD evaluation
            split_data['test_X_mean'] = X_test_mean_device
            split_data['test_X_covar'] = X_test_covar if model_name != 'point' else None

            # Compute the metrics (using batched functions to avoid OOM)
            try:
                mse_train = compute_rmse_with_sampling(
                    model=model,
                    y_true=Y_train,
                    X_mean=X_train_mean_device,
                    X_covar=X_train_covar,
                    model_name=model_name,
                    n_samples=32,
                    enable_missing_data=config.enable_missing_data,
                    batch_size=config.batch_size
                )
                
                mse_test = compute_rmse_with_sampling(
                    model=model,
                    y_true=Y_test,
                    X_mean=X_test_mean_device,
                    X_covar=X_test_covar,
                    model_name=model_name,
                    n_samples=32,
                    enable_missing_data=config.enable_missing_data,
                    batch_size=config.batch_size
                )
                
                print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
                print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
            except Exception as e:
                print(f"Warning: Failed to compute RMSE for {model_name} due to error: {e}")
                mse_train = float('nan')
                mse_test = float('nan')
            
            # Compute NLPD
            obs_noise = likelihood.noise
            try:
                train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean_device, model_name, loss_list, X_train_covar, obs_noise, enable_missing_data=config.enable_missing_data, batch_size=config.batch_size)
                test_nlpd = compute_nlpd_nll_based(model, Y_test_for_encoder, X_test_mean_device, model_name, losses_test if model_name == 'gauss' else loss_list, X_test_covar, obs_noise, enable_missing_data=config.enable_missing_data, batch_size=config.batch_size)

                print(f'Train NLPD {model_name} = ' + str(train_nlpd))
                print(f'Test NLPD {model_name} = ' + str(test_nlpd))
            except Exception as e:
                print(f"Warning: Failed to compute NLPD for {model_name} due to error: {e}")
                train_nlpd = float('nan')
                test_nlpd = float('nan')

            # Continuous Ranked Probability Score family (same predictive path as NLPD)
            try:
                train_crps_all = compute_all_crps_nll_based(
                    model, Y_train, X_train_mean_device, model_name, X_train_covar, obs_noise,
                    enable_missing_data=config.enable_missing_data, batch_size=config.batch_size
                )
                test_crps_all = compute_all_crps_nll_based(
                    model, Y_test_for_encoder, X_test_mean_device, model_name, X_test_covar, obs_noise,
                    enable_missing_data=config.enable_missing_data, batch_size=config.batch_size
                )
                train_crps = train_crps_all['crps']
                test_crps = test_crps_all['crps']
                crps = test_crps  # held-out scalar for evaluation output
                train_crps_pp_sample = train_crps_all['crps_posterior_predictive_sample']
                test_crps_pp_sample = test_crps_all['crps_posterior_predictive_sample']
                crps_posterior_predictive_sample = test_crps_pp_sample
                train_crps_pp_mixture = train_crps_all['crps_posterior_predictive_mixture']
                test_crps_pp_mixture = test_crps_all['crps_posterior_predictive_mixture']
                crps_posterior_predictive_mixture = test_crps_pp_mixture

                print(f'Train CRPS {model_name} = ' + str(train_crps))
                print(f'Test CRPS (crps) {model_name} = ' + str(crps))
                print(f'Train/Test crps_pp_sample {model_name} = {train_crps_pp_sample:.4f} / {test_crps_pp_sample:.4f}')
                print(f'Train/Test crps_pp_mixture {model_name} = {train_crps_pp_mixture:.4f} / {test_crps_pp_mixture:.4f}')
            except Exception as e:
                print(f"Warning: Failed to compute CRPS for {model_name} due to error: {e}")
                train_crps = float('nan')
                test_crps = float('nan')
                crps = float('nan')
                train_crps_pp_sample = float('nan')
                test_crps_pp_sample = float('nan')
                crps_posterior_predictive_sample = float('nan')
                train_crps_pp_mixture = float('nan')
                test_crps_pp_mixture = float('nan')
                crps_posterior_predictive_mixture = float('nan')

            # Predictive interval coverage (same predictive path as NLPD)
            try:
                train_cov = compute_coverage_nll_based(
                    model, Y_train, X_train_mean_device, model_name, X_train_covar, obs_noise,
                    enable_missing_data=config.enable_missing_data, batch_size=config.batch_size
                )
                test_cov = compute_coverage_nll_based(
                    model, Y_test_for_encoder, X_test_mean_device, model_name, X_test_covar, obs_noise,
                    enable_missing_data=config.enable_missing_data, batch_size=config.batch_size
                )
                coverage_90 = test_cov.get('coverage_90', float('nan'))
                coverage_95 = test_cov.get('coverage_95', float('nan'))
                interval_width_90 = test_cov.get('interval_width_90', float('nan'))
                interval_width_95 = test_cov.get('interval_width_95', float('nan'))
                print(f'Train coverage_90/95 {model_name} = {train_cov.get("coverage_90"):.4f} / {train_cov.get("coverage_95"):.4f}')
                print(f'Test coverage_90/95 {model_name} = {coverage_90:.4f} / {coverage_95:.4f}')
                print(f'Train interval_width_90/95 {model_name} = {train_cov.get("interval_width_90"):.4f} / {train_cov.get("interval_width_95"):.4f}')
                print(f'Test interval_width_90/95 {model_name} = {interval_width_90:.4f} / {interval_width_95:.4f}')
            except Exception as e:
                print(f"Warning: Failed to compute coverage for {model_name} due to error: {e}")
                train_cov = {
                    'coverage_90': float('nan'), 'coverage_95': float('nan'),
                    'interval_width_90': float('nan'), 'interval_width_95': float('nan'),
                }
                coverage_90 = float('nan')
                coverage_95 = float('nan')
                interval_width_90 = float('nan')
                interval_width_95 = float('nan')
                
            if model_name not in results_summary:
                results_summary[model_name] = {}

            # Missing Data metrics
            if config.enable_missing_data and model_name in ['gauss', 'nn_gauss']:
                try:
                    # Extract standard deviations from X_covar (handles both gauss and nn_gauss)
                    if model_name == 'nn_gauss':
                        X_train_std = torch.stack([torch.sqrt(x.diag()) for x in X_train_covar])
                        X_test_std = torch.stack([torch.sqrt(x.diag()) for x in X_test_covar])
                    else:  # gauss model
                        X_train_std = X_train_covar
                        X_test_std = X_test_covar

                    train_missing_nlpd = compute_missing_data_nlpd_gauss(
                        model=model,
                        Y_original=Y_train_original,
                        missing_mask=train_missing_mask,
                        X_mean=X_train_mean_device,
                        X_std=X_train_std,
                        obs_noise=obs_noise,
                        n_samples=config.reconstruction_samples,
                        batch_size=config.batch_size
                    )

                    test_missing_nlpd = compute_missing_data_nlpd_gauss(
                        model=model,
                        Y_original=Y_test_original,
                        missing_mask=test_missing_mask,
                        X_mean=X_test_mean_device,
                        X_std=X_test_std,
                        obs_noise=obs_noise,
                        n_samples=config.reconstruction_samples,
                        batch_size=config.batch_size
                    )

                    print(f'Train Missing Data NLPD {model_name} = ' + str(train_missing_nlpd))
                    print(f'Test Missing Data NLPD {model_name} = ' + str(test_missing_nlpd))

                    results_summary[model_name]['train_missing_nlpd'] = float(train_missing_nlpd)
                    results_summary[model_name]['test_missing_nlpd'] = float(test_missing_nlpd)

                    train_missing_rmse = compute_missing_data_rmse_with_sampling(
                        model=model,
                        Y_original=Y_train_original,
                        missing_mask=train_missing_mask,
                        X_mean=X_train_mean_device,
                        X_covar=X_train_covar,
                        model_name=model_name,
                        n_samples=config.reconstruction_samples,
                        batch_size=config.batch_size
                    )

                    test_missing_rmse = compute_missing_data_rmse_with_sampling(
                        model=model,
                        Y_original=Y_test_original,
                        missing_mask=test_missing_mask,
                        X_mean=X_test_mean_device,
                        X_covar=X_test_covar,
                        model_name=model_name,
                        n_samples=config.reconstruction_samples,
                        batch_size=config.batch_size
                    )

                    print(f'Train Missing Data RMSE {model_name} = ' + str(train_missing_rmse))
                    print(f'Test Missing Data RMSE {model_name} = ' + str(test_missing_rmse))

                    results_summary[model_name]['train_missing_rmse'] = float(train_missing_rmse)
                    results_summary[model_name]['test_missing_rmse'] = float(test_missing_rmse)

                except Exception as e:
                    print(f"Warning: Failed to compute missing data metrics for {model_name} due to error: {e}")
                    results_summary[model_name]['train_missing_nlpd'] = float('nan')
                    results_summary[model_name]['test_missing_nlpd'] = float('nan')
                    results_summary[model_name]['train_missing_rmse'] = float('nan')
                    results_summary[model_name]['test_missing_rmse'] = float('nan')

            # Save training results
            loss_list_floats = [float(x) for x in loss_list]
            noise_trace_floats = [float(x) for x in noise_trace]
            logger.save_training_results(model_name, config.random_seed, loss_list_floats, noise_trace_floats,
                                        float(mse_train.item()) if hasattr(mse_train, 'item') else float(mse_train), 
                                        float(mse_test.item()) if hasattr(mse_test, 'item') else float(mse_test))
            
            # Store results
            train_missing_nlpd_value = results_summary[model_name].get('train_missing_nlpd', float('nan'))
            test_missing_nlpd_value = results_summary[model_name].get('test_missing_nlpd', float('nan'))
            train_missing_rmse_value = results_summary[model_name].get('train_missing_rmse', float('nan'))
            test_missing_rmse_value = results_summary[model_name].get('test_missing_rmse', float('nan'))
            
            results_summary[model_name] = {
                'train_rmse': float(mse_train.item()) if hasattr(mse_train, 'item') else float(mse_train),
                'test_rmse': float(mse_test.item()) if hasattr(mse_test, 'item') else float(mse_test),
                'train_nlpd': float(train_nlpd),
                'test_nlpd': float(test_nlpd),
                'train_crps': float(train_crps),
                'test_crps': float(test_crps),
                'crps': float(crps),  # held-out CRPS (same as test_crps)
                'train_crps_posterior_predictive_sample': float(train_crps_pp_sample),
                'test_crps_posterior_predictive_sample': float(test_crps_pp_sample),
                'crps_posterior_predictive_sample': float(crps_posterior_predictive_sample),
                'train_crps_posterior_predictive_mixture': float(train_crps_pp_mixture),
                'test_crps_posterior_predictive_mixture': float(test_crps_pp_mixture),
                'crps_posterior_predictive_mixture': float(crps_posterior_predictive_mixture),
                'train_coverage_90': float(train_cov.get('coverage_90', float('nan'))),
                'train_coverage_95': float(train_cov.get('coverage_95', float('nan'))),
                'test_coverage_90': float(coverage_90),
                'test_coverage_95': float(coverage_95),
                'coverage_90': float(coverage_90),
                'coverage_95': float(coverage_95),
                'train_interval_width_90': float(train_cov.get('interval_width_90', float('nan'))),
                'train_interval_width_95': float(train_cov.get('interval_width_95', float('nan'))),
                'test_interval_width_90': float(interval_width_90),
                'test_interval_width_95': float(interval_width_95),
                'interval_width_90': float(interval_width_90),
                'interval_width_95': float(interval_width_95),
                'train_missing_nlpd': train_missing_nlpd_value,
                'test_missing_nlpd': test_missing_nlpd_value,
                'train_missing_rmse': train_missing_rmse_value,
                'test_missing_rmse': test_missing_rmse_value,
                'final_batch_loss': float(loss_list[-1]),
                'final_full_loss': float(full_loss_list[-1]) if full_loss_list else float(loss_list[-1]),
                'final_noise': float(noise_trace[-1])
            }
            
            logger.log_training(
                f"Model {model_name} - Train RMSE: {mse_train:.4f}, Test RMSE: {mse_test:.4f}, "
                f"Train NLPD: {train_nlpd:.4f}, Test NLPD: {test_nlpd:.4f}, "
                f"Train CRPS: {train_crps:.4f}, Test CRPS (crps): {crps:.4f}, "
                f"crps_pp_sample: {crps_posterior_predictive_sample:.4f}, crps_pp_mixture: {crps_posterior_predictive_mixture:.4f}, "
                f"coverage_90: {coverage_90:.4f}, coverage_95: {coverage_95:.4f}, "
                f"interval_width_90: {interval_width_90:.4f}, interval_width_95: {interval_width_95:.4f}"
            )
            
            # Visualize missing data reconstruction for gauss model if enabled
            if config.enable_missing_data and model_name == 'gauss':
                print("\n=== Missing Data Reconstruction Visualization (Gauss) ===")
                
                X_train_mean = model.X.q_mu.detach()
                X_train_covar = torch.nn.functional.softplus(model.X.q_log_sigma).detach()
                
                plots_dir = os.path.join(logger.get_results_dir(), "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                simple_viz_path = os.path.join(plots_dir, f"missing_data_reconstruction_gauss_{config.random_seed}.png")
                visualize_mnist_missing_reconstruction_gauss(
                    model=model,
                    y_train=Y_train,
                    missing_mask=train_missing_mask,
                    y_original=Y_train_original,
                    X_mean=X_train_mean,
                    X_covar=X_train_covar,
                    n_images=10,
                    n_samples=5,
                    save_path=simple_viz_path,
                    show_plot=False,
                    scaler=split_data.get('scaler')
                )
                
                detailed_viz_path = os.path.join(plots_dir, f"missing_data_reconstruction_gauss_detailed_{config.random_seed}.png")
                visualize_mnist_missing_reconstruction_gauss_detailed(
                    model=model,
                    y_train=Y_train,
                    missing_mask=train_missing_mask,
                    y_original=Y_train_original,
                    X_mean=X_train_mean,
                    X_covar=X_train_covar,
                    n_images=10,
                    n_samples=5,
                    save_path=detailed_viz_path,
                    show_plot=False,
                    scaler=split_data.get('scaler')
                )
                
                print("=" * 50)
            
            # === OOD Detection Evaluation ===
            print("\n" + "=" * 60)
            ood_results = evaluate_ood_detection_metrics(model, split_data, config, model_name, likelihood, logger)
            all_ood_results[model_name] = ood_results
            
            # Plot OOD score histograms
            print("\nGenerating OOD score histograms...")
            plot_ood_score_histograms(ood_results, config, model_name, logger)
            
            # Print OOD summary
            print("\n" + "=" * 60)
            print(f"OOD DETECTION SUMMARY ({model_name})")
            print("=" * 60)
            print(f"Dataset Pair: {config.id_dataset} (ID) vs {config.get_ood_dataset()} (OOD)")
            print("-" * 60)
            for score_type, result in ood_results.items():
                if 'error' not in result:
                    metrics = result['metrics']
                    print(f"\n{score_type.upper()} Scores:")
                    print(f"  AUROC:  {metrics['auroc']:.4f}")
                    print(f"  AUPRC:  {metrics['auprc']:.4f}")
                    print(f"  FPR@80: {metrics['fpr_at_80_tpr']:.4f}")
            print("=" * 60)
            
            # Add OOD results to results_summary
            for score_type, result in ood_results.items():
                if 'error' not in result:
                    results_summary[model_name][f'ood_{score_type}_auroc'] = result['metrics']['auroc']
                    results_summary[model_name][f'ood_{score_type}_auprc'] = result['metrics']['auprc']
                    results_summary[model_name][f'ood_{score_type}_fpr80'] = result['metrics']['fpr_at_80_tpr']
            
            # Add k-NN clustering quality metrics if available
            if knn_metrics_history:
                # Use final k-NN results from history
                final_knn_results = knn_metrics_history[-1]
                for key, value in final_knn_results.items():
                    results_summary[model_name][f'knn_{key}'] = value
            
            # Generate prior samples visualization
            print(f"\n=== Prior Samples Visualization ({model_name}) ===")
            plots_dir = os.path.join(logger.get_results_dir(), "plots")
            os.makedirs(plots_dir, exist_ok=True)
            prior_samples_path = os.path.join(plots_dir, f"prior_samples_{model_name}_{config.random_seed}.png")
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
            
            # Generate prior samples visualization averaged over multiple GP samples
            prior_samples_avg_path = os.path.join(plots_dir, f"prior_samples_averaged_{model_name}_{config.random_seed}.png")
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
            
            # Generate test image predictions visualization
            print(f"\n=== Test Image Predictions Visualization ({model_name}) ===")
            test_image_predictions_path = os.path.join(plots_dir, f"test_image_predictions_{model_name}_{config.random_seed}.png")
            visualize_test_image_predictions_original_gplvm(
                model=model,
                likelihood=likelihood,
                y_test=split_data['y_test'],
                config=config,
                model_name=model_name,
                n_images=24,
                save_path=test_image_predictions_path,
                show_plot=False,
                dataset_name=config.id_dataset,
                image_shape=(28, 28),
                scaler=split_data.get('scaler')
            )
            print("=" * 50)
    
    # Save final experiment summary
    logger.save_final_summary(results_summary)
    logger.log_training("=== Experiment completed ===")
    
    return model_dict, noise_trace_dict, results_summary, all_ood_results


def create_ood_argument_parser():
    """Create argument parser with OOD-specific options"""
    parser = create_argument_parser()
    
    # ID dataset configuration
    parser.add_argument('--id-dataset', type=str, default=None,
                       choices=['MNIST', 'FashionMNIST'],
                       help='In-distribution dataset (default from config: FashionMNIST)')
    parser.add_argument('--id-classes', type=int, nargs='+', default=None,
                       help='ID class indices (e.g., --id-classes 0 1 2 3 4 5 6 7)')
    parser.add_argument('--id-samples-per-class', type=int, default=None,
                       help='Samples per ID class')
    
    # OOD dataset configuration
    parser.add_argument('--ood-dataset', type=str, default=None,
                       choices=['MNIST', 'FashionMNIST'],
                       help='OOD dataset (default from config: MNIST)')
    parser.add_argument('--ood-classes', type=int, nargs='+', default=None,
                       help='OOD class indices (e.g., --ood-classes 8 9)')
    parser.add_argument('--ood-samples-per-class', type=int, default=None,
                       help='Samples per OOD class')
    
    # Convenience: same-dataset OOD mode
    parser.add_argument('--same-dataset-ood', action='store_true',
                       help='Use same-dataset OOD mode (requires --ood-classes)')
    
    # OOD evaluation configuration
    parser.add_argument('--ood-n-samples', type=int, default=None,
                       help='Number of samples for OOD evaluation (default: 4)')
    
    return parser


if __name__ == '__main__':
    # Parse command line arguments
    parser = create_ood_argument_parser()
    args = parser.parse_args()
    
    # Ensure config directory exists
    ensure_config_dir_exists()
    
    # Default config name
    config_name = "fast"
    
    if args.config is None:
        print("No config file specified. Using default OOD fast_config...\n")
        config = get_fast_config()
        config_name = "fast"
    else:
        # Load from YAML file
        print(f"Loading configuration from: {args.config}")
        from src.original_gplvm_config_utils import load_config_from_yaml, dict_to_dataclass, get_config_name_from_path
        if os.path.exists(args.config):
            config_dict = load_config_from_yaml(args.config)
            config = dict_to_dataclass(config_dict, OODOriginalGPLVMConfig)
            config_name = get_config_name_from_path(args.config)
            config_dir = os.path.dirname(args.config)
            config.results_dir = os.path.join(config_dir, "ood_run_results")
        else:
            print(f"Warning: Config file {args.config} not found, using default OOD fast config")
            config = get_fast_config()
            config_name = "fast"
    
    # Apply command line overrides
    if hasattr(args, 'config_name') and args.config_name is not None:
        config_name = args.config_name
    if hasattr(args, 'device') and args.device is not None:
        config.device = args.device
    if hasattr(args, 'training_steps') and args.training_steps is not None:
        # Update training steps for all models
        for model_name in config.models:
            config.training_steps[model_name] = args.training_steps
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.batch_size = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    
    # Apply OOD-specific overrides (only if explicitly provided on command line)
    if hasattr(args, 'id_dataset') and args.id_dataset is not None:
        config.id_dataset = args.id_dataset
    if hasattr(args, 'id_classes') and args.id_classes is not None:
        config.id_classes = args.id_classes
    if hasattr(args, 'id_samples_per_class') and args.id_samples_per_class is not None:
        config.id_samples_per_class = args.id_samples_per_class
    if hasattr(args, 'ood_dataset') and args.ood_dataset is not None:
        config.ood_dataset = args.ood_dataset
    if hasattr(args, 'ood_classes') and args.ood_classes is not None:
        config.ood_classes = args.ood_classes
    if hasattr(args, 'ood_samples_per_class') and args.ood_samples_per_class is not None:
        config.ood_samples_per_class = args.ood_samples_per_class
    # Handle same-dataset OOD convenience flag
    if hasattr(args, 'same_dataset_ood') and args.same_dataset_ood:
        config.ood_dataset = config.id_dataset
    if hasattr(args, 'ood_n_samples') and args.ood_n_samples is not None:
        config.ood_n_samples = args.ood_n_samples
    
    # Override results directory for OOD experiments when using fast config
    if config_name == "fast":
        config.results_dir = "ood_results/original_gplvm_results"
    
    print(f"Using configuration: {config_name}")
    if args.config:
        print(f"Config file: {args.config}")
    
    # Run main function with loaded config
    main(config, config_name)
