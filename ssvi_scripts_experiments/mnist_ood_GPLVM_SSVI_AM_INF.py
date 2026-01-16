# %% 
"""
MNIST OOD Detection with SSVI GP-LVM with Amortized Inference

Extended from mnist_GPLVM_SSVI_AM_INF.py to add OOD detection capability.

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

Usage examples:
    # Run with default fast config (cross-dataset: FashionMNIST as ID, MNIST as OOD)
    python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py
    
    # Run with MNIST as in-distribution (cross-dataset: MNIST as ID, FashionMNIST as OOD)
    python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py --id-dataset MNIST
    
    # Same-dataset OOD: 8 FashionMNIST classes as ID, classes 8,9 as OOD
    python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py \\
        --id-dataset FashionMNIST --id-classes 0 1 2 3 4 5 6 7 \\
        --ood-dataset FashionMNIST --ood-classes 8 9
    
    # Same-dataset OOD with convenience flag
    python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py \\
        --id-dataset FashionMNIST --id-classes 0 1 2 3 4 5 6 7 \\
        --same-dataset-ood --ood-classes 8 9
    
    # Run with base config
    python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py --config ssvi_scripts_experiments/ssvi_configs/ssvi_base_config.yaml
"""

import torch 
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Any, List
import numpy as np
import urllib.request
import tarfile
import os
import sys
import json
from datetime import datetime
import time

# Get the project root directory (parent of ssvi_scripts_experiments)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add both project root and generalised_gplvm_code to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "generalised_gplvm_code"))

from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF

from src.ssvi_utils import SSVIResultsLogger, plot_metrics_history, plot_final_metrics_summary, plot_missing_nlpd_metrics_progression, plot_missing_rmse_metrics_progression
from src.ssvi_metrics import SSVIMetricsComputer, create_train_test_split
from src.missing_data_utils import preprocess_data_for_missing_training
from src.save_utils import print_final_model_parameters
from src.mnist_missing_reconstruction import visualize_mnist_missing_reconstruction, visualize_mnist_missing_reconstruction_detailed, visualize_prior_samples, visualize_prior_samples_averaged, visualize_test_image_predictions
from src.knn_clustering_metrics import compute_knn_accuracies_ssvi
from src.knn_progression_plot import plot_knn_metrics_progression_ssvi
from src.ssvi_config_utils import (
    load_config_with_overrides,
    create_argument_parser,
    get_base_config_path,
    ensure_config_dir_exists,
    print_available_configs,
)
from sklearn.model_selection import train_test_split

# Import OilFlowSSVIConfig from shared config module
from src.ssvi_config import OilFlowSSVIConfig

# Import data loaders
from src.data_utils import load_mnist_data, filter_mnist_by_digits, load_image_dataset

# Import OOD metrics
from src.ood_metrics import evaluate_ood_detection, compute_all_ood_metrics, OODMetrics
from src.pytorch_scalers import MeanCenter, StandardScalerPytorch, MinMaxScalerPytorch, NoTransform

# Device will be set in main() from config


# === OOD-Specific Configuration Extension ===
@dataclass
class OODSSVIConfig(OilFlowSSVIConfig):
    """Extended configuration for OOD detection experiments.
    
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
    ood_score_types: list = None           # Will be set to ['nlpd', 'rmse', 'variance'] if None
    ood_n_samples: int = None               # Number of P samples for OOD evaluation (None = use model.P)
    
    # === Data Standardization Configuration ===
    standardization: str = (
        "standard"  # "standard" (mean=0, std=1), "minmax" ([0,1]), or "none" (no transform)
    )

    def __post_init__(self):
        super().__post_init__()
        if self.ood_score_types is None:
            self.ood_score_types = ["nlpd", "rmse", "variance"]

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
        print(
            f"  ID Samples/Class: {self.id_samples_per_class if self.id_samples_per_class else 'all'}"
        )
        print(f"OOD Dataset: {self.get_ood_dataset()}")
        print(f"  OOD Classes: {self.ood_classes if self.ood_classes else 'all'}")
        print(
            f"  OOD Samples/Class: {self.ood_samples_per_class if self.ood_samples_per_class else 'all'}"
        )
        if self.is_same_dataset_ood():
            print(f"  Mode: Same-dataset OOD (class-based)")
        else:
            print(f"  Mode: Cross-dataset OOD")
        print(f"OOD Score Types: {self.ood_score_types}")
        print(
            f"OOD N Samples: {self.ood_n_samples if self.ood_n_samples else 'same as training P'}"
        )
        print(f"Data Standardization: {self.standardization}")
        print("=" * 50)


def load_ood_data(config: OODSSVIConfig):
    """Load and preprocess data with optional train/test split, plus OOD data.

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
        Y, labels, selected_digits=id_classes, samples_per_digit=id_samples_per_class
    )

    N = len(Y)

    print(f"Using {N} {config.id_dataset} samples for training/testing")
    if id_classes is not None:
        print(f"  ID classes: {id_classes}")

    # Debug: print labels type and shape
    print(
        f"Labels type: {type(labels)}, Labels shape/length: {getattr(labels, 'shape', len(labels) if hasattr(labels, '__len__') else 'no length')}"
    )

    # Ensure data is on CPU before converting to numpy for sklearn
    y_full = Y.to(config.device)
    y_cpu = y_full.cpu() if y_full.is_cuda else y_full

    # Labels are already numpy array
    labels_cpu = labels

    y_np = y_cpu.numpy()

    # Create train/test split
    print(f"Creating train/test split (test_size={config.test_size})...")
    (
        train_indices,
        test_indices,
        y_train_np,
        y_test_np,
        labels_train_np,
        labels_test_np,
    ) = train_test_split(
        np.arange(len(y_full)),
        y_np,
        labels_cpu,
        test_size=config.test_size,
        random_state=config.random_seed,
    )

    # Convert back to tensors on original device
    y_train = torch.tensor(y_train_np, dtype=torch.float64, device=config.device)
    y_test = torch.tensor(y_test_np, dtype=torch.float64, device=config.device)
    labels_train = labels_train_np
    labels_test = labels_test_np

    # Convert indices to tensors for consistency
    train_indices_tensor = torch.tensor(
        train_indices, dtype=torch.int64, device=config.device
    )
    test_indices_tensor = torch.tensor(
        test_indices, dtype=torch.int64, device=config.device
    )

    print(f"Data split shapes - Train: {y_train.shape}, Test: {y_test.shape}")
    print(
        f"Labels split info - Train: {type(labels_train)}, length: {len(labels_train) if hasattr(labels_train, '__len__') else 'no length'}"
    )
    print(
        f"Labels split info - Test: {type(labels_test)}, length: {len(labels_test) if hasattr(labels_test, '__len__') else 'no length'}"
    )

    # === Load OOD data ===
    ood_dataset = config.get_ood_dataset()
    is_same_dataset = config.is_same_dataset_ood()

    if is_same_dataset:
        print(
            f"\nLoading {ood_dataset} data as OOD (same-dataset, different classes)..."
        )
    else:
        print(f"\nLoading {ood_dataset} data as OOD (cross-dataset)...")

    Y_ood, labels_ood = load_image_dataset(ood_dataset, test_only=True)
    Y_ood = Y_ood / 255.0  # Normalize to [0, 1]

    # Apply OOD class filtering
    ood_classes, ood_samples_per_class = config.get_ood_filter_params()
    if ood_classes is not None or ood_samples_per_class is not None:
        Y_ood, labels_ood = filter_mnist_by_digits(
            Y_ood,
            labels_ood,
            selected_digits=ood_classes,
            samples_per_digit=ood_samples_per_class,
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
        raise ValueError(
            f"Unknown standardization method: {config.standardization}. Must be 'standard', 'minmax', or 'none'"
        )

    # Fit scaler on training data only (critical: don't mix test data)
    print(f"Fitting scaler on training data only ({y_train.shape[0]} samples)...")
    scaler.fit(y_train)

    # Transform all data using the scaler fitted on training data
    print(f"Transforming train/test/OOD data...")
    print(
        f"  Before transform - Train shape: {y_train.shape}, Test shape: {y_test.shape}, OOD shape: {y_ood.shape}"
    )
    print(f"  Scaler mean shape: {scaler.mean.shape}, std shape: {scaler.std.shape}")
    print(
        f"  Before transform - Train mean: {y_train.mean().item():.6f}, Test mean: {y_test.mean().item():.6f}, OOD mean: {y_ood.mean().item():.6f}"
    )
    print(
        f"  Before transform - Train std: {y_train.std().item():.6f}, Test std: {y_test.std().item():.6f}, OOD std: {y_ood.std().item():.6f}"
    )

    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    y_ood = scaler.transform(y_ood)

    # Debug: Show statistics after transform
    print(
        f"  After transform - Train mean: {y_train.mean().item():.6f}, Test mean: {y_test.mean().item():.6f}, OOD mean: {y_ood.mean().item():.6f}"
    )
    print(
        f"  After transform - Train std: {y_train.std().item():.6f}, Test std: {y_test.std().item():.6f}, OOD std: {y_ood.std().item():.6f}"
    )
    print(
        f"  After transform - Train min: {y_train.min().item():.6f}, Test min: {y_test.min().item():.6f}, OOD min: {y_ood.min().item():.6f}"
    )
    print(
        f"  After transform - Train max: {y_train.max().item():.6f}, Test max: {y_test.max().item():.6f}, OOD max: {y_ood.max().item():.6f}"
    )

    print(f"Standardization completed.")
    print("=" * 50)

    # Use training data for model initialization
    y = y_train
    labels = labels_train
    split_data = {
        "y_train": y_train,
        "y_test": y_test,
        "labels_train": labels_train,
        "labels_test": labels_test,
        "train_indices": train_indices_tensor,
        "test_indices": test_indices_tensor,
        # OOD data
        "y_ood": y_ood,
        "labels_ood": labels_ood,
        # Scaler for inverse transforms (needed for visualization)
        "scaler": scaler,
    }

    # Initialize inducing points and latent variables (following original)
    Vx = torch.randn(
        config.n_inducing_points,
        config.latent_dim,
        dtype=torch.float64,
        device=config.device,
    )
    mu_x, _, _ = torch.pca_lowrank(y, config.latent_dim)

    print(f"Data loaded: {y.shape[0]} samples, {y.shape[1]} dimensions")
    print(f"Latent initialization: {mu_x.shape}")
    print(f"Inducing points: {Vx.shape}")
    config.total_size = y.shape[0]

    # Apply missing data preprocessing if enabled
    if config.enable_missing_data:
        print(f"\n=== Missing Data Preprocessing ===")

        # Store original training data before preprocessing
        y_original_train = y.clone()

        y_train_missing, train_missing_mask, train_missing_info = (
            preprocess_data_for_missing_training(split_data["y_train"], config)
        )
        y_test_missing, test_missing_mask, test_missing_info = (
            preprocess_data_for_missing_training(split_data["y_test"], config)
        )

        split_data.update(
            {
                "y_train_original": split_data["y_train"].clone(),
                "y_test_original": split_data["y_test"].clone(),
                "y_train": y_train_missing,
                "y_test": y_test_missing,
                "train_missing_mask": train_missing_mask,
                "test_missing_mask": test_missing_mask,
            }
        )

        # Use training data with missing values for model training
        y = y_train_missing
        missing_mask = train_missing_mask
        missing_info = train_missing_info

        # Store missing data info for later use
        config.missing_info = missing_info
        config.y_original_train = y_original_train
        config.missing_mask = missing_mask

        print(f"Missing data preprocessing completed.")
        print("=" * 50)

    return y, labels, Vx, mu_x, split_data


class OilFlow_SSVI_GPLVM(GPLVM_SSVI_AM_INF):
    """MNIST/FashionMNIST SSVI GP-LVM with Amortized Inference"""

    def __init__(self, config: OODSSVIConfig, Vx, mu_x, M):
        # Use init_lengthscales to create beta values
        beta = [[config.init_lengthscales] * config.latent_dim] * 1  # C=1 for GP-LVM

        # Use init_output_noise_scale for sigma2 initialization
        sigma2 = config.init_output_noise_scale

        # Initialize underlying SSVI model
        super().__init__(
            Vx=Vx,
            M=M,  # Observation dimension
            mu_x=mu_x,
            sigma2=sigma2,
            beta=beta,
            sigma2_s=config.sigma2_s,
            P=config.P,
            trainable_V=config.trainable_V,
            encoder_hidden_dims=config.encoder_hidden_dims,
            local_lr=config.local_lr,
            global_lr=config.global_lr,
            device=config.device,
            enable_missing_data=config.enable_missing_data,
        )

        # Store config
        self.config = config

    def train_step(
        self,
        Y_batch: torch.Tensor,
        batch_indices: torch.Tensor,
        prior_weight: float,
        U_samples: torch.Tensor = None,
    ):
        """
        Modified train_step: proper SSVI separation of local/global optimization
        """
        # Zero gradients for local optimizer (encoder) only initially
        self.optimizer_local.zero_grad()

        # Sample U before amortized inference step
        epsilon = self.get_epsilon()
        U_samples = self.get_u_samples(epsilon)

        # Handle missing data preprocessing for encoder input
        if self.enable_missing_data:
            # Replace NaN values with -9.0 for encoder input (encoder expects clean data)
            Y_batch_clean = torch.where(
                torch.isnan(Y_batch),
                torch.full_like(Y_batch, -9.0, dtype=torch.float64),
                Y_batch,
            )
            # Get local parameters via amortized inference (gradients only to encoder)
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                Y_batch_clean, epsilon
            )
        else:
            # Standard case: no missing data preprocessing needed
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                Y_batch, epsilon
            )

        # Now zero gradients for global optimizer before global computation
        self.optimizer_global.zero_grad()

        # Compute global ELBO loss (gradients flow to all parameters)
        if self.enable_missing_data:
            # Use missing data aware ELBO (coordinator handles the missing mask automatically)
            missing_mask = torch.isnan(Y_batch)
            # Use the same clean data for ELBO computation to maintain consistency with encoder
            loss = -self.ssvi_elbo(
                Y_batch_clean,
                mu_n_batch,
                sigma_n_batch,
                U_samples=U_samples,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                missing_mask=missing_mask,
                prior_weight=prior_weight
            )
        else:
            # Standard ELBO computation
            loss = -self.ssvi_elbo(
                Y_batch,
                mu_n_batch,
                sigma_n_batch,
                U_samples=U_samples,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                missing_mask=None,
                prior_weight=prior_weight
            )

        # Backward pass (gradients computed for all parameters)
        loss.backward()

        # Update parameters with different learning rates
        self.optimizer_local.step()
        self.optimizer_global.step()

        return loss.item()

    @property
    def mu_x(self):
        return self.mu_x_train

    @property
    def Sigma_x(self):
        return self.Sigma_x_train

    @torch.no_grad()
    def compute_full_elbo(
        self, y_full: torch.Tensor, missing_mask: torch.Tensor = None
    ) -> float:
        """
        Compute ELBO on full dataset using minibatches (for metrics and plots, not optimization).
        Processes data in minibatches to reduce memory usage, using the same parameters as training.

        Args:
            y_full: Full dataset [N, M]
            missing_mask: Optional boolean mask for missing data [N, M]

        Returns:
            Full dataset ELBO value (scalar)
        """
        # Sample U for ELBO computation (same U for all batches)
        epsilon = self.get_epsilon()
        U_samples = self.get_u_samples(epsilon)

        # Use same batch size as training
        batch_size = self.config.batch_size
        N = y_full.shape[0]

        # Process data in minibatches and collect ELBOs (same as training loop)
        batch_elbos = []

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            y_batch = y_full[start_idx:end_idx]

            # Handle missing data preprocessing for encoder input (same as training)
            if self.enable_missing_data and missing_mask is not None:
                mask_batch = missing_mask[start_idx:end_idx]
                y_batch_clean = torch.where(
                    torch.isnan(y_batch),
                    torch.full_like(y_batch, -9.0, dtype=torch.float64),
                    y_batch,
                )
            else:
                y_batch_clean = y_batch
                mask_batch = None

            # Get latent parameters via amortized inference for this batch
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                y_batch_clean, epsilon
            )

            # Compute ELBO for this batch (EXACTLY same as in training: batch_size=config.batch_size, total_size=config.total_size)
            if self.enable_missing_data and mask_batch is not None:
                batch_elbo = self.ssvi_elbo(
                    y_batch_clean,
                    mu_n_batch,
                    sigma_n_batch,
                    U_samples,
                    batch_size=self.config.batch_size,
                    total_size=self.config.total_size,
                    missing_mask=mask_batch,
                    prior_weight=1
                )
            else:
                batch_elbo = self.ssvi_elbo(
                    y_batch_clean,
                    mu_n_batch,
                    sigma_n_batch,
                    U_samples,
                    batch_size=self.config.batch_size,
                    total_size=self.config.total_size,
                    missing_mask=None,
                    prior_weight=1
                )

            batch_elbos.append(batch_elbo)

        # Average the batch ELBOs (same approach as would be used in training)
        mean_elbo = sum(batch_elbos) / len(batch_elbos)

        return mean_elbo.item()


def train_oil_flow_ssvi(
    model: OilFlow_SSVI_GPLVM,
    y_train,
    config: OODSSVIConfig,
    logger: SSVIResultsLogger = None,
    split_data=None,
    labels_train=None,
):
    """Training function for SSVI GP-LVM with optional logging and metrics"""
    losses = []  # Batch ELBO losses (for training monitoring)
    full_elbo_losses = []  # Full dataset ELBO (for metrics and plots)
    full_elbo_iterations = []  # Iterations at which full ELBO was computed

    print(f"Training {config.id_dataset} SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")

    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(
            f"Metrics computation enabled (frequency: every {config.metrics_frequency} iterations)"
        )
        metrics_computer = SSVIMetricsComputer(
            device=config.device, metrics_batch_size=config.batch_size
        )

    # Initialize k-NN metrics tracking
    knn_metrics_history = []
    knn_metrics_iterations = []

    start_time = time.time()
    init_prior_weight = 10
    for i in range(config.training_iterations):
        # Sample batch indices
        N = y_train.shape[0]
        batch_indices = torch.randperm(N, device=y_train.device)[: config.batch_size]
        y_batch = y_train[batch_indices]

        # Training step - U sampling happens inside automatically each step
        # prior_weight = init_prior_weight 
        # prior_weight = init_prior_weight + (i/config.training_iterations)*(1-init_prior_weight)
        prior_weight = (init_prior_weight-1)/2 + (init_prior_weight-1)/2*torch.cos(torch.tensor(torch.pi*i/config.training_iterations)) + 1
        loss = model.train_step(y_batch, batch_indices, prior_weight= prior_weight)
        losses.append(loss)

        # Metrics computation (periodic)
        if (
            metrics_computer is not None
            and config.enable_metrics
            and i % config.metrics_frequency == 0
        ):

            print(f"  Computing metrics at iteration {i}...")

            # Compute full dataset loss (negative ELBO, for metrics and plots)
            if config.enable_missing_data:
                full_elbo = model.compute_full_elbo(
                    split_data["y_train"], missing_mask=split_data["train_missing_mask"]
                )
            else:
                full_elbo = model.compute_full_elbo(split_data["y_train"])

            # Store as negative ELBO (loss) for consistency with batch losses
            full_elbo_losses.append(-full_elbo)
            full_elbo_iterations.append(i)
            print(f"    Full Dataset Loss: {-full_elbo:.4f} (batch loss: {loss:.4f})")

            # Use full datasets for proper evaluation
            # Compute metrics with missing data info if enabled
            if config.enable_missing_data:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data["y_train"],
                    y_test=split_data["y_test"],
                    iteration=i,
                    elbo_loss=full_elbo,  # Use full dataset ELBO
                    n_samples=32,  # Fixed number of samples for metrics computation
                    y_train_original=split_data["y_train_original"],
                    y_test_original=split_data["y_test_original"],
                    train_missing_mask=split_data["train_missing_mask"],
                    test_missing_mask=split_data["test_missing_mask"],
                )
            else:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data["y_train"],
                    y_test=split_data["y_test"],
                    iteration=i,
                    elbo_loss=full_elbo,  # Use full dataset ELBO
                    n_samples=32,  # Fixed number of samples for metrics computation
                )

            print(
                f"    Train RMSE: {metrics.train_rmse:.4f}, Test RMSE: {metrics.test_rmse:.4f} "
            )
            print(
                f"    Train NLPD: {metrics.train_nlpd:.3f}, Test NLPD: {metrics.test_nlpd:.3f}"
            )
            if config.enable_missing_data and not np.isnan(metrics.train_missing_nlpd):
                print(f"    Train Missing Data NLPD: {metrics.train_missing_nlpd:.3f}")
                print(f"    Test Missing Data NLPD: {metrics.test_missing_nlpd:.3f}")
            if config.enable_missing_data and not np.isnan(metrics.train_missing_rmse):
                print(f"    Train Missing Data RMSE: {metrics.train_missing_rmse:.4f}")
                print(f"    Test Missing Data RMSE: {metrics.test_missing_rmse:.4f}")
            print(
                f"    Effective dims: {metrics.effective_latent_dims}/{metrics.total_latent_dims}"
            )
            print(
                f"    Full datasets: Train={split_data['y_train'].shape[0]}, Test={split_data['y_test'].shape[0]} samples"
            )

            # Compute k-NN clustering quality metrics
            print(f"    Computing k-NN clustering metrics...")
            knn_results = compute_knn_accuracies_ssvi(
                model=model,
                y_train=split_data['y_test'],
                labels_train=split_data['labels_test'],
                k_values=[1, 3, 7, 12],
                batch_size=config.batch_size,
            )
            print(f"      k=1: {knn_results['knn_1']:.4f}")
            print(f"      k=3: {knn_results['knn_3']:.4f}")
            print(f"      k=7: {knn_results['knn_7']:.4f}")
            print(f"      k=12: {knn_results['knn_12']:.4f}")
            knn_metrics_history.append(knn_results)
            knn_metrics_iterations.append(i)

            # Log metrics
            if logger:
                logger.log_training(f"Metrics at iteration {i}: {metrics.to_dict()}")
                logger.log_training(f"k-NN metrics at iteration {i}: {knn_results}")

        # Checkpoint saving (models and plots together)
        if logger and i > 0 and i % config.checkpoint_frequency == 0:
            print(f"  Saving checkpoint at iteration {i}...")
            # Save checkpoint model
            checkpoint_filename = f"ood_ssvi_iter_{i}_seed_{config.random_seed}.pkl"
            logger.save_model(model, checkpoint_filename, is_checkpoint=True)
            print(f"    Checkpoint model saved: {checkpoint_filename}")

            # Save checkpoint plot (if enabled)
            if config.save_training_plots:
                plot_filename = f"training_results_iter_{i}.png"
                from src.mnist_ood_plot_helper import visualize_ssvi_results

                # Use full ELBO losses for checkpoint plots
                visualize_ssvi_results(
                    model=model,
                    y_train=y_train,
                    labels_train=labels_train,
                    losses=full_elbo_losses,  # Use full dataset ELBO for plots
                    config=config,
                    logger=logger,
                    save_filename=plot_filename,
                    show_plot=False,
                    is_checkpoint=True,
                    train_indices=split_data["train_indices"],
                    dataset_name=config.id_dataset,
                    is_ood=False,
                    full_elbo_iterations=full_elbo_iterations,
                    batch_losses=losses  # Batch losses up to current iteration
                )
                print(f"    Checkpoint plot saved: {plot_filename}")
                

                # Save prior samples plots at checkpoint
                
                plots_checkpoints_dir = os.path.join(
                    logger.plots_dir, "checkpoints"
                )
                os.makedirs(plots_checkpoints_dir, exist_ok=True)

                prior_samples_path = os.path.join(
                    plots_checkpoints_dir,
                    f"prior_samples_iter_{i}_{config.random_seed}.png",
                )
                visualize_prior_samples(
                    model=model,
                    n_images=24,
                    save_path=prior_samples_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get("scaler"),
                )
                print(
                    f"    Prior samples checkpoint saved: prior_samples_iter_{i}_{config.random_seed}.png"
                )

                prior_samples_avg_path = os.path.join(
                    plots_checkpoints_dir,
                    f"prior_samples_averaged_iter_{i}_{config.random_seed}.png",
                )
                visualize_prior_samples_averaged(
                    model=model,
                    n_images=8,
                    save_path=prior_samples_avg_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get("scaler"),
                )
                print(
                    f"    Prior samples averaged checkpoint saved: prior_samples_averaged_iter_{i}_{config.random_seed}.png"
                )

                test_image_predictions_checkpoint_path = os.path.join(plots_checkpoints_dir, f"test_image_predictions_iter_{i}_{config.random_seed}.png")
                visualize_test_image_predictions(
                    model=model,
                    y_train=split_data['y_test'],
                    config=config,
                    n_images=24,
                    save_path=test_image_predictions_checkpoint_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get('scaler')
                )
                print(f"    Test image predictions checkpoint saved: test_image_predictions_iter_{i}_{config.random_seed}.png")
                    

            

        # Gradient logging (only if logger provided and enabled)
        if (
            logger
            and config.enable_gradient_logging
            and i % config.gradient_log_frequency == 0
        ):
            gradient_info = model.get_gradient_info()
            logger.log_gradients(i, loss, gradient_info)

        # Progress reporting
        if i % config.log_frequency == 0:
            elapsed = time.time() - start_time
            message = f"Iteration {i}/{config.training_iterations}: Loss={loss:.4f}, Time={elapsed:.1f}s"
            print(f"Iteration {i}/{config.training_iterations}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            # Log to file
            if logger:
                logger.log_training(message)

    total_time = time.time() - start_time

    # Compute final full dataset loss (negative ELBO)
    if config.enable_missing_data and split_data is not None:
        final_full_elbo = model.compute_full_elbo(
            split_data["y_train"], missing_mask=split_data["train_missing_mask"]
        )
    elif split_data is not None:
        final_full_elbo = model.compute_full_elbo(split_data["y_train"])
    else:
        if config.enable_missing_data:
            missing_mask = torch.isnan(y_train)
        else:
            missing_mask = None
        final_full_elbo = model.compute_full_elbo(y_train, missing_mask=missing_mask)

    # Store as negative ELBO (loss) for consistency with batch losses
    full_elbo_losses.append(-final_full_elbo)
    full_elbo_iterations.append(config.training_iterations)

    final_message = f"Training completed! Final Full Loss: {-final_full_elbo:.4f} (batch loss: {losses[-1]:.4f}), Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    print(f"Training completed!")
    print(f"Final Full Dataset Loss: {-final_full_elbo:.4f}")
    print(f"Final Batch Loss: {losses[-1]:.4f}")
    print(
        f"Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    )
    # Log to file
    if logger:
        logger.log_training(final_message)

    # Return full ELBO losses for plots (with iterations for x-axis)
    return {
        "batch_losses": losses,
        "full_elbo_losses": full_elbo_losses,
        "full_elbo_iterations": full_elbo_iterations,
        "knn_metrics_history": knn_metrics_history,
        "knn_metrics_iterations": knn_metrics_iterations,
    }, metrics_computer


# === OOD Evaluation Functions ===


def evaluate_ood_detection_metrics(
    model, split_data, config: OODSSVIConfig, logger=None
):
    """Evaluate OOD detection performance and return results"""
    print(f"\n=== OOD Detection Evaluation ===")
    print(f"ID Dataset: {config.id_dataset}")
    print(f"OOD Dataset: {config.get_ood_dataset()}")

    # Use original test data (without missing values) for OOD evaluation
    # This ensures fair comparison with OOD data which has no missing values
    if config.enable_missing_data and "y_test_original" in split_data:
        id_test_data = split_data["y_test_original"]
        print(f"Using original test data (no missing values) for OOD evaluation")
    else:
        id_test_data = split_data["y_test"]

    print(f"ID Test samples: {id_test_data.shape[0]}")
    print(f"OOD samples: {split_data['y_ood'].shape[0]}")

    # Temporarily change model.P if ood_n_samples is specified
    original_P = model.P
    if config.ood_n_samples is not None:
        model.P = config.ood_n_samples
        print(
            f"Using P={config.ood_n_samples} samples for OOD evaluation (training P={original_P})"
        )
    else:
        print(f"Using P={model.P} samples for OOD evaluation (same as training)")

    ood_results = {}

    for score_type in config.ood_score_types:
        print(f"\nEvaluating with {score_type.upper()} scores...")

        metrics, id_scores, ood_scores = evaluate_ood_detection(
            model=model,
            id_test_data=id_test_data,
            ood_data=split_data["y_ood"],
            score_type=score_type,
        )

        print(f"  AUROC: {metrics.auroc:.4f}")
        print(f"  AUPRC: {metrics.auprc:.4f}")
        print(f"  FPR@80: {metrics.fpr_at_80_tpr:.4f}")

        ood_results[score_type] = {
            "metrics": metrics.to_dict(),
            "id_scores_mean": float(np.mean(id_scores)),
            "id_scores_std": float(np.std(id_scores)),
            "ood_scores_mean": float(np.mean(ood_scores)),
            "ood_scores_std": float(np.std(ood_scores)),
            "id_scores": id_scores.tolist(),
            "ood_scores": ood_scores.tolist(),
        }

        # Log to file
        if logger:
            logger.log_training(
                f"OOD {score_type.upper()}: AUROC={metrics.auroc:.4f}, AUPRC={metrics.auprc:.4f}, FPR@80={metrics.fpr_at_80_tpr:.4f}"
            )


    # Restore original P value
    model.P = original_P

    return ood_results


def plot_ood_score_histograms(ood_results, config: OODSSVIConfig, logger=None):
    """Plot histograms of OOD scores"""
    if logger is None or logger.plots_dir is None:
        return

    for score_type, result in ood_results.items():


        id_scores = np.array(result["id_scores"])
        ood_scores = np.array(result["ood_scores"])

        plt.figure(figsize=(10, 6))

        plt.hist(
            id_scores,
            bins=50,
            alpha=0.7,
            label=f"{config.id_dataset} (ID)",
            density=True,
        )
        plt.hist(
            ood_scores,
            bins=50,
            alpha=0.7,
            label=f"{config.get_ood_dataset()} (OOD)",
            density=True,
        )

        plt.xlabel(f"{score_type.upper()} Score")
        plt.ylabel("Density")
        plt.title(
            f"OOD Detection: {config.id_dataset} vs {config.get_ood_dataset()}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(
            logger.plots_dir, f"ood_hist_{score_type}_{config.random_seed}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  OOD histogram saved: {save_path}")



# === Pre-defined Configurations ===


def get_fast_config():
    """Fast configuration for testing with OOD detection enabled.

    Default: Cross-dataset OOD with FashionMNIST as ID, MNIST as OOD.
    """
    return OODSSVIConfig(
        training_iterations=2000,
        batch_size=50,
        n_inducing_points=16,
        encoder_hidden_dims=[128, 64],
        local_lr=2e-3,
        global_lr=1e-2,
        P=16,
        log_frequency=200,
        checkpoint_frequency=50,
        enable_gradient_logging=True,
        gradient_log_frequency=25,
        enable_metrics=True,
        metrics_frequency=50,
        save_training_plots=True,
        plot_training=True,
        plot_latent_space=True,
        plot_hyperparams=True,
        # === GP Hyperparameters ===
        init_lengthscales=1.0,
        init_output_noise_scale=1.0,
        # === Missing Data Configuration ===
        enable_missing_data=False,
        missing_data_ratio=0.3,
        reconstruction_samples=50,
        # === Random Seed Configuration ===
        random_seed=7,
        # === Test Size Configuration ===
        test_size=0.2,
        # === OOD Configuration (Cross-dataset mode) ===
        # FashionMNIST classes: 0=T-shirt, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,
        #                       5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot
        # MNIST classes: 0-9 = Digits 0-9
        id_dataset="FashionMNIST",
        id_classes=[1, 2],  # Trouser, Pullover
        id_samples_per_class=100,
        ood_dataset="MNIST",  # Explicitly specify OOD dataset
        ood_classes=[1, 2],  # Digits 1, 2
        ood_samples_per_class=100,
        # === OOD Evaluation Configuration ===
        ood_n_samples=16,  # Number of samples for OOD evaluation
    )


def main(config: OODSSVIConfig = None, config_name: str = "fast"):
    """Main function for OOD Detection SSVI GP-LVM example"""

    # Use default config if none provided
    if config is None:
        config = get_fast_config()
        # Override results directory for OOD experiments when using fast config
        if config_name == "fast":
            config.results_dir = "ood_results/ssvi_results"

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

    # Load data (including OOD data)
    y_train, labels_train, Vx, mu_x, split_data = load_ood_data(config)

    # Create model
    print(f"\nInitializing {config.id_dataset} SSVI GP-LVM...")
    model = OilFlow_SSVI_GPLVM(
        config=config, Vx=Vx, mu_x=mu_x, M=y_train.shape[1]  # Observation dimension
    )

    print(f"Model initialized:")
    print(f"  Mu Encoder: {model.mu_encoder}")
    print(f"  Sigma Encoder: {model.sigma_encoder}")
    print(f"  Inducing points: {model.gp.Vx.shape}")
    print(f"  Latent variables: {model.mu_x.shape}")

    # Initialize logger with config-based naming
    experiment_name = f"ood_ssvi_{config.id_dataset}_{config_name}"
    logger = SSVIResultsLogger(config, experiment_name=experiment_name)
    if logger.results_dir:
        print(f"Results will be saved to: {logger.results_dir}")
        # Log initial info
        logger.log_training(f"=== OOD Detection SSVI GP-LVM Training Session ===")
        logger.log_training(
            f"ID Dataset: {config.id_dataset}, OOD Dataset: {config.get_ood_dataset()}"
        )
        logger.log_training(
            f"Model: {config.n_inducing_points} inducing points, latent_dim={config.latent_dim}"
        )
        logger.log_training(
            f"Training: {config.training_iterations} iter, batch={config.batch_size}"
        )
        logger.log_training(
            f"Learning rates: Local={config.local_lr}, Global={config.global_lr}"
        )

    # Train model
    print(f"\nTraining model...")
    training_result, metrics_computer = train_oil_flow_ssvi(
        model, y_train, config, logger, split_data, labels_train
    )

    # Extract losses from training result
    batch_losses = training_result["batch_losses"]
    full_elbo_losses = training_result["full_elbo_losses"]
    full_elbo_iterations = training_result["full_elbo_iterations"]
    knn_metrics_history = training_result.get("knn_metrics_history", [])
    knn_metrics_iterations = training_result.get("knn_metrics_iterations", [])

    # Visualize results (use full ELBO losses for plots)
    print(f"\nVisualizing results...")
    from src.mnist_ood_plot_helper import visualize_ssvi_results

    visualize_ssvi_results(
        model,
        y_train,
        labels_train,
        full_elbo_losses,
        config,
        logger,
        show_plot=False,
        train_indices=split_data["train_indices"],
        dataset_name=config.id_dataset,
        is_ood=False,
    )
    
    # Generate final metrics and plots if metrics were computed
    summary = None
    if (
        metrics_computer is not None
        and config.enable_metrics
        and split_data is not None
    ):
        print(f"\nGenerating final metrics and plots...")

        # Compute final comprehensive metrics on full test set
        print("Computing final metrics on full datasets...")

        # Compute final metrics with missing data info if enabled
        # Use final full ELBO for metrics (not batch ELBO)
        # Note: full_elbo_losses stores -elbo (loss), so negate to get ELBO
        final_full_elbo = -full_elbo_losses[-1] if full_elbo_losses else 0.0

        if config.enable_missing_data:
            final_metrics = metrics_computer.compute_full_metrics(
                model=model,
                y_train=split_data["y_train"],
                y_test=split_data["y_test"],
                iteration=config.training_iterations,
                elbo_loss=final_full_elbo,
                n_samples=32,
                y_train_original=split_data["y_train_original"],
                y_test_original=split_data["y_test_original"],
                train_missing_mask=split_data["train_missing_mask"],
                test_missing_mask=split_data["test_missing_mask"],
            )
        else:
            final_metrics = metrics_computer.compute_full_metrics(
                model=model,
                y_train=split_data["y_train"],
                y_test=split_data["y_test"],
                iteration=config.training_iterations,
                elbo_loss=final_full_elbo,
                n_samples=32,
            )

        # Print final metrics summary
        summary = metrics_computer.get_metrics_summary()

        # Log the final metrics to the training log
        if logger:
            logger.log_training(
                f"Final comprehensive metrics computed: {final_metrics.to_dict()}"
            )

        print(f"\n=== Final Model Performance ===")
        print(f"Final Train RMSE: {summary['final_train_rmse']:.4f}")
        print(f"Final Test RMSE: {summary['final_test_rmse']:.4f}")
        print(f"Final Train NLPD: {summary['final_train_nlpd']:.3f}")
        print(f"Final Test NLPD: {summary['final_test_nlpd']:.3f}")
        if config.enable_missing_data and not np.isnan(
            summary.get("final_train_missing_nlpd", float("nan"))
        ):
            print(
                f"Final Train Missing Data NLPD: {summary['final_train_missing_nlpd']:.3f}"
            )
            print(
                f"Final Test Missing Data NLPD: {summary['final_test_missing_nlpd']:.3f}"
            )
        if config.enable_missing_data and not np.isnan(
            summary.get("final_train_missing_rmse", float("nan"))
        ):
            print(
                f"Final Train Missing Data RMSE: {summary['final_train_missing_rmse']:.4f}"
            )
            print(
                f"Final Test Missing Data RMSE: {summary['final_test_missing_rmse']:.4f}"
            )
        print(
            f"Overfitting Ratio (Test/Train RMSE): {summary['overfitting_ratio']:.3f}"
        )
        print(
            f"Effective Latent Dimensions: {summary['final_effective_dims']}/{summary['total_latent_dims']}"
        )
        print("=" * 35)

        # Generate metrics plots
        print("Generating metrics progression plots...")
        plot_metrics_history(metrics_computer, logger, show_plot=False, batch_losses=batch_losses)
        
        print("Generating final metrics summary plots...")
        plot_final_metrics_summary(metrics_computer, model, logger, show_plot=False)

        # Generate missing NLPD metrics progression plot if missing data is enabled
        if config.enable_missing_data:
            print("Generating missing NLPD metrics progression plot...")
            plot_missing_nlpd_metrics_progression(
                metrics_computer, logger, show_plot=False
            )
            print("Generating missing RMSE metrics progression plot...")
            plot_missing_rmse_metrics_progression(
                metrics_computer, logger, show_plot=False
            )

        # Save checkpoint metrics and training losses in structured format
        if logger:
            print("Saving checkpoint metrics...")
            logger.save_checkpoint_metrics(
                metrics_computer, "ood_ssvi", config.random_seed
            )
            print("Saving training losses (full dataset loss)...")
            # Save full dataset losses (negative ELBO, for metrics/plots)
            logger.save_training_losses(
                full_elbo_losses, "ood_ssvi_full_loss", config.random_seed
            )
            # Also save batch losses for reference
            logger.save_training_losses(
                batch_losses, "ood_ssvi_batch", config.random_seed
            )

        # Save k-NN metrics and create progression plot
        if knn_metrics_history and knn_metrics_iterations and logger:
            print("Saving k-NN clustering metrics...")
            # Save k-NN metrics to JSON file
            knn_checkpoint_data = {
                "model_name": "ood_ssvi",
                "seed": config.random_seed,
                "total_iterations": config.training_iterations,
                "checkpoints": [],
            }

            for iteration, knn_metrics in zip(
                knn_metrics_iterations, knn_metrics_history
            ):
                knn_checkpoint_data["checkpoints"].append(
                    {
                        "iteration": iteration,
                        **knn_metrics,  # Include all k-NN metrics (train_knn_1, test_knn_1, etc.)
                    }
                )

            knn_checkpoint_file = os.path.join(
                logger.training_results_dir,
                f"ood_ssvi_{config.random_seed}_knn_metrics.json",
            )
            with open(knn_checkpoint_file, "w") as f:
                json.dump(knn_checkpoint_data, f, indent=2)
            logger.log_training(
                f"k-NN metrics saved: training_results/{os.path.basename(knn_checkpoint_file)}"
            )

            print("Generating k-NN metrics progression plot...")
            plot_knn_metrics_progression_ssvi(
                knn_metrics_history,
                knn_metrics_iterations,
                logger.results_dir,
                config.random_seed,
            )

    # === OOD Detection Evaluation ===
    print("\n" + "=" * 60)
    ood_results = evaluate_ood_detection_metrics(model, split_data, config, logger)

    # Plot OOD score histograms
    print("\nGenerating OOD score histograms...")
    plot_ood_score_histograms(ood_results, config, logger)

    # Print OOD summary
    print("\n" + "=" * 60)
    print("OOD DETECTION SUMMARY")
    print("=" * 60)
    print(f"Dataset Pair: {config.id_dataset} (ID) vs {config.get_ood_dataset()} (OOD)")
    print("-" * 60)
    for score_type, result in ood_results.items():
        if "error" not in result:
            metrics = result["metrics"]
            print(f"\n{score_type.upper()} Scores:")
            print(f"  AUROC:  {metrics['auroc']:.4f}")
            print(f"  AUPRC:  {metrics['auprc']:.4f}")
            print(f"  FPR@80: {metrics['fpr_at_80_tpr']:.4f}")
    print("=" * 60)

    # Print final model parameters
    print_final_model_parameters(model, "GP LVM SSVI")

    # Save final model
    if logger:
        print("Saving final model...")
        logger.save_model(
            model, f"ood_ssvi_{config.random_seed}.pkl", is_checkpoint=False
        )

    # Return final_metrics if computed, otherwise None
    final_metrics = None
    if metrics_computer is not None and metrics_computer.metrics_history:
        final_metrics = metrics_computer.metrics_history[-1]

    # Save experiment summary
    if logger:
        # Create experiment summary with model-specific performance metrics
        experiment_summary = {}

        # Add model-specific performance metrics if available
        if summary:
            # Convert tensor to float for JSON serialization
            final_noise = getattr(model, "sigma2", 1.0)
            if hasattr(final_noise, "item"):
                final_noise = final_noise.item()

            experiment_summary["ood_ssvi"] = {
                "id_dataset": config.id_dataset,
                "ood_dataset": config.get_ood_dataset(),
                "train_rmse": summary.get("final_train_rmse", 0.0),
                "test_rmse": summary.get("final_test_rmse", 0.0),
                "train_nlpd": summary.get("final_train_nlpd", 0.0),
                "test_nlpd": summary.get("final_test_nlpd", 0.0),
                "train_missing_data_nlpd": summary.get(
                    "final_train_missing_nlpd", float("nan")
                ),
                "test_missing_data_nlpd": summary.get(
                    "final_test_missing_nlpd", float("nan")
                ),
                "train_missing_data_rmse": summary.get(
                    "final_train_missing_rmse", float("nan")
                ),
                "test_missing_data_rmse": summary.get(
                    "final_test_missing_rmse", float("nan")
                ),
                "final_full_loss": full_elbo_losses[-1] if full_elbo_losses else 0.0,
                "final_batch_loss": batch_losses[-1] if batch_losses else 0.0,
                "final_noise": float(final_noise),
            }

            # Add OOD results (without raw scores to keep file small)
            for score_type, result in ood_results.items():
                if "error" not in result:
                    experiment_summary["ood_ssvi"][f"ood_{score_type}_auroc"] = result[
                        "metrics"
                    ]["auroc"]
                    experiment_summary["ood_ssvi"][f"ood_{score_type}_auprc"] = result[
                        "metrics"
                    ]["auprc"]
                    experiment_summary["ood_ssvi"][f"ood_{score_type}_fpr80"] = result[
                        "metrics"
                    ]["fpr_at_80_tpr"]

            # Add k-NN clustering quality metrics if available
            if knn_metrics_history:
                # Use final k-NN results from history
                final_knn_results = knn_metrics_history[-1]
                for key, value in final_knn_results.items():
                    experiment_summary["ood_ssvi"][f"knn_{key}"] = value

        logger.save_experiment_summary(experiment_summary)

    # Visualize missing data reconstruction if enabled
    if config.enable_missing_data and split_data is not None:
        print("\n=== Missing Data Reconstruction Visualization ===")

        # Simple visualization (3 columns: original, masked, samples)
        simple_viz_path = os.path.join(
            logger.plots_dir, f"missing_data_reconstruction_{config.random_seed}.png"
        )
        visualize_mnist_missing_reconstruction(
            model=model,
            y_train=split_data["y_train"],
            missing_mask=split_data["train_missing_mask"],
            y_original=split_data["y_train_original"],
            n_images=10,
            n_samples=5,
            save_path=simple_viz_path,
            show_plot=False,
            dataset_name=config.id_dataset,
            scaler=split_data.get("scaler"),
        )

        # Detailed visualization (with mean and uncertainty)
        detailed_viz_path = os.path.join(
            logger.plots_dir,
            f"missing_data_reconstruction_detailed_{config.random_seed}.png",
        )
        visualize_mnist_missing_reconstruction_detailed(
            model=model,
            y_train=split_data["y_train"],
            missing_mask=split_data["train_missing_mask"],
            y_original=split_data["y_train_original"],
            n_images=10,
            n_samples=5,
            save_path=detailed_viz_path,
            show_plot=False,
            dataset_name=config.id_dataset,
            scaler=split_data.get("scaler"),
        )

        print("=" * 50)

    # Generate prior samples visualization (works for both MNIST and FashionMNIST)
    print("\n=== Prior Samples Visualization ===")
    prior_samples_path = os.path.join(
        logger.plots_dir, f"prior_samples_{config.random_seed}.png"
    )
    visualize_prior_samples(
        model=model,
        n_images=24,
        save_path=prior_samples_path,
        show_plot=False,
        dataset_name=config.id_dataset,
        image_shape=(28, 28),  # MNIST/FashionMNIST image shape
        scaler=split_data.get("scaler"),
    )

    # Generate prior samples visualization averaged over P (alternative view)
    prior_samples_avg_path = os.path.join(
        logger.plots_dir, f"prior_samples_averaged_{config.random_seed}.png"
    )
    visualize_prior_samples_averaged(
        model=model,
        n_images=8,
        save_path=prior_samples_avg_path,
        show_plot=False,
        dataset_name=config.id_dataset,
        image_shape=(28, 28),  # MNIST/FashionMNIST image shape
        scaler=split_data.get("scaler"),
    )
    print("=" * 50)

    # Generate test image predictions visualization
    print("\n=== Test Image Predictions Visualization ===")
    test_image_predictions_path = os.path.join(logger.plots_dir, f"test_image_predictions_{config.random_seed}.png")
    visualize_test_image_predictions(
        model=model,
        y_train=split_data['y_test'],
        config=config,
        n_images=24,
        save_path=test_image_predictions_path,
        show_plot=False,
        dataset_name=config.id_dataset,
        image_shape=(28, 28),  # MNIST/FashionMNIST image shape
        scaler=split_data.get('scaler')
    )
    print("=" * 50)

    return model, training_result, config, metrics_computer, final_metrics, ood_results


def create_ood_argument_parser():
    """Create argument parser with OOD-specific options"""
    # Start with base parser
    parser = create_argument_parser()

    # ID dataset configuration
    parser.add_argument(
        "--id-dataset",
        type=str,
        default=None,
        choices=["MNIST", "FashionMNIST"],
        help="In-distribution dataset (default from config: FashionMNIST)",
    )
    parser.add_argument(
        "--id-classes",
        type=int,
        nargs="+",
        default=None,
        help="ID class indices (e.g., --id-classes 0 1 2 3 4 5 6 7)",
    )
    parser.add_argument(
        "--id-samples-per-class", type=int, default=None, help="Samples per ID class"
    )

    # OOD dataset configuration
    parser.add_argument(
        "--ood-dataset",
        type=str,
        default=None,
        choices=["MNIST", "FashionMNIST"],
        help="OOD dataset (default from config: MNIST)",
    )
    parser.add_argument(
        "--ood-classes",
        type=int,
        nargs="+",
        default=None,
        help="OOD class indices (e.g., --ood-classes 8 9)",
    )
    parser.add_argument(
        "--ood-samples-per-class", type=int, default=None, help="Samples per OOD class"
    )

    # Convenience: same-dataset OOD mode
    parser.add_argument(
        "--same-dataset-ood",
        action="store_true",
        help="Use same-dataset OOD mode (requires --ood-classes)",
    )

    # OOD evaluation configuration
    parser.add_argument(
        "--ood-n-samples",
        type=int,
        default=None,
        help="Number of samples for OOD evaluation (default: same as training P)",
    )

    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_ood_argument_parser()
    args = parser.parse_args()

    # Ensure config directory exists
    ensure_config_dir_exists()

    # Determine config name
    config_name = "fast"

    if args.config is None:
        # Use our OOD-specific fast config directly (not from oil_flow)
        print("No config file specified. Using default OOD fast_config...\n")
        config = get_fast_config()
        config_name = "fast"
    else:
        # Load from YAML file
        print(f"Loading configuration from: {args.config}")
        from src.ssvi_config_utils import (
            load_config_from_yaml,
            dict_to_dataclass,
            get_config_name_from_path,
        )
        import os

        if os.path.exists(args.config):
            config_dict = load_config_from_yaml(args.config)
            config = dict_to_dataclass(config_dict, OODSSVIConfig)
            config_name = get_config_name_from_path(args.config)
            # Set results_dir to config directory
            config_dir = os.path.dirname(args.config)
            config.results_dir = os.path.join(config_dir, "ood_run_results")
        else:
            raise Exception(f"Config file {args.config} not found, using default OOD fast config")

            config = get_fast_config()
            config_name = "fast"

    # Apply command line overrides
    if args.config_name is not None:
        config_name = args.config_name
    if args.device is not None:
        config.device = args.device
    if args.training_iterations is not None:
        config.training_iterations = args.training_iterations
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.local_lr is not None:
        config.local_lr = args.local_lr
    if args.global_lr is not None:
        config.global_lr = args.global_lr
    if hasattr(args, "no_metrics") and args.no_metrics:
        config.enable_metrics = False
    if hasattr(args, "no_plots") and args.no_plots:
        config.plot_training = False
        config.plot_latent_space = False
        config.plot_hyperparams = False
        config.save_training_plots = False
    if hasattr(args, "no_save") and args.no_save:
        config.save_results = False

    # Apply OOD-specific overrides (only if explicitly provided on command line)
    if hasattr(args, "id_dataset") and args.id_dataset is not None:
        config.id_dataset = args.id_dataset
    if hasattr(args, "id_classes") and args.id_classes is not None:
        config.id_classes = args.id_classes
    if hasattr(args, "id_samples_per_class") and args.id_samples_per_class is not None:
        config.id_samples_per_class = args.id_samples_per_class
    if hasattr(args, "ood_dataset") and args.ood_dataset is not None:
        config.ood_dataset = args.ood_dataset
    if hasattr(args, "ood_classes") and args.ood_classes is not None:
        config.ood_classes = args.ood_classes
    if (
        hasattr(args, "ood_samples_per_class")
        and args.ood_samples_per_class is not None
    ):
        config.ood_samples_per_class = args.ood_samples_per_class
    # Handle same-dataset OOD convenience flag
    if hasattr(args, "same_dataset_ood") and args.same_dataset_ood:
        config.ood_dataset = config.id_dataset
    if hasattr(args, "ood_n_samples") and args.ood_n_samples is not None:
        config.ood_n_samples = args.ood_n_samples

    # Override results directory for OOD experiments when using fast config
    if config_name == "fast":
        config.results_dir = "ood_results/ssvi_results"

    print(f"Using configuration: {config_name}")
    if args.config:
        print(f"Config file: {args.config}")

    # Run main function with loaded config
    main(config, config_name)
