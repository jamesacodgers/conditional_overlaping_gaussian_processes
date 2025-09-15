#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Original GP LVM experiments
Based on the paper "GP LVM with SVI" implementation
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import torch

@dataclass
class OriginalGPLVMConfig:
    """Configuration for Original GP LVM experiments"""
    
    # === Device Configuration ===
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"
    
    # === Data Configuration ===
    dataset: str = "oilflow"  # Dataset to use
    test_size: int = 50  # Number of test samples
    random_seed: int = 7  # Random seed for reproducibility
    
    # === Model Configuration ===
    latent_dim: int = 12  # Latent dimension
    n_inducing: int = 25  # Number of inducing points
    pca_init: bool = False  # Whether to initialize with PCA
    
    # === Model Selection ===
    models: List[str] = None  # Will be set to ['gauss', 'nn_gauss'] if None
    
    # === Training Configuration ===
    training_steps: dict = None  # Will be set to {'gauss': 200, 'nn_gauss': 200} if None
    batch_size: int = 100
    learning_rate: float = 0.001
    
    # === Model-specific Configuration ===
    # For nn_gauss model
    nn_layers: tuple = (10, 5)  # Neural network layers for encoder
    
    # === Results Configuration ===
    save_results: bool = True
    results_dir: str = "ssvi_scripts_experiments/oil_flow_results/original_gplvm_results"
    experiment_name: str = "fast"
    
    # === Visualization Configuration ===
    save_plots: bool = True
    plot_colors: List[str] = None  # Will be set to ['r', 'b', 'g'] if None
    
    # === Testing Configuration ===
    enable_testing: bool = True  # Enable test evaluation
    test_iterations: int = 200  # Number of iterations for test phase (default for fast config)
    
    # === Additional Configuration from oilflow_expt.py ===
    # Prior configuration for different models
    prior_type: dict = None  # Will be set to {'gauss': 'normal', 'nn_gauss': 'multivariate'} if None
    
    # Kernel configuration
    kernel_type: str = "rbf"  # Kernel type (RBF from oilflow_expt.py)
    use_ard: bool = True  # ARD (Automatic Relevance Determination) for RBF kernel
    
    # Mean function configuration
    mean_function: str = "zero"  # ZeroMean() in oilflow_expt.py
    
    # Variational configuration
    variational_strategy: str = "cholesky"  # CholeskyVariationalDistribution in oilflow_expt.py
    learn_inducing_locations: bool = True  # learn_inducing_locations=True in oilflow_expt.py
    
    def __post_init__(self):
        """Set default values and resolve device if not specified"""
        if self.models is None:
            self.models = ['gauss', 'nn_gauss']
        
        if self.training_steps is None:
            self.training_steps = {'gauss': 200, 'nn_gauss': 200}
        
        if self.plot_colors is None:
            self.plot_colors = ['r', 'b', 'g']
        
        if self.prior_type is None:
            self.prior_type = {'gauss': 'normal', 'nn_gauss': 'multivariate'}
        
        # Resolve device configuration
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def print_config(self):
        """Print configuration summary"""
        print("=== Original GP LVM Configuration ===")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0) if self.device.startswith('cuda') else 'Not using CUDA'}")
        else:
            print("CUDA available: No")
        print(f"Dataset: {self.dataset}")
        print(f"Models: {self.models}")
        print(f"Latent dim: {self.latent_dim}, Inducing points: {self.n_inducing}")
        print(f"Training steps: {self.training_steps}")
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
        print(f"Save results: {self.save_results}")
        if self.save_results:
            print(f"Results dir: {self.results_dir}")
        print("=" * 50)

def get_fast_config():
    """Fast configuration for testing"""
    return OriginalGPLVMConfig(
        device="cuda",  # Use CUDA device
        training_steps={'gauss': 200, 'nn_gauss': 200},
        batch_size=50,
        n_inducing=15,
        learning_rate=0.002,
        random_seed=42,
        test_iterations=200
    )

def get_base_config():
    """Base configuration matching oilflow_expt.py settings"""
    return OriginalGPLVMConfig(
        training_steps={'gauss': 60000, 'nn_gauss': 60000},
        batch_size=100,
        n_inducing=25,
        learning_rate=0.001,
        random_seed=7,
        test_iterations=5000,
        latent_dim=12,
        pca_init=False,
        prior_type={'gauss': 'normal', 'nn_gauss': 'multivariate'},
        kernel_type="rbf",
        use_ard=True,
        mean_function="zero",
        variational_strategy="cholesky",
        learn_inducing_locations=True
    )
