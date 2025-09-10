# %% 
"""
Oil Flow Custom ELBO GP-LVM

Usage examples:
    # Run with default fast config
    python ssvi_scripts_experiments/oil_flow_GPLVM_custom_elbo.py
    
    # Run with base config
    python ssvi_scripts_experiments/oil_flow_GPLVM_custom_elbo.py --config ssvi_scripts_experiments/original_gp_lvm_configs/base_config.yaml
    
    # Run with base config and overrides
    python ssvi_scripts_experiments/oil_flow_GPLVM_custom_elbo.py --config ssvi_scripts_experiments/original_gp_lvm_configs/base_config.yaml --training-iterations 500 --device cuda
    
    # Run with custom config name for results
    python ssvi_scripts_experiments/oil_flow_GPLVM_custom_elbo.py --config ssvi_scripts_experiments/original_gp_lvm_configs/base_config.yaml --config-name my_experiment
"""

import torch 
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import tarfile
import os
import sys
import json
from datetime import datetime
import time

sys.path.insert(0, os.getcwd())

from src.struc_gplvm import Struc_GPLVM_Custom_ELBO
from src.original_gplvm_config_utils import (
    OilFlowGPLVMConfig, create_gplvm_argument_parser, 
    load_gplvm_config_with_overrides, get_gplvm_base_config_path, 
    ensure_gplvm_config_dir_exists, print_available_gplvm_configs
)
from src.ssvi_utils import (
    SSVIResultsLogger, visualize_ssvi_results, plot_metrics_history, 
    plot_final_metrics_summary, save_metrics_to_json, save_metrics_to_text
)
from src.ssvi_metrics import SSVIMetricsComputer, create_train_test_split
from src.missing_data_utils import preprocess_data_for_missing_training
from src.save_utils import save_paper_style_metrics, print_final_model_parameters
from src.missing_data_experiment import run_missing_data_reconstruction_experiment

# Device will be set in main() from config

class OilFlow_GPLVM(Struc_GPLVM_Custom_ELBO):
    """Simplified Oil Flow GP-LVM using custom ELBO base class"""
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64, trainable_V=True, enable_missing_data=False):
        super().__init__(Vx=Vx, M=M, mu_x=mu_x, sigma2=sigma2, beta=beta, sigma2_s=sigma2_s, P=P, trainable_V=trainable_V)
        
        # Missing data capability for GP-LVM
        self.enable_missing_data = enable_missing_data
        if self.enable_missing_data:
            print("Missing data reconstruction enabled for GP-LVM")

    def gplvm_elbo(self, y, mu_x=None, Sigma_x=None, U_samples=None, batch_size=None, total_size=None):
        """
        Compute GP-LVM ELBO using custom ELBO base class.
        Now requires U_samples to be passed in.
        """
        if mu_x is None:
            mu_x = self.mu_x
        if Sigma_x is None:
            Sigma_x = self.Sigma_x
        if U_samples is None:
            # Generate U samples if not provided
            U_samples = self.get_u_samples()
        
        # Handle missing data if enabled
        if self.enable_missing_data:
            missing_mask = torch.isnan(y)
            if missing_mask.any():
                # Store missing mask in coordinator for likelihood computation
                self.coordinator.missing_mask = missing_mask
                elbo_result = super().gplvm_elbo(y, mu_x, Sigma_x, U_samples, batch_size, total_size)
                # Clean up
                if hasattr(self.coordinator, 'missing_mask'):
                    delattr(self.coordinator, 'missing_mask')
                return elbo_result
        
        # Use the custom ELBO from base class
        return super().gplvm_elbo(y, mu_x, Sigma_x, U_samples, batch_size, total_size)
    
    def get_u_samples(self):
        """Generate U samples for ELBO computation"""
        dist = self.gp.get_q_u_dist()
        samples = dist.rsample((self.P,))
        return samples.reshape(self.P, self.M, self.C, self.gp.Lx)
    
    def get_mu_sigma_x_train_samples(self, Y_train_batch=None, U_samples=None):
        """
        Get latent variable samples for training data
        
        Args:
            Y_train_batch: Training observations (used to determine batch size)
            U_samples: Not used for oil_flow_GPLVM (uses stored latent variables)
            
        Returns:
            (mu_x, sigma_x): Training latent variable samples (P, batch_size, D)
        """
        if Y_train_batch is not None:
            batch_size = Y_train_batch.shape[0]
            # Return training data subset (assumes Y_batch corresponds to first batch_size training points)
            mu_batch = self.mu_x_train[:, :batch_size, :]  # (P, batch_size, D)
            sigma_batch = self.Sigma_x_train[:, :batch_size, :]  # (P, batch_size, D)
        else:
            # Return all training data
            mu_batch = self.mu_x_train  # (P, N_train, D)
            sigma_batch = self.Sigma_x_train  # (P, N_train, D)
        
        return mu_batch, sigma_batch
    
    def get_mu_sigma_x_test_samples(self, Y_test_batch=None, U_samples=None):
        """
        Get latent variable samples for test data
        
        Args:
            Y_test_batch: Test observations (used to determine batch size and validate setup)
            U_samples: Not used for oil_flow_GPLVM (uses optimized test latent variables)
            
        Returns:
            (mu_x, sigma_x): Test latent variable samples (P, batch_size, D)
        """
        if not hasattr(self, 'mu_x_test_predict') or self.mu_x_test_predict is None:
            raise ValueError("Test data not initialized. Call setup_test_data() and prepare_for_test_metrics() first.")
        
        if Y_test_batch is not None:
            batch_size = Y_test_batch.shape[0]
            if batch_size != self._test_data_size:
                raise ValueError(f"Test batch size {batch_size} doesn't match initialized test size {self._test_data_size}")
        
        # Return optimized test latent variables
        mu_batch = self.mu_x_test_predict  # (P, N_test, D)
        sigma_batch = self.Sigma_x_test_predict  # (P, N_test, D)
        return mu_batch, sigma_batch
    
    def prepare_for_test_metrics(self, y_test, config, n_optimization_steps=100):
        """
        Prepare model for computing test metrics by optimizing test latent variables
        
        Args:
            y_test: Test observations
            config: Configuration object with learning rate info
            n_optimization_steps: Number of optimization steps for test variables
        """
        if not hasattr(self, 'mu_x_test_predict') or self.mu_x_test_predict is None:
            print("Warning: Test data not initialized. Call setup_test_data() first.")
            return
        
        print(f"Optimizing test latent variables for metrics computation ({n_optimization_steps} steps)...")
        
        # Setup test optimizer (only test parameters)
        test_params = self.get_test_parameters()
        if not test_params:
            print("Warning: No test parameters found for optimization.")
            return
        
        test_optimizer = torch.optim.Adam(test_params, lr=config.learning_rate)
        
        # Store original training state
        was_training = self.training
        self.train()
        
        initial_loss = None
        for i in range(n_optimization_steps):
            test_optimizer.zero_grad()
            
            # Compute test ELBO (only optimizes test latent variables)
            test_loss = -self.test_prediction_elbo(y_test)
            test_loss.backward()
            test_optimizer.step()
            
            if i == 0:
                initial_loss = test_loss.item()
            
            # Minimal progress reporting
            if i % (n_optimization_steps // 4) == 0 or i == n_optimization_steps - 1:
                print(f"  Step {i}/{n_optimization_steps}: Loss={test_loss.item():.4f}")
        
        final_loss = test_loss.item()
        print(f"Test optimization completed. Loss: {initial_loss:.4f} -> {final_loss:.4f} (improvement: {initial_loss - final_loss:.4f})")
        
        # Restore original training state
        self.train(was_training)
    
    def setup_test_data(self, mu_x_test_init):
        """Initialize new test data parameters for separate optimization (separate from parent class test vars)"""
        if mu_x_test_init is not None:
            # Use new variable names to avoid touching parent class test variables
            self.mu_x_test_predict = torch.nn.Parameter(mu_x_test_init[None,...].repeat(self.P,1,1))
            self._log_sigma_x_test_predict = torch.nn.Parameter(torch.zeros_like(self.mu_x_test_predict))
            
            # Keep track of test data size for validation
            self._test_data_size = mu_x_test_init.shape[0]
    
    def get_test_parameters(self):
        """Get only our new test parameters for separate optimization"""
        if hasattr(self, 'mu_x_test_predict') and self.mu_x_test_predict is not None:
            return [self.mu_x_test_predict, self._log_sigma_x_test_predict]
        return []
    
    def get_training_parameters(self):
        """Get training parameters (excluding our new test parameters)"""
        training_params = []
        for name, param in self.named_parameters():
            # Exclude only our new test parameters from training optimizer
            if 'mu_x_test_predict' not in name and '_log_sigma_x_test_predict' not in name:
                training_params.append(param)
        return training_params
    
    @property
    def Sigma_x_test_predict(self):
        """Variance for our NEW test parameters"""
        if hasattr(self, '_log_sigma_x_test_predict'):
            return torch.exp(self._log_sigma_x_test_predict)
        return None
    
    def reconstruct_missing_data(self, y_observed, missing_mask=None, n_samples=50):
        """
        Reconstruct missing dimensions using GP-LVM optimization and GP decoding.
        For GP-LVM, this requires optimizing latent variables for the observed data.
        
        Args:
            y_observed: Data with missing values as NaN [batch_size, output_dim]
            missing_mask: Boolean mask [batch_size, output_dim] where True=missing (optional)
            n_samples: Number of samples for Monte Carlo reconstruction
            
        Returns:
            y_reconstructed: Complete data with reconstructed missing values
            reconstruction_std: Standard deviation of reconstructions
        """
        if not self.enable_missing_data:
            raise ValueError("Missing data reconstruction not enabled. Set enable_missing_data=True in model constructor.")
        
        # Create missing mask from NaN values if not provided
        if missing_mask is None:
            missing_mask = torch.isnan(y_observed)
        
        batch_size = y_observed.shape[0]
        
        # Initialize latent variables for the data with missing values
        # Use PCA initialization on observed data only
        y_clean = torch.where(missing_mask, torch.zeros_like(y_observed), y_observed)
        mu_x_init, _, _ = torch.pca_lowrank(y_clean, self.A)
        mu_x_init = mu_x_init / mu_x_init.std(dim=0)
        
        # Setup temporary latent variables for this data
        mu_x_temp = torch.nn.Parameter(mu_x_init[None,...].repeat(self.P,1,1))
        log_sigma_x_temp = torch.nn.Parameter(torch.zeros_like(mu_x_temp))
        
        # Optimizer for latent variables
        temp_optimizer = torch.optim.Adam([mu_x_temp, log_sigma_x_temp], lr=0.01)
        
        # Optimize latent variables using only observed dimensions
        self.eval()
        for i in range(min(100, n_samples * 2)):  # Optimize before sampling
            temp_optimizer.zero_grad()
            
            # Get GP expectation for current latent variables
            sigma_x_temp = torch.exp(log_sigma_x_temp)
            
            # Sample latent variables
            sample_idx = torch.randint(0, self.P, (1,)).item()
            mu_sample = mu_x_temp[sample_idx]
            sigma_sample = sigma_x_temp[sample_idx]
            
            epsilon = torch.randn_like(sigma_sample)
            x_sample = mu_sample + sigma_sample * epsilon
            
            # Get GP prediction
            y_pred, _ = self.get_gp_expectation(x_sample)
            
            # Compute likelihood only for observed dimensions using GP-LVM's ELBO
            # Create clean data for ELBO computation
            y_clean = torch.where(missing_mask, torch.zeros_like(y_observed), y_observed)
            
            # Store missing mask for coordinator and compute ELBO
            if hasattr(self, 'coordinator'):
                self.coordinator.missing_mask = missing_mask
                # Generate U samples for ELBO computation
                U_samples = self.get_u_samples()
                log_likelihood = self.gplvm_elbo(y_clean, U_samples=U_samples)
                # Clean up
                if hasattr(self.coordinator, 'missing_mask'):
                    delattr(self.coordinator, 'missing_mask')
            else:
                # Proper Gaussian log-likelihood for observed dimensions (following SVI paper formulation)
                observed_mask = ~missing_mask
                n_observed = observed_mask.sum()
                if n_observed > 0:
                    observed_errors = (y_pred - y_clean)[observed_mask]
                    # Complete Gaussian log-likelihood: -0.5 * N * log(2*pi*sigma^2) - 0.5 * sum((y-mu)^2)/sigma^2
                    log_likelihood = -0.5 * n_observed * torch.log(2 * torch.pi * self.sigma2) - 0.5 * torch.sum(observed_errors**2) / self.sigma2
                else:
                    log_likelihood = torch.zeros(1, device=y_pred.device)
            
            # Add KL regularization
            kl_x = 0.5 * torch.sum(mu_sample**2 + sigma_sample**2 - 1 - 2*torch.log(sigma_sample + 1e-8))
            
            loss = -log_likelihood + 0.01 * kl_x  # Small KL weight
            loss.backward()
            temp_optimizer.step()
        
        # Sample reconstructions
        reconstructions = []
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample latent variables
                sample_idx = torch.randint(0, self.P, (1,)).item()
                mu_sample = mu_x_temp[sample_idx]
                sigma_sample = torch.exp(log_sigma_x_temp[sample_idx])
                
                epsilon = torch.randn_like(sigma_sample)
                x_sample = mu_sample + sigma_sample * epsilon
                
                # Decode to observation space using GP
                y_mean, _ = self.get_gp_expectation(x_sample)
                reconstructions.append(y_mean)
        
        # Stack reconstructions and compute statistics
        reconstructions = torch.stack(reconstructions, dim=0)  # [n_samples, batch_size, output_dim]
        
        # Compute mean and std across samples
        y_reconstructed_mean = reconstructions.mean(dim=0)
        y_reconstructed_std = reconstructions.std(dim=0)
        
        # For complete reconstruction, use observed values where available
        y_complete = torch.where(missing_mask, 
                                y_reconstructed_mean, 
                                torch.where(torch.isnan(y_observed), 
                                           y_reconstructed_mean, 
                                           y_observed))
        
        return y_complete, y_reconstructed_std
    
    def test_prediction_elbo(self, y_test):
        """
        Simplified test ELBO computation using custom ELBO base class.
        This is much simpler now since we can use the custom ELBO directly.
        """
        if not hasattr(self, 'mu_x_test_predict') or self.mu_x_test_predict is None:
            raise ValueError("Test data not initialized. Call setup_test_data() first.")
        
        # Validate test data size matches
        if y_test.shape[0] != self._test_data_size:
            raise ValueError(f"Test data size mismatch: expected {self._test_data_size}, got {y_test.shape[0]}")
        
        # Generate U samples for test ELBO computation
        U_samples = self.get_u_samples()
        
        # Use the custom ELBO from base class with test data
        # The base class elbo method handles both training and test data
        return self.elbo(
            D_train=y_test,  # Test data as "training" data for ELBO computation
            R_train=torch.ones((y_test.shape[0], 1), device=y_test.device, dtype=y_test.dtype),
            mu_x_train=self.mu_x_test_predict,  # Use test latent variables
            Sigma_x_train=self.Sigma_x_test_predict,  # Use test latent variables
            U_samples=U_samples
        )


def get_fast_gplvm_config():
    """Fast configuration for testing GP-LVM with missing data enabled"""
    return OilFlowGPLVMConfig(
        training_iterations=200,
        learning_rate=1e-2,
        n_inducing_points=8,
        latent_dim=4,
        P=4,
        log_frequency=100,
        enable_gradient_logging=True,  # Enable gradient logging like SSVI
        gradient_log_frequency=50,     # Add gradient log frequency
        enable_metrics=True,           
        metrics_frequency=100,
        save_training_plots=True,
        plot_save_frequency=100,        
        plot_training=True,
        plot_latent_space=True,
        plot_hyperparams=True,
        test_optimization_steps=50,  # Config value for all test preparation
        # === Missing Data Configuration ===
        enable_missing_data=True,      # Enable missing data reconstruction capability
        missing_data_ratio=0.3,        # Fraction of dimensions to artificially mask (for testing)
        reconstruction_samples=50,     # Number of samples for missing data reconstruction
        missing_data_seed=123          # Seed for reproducible missing data patterns
    )

def create_gplvm_model(config: OilFlowGPLVMConfig, Vx, mu_x, M):
    """Create GP-LVM model"""
    model = OilFlow_GPLVM(
        Vx=Vx, 
        M=M, 
        mu_x=mu_x, 
        sigma2=config.sigma2, 
        beta=config.beta,
        sigma2_s=config.sigma2_s,
        P=config.P,
        trainable_V=config.trainable_V,
        enable_missing_data=config.enable_missing_data
    )
    return model

def load_oil_flow_data_gplvm(config):
    """Load and preprocess oil flow data for GP-LVM with optional train/test split"""
    print("Downloading and loading oil flow data...")
    
    # Download data
    urllib.request.urlretrieve(config.data_url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')
    
    # Load data and move to device
    y_full = torch.Tensor(np.loadtxt(fname='DataTrn.txt')).to(config.device)
    labels_full = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt')).to(config.device)
    labels_full = (labels_full @ torch.diag(torch.tensor([1, 2, 3], device=config.device, dtype=labels_full.dtype))).sum(axis=1)
    
    # Create train/test split if metrics enabled
    if config.enable_metrics:
        print(f"Creating train/test split (test_size={config.test_size})...")
        y_train, y_test, labels_train, labels_test, train_indices, test_indices = create_train_test_split(
            y_full, labels_full, 
            test_size=config.test_size, 
            random_seed=config.split_random_seed
        )
        print(f"Train set: {y_train.shape[0]} samples")
        print(f"Test set: {y_test.shape[0]} samples")
        
        # Use training data for model initialization
        y = y_train
        labels = labels_train
        split_data = {
            'y_train': y_train,
            'y_test': y_test,
            'labels_train': labels_train,
            'labels_test': labels_test,
            'train_indices': train_indices,
            'test_indices': test_indices
        }
    else:
        # Use full dataset
        y = y_full
        labels = labels_full
        split_data = None

    # Initialize inducing points and latent variables (following original)
    Vx = torch.randn(config.n_inducing_points, config.latent_dim, device=config.device)
    mu_x, _, _ = torch.pca_lowrank(y, config.latent_dim)
    mu_x = mu_x / mu_x.std(dim=0)
    
    print(f"Data loaded: {y.shape[0]} samples, {y.shape[1]} dimensions")
    print(f"Latent initialization: {mu_x.shape}")
    print(f"Inducing points: {Vx.shape}")
    config.total_size = y.shape[0]
    
    # Apply missing data preprocessing if enabled
    if config.enable_missing_data:
        print(f"\n=== Missing Data Preprocessing (GP-LVM) ===")
        y_missing, missing_mask, missing_info = preprocess_data_for_missing_training(y, config)
        
        # Store original data for evaluation
        y_original = y.clone()
        y = y_missing  # Use missing data for training
        
        # Update split data if it exists
        if split_data is not None:
            y_train_missing, train_missing_mask, _ = preprocess_data_for_missing_training(split_data['y_train'], config)
            y_test_missing, test_missing_mask, _ = preprocess_data_for_missing_training(split_data['y_test'], config)
            
            split_data.update({
                'y_train_original': split_data['y_train'].clone(),
                'y_test_original': split_data['y_test'].clone(),
                'y_train': y_train_missing,
                'y_test': y_test_missing,
                'train_missing_mask': train_missing_mask,
                'test_missing_mask': test_missing_mask
            })
        
        # Store missing data info for later use
        config.missing_info = missing_info
        config.y_original = y_original
        config.missing_mask = missing_mask
        
        print(f"Missing data preprocessing completed.")
        print("=" * 50)

    return y, labels, Vx, mu_x, split_data

def train_gplvm_with_logging(model, y, config, logger=None, split_data=None, labels=None):
    """Training function for GP-LVM with logging and metrics"""
    losses = []
    
    print(f"Training GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Learning rate: {config.learning_rate}")
    
    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(f"Metrics computation enabled (frequency: every {config.metrics_frequency} iterations)")
        metrics_computer = SSVIMetricsComputer(device=config.device)
        print(f"Debug: Model mu_x shape: {model.mu_x.shape}")
        print(f"Debug: Model Sigma_x shape: {model.Sigma_x.shape}")
        
        # Setup test data for separate optimization if metrics enabled
        if split_data is not None:
            # Initialize test latent variables using PCA of test data
            print("Setting up test data for separate optimization...")
            y_test = split_data['y_test']
            mu_x_test_init, _, _ = torch.pca_lowrank(y_test, config.latent_dim)
            mu_x_test_init = mu_x_test_init / mu_x_test_init.std(dim=0)
            model.setup_test_data(mu_x_test_init)
            print(f"Test latent variables initialized: {model.mu_x_test_predict.shape}")
    
    # Setup training optimizer (excludes test parameters) 
    training_optimizer = torch.optim.Adam(model.get_training_parameters(), lr=config.learning_rate)
    
    # Setup test optimizer (only our NEW test parameters)
    test_optimizer = None
    if hasattr(model, 'mu_x_test_predict') and model.mu_x_test_predict is not None:
        test_params = model.get_test_parameters()
        if test_params:
            test_optimizer = torch.optim.Adam(test_params, lr=config.learning_rate)
            print(f"Test optimizer initialized with {len(test_params)} parameters")
    
    start_time = time.time()
    
    for i in range(config.training_iterations):
        training_optimizer.zero_grad()
        
        # Generate U samples for this iteration
        U_samples = model.get_u_samples()
        
        # Compute ELBO using custom ELBO method
        loss = -model.gplvm_elbo(y=y, U_samples=U_samples)
        loss.backward()
        training_optimizer.step()
        losses.append(loss.detach().item())
        
        # Metrics computation (periodic) - reuse SSVI infrastructure
        if (metrics_computer is not None and 
            config.enable_metrics and 
            i % config.metrics_frequency == 0):
            
            print(f"  Computing metrics at iteration {i}...")
            try:
                # Use full datasets for proper metrics computation
                y_train_full = split_data['y_train']
                y_test_full = split_data['y_test']
                
                # Prepare model for test metrics computation
                if hasattr(model, 'prepare_for_test_metrics'):
                    print(f"    Preparing model for test metrics computation...")
                    model.prepare_for_test_metrics(y_test_full, config, n_optimization_steps=config.test_optimization_steps)
                
                print(f"    Computing metrics with y_train shape: {y_train_full.shape}")
                print(f"    Computing metrics with y_test shape: {y_test_full.shape}")
                
                metrics = metrics_computer.compute_full_metrics(
                    model=model,  
                    y_train=y_train_full,
                    y_test=y_test_full,
                    iteration=i,
                    elbo_loss=loss.item(),
                    n_samples=config.P  
                )
                
                print(f"    Train RMSE: {metrics.train_rmse:.4f}, Test RMSE: {metrics.test_rmse:.4f}")
                print(f"    Train NLPD: {metrics.train_nlpd:.3f}, Test NLPD: {metrics.test_nlpd:.3f}")
                print(f"    Effective dims: {metrics.effective_latent_dims}/{metrics.total_latent_dims}")
                
                # Log metrics
                if logger:
                    logger.log_training(f"Metrics at iteration {i}: {metrics.to_dict()}")
                    
            except Exception as e:
                print(f"    Error computing metrics at iteration {i}: {e}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                # Re-raise the error to see it in the main output
                raise e
        
        # Training plot saving (periodic)  
        if (logger and config.save_training_plots and 
            i > 0 and i % config.plot_save_frequency == 0):
            
            print(f"  Saving training plots at iteration {i}...")
            try:
                # Generate filename with iteration number
                plot_filename = f"training_results_iter_{i}.png"
                
                # Use same SSVI visualization for identical output structure
                visualize_ssvi_results(
                    model=model,
                    y=y,
                    labels=labels,
                    losses=losses,
                    config=config,
                    logger=logger,
                    save_filename=plot_filename,
                    show_plot=False,
                    is_test_data=False  # This is training data visualization
                )
                
                print(f"    Training plots saved: {plot_filename}")
                
            except Exception as e:
                print(f"    Warning: Error saving training plots at iteration {i}: {e}")
        
        # Gradient logging (if enabled)
        if logger and config.enable_gradient_logging and i % config.gradient_log_frequency == 0:
            # Simple gradient info for GP-LVM
            gradient_info = {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'parameters_with_grad': sum(p.numel() for p in model.parameters() if p.grad is not None)
            }
            logger.log_gradients(i, loss.item(), gradient_info)
        
        # Progress reporting
        if i % config.log_frequency == 0:
            elapsed = time.time() - start_time
            message = f"Iteration {i}/{config.training_iterations}: Loss={loss.item():.4f}, Time={elapsed:.1f}s"
            print(f"Iteration {i}/{config.training_iterations}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            # Log to file
            if logger:
                logger.log_training(message)
    
    total_time = time.time() - start_time
    final_message = f"Training completed! Final Loss: {losses[-1]:.4f}, Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    print(f"Training completed!")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)")
    # Log to file
    if logger:
        logger.log_training(final_message)
    
    return losses, metrics_computer

def predict_test_data_with_separate_optimizer(model, y_test, config, n_test_iterations=100, test_lr=None):
    """
    Predict test data using separate optimizer for our NEW test latent variables.
    Only optimizes mu_x_test_predict and sigma_x_test_predict, keeping GP parameters fixed.
    """
    if not hasattr(model, 'mu_x_test_predict') or model.mu_x_test_predict is None:
        raise ValueError("Test data not initialized. Ensure model.setup_test_data() was called.")
    
    if test_lr is None:
        test_lr = config.learning_rate
    
    print(f"Optimizing test latent variables for {y_test.shape[0]} test points...")
    print(f"Test iterations: {n_test_iterations}, Test learning rate: {test_lr}")
    
    # Setup test optimizer (only test parameters)
    test_params = model.get_test_parameters()
    if not test_params:
        raise ValueError("No test parameters found for optimization.")
    
    test_optimizer = torch.optim.Adam(test_params, lr=test_lr)
    test_losses = []
    
    # Store original training state
    model.train()
    
    start_time = time.time()
    
    for i in range(n_test_iterations):
        test_optimizer.zero_grad()
        
        # Compute test ELBO (only optimizes test latent variables)
        test_loss = -model.test_prediction_elbo(y_test)
        test_loss.backward()
        test_optimizer.step()
        
        test_losses.append(test_loss.detach().item())
        
        # Progress reporting
        if i % (n_test_iterations // 5) == 0 or i == n_test_iterations - 1:
            elapsed = time.time() - start_time
            print(f"  Test iteration {i}/{n_test_iterations}: Loss={test_loss.item():.4f}, Time={elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Test optimization completed! Final Loss: {test_losses[-1]:.4f}, Total time: {total_time:.1f}s")
    
    return test_losses

def main(config: OilFlowGPLVMConfig = None, config_name: str = "fast"):
    """Main function for Oil Flow GP-LVM example"""
    
    # Use default config if none provided
    if config is None:
        config = get_fast_gplvm_config()
    
    # Set device from config
    torch.set_default_device(config.device)
    
    # Print configuration
    config.print_config()
    
    # Load oil flow data
    y, labels, Vx, mu_x, split_data = load_oil_flow_data_gplvm(config)
    
    # Create model
    print(f"\nInitializing Oil Flow GP-LVM...")
    model = create_gplvm_model(
        config=config,
        Vx=Vx,
        mu_x=mu_x,
        M=y.shape[1]  # Observation dimension
    )
    
    print(f"Model initialized:")
    print(f"  Type: GP-LVM")
    print(f"  Inducing points: {model.gp.Vx.shape}")
    print(f"  Latent variables: {model.mu_x.shape}")
    print(f"  Structured samples: P={config.P}")
    
    # Initialize logger with GP-LVM naming (reuse SSVI logger but with GP-LVM results folder)
    experiment_name = f"oil_flow_gplvm_{config_name}"
    logger = SSVIResultsLogger(config, experiment_name=experiment_name)
    if logger.results_dir:
        print(f"Results will be saved to: {logger.results_dir}")
        # Log initial info (match SSVI format)
        logger.log_training("=== Oil Flow Original GP-LVM Training Session ===")
        logger.log_training(f"Model: Structured GP-LVM with {config.n_inducing_points} inducing points")
        logger.log_training(f"Latent dimensions: {config.latent_dim}")
        logger.log_training(f"Training: {config.training_iterations} iter, lr={config.learning_rate}")
    
    # Train model
    print(f"\nTraining model...")
    losses, metrics_computer = train_gplvm_with_logging(model, y, config, logger, split_data, labels)
    
    # Visualize results (use same SSVI visualization for identical output structure)
    print(f"\nVisualizing results...")
    visualize_ssvi_results(model, y, labels, losses, config, logger, is_test_data=False)  # Training data visualization
    
    # Generate final metrics and plots if metrics were computed
    if metrics_computer is not None and config.enable_metrics and split_data is not None:
        print(f"\nGenerating final metrics and plots...")
        
        # Optimize test latent variables on full test set for final evaluation (GP-LVM specific feature)
        if hasattr(model, 'mu_x_test_predict') and model.mu_x_test_predict is not None:
            print("Performing final test optimization on full test set...")
            final_test_losses = predict_test_data_with_separate_optimizer(
                model, split_data['y_test'], config, n_test_iterations=200, test_lr=config.learning_rate
            )
            
            # Log test optimization results (GP-LVM specific feature)
            if logger:
                logger.log_training(f"Test optimization completed. Initial test loss: {final_test_losses[0]:.4f}, Final test loss: {final_test_losses[-1]:.4f}")
                logger.log_training(f"Test loss improvement: {final_test_losses[0] - final_test_losses[-1]:.4f}")
            
            print(f"Test optimization results: Initial loss: {final_test_losses[0]:.4f} -> Final loss: {final_test_losses[-1]:.4f}")
            print(f"Test loss improvement: {final_test_losses[0] - final_test_losses[-1]:.4f}")
        
        # Compute final comprehensive metrics on full test set
        print("Computing final metrics on full datasets...")
        
        # Prepare model for final test metrics computation
        if hasattr(model, 'prepare_for_test_metrics'):
            print("Preparing model for final test metrics computation...")
            model.prepare_for_test_metrics(split_data['y_test'], config, n_optimization_steps=config.test_optimization_steps)
        
        final_metrics = metrics_computer.compute_full_metrics(
            model=model,  # Use model directly (now handles P samples internally)
            y_train=split_data['y_train'],
            y_test=split_data['y_test'],
            iteration=config.training_iterations,
            elbo_loss=losses[-1] if losses else 0.0,
            n_samples=config.P  # Use proper P samples from model configuration
        )
        
        # Print final metrics summary using the computed final_metrics
        summary = metrics_computer.get_metrics_summary()
        
        # Log the final metrics to the training log
        if logger:
            logger.log_training(f"Final comprehensive metrics computed: {final_metrics.to_dict()}")
        
        print(f"\n=== Final Model Performance ===")
        print(f"Final Train RMSE: {summary['final_train_rmse']:.4f}")
        print(f"Final Test RMSE: {summary['final_test_rmse']:.4f}")
        print(f"Final Train NLPD: {summary['final_train_nlpd']:.3f}")
        print(f"Final Test NLPD: {summary['final_test_nlpd']:.3f}")
        print(f"Overfitting Ratio (Test/Train RMSE): {summary['overfitting_ratio']:.3f}")
        print(f"Effective Latent Dimensions: {summary['final_effective_dims']}/{summary['total_latent_dims']}")
        print("=" * 35)
        
        # Generate metrics plots
        print("Generating metrics progression plots...")
        plot_metrics_history(metrics_computer, logger, show_plot=False)
        
        print("Generating final metrics summary plots...")
        plot_final_metrics_summary(metrics_computer, model, logger, show_plot=False)
        
        # Save metrics to JSON and text
        if logger:
            print("Saving complete metrics to JSON...")
            save_metrics_to_json(metrics_computer, logger)
            print("Saving metrics summary to text...")
            save_metrics_to_text(metrics_computer, logger, model)
    else:
        print(f"\nFinal training loss: {losses[-1]:.4f}")
        if logger:
            logger.log_training(f"Training completed. Final loss: {losses[-1]:.4f}")
    
    # Comprehensive missing data reconstruction experiment if enabled
    if config.enable_missing_data:
        run_missing_data_reconstruction_experiment(
            model, config, y, split_data, logger,
            model_type="GP LVM",
            model_description="Gaussian Process Latent Variable Model"
        )

    
    # Print final model parameters
    print_final_model_parameters(model, "GP LVM")
    
    # Return final_metrics if computed, otherwise None
    final_metrics = None
    if metrics_computer is not None and metrics_computer.metrics_history:
        final_metrics = metrics_computer.metrics_history[-1]
    
    # Save standard paper metrics for all experiments (missing data enabled/disabled)
    if logger and split_data is not None:
        save_paper_style_metrics(
            model, split_data, losses, logger, config, metrics_computer, final_metrics,
            model_type="GP LVM", 
            model_description="Gaussian Process Latent Variable Model"
        )
    
    return model, losses, config, metrics_computer, final_metrics

if __name__ == "__main__":
    # Parse command line arguments
    parser = create_gplvm_argument_parser()
    args = parser.parse_args()
    
    # Ensure config directory exists
    ensure_gplvm_config_dir_exists()
    
    # Show available configs if no config specified
    if args.config is None:
        print("No config file specified. Available GP-LVM configurations:")
        print_available_gplvm_configs()
        print("\nUsing default fast_config...\n")
    
    # Load configuration with overrides
    try:
        config, config_name = load_gplvm_config_with_overrides(
            config_path=args.config,
            config_class=OilFlowGPLVMConfig,
            args=args
        )
        
        print(f"Using GP-LVM configuration: {config_name}")
        if args.config:
            print(f"Config file: {args.config}")
        
        # Run main function with loaded config
        main(config, config_name)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nAvailable GP-LVM configurations:")
        print_available_gplvm_configs()
        sys.exit(1)