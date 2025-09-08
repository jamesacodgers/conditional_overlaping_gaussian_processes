# %% 
"""
Oil Flow Original GP-LVM

Usage examples:
    # Run with default fast config
    python ssvi_scripts_experiments/oil_flow_GPLVM.py
    
    # Run with base config
    python ssvi_scripts_experiments/oil_flow_GPLVM.py --config ssvi_scripts_experiments/original_gp_lvm_configs/base_config.yaml
    
    # Run with base config and overrides
    python ssvi_scripts_experiments/oil_flow_GPLVM.py --config ssvi_scripts_experiments/original_gp_lvm_configs/base_config.yaml --training-iterations 500 --device cuda
    
    # Run with custom config name for results
    python ssvi_scripts_experiments/oil_flow_GPLVM.py --config ssvi_scripts_experiments/original_gp_lvm_configs/base_config.yaml --config-name my_experiment
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

from src.struc_models import Struc_WS_GPLVM
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

# Device will be set in main() from config

class Struc_GPLVM(Struc_WS_GPLVM):
    """Structured GP-LVM wrapper with proper P-sample handling like SSVI"""
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64, trainable_V=True):
        super().__init__(P=P, V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, 
                        sigma2=sigma2, weight_distribution="continuous", beta=beta, 
                        sigma2_s=sigma2_s, trainable_V=trainable_V)

    @property 
    def mu_x(self): 
        return self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return self.Sigma_x_train
    
    @property
    def mu_x_raw(self):
        """Access to raw P-sample latent means (P, N, D) for internal use"""
        return self.mu_x_train
    
    @property
    def Sigma_x_raw(self):
        """Access to raw P-sample latent variances (P, N, D) for internal use"""
        return self.Sigma_x_train
    
    def gplvm_elbo(self, y):
        R_train = torch.ones((y.shape[0], 1))
        return super(Struc_GPLVM, self).elbo(y, R_train)
    
    # Add minimal compatibility methods for SSVI visualization
    def get_u_samples(self):
        """Simple U samples for SSVI visualization compatibility"""
        dist = self.gp.get_q_u_dist()
        samples = dist.rsample((self.P,))
        return samples.reshape(self.P, self.M, self.C, self.gp.Lx)
    
    def get_mu_sigma_x_samples(self, Y_batch, U_samples=None):
        """Simple method for SSVI visualization compatibility"""
        batch_size = Y_batch.shape[0]
        # Return subset for batch (assumes Y_batch corresponds to training data)
        mu_batch = self.mu_x_train[:, :batch_size, :]  # (P, batch_size, D)
        sigma_batch = self.Sigma_x_train[:, :batch_size, :]  # (P, batch_size, D)
        return mu_batch, sigma_batch
    
    
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
    
    def test_prediction_elbo(self, y_test):
        """Clean ELBO computation for test data using our new test variables"""
        if not hasattr(self, 'mu_x_test_predict') or self.mu_x_test_predict is None:
            raise ValueError("Test data not initialized. Call setup_test_data() first.")
        
        # Validate test data size matches
        if y_test.shape[0] != self._test_data_size:
            raise ValueError(f"Test data size mismatch: expected {self._test_data_size}, got {y_test.shape[0]}")
        
        # Compute test ELBO directly using our test variables
        # This follows the same computation as the parent class but isolated to test data only
        
        # 1. Compute kernel expectations for our test variables
        kernel_test_expectation = self._get_lv_kernel_expectation(
            self.mu_x_test_predict, self.Sigma_x_test_predict
        )
        kernel_test_outer_expectation = self._get_lv_kernel_outer_expectation(
            self.mu_x_test_predict, self.Sigma_x_test_predict
        )
        
        # 2. Get GP signal expectations
        s_test_expectation = self.gp.get_s_expectation(kernel_test_expectation)
        s_test_outer_expectation = self.gp.get_s_outer_expectation(kernel_test_outer_expectation)
        
        # 3. Compute test likelihood term (reconstruction term)
        R_test = torch.ones((y_test.shape[0], 1), device=y_test.device, dtype=y_test.dtype)
        assert R_test.shape[1] == self.C_obs
        
        # Get observed part of signal expectation
        s_test_obs_expectation = s_test_expectation[...,:self.C_obs]  # [P, N_test, C_obs]
        s_test_obs_outer_expectation = s_test_outer_expectation[...,:self.C_obs, :self.C_obs]  # [P, N_test, C_obs, C_obs]
        
        # Compute likelihood term for test data
        # Following the pattern from NormalCoordinator._elbo
        A_test = self.coordinator.get_A_test(s_test_obs_outer_expectation)
        b_test = self.coordinator.get_b_test(y_test, s_test_obs_expectation)
        const_test = self.coordinator.get_const_test(y_test, s_test_obs_expectation, s_test_obs_outer_expectation)
        
        test_inner_elbo = self.coordinator.inner_test_solver(A_test, b_test).mean() + const_test
        
        # 4. Compute KL term for our test variables only (no GP KL since GP params are fixed)
        test_kl_x = self.get_kl_x_from(self.mu_x_test_predict, self.Sigma_x_test_predict)
        
        # 5. Return test ELBO (likelihood - KL)
        return test_inner_elbo - test_kl_x



def get_fast_gplvm_config():
    """Fast configuration for testing GP-LVM"""
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
        plot_hyperparams=True
    )

def create_gplvm_model(config: OilFlowGPLVMConfig, Vx, mu_x, M):
    """Create Structured GP-LVM model"""
    model = Struc_GPLVM(
        Vx=Vx, 
        M=M, 
        mu_x=mu_x, 
        sigma2=config.sigma2,
        beta=config.beta,
        sigma2_s=config.sigma2_s,
        P=config.P,
        trainable_V=config.trainable_V
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

    return y, labels, Vx, mu_x, split_data

def train_gplvm_with_logging(model, y, config, logger=None, split_data=None, labels=None):
    """Training function for GP-LVM with logging and metrics"""
    losses = []
    
    print(f"Training Original GP-LVM...")
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
        loss = -model.gplvm_elbo(y=y)
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
                
                # Optimize test latent variables using separate test optimizer
                if test_optimizer is not None:
                    print(f"    Optimizing test latent variables for metrics...")
                    
                    # Test optimization for metrics computation (using full test set)
                    test_losses = predict_test_data_with_separate_optimizer(
                        model, y_test_full, config, n_test_iterations=100, test_lr=config.learning_rate
                    )
                    
                    print(f"    Debug: About to compute metrics with y_train shape: {y_train_full.shape}")
                    print(f"    Debug: About to compute metrics with y_test shape: {y_test_full.shape}")
                    
                    metrics = metrics_computer.compute_full_metrics(
                        model=model,  # Use model directly (now handles P samples internally)
                        y_train=y_train_full,
                        y_test=y_test_full,
                        iteration=i,
                        elbo_loss=loss.item(),
                        n_samples=config.P  # Use proper P samples from model configuration
                    )
                else:
                    # Fallback to original metrics computation without test optimization
                    print(f"    Debug: About to compute metrics with y_train shape: {y_train_full.shape}")
                    print(f"    Debug: About to compute metrics with y_test shape: {y_test_full.shape}")
                    
                    metrics = metrics_computer.compute_full_metrics(
                        model=model,  # Use model directly (now handles P samples internally)
                        y_train=y_train_full,
                        y_test=y_test_full,
                        iteration=i,
                        elbo_loss=loss.item(),
                        n_samples=config.P  # Use proper P samples from model configuration
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
                    show_plot=False
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
    """Main function for Oil Flow Original GP-LVM example"""
    
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
    print(f"\nInitializing Oil Flow Structured GP-LVM...")
    model = create_gplvm_model(
        config=config,
        Vx=Vx,
        mu_x=mu_x,
        M=y.shape[1]  # Observation dimension
    )
    
    print(f"Model initialized:")
    print(f"  Type: Structured GP-LVM")
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
    visualize_ssvi_results(model, y, labels, losses, config, logger)
    
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
    
    # Print final model parameters
    print(f"\nFinal Model Parameters:")
    print(f"  Observation noise (sigma2): {model.sigma2:.4f}")
    if hasattr(model, 'gp'):
        print(f"  Signal variances (sigma2_s): {model.gp.sigma2_s}")
        print(f"  Lengthscales (1/beta): {(1/model.gp.beta).flatten()}")
    
    # Return final_metrics if computed, otherwise None
    final_metrics = None
    if metrics_computer is not None and metrics_computer.metrics_history:
        final_metrics = metrics_computer.metrics_history[-1]
    
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