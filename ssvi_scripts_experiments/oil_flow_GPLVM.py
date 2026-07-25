#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for experiments with oilflow data

2 different inference modes:
    
   models = ['gaussian','nn_gaussian']

Usage examples:
    # Run with default fast config
    python ssvi_scripts_experiments/oil_flow_GPLVM.py
    
    # Run with base config
    python ssvi_scripts_experiments/oil_flow_GPLVM.py --config ssvi_scripts_experiments/original_gp_lvm_configs/base_lr_0001_s7.yaml
    
    # Run with fast config and overrides
    python ssvi_scripts_experiments/oil_flow_GPLVM.py --config ssvi_scripts_experiments/original_gp_lvm_configs/original_gplvm_fast_config.yaml --training-steps 100 --device cuda
"""

import sys
import os
# Get the project root directory (parent of ssvi_scripts_experiments)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add both project root and generalised_gplvm_code to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

from utils.data import load_real_data 
import pickle as pkl
from models.bayesianGPLVM import BayesianGPLVM
from models.latent_variable import PointLatentVariable, MAPLatentVariable, VariationalLatentVariable, NNEncoder
from matplotlib import pyplot as plt
import torch
import numpy as np
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
# Import metrics
from utils.metrics import rmse

# Import configuration system
from src.original_gplvm_config import OriginalGPLVMConfig, get_fast_config
from src.original_gplvm_config_utils import (
    create_argument_parser, load_config_with_overrides, 
    ensure_config_dir_exists, print_available_configs
)
from src.original_gplvm_results_logger import OriginalGPLVMResultsLogger
from src.data_utils import download_oilflow_data_if_needed
from src.nlpd_utils import compute_nlpd_nll_based, compute_crps_nll_based, compute_all_crps_nll_based, compute_coverage_nll_based, compute_missing_data_nlpd_gauss
from src.original_gplvm_metrics_plot import plot_metrics_progression, compute_effective_latent_dims, plot_missing_nlpd_metrics_progression, plot_missing_rmse_metrics_progression, plot_crps_metrics_progression, plot_crps_posterior_predictive_progression, plot_latent_slice_original_gplvm
from src.missing_data_utils import preprocess_data_for_missing_training, rmse_missing_improved
from src.missing_data_experiment import run_missing_data_reconstruction_experiment
from src.unified_rmse_utils import compute_rmse_with_sampling, compute_missing_data_rmse_with_sampling

plt.style.use('ggplot')

# Import shared function from mnist_ood_GPLVM.py
from ssvi_scripts_experiments.mnist_ood_GPLVM import compute_full_elbo_minibatched

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))

class OilFlowModel(BayesianGPLVM):
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

def main(config: OriginalGPLVMConfig = None, config_name: str = "fast"):
    """Main function for Original GP LVM experiments"""
    
    # Use default config if none provided
    if config is None:
        config = get_fast_config()
    
    # Set device and dtype from config
    torch.set_default_device(config.device)
    torch.set_default_dtype(torch.float64)
    
    # Print configuration
    config.print_config()
    
    # Initialize results logger
    logger = OriginalGPLVMResultsLogger(config, experiment_name=config_name)
    if logger.get_results_dir():
        print(f"Results will be saved to: {logger.get_results_dir()}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
    
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
    Y_train = torch.tensor(Y_train_np, dtype=torch.float64, device=config.device)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float64, device=config.device)
    lb_train = lb_train_np  # Keep as numpy array for compatibility
    lb_test = lb_test_np    # Keep as numpy array for compatibility
    
    print(f"Data split shapes - Train: {Y_train.shape}, Test: {Y_test.shape}")
    print(f"Labels split shapes - Train: {lb_train.shape}, Test: {lb_test.shape}")
    
    # Apply missing data preprocessing if enabled (using same function as SSVI for alignment)
    if config.enable_missing_data:
        print(f"\n=== Missing Data Preprocessing (Aligned with SSVI) ===")
        
        # Store original data for evaluation
        Y_train_original = Y_train.clone()
        Y_test_original = Y_test.clone()
        
        # Use the same preprocessing function as SSVI to ensure identical NaN patterns
        Y_train_missing, train_missing_mask, missing_info = preprocess_data_for_missing_training(Y_train, config)
        Y_test_missing, test_missing_mask, _ = preprocess_data_for_missing_training(Y_test, config)
        
        # Update the data to use missing versions
        Y_train = Y_train_missing
        Y_test = Y_test_missing

        print(f"Missing data preprocessing completed.")
        print("=" * 50)
    else:
        # No missing data - store None for consistency
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
    steps_per_model = config.training_steps
    
    model_dict = {}
    noise_trace_dict = {}
    results_summary = {}
    
    for model_name in models:
        
        print(f"\n=== Training {model_name} model ===")
        logger.log_training(f"Starting training for model: {model_name}")
        
        # Always apply PCA initialization (matching SSVI approach)
        # Use original data for PCA to avoid NaN propagation from missing data preprocessing
        if config.enable_missing_data and Y_train_original is not None:
            # Use original complete data for PCA initialization
            mu_x_pca, _, _ = torch.pca_lowrank(Y_train_original, latent_dim)
        else:
            # Use training data if no missing data or if Y_train_original is None
            mu_x_pca, _, _ = torch.pca_lowrank(Y_train, latent_dim)
        X_init = torch.nn.Parameter(mu_x_pca)
        
        # Define prior for X using PCA-derived means (matching SSVI approach)
        X_prior_mean = mu_x_pca  # Use PCA-derived means instead of zeros
        X_prior_mean_test = X_prior_mean[0:len(Y_test),:]
        
        # Each inference model differs in its latent variable configuration / 
        # LatentVariable (X)
        
        # defaults - if a model needs them they are internally assigned
        nn_layers = None
        prior_x = None
        prior_x_test = None
        
            
        if model_name == 'gauss':
            
            ae = False
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean, dtype=torch.float64))
            prior_x_test = NormalPrior(X_prior_mean_test, torch.ones_like(X_prior_mean_test, dtype=torch.float64))
            X = VariationalLatentVariable(X_init, prior_x, data_dim)  # CORRECT: constructor expects data_dim
        
        elif model_name == 'nn_gauss':
            
            ae = True
            nn_layers = config.nn_layers
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1], dtype=torch.float64, device=config.device))
            prior_x_test = MultivariateNormalPrior(X_prior_mean_test, torch.eye(X_prior_mean.shape[1], dtype=torch.float64, device=config.device))
            X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
            
        # Initialise model, likelihood, elbo and optimizer
        
        model = OilFlowModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers, device=config.device, config=config)
        
        # Use missing data aware likelihood if enabled (original approach)
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

        # Initialize likelihood noise from config if provided
        if config is not None:
            with torch.no_grad():
                # Use initialize method for likelihood noise too
                noise_scale = torch.tensor(config.init_output_noise_scale, dtype=torch.float64, device=config.device)
                likelihood.initialize(noise=noise_scale)
        elbo = VariationalELBO(likelihood, model, num_data=len(Y_train))

        # Move model and likelihood to CUDA if available (like original experiments)
        if config.device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()
            # Also move the latent variable (X) to CUDA using proper to() method
            if hasattr(model, 'X'):
                model.X = model.X.to(config.device)

        # Fix device placement for NNEncoder tensors (created on CPU in original code)
        if model_name == 'nn_gauss' and hasattr(model.X, 'jitter'):
            # Use the proper to() method which handles jitter tensor
            model.X = model.X.to(config.device)
            # Also ensure prior_x tensors are on correct device
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
    
        # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
        # using the optimizer provided.
        
        loss_list = []  # Batch losses (negative ELBO)
        full_loss_list = []  # Full dataset losses (negative ELBO)
        full_loss_iterations = []  # Iterations at which full loss was computed
        noise_trace = []
        
        # Metrics tracking for progression plot (train only)
        train_rmse_list = []
        train_nlpd_list = []
        train_crps_list = []
        train_crps_pp_sample_list = []
        train_crps_pp_mixture_list = []
        train_coverage_90_list = []
        train_coverage_95_list = []
        train_interval_width_90_list = []
        train_interval_width_95_list = []
        train_missing_nlpd_list = []  # For missing data NLPD tracking
        train_missing_rmse_list = []  # For missing data RMSE tracking
        effective_dims_list = []
        metrics_iterations = []
        metrics_losses = []  # Will store full dataset losses
        
        iterator = trange(steps_per_model[model_name], leave=True)
        batch_size = config.batch_size
        for i in iterator: 
            batch_index = model._get_batch_idx(batch_size)
            # Convert batch_index to torch tensor on correct device for CUDA compatibility
            batch_index_tensor = torch.tensor(batch_index, dtype=torch.int64, device=config.device)
            optimizer.zero_grad()
            if model_name in ['point','map', 'gauss']:
                sample = model.sample_latent_variable()  # a full sample returns latent x across all N
            else:
                # For nn_gauss with missing data, replace NaN values for encoder input
                if config.enable_missing_data:
                    # Replace NaN values with zeros for encoder input (encoder can't handle NaN)
                    Y_train_for_encoder = Y_train.clone()
                    Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -999.0
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
            
            # Compute training metrics periodically for progression plot (including iter 0 and final iteration)
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
                        
                        # Get training latent variables
                        if config.enable_missing_data and model_name == 'nn_gauss':
                            # For nn_gauss with missing data, use cleaned input for encoder
                            Y_train_for_encoder = Y_train.clone()
                            Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -999.0
                            X_train_mean = model.get_X_mean(Y_train_for_encoder)
                            X_train_covar = model.X.sigma(Y_train_for_encoder).detach()
                            Y_train_recon, _ = model.reconstruct_y(X_train_mean, Y_train_for_encoder, ae=ae,
                                                                   model_name=model_name)
                        else:
                            X_train_mean = model.get_X_mean(Y_train)
                            Y_train_recon, _ = model.reconstruct_y(X_train_mean, Y_train, ae=ae, model_name=model_name)

                            if model_name == "nn_gauss":
                                X_train_covar = model.X.sigma(Y_train).detach()
                            else:
                                X_train_covar = torch.nn.functional.softplus(model.X.q_log_sigma).detach()

                        # Compute training RMSE using unified sampling approach
                        train_rmse = compute_rmse_with_sampling(
                            model=model,
                            y_true=Y_train,
                            X_mean=X_train_mean,
                            X_covar=X_train_covar,
                            model_name=model_name,
                            n_samples=32,
                            enable_missing_data=config.enable_missing_data
                        )
                        # Compute training NLPD - both gauss and nn_gauss models now use the same sampling-based approach
                        try:
                            obs_noise = likelihood.noise
                            train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean, model_name, loss_list, X_train_covar, obs_noise, enable_missing_data=config.enable_missing_data)
                        except Exception as e:
                            print(f"  Warning: Could not get covariance for NLPD: {e}")
                            train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean, model_name, loss_list, enable_missing_data=config.enable_missing_data)

                        # Compute training CRPS family (same predictive path as NLPD)
                        try:
                            obs_noise = likelihood.noise
                            train_crps_all = compute_all_crps_nll_based(
                                model, Y_train, X_train_mean, model_name, X_train_covar, obs_noise,
                                enable_missing_data=config.enable_missing_data
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
                                enable_missing_data=config.enable_missing_data
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
                        
                        # Compute missing data NLPD and RMSE for gauss model during training (for checkpoints)
                        train_missing_nlpd = None
                        train_missing_rmse = None
                        if config.enable_missing_data and model_name == 'gauss':
                            try:
                                obs_noise = likelihood.noise
                                train_missing_nlpd = compute_missing_data_nlpd_gauss(
                                    model=model,
                                    Y_original=Y_train_original,
                                    missing_mask=train_missing_mask,
                                    X_mean=X_train_mean,
                                    X_std=X_train_covar,
                                    obs_noise=obs_noise,
                                    n_samples=config.reconstruction_samples
                                )
                                train_missing_rmse = compute_missing_data_rmse_with_sampling(
                                    model=model,
                                    Y_original=Y_train_original,
                                    missing_mask=train_missing_mask,
                                    X_mean=X_train_mean,
                                    X_covar=X_train_covar,
                                    model_name=model_name,
                                    n_samples=config.reconstruction_samples
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
                        
                        print(f"  Train metrics at iter {i}: RMSE={train_rmse:.4f}, NLPD={train_nlpd:.3f}, CRPS={train_crps:.4f}, pp_sample={train_crps_pp_sample:.4f}, pp_mix={train_crps_pp_mixture:.4f}, "
                              f"cov90={train_coverage_90:.3f}, cov95={train_coverage_95:.3f}, "
                              f"width90={train_interval_width_90:.4f}, width95={train_interval_width_95:.4f}, "
                              f"Effective dims: {effective_dims}/{total_dims}")
                        
                except Exception as e:
                    print(f"  Warning: Could not compute metrics at iteration {i}: {e}")
                finally:
                    model.train()

            # Save iteration-based models and plots (like SSVI does)
            if logger and logger.get_results_dir() and i > 0 and i % config.checkpoint_frequency == 0:
                from src.ssvi_alignment_utils import save_iteration_model, create_iteration_plot
                print(f"  Saving iteration {i} model and plot...")
                save_iteration_model(model, model_name, i, config.random_seed, logger.get_results_dir())
                # Use full_loss_list for checkpoint plots (consistent with SSVI)
                create_iteration_plot(model, full_loss_list, lb_train, config, logger.get_results_dir(), model_name, config.random_seed, Y_train, i, "original", train_indices=torch.from_numpy(train_indices).to(dtype=torch.int64, device=Y_train.device), full_loss_iterations=full_loss_iterations, batch_losses=loss_list)
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
        
        # Save metrics progression plot (train metrics only)
        if logger.get_results_dir() and train_rmse_list and train_nlpd_list:
            plot_metrics_progression(train_rmse_list, train_nlpd_list, metrics_iterations, 
                                   metrics_losses, effective_dims_list, logger.get_results_dir(), model_name, config.random_seed, batch_losses=loss_list)

            # Separate CRPS + coverage progression plot (comparable across models)
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
            
            # Save checkpoint metrics to JSON (train metrics only)
            logger.save_checkpoint_metrics(train_rmse_list, train_nlpd_list, metrics_iterations, 
                                         metrics_losses, model_name, config.random_seed, effective_dims_list, train_missing_nlpd_list, train_missing_rmse_list, train_crps_list=train_crps_list,
                                         train_coverage_90_list=train_coverage_90_list, train_coverage_95_list=train_coverage_95_list,
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
            
            # Generate missing NLPD metrics progression plot if missing data is enabled and we have missing NLPD data
            if config.enable_missing_data and train_missing_nlpd_list:
                plot_missing_nlpd_metrics_progression(train_missing_nlpd_list, metrics_iterations, 
                                                    logger.get_results_dir(), model_name, config.random_seed)
            
            # Generate missing RMSE metrics progression plot if missing data is enabled and we have missing RMSE data
            if config.enable_missing_data and train_missing_rmse_list:
                plot_missing_rmse_metrics_progression(train_missing_rmse_list, metrics_iterations, 
                                                    logger.get_results_dir(), model_name, config.random_seed)
            
        # Save models & training info
        
        print(model.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
        model_dict[model_name + '_' + str(config.random_seed)] = model
        noise_trace_dict[model_name + '_' + str(config.random_seed)] = noise_trace
        
        ### Saving training report
        
        # Get training latent variables (handle missing data for nn_gauss)
        if config.enable_missing_data and model_name == 'nn_gauss':
            # For nn_gauss with missing data, use cleaned input for encoder
            Y_train_for_encoder = Y_train.clone()
            Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -999.0
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
            # Import function only when needed to avoid directory creation issues
            from src.ssvi_alignment_utils import create_standardized_scattered_plot
            # lb_train is already a numpy array from train_test_split, no need to convert
            # Use Y_train on the same device as model to avoid device mismatch
            # Use full_loss_list for final plots (consistent with SSVI)
            create_standardized_scattered_plot(model, full_loss_list, lb_train, config, logger.get_results_dir(), model.X.__class__.__name__, config.random_seed, Y_train, model_type="original", train_indices=torch.from_numpy(train_indices).to(dtype=torch.int64, device=Y_train.device), full_loss_iterations=full_loss_iterations, batch_losses=loss_list)
        else:
            pass
        
        #### Saving model with seed 
        print(f'Saving {model_name} {config.random_seed}')
        
        # Save model using logger
        logger.save_model(model, model_name, config.random_seed)

        ####################### Testing Framework ################################################
        if config.enable_testing:
        #Compute latent test & reconstructions
            with torch.no_grad():
                model.eval()
                likelihood.eval()
                # Ensure encoder is in eval mode for nn_gauss
                if hasattr(model.X, 'eval'):
                    model.X.eval()
            
            if ae is True:
                if model_name != 'iaf':
                    # For nn_gauss, fix jitter size mismatch by temporarily adjusting it
                    if model_name == 'nn_gauss':
                        # Store original jitter
                        original_jitter = model.X.jitter
                        # Create jitter for test data size
                        test_jitter = torch.eye(latent_dim, dtype=torch.float64).unsqueeze(0) * 1e-5
                        model.X.jitter = torch.cat([test_jitter for i in range(len(Y_test))], axis=0)
                    
                    # For nn_gauss with missing data, use cleaned input for predict_latent
                    if config.enable_missing_data:
                        Y_train_for_encoder = Y_train.clone()
                        Y_train_for_encoder[torch.isnan(Y_train_for_encoder)] = -999.0
                        X_test_mean, X_test_covar = model.predict_latent(Y_train_for_encoder, 
                                                                          Y_test, 
                                                                          optimizer.defaults['lr'], 
                                                                          likelihood, 
                                                                          config.random_seed,
                                                                          prior_x=prior_x_test, 
                                                                          ae=True, 
                                                                          model_name='nn_gauss', 
                                                                          pca=pca)
                    else:
                        X_test_mean, X_test_covar = model.predict_latent(Y_train, 
                                                                          Y_test, 
                                                                          optimizer.defaults['lr'], 
                                                                          likelihood, 
                                                                          config.random_seed,
                                                                          prior_x=prior_x_test, 
                                                                          ae=True, 
                                                                          model_name='nn_gauss', 
                                                                          pca=pca)
                    
                    # Restore original jitter for nn_gauss
                    if model_name == 'nn_gauss':
                        model.X.jitter = original_jitter
                else:
                    X_test_mean, X_flow_samples = model.predict_latent(Y_train, 
                                                                        Y_test, 
                                                                        optimizer.defaults['lr'], 
                                                                        likelihood, 
                                                                        prior_x = prior_x,
                                                                        ae=True, 
                                                                        model_name='iaf',
                                                                        pca=pca)
            
            else: # either point, map or gauss
                losses_test,  X_test = model.predict_latent(Y_train, Y_test, optimizer.defaults['lr'], 
                                          likelihood, config.random_seed, prior_x=prior_x_test, ae=ae, 
                                          model_name=model_name,pca=pca, steps=config.test_iterations)
                    
            
            # Compute training and test reconstructions
            if model_name in ('point', 'map'):
                    X_test_mean = X_test.X
            elif model_name == 'gauss':
                   X_test_mean = X_test.q_mu.detach()
                   X_test_covar = torch.nn.functional.softplus(X_test.q_log_sigma).detach()
            # Note: For nn_gauss, X_test_mean is already computed above in the ae=True block
       
            # Ensure tensors are on correct device and dtype (X_test_mean and X_train_mean are already tensors)
            X_test_mean_device = X_test_mean.to(dtype=torch.float64, device=config.device)
            X_train_mean_device = X_train_mean.to(dtype=torch.float64, device=config.device)


            # Compute reconstructions with proper missing data handling
            if config.enable_missing_data and model_name == 'nn_gauss':
                # For nn_gauss with missing data, use cleaned input for reconstruction
                Y_test_for_recon = Y_test.clone()
                Y_test_for_recon[torch.isnan(Y_test_for_recon)] = -999.0
                Y_train_for_recon = Y_train.clone()
                Y_train_for_recon[torch.isnan(Y_train_for_recon)] = -999.0
                
                Y_test_recon, Y_test_pred_covar = model.reconstruct_y(X_test_mean_device, Y_test_for_recon, ae=ae, model_name=model_name)
                Y_train_recon, Y_train_pred_covar = model.reconstruct_y(X_train_mean_device, Y_train_for_recon, ae=ae, model_name=model_name)
            else:
                Y_test_recon, Y_test_pred_covar = model.reconstruct_y(X_test_mean_device, Y_test, ae=ae, model_name=model_name)
                Y_train_recon, Y_train_pred_covar = model.reconstruct_y(X_train_mean_device, Y_train, ae=ae, model_name=model_name)

            # ################################
            # Compute the metrics:
            
            # 1) Reconstruction error - Train & Test using unified sampling approach
            try:
                mse_train = compute_rmse_with_sampling(
                    model=model,
                    y_true=Y_train,
                    X_mean=X_train_mean_device,
                    X_covar=X_train_covar,
                    model_name=model_name,
                    n_samples=32,
                    enable_missing_data=config.enable_missing_data
                )
                
                mse_test = compute_rmse_with_sampling(
                    model=model,
                    y_true=Y_test,
                    X_mean=X_test_mean_device,
                    X_covar=X_test_covar,
                    model_name=model_name,
                    n_samples=32,
                    enable_missing_data=config.enable_missing_data
                )
                
                print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
                print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
            except Exception as e:
                print(f"Warning: Failed to compute RMSE for {model_name} due to error: {e}")
                print("This is likely due to NaN values in latent variables. Setting RMSE to NaN...")
                mse_train = float('nan')
                mse_test = float('nan')
                print(f'Train Reconstruction error {model_name} = NaN (error)')
                print(f'Test Reconstruction error {model_name} = NaN (error)')
            
            # 2) Negative Log Predictive Density (NLPD) - using NLL-based approach inspired by movie_lens_large_expt.py

            # Train NLPD using NLL-based approach (inspired by movie_lens_large_expt.py)
            obs_noise = likelihood.noise
            try:
                # Both gauss and nn_gauss models now use the same sampling-based NLPD approach
                train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean_device, model_name, loss_list, X_train_covar, obs_noise, enable_missing_data=config.enable_missing_data)
                test_nlpd = compute_nlpd_nll_based(model, Y_test, X_test_mean_device, model_name, losses_test, X_test_covar, obs_noise, enable_missing_data=config.enable_missing_data)

                print(f'Train NLPD {model_name} = ' + str(train_nlpd))
                print(f'Test NLPD {model_name} = ' + str(test_nlpd))
            except Exception as e:
                print(f"Warning: Failed to compute NLPD for {model_name} due to error: {e}")
                print("This is likely due to NaN values in latent variables. Setting NLPD to NaN...")
                train_nlpd = float('nan')
                test_nlpd = float('nan')
                print(f'Train NLPD {model_name} = NaN (error)')
                print(f'Test NLPD {model_name} = NaN (error)')

            # 2b) Continuous Ranked Probability Score family (same predictive path as NLPD)
            try:
                train_crps_all = compute_all_crps_nll_based(
                    model, Y_train, X_train_mean_device, model_name, X_train_covar, obs_noise,
                    enable_missing_data=config.enable_missing_data
                )
                test_crps_all = compute_all_crps_nll_based(
                    model, Y_test, X_test_mean_device, model_name, X_test_covar, obs_noise,
                    enable_missing_data=config.enable_missing_data
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
                print(f'Train CRPS {model_name} = NaN (error)')
                print(f'Test CRPS (crps) {model_name} = NaN (error)')

            # 2c) Predictive interval coverage (same predictive path as NLPD)
            try:
                train_cov = compute_coverage_nll_based(
                    model, Y_train, X_train_mean_device, model_name, X_train_covar, obs_noise,
                    enable_missing_data=config.enable_missing_data
                )
                test_cov = compute_coverage_nll_based(
                    model, Y_test, X_test_mean_device, model_name, X_test_covar, obs_noise,
                    enable_missing_data=config.enable_missing_data
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
                
            # Initialize results_summary[model_name] if it doesn't exist
            if model_name not in results_summary:
                results_summary[model_name] = {}

            # 3) Missing Data NLPD for gauss model (similar to SSVI approach)
            if config.enable_missing_data and model_name == 'gauss':
                try:
                    # Use already computed X_covar for gauss model
                    X_train_std = X_train_covar
                    X_test_std = X_test_covar

                    # Compute missing data NLPD for training data
                    train_missing_nlpd = compute_missing_data_nlpd_gauss(
                        model=model,
                        Y_original=Y_train_original,
                        missing_mask=train_missing_mask,
                        X_mean=X_train_mean_device,
                        X_std=X_train_std,
                        obs_noise=obs_noise,
                        n_samples=config.reconstruction_samples
                    )

                    # Compute missing data NLPD for test data
                    test_missing_nlpd = compute_missing_data_nlpd_gauss(
                        model=model,
                        Y_original=Y_test_original,
                        missing_mask=test_missing_mask,
                        X_mean=X_test_mean_device,
                        X_std=X_test_std,
                        obs_noise=obs_noise,
                        n_samples=config.reconstruction_samples
                    )

                    print(f'Train Missing Data NLPD {model_name} = ' + str(train_missing_nlpd))
                    print(f'Test Missing Data NLPD {model_name} = ' + str(test_missing_nlpd))

                    # Store missing data NLPD results for summary
                    results_summary[model_name]['train_missing_nlpd'] = float(train_missing_nlpd)
                    results_summary[model_name]['test_missing_nlpd'] = float(test_missing_nlpd)

                    # Compute missing data RMSE for training data
                    train_missing_rmse = compute_missing_data_rmse_with_sampling(
                        model=model,
                        Y_original=Y_train_original,
                        missing_mask=train_missing_mask,
                        X_mean=X_train_mean_device,
                        X_covar=X_train_covar,
                        model_name=model_name,
                        n_samples=config.reconstruction_samples
                    )

                    # Compute missing data RMSE for test data
                    test_missing_rmse = compute_missing_data_rmse_with_sampling(
                        model=model,
                        Y_original=Y_test_original,
                        missing_mask=test_missing_mask,
                        X_mean=X_test_mean_device,
                        X_covar=X_test_covar,
                        model_name=model_name,
                        n_samples=config.reconstruction_samples
                    )

                    print(f'Train Missing Data RMSE {model_name} = ' + str(train_missing_rmse))
                    print(f'Test Missing Data RMSE {model_name} = ' + str(test_missing_rmse))

                    # Store missing data RMSE results for summary
                    results_summary[model_name]['train_missing_rmse'] = float(train_missing_rmse)
                    results_summary[model_name]['test_missing_rmse'] = float(test_missing_rmse)

                except Exception as e:
                    print(f"Warning: Failed to compute missing data metrics for {model_name} due to error: {e}")
                    print("Setting missing data NLPD and RMSE to NaN...")
                    train_missing_nlpd = float('nan')
                    test_missing_nlpd = float('nan')
                    train_missing_rmse = float('nan')
                    test_missing_rmse = float('nan')
                    print(f'Train Missing Data NLPD {model_name} = NaN (error)')
                    print(f'Test Missing Data NLPD {model_name} = NaN (error)')
                    print(f'Train Missing Data RMSE {model_name} = NaN (error)')
                    print(f'Test Missing Data RMSE {model_name} = NaN (error)')

                    # Store NaN values for missing data metrics results
                    results_summary[model_name]['train_missing_nlpd'] = float(train_missing_nlpd)
                    results_summary[model_name]['test_missing_nlpd'] = float(test_missing_nlpd)
                    results_summary[model_name]['train_missing_rmse'] = float(train_missing_rmse)
                    results_summary[model_name]['test_missing_rmse'] = float(test_missing_rmse)

            # Save training results (convert tensors to floats for JSON serialization)
            # Ensure all values in loss_list and noise_trace are Python floats
            loss_list_floats = [float(x) for x in loss_list]
            noise_trace_floats = [float(x) for x in noise_trace]
            logger.save_training_results(model_name, config.random_seed, loss_list_floats, noise_trace_floats,
                                        float(mse_train.item()) if hasattr(mse_train, 'item') else float(mse_train), 
                                        float(mse_test.item()) if hasattr(mse_test, 'item') else float(mse_test))
            
            # Store results for summary (convert tensors to floats for JSON serialization)
            # Preserve existing missing data NLPD and RMSE values if they were computed
            train_missing_nlpd_value = results_summary[model_name].get('train_missing_nlpd', float('nan'))
            test_missing_nlpd_value = results_summary[model_name].get('test_missing_nlpd', float('nan'))
            train_missing_rmse_value = results_summary[model_name].get('train_missing_rmse', float('nan'))
            test_missing_rmse_value = results_summary[model_name].get('test_missing_rmse', float('nan'))
            
            results_summary[model_name] = {
                'train_rmse': float(mse_train.item()) if hasattr(mse_train, 'item') else float(mse_train),
                'test_rmse': float(mse_test.item()) if hasattr(mse_test, 'item') else float(mse_test),
                'train_nlpd': float(train_nlpd),  # train_nlpd is already a float from compute_nlpd_nll_based
                'test_nlpd': float(test_nlpd),    # test_nlpd is already a float from compute_nlpd_nll_based
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
                'train_missing_nlpd': train_missing_nlpd_value,  # Preserve computed value or use NaN
                'test_missing_nlpd': test_missing_nlpd_value,   # Preserve computed value or use NaN
                'train_missing_rmse': train_missing_rmse_value,  # Preserve computed value or use NaN
                'test_missing_rmse': test_missing_rmse_value,   # Preserve computed value or use NaN
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

        # Generate latent slice visualization (after training and testing)
        try:
            print(f"\nGenerating latent slice visualizations (adaptive grid from data)...")
            
            # Plot slice centered at origin
            if logger.get_results_dir():
                plots_dir = os.path.join(logger.get_results_dir(), 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                
                latent_slice_path_origin = os.path.join(plots_dir, f"latent_slice_origin_{model_name}_seed_{config.random_seed}.png")
                plot_latent_slice_original_gplvm(
                    model=model,
                    y_train=Y_train,
                    model_name=model_name,
                    center="zeros",
                    save_path=latent_slice_path_origin
                )
                print(f"  Saved latent slice (origin) to: {latent_slice_path_origin}")
                
                # Plot slices centered at a few interesting data points
                N_train = Y_train.shape[0]  # Number of training points from actual data
                # Select 3 representative points (beginning, middle, end)
                representative_indices = [0, N_train // 2, N_train - 1]
                
                for idx in representative_indices:
                    latent_slice_path = os.path.join(plots_dir, f"latent_slice_point_{idx}_{model_name}_seed_{config.random_seed}.png")
                    plot_latent_slice_original_gplvm(
                        model=model,
                        y_train=Y_train,
                        model_name=model_name,
                        center="data_point",
                        n_idx=idx,
                        save_path=latent_slice_path
                    )
                    print(f"  Saved latent slice (point {idx}) to: {latent_slice_path}")
        except Exception as e:
            print(f"  Warning: Failed to generate latent slice visualizations: {e}")
    
    # Save final experiment summary
    logger.save_final_summary(results_summary)
    logger.log_training("=== Experiment completed ===")
    
    return model_dict, noise_trace_dict, results_summary

if __name__ == '__main__':
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Ensure config directory exists
    ensure_config_dir_exists()
    
    # Default to fast config when running in PyCharm or without arguments
    if args.config is None and len(sys.argv) == 1:
        print("Running in PyCharm or without arguments - using fast config by default")
        print("Available configurations:")
        print_available_configs()
        print("\nUsing fast_config (200 iterations, batch_size=50)...\n")
    elif args.config is None:
        print("No config file specified. Available configurations:")
        print_available_configs()
        print("\nUsing default fast_config...\n")
    
    # Load configuration with overrides
    try:
        config, config_name = load_config_with_overrides(
            config_path=args.config,
            config_class=OriginalGPLVMConfig,
            args=args
        )
        
        print(f"Using configuration: {config_name}")
        if args.config:
            print(f"Config file: {args.config}")
        
        # Run main function with loaded config
        main(config, config_name)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nAvailable configurations:")
        print_available_configs()
        sys.exit(1)
    
