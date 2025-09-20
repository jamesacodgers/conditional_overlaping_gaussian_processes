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
    python ssvi_scripts_experiments/oil_flow_GPLVM.py --config ssvi_scripts_experiments/original_gplvm_configs/base_lr_0001_s7.yaml
    
    # Run with fast config and overrides
    python ssvi_scripts_experiments/oil_flow_GPLVM.py --config ssvi_scripts_experiments/original_gplvm_configs/original_gplvm_fast_config.yaml --training-steps 100 --device cuda
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
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from sklearn.model_selection import train_test_split

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
from src.nlpd_utils import compute_nlpd_nll_based
from src.original_gplvm_metrics_plot import plot_metrics_progression, compute_effective_latent_dims

plt.style.use('ggplot')

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
        self.inducing_inputs = torch.randn(n_inducing, latent_dim, device=device)
    
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
                lengthscale_device = torch.sqrt(torch.tensor(config.init_lengthscales, device=device))
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
    
    # Set device from config
    torch.set_default_device(config.device)
    
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
    Y_train = torch.tensor(Y_train_np, dtype=torch.float32, device=config.device)
    Y_test = torch.tensor(Y_test_np, dtype=torch.float32, device=config.device)
    lb_train = lb_train_np  # Keep as numpy array for compatibility
    lb_test = lb_test_np    # Keep as numpy array for compatibility
    
    print(f"Data split shapes - Train: {Y_train.shape}, Test: {Y_test.shape}")
    print(f"Labels split shapes - Train: {lb_train.shape}, Test: {lb_test.shape}")
    
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
        
        # Define prior for X
        X_prior_mean = torch.zeros(N, latent_dim, device=config.device)  # shape: N x Q
        X_prior_mean_test = X_prior_mean[0:len(Y_test),:]
    
        # Initialise X with PCA or 0s.
        if pca == True:
              X_init = _init_pca(Y_train, latent_dim) # Initialise X to PCA 
        else:
              X_init = torch.nn.Parameter(torch.zeros(N, latent_dim, device=config.device))
        
        # Each inference model differs in its latent variable configuration / 
        # LatentVariable (X)
        
        # defaults - if a model needs them they are internally assigned
        nn_layers = None
        prior_x = None
        prior_x_test = None
        
            
        if model_name == 'gauss':
            
            ae = False
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
            prior_x_test = NormalPrior(X_prior_mean_test, torch.ones_like(X_prior_mean_test))
            X = VariationalLatentVariable(X_init, prior_x, data_dim)  # CORRECT: constructor expects data_dim
        
        elif model_name == 'nn_gauss':
            
            ae = True
            nn_layers = config.nn_layers
            prior_x = MultivariateNormalPrior(X_prior_mean, torch.eye(X_prior_mean.shape[1], device=config.device))
            prior_x_test = MultivariateNormalPrior(X_prior_mean_test, torch.eye(X_prior_mean.shape[1], device=config.device))
            X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)
            
        # Initialise model, likelihood, elbo and optimizer
        
        model = OilFlowModel(N, data_dim, latent_dim, n_inducing, X, nn_layers=nn_layers, device=config.device, config=config)
        likelihood = GaussianLikelihood()
        
        # Initialize likelihood noise from config if provided
        if config is not None:
            with torch.no_grad():
                # Use initialize method for likelihood noise too
                noise_scale = torch.tensor(config.init_output_noise_scale, device=config.device)
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
        
        loss_list = []
        noise_trace = []
        
        # Metrics tracking for progression plot (train only)
        train_rmse_list = []
        train_nlpd_list = []
        effective_dims_list = []
        metrics_iterations = []
        metrics_losses = []
        
        iterator = trange(steps_per_model[model_name], leave=True)
        batch_size = config.batch_size
        for i in iterator: 
            batch_index = model._get_batch_idx(batch_size)
            # Convert batch_index to torch tensor on correct device for CUDA compatibility
            batch_index_tensor = torch.tensor(batch_index, device=config.device)
            optimizer.zero_grad()
            if model_name in ['point','map', 'gauss']:
                sample = model.sample_latent_variable()  # a full sample returns latent x across all N
            else:
                sample = model.sample_latent_variable(Y_train)
            sample_batch = sample[batch_index_tensor]
            output_batch = model(sample_batch)
            loss = -elbo(output_batch, Y_train[batch_index_tensor].T).sum()
            loss_list.append(float(loss.item()))
            noise_trace.append(float(np.round(likelihood.noise_covar.noise.item(),3)))
            iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
            
            # Compute training metrics periodically for progression plot (including iter 0 and final iteration)
            if i % config.metrics_frequency == 0 or i == steps_per_model[model_name] - 1:
                try:
                    with torch.no_grad():
                        model.eval()
                        # Get training latent variables
                        X_train_mean = model.get_X_mean(Y_train)
                        # Compute training reconstruction
                        Y_train_recon, _ = model.reconstruct_y(X_train_mean, Y_train, ae=ae, model_name=model_name)
                        # Compute training RMSE
                        train_rmse = rmse(Y_train, Y_train_recon.T)
                        # Compute training NLPD
                        train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean, model_name, loss_list)
                        
                        # Compute effective dimensions
                        effective_dims, total_dims = compute_effective_latent_dims(model)
                        
                        # Store metrics
                        train_rmse_list.append(float(train_rmse.item()) if hasattr(train_rmse, 'item') else float(train_rmse))
                        train_nlpd_list.append(float(train_nlpd))
                        effective_dims_list.append(effective_dims)
                        metrics_iterations.append(i)
                        metrics_losses.append(float(loss.item()))
                        
                        print(f"  Train metrics at iter {i}: RMSE={train_rmse:.4f}, NLPD={train_nlpd:.3f}, Effective dims: {effective_dims}/{total_dims}")
                        
                except Exception as e:
                    print(f"  Warning: Could not compute metrics at iteration {i}: {e}")
                finally:
                    model.train()

            # Save iteration-based models and plots (like SSVI does)
            if logger and logger.get_results_dir() and i > 0 and i % config.checkpoint_frequency == 0:
                from src.ssvi_alignment_utils import save_iteration_model, create_iteration_plot
                print(f"  Saving iteration {i} model and plot...")
                save_iteration_model(model, model_name, i, config.random_seed, logger.get_results_dir())
                create_iteration_plot(model, loss_list, lb_train, config, logger.get_results_dir(), model_name, config.random_seed, Y_train, i, "original")
        model.store(loss_list, likelihood)
        
        logger.log_training(f"Training completed for {model_name}. Final loss: {loss_list[-1]:.4f}")
        
        # Save metrics progression plot (train metrics only)
        if logger.get_results_dir() and train_rmse_list and train_nlpd_list:
            plot_metrics_progression(train_rmse_list, train_nlpd_list, metrics_iterations, 
                                   metrics_losses, effective_dims_list, logger.get_results_dir(), model_name, config.random_seed)
            
            # Save checkpoint metrics to JSON (train metrics only)
            logger.save_checkpoint_metrics(train_rmse_list, train_nlpd_list, metrics_iterations, 
                                         metrics_losses, model_name, config.random_seed, effective_dims_list)
            
        # Save models & training info
        
        print(model.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
        model_dict[model_name + '_' + str(config.random_seed)] = model
        noise_trace_dict[model_name + '_' + str(config.random_seed)] = noise_trace
        
        ### Saving training report
        
        X_train_mean = model.get_X_mean(Y_train)
        try:
            X_train_scales = model.get_X_scales(Y_train)
        except (RuntimeError, TypeError) as e:
            if "can't convert cuda" in str(e).lower():
                # For nn_gauss model, get_X_scales returns CUDA tensors that need CPU conversion
                if model_name == 'nn_gauss':
                    sigma_tensors = model.X.sigma(Y_train).detach()
                    X_train_scales = np.array([torch.sqrt(x.diag()).cpu().numpy() for x in sigma_tensors])
                else:
                    raise e
        
        # Save plots to results directory
        if logger.get_results_dir():
            # Import function only when needed to avoid directory creation issues
            from src.ssvi_alignment_utils import create_standardized_scattered_plot
            # lb_train is already a numpy array from train_test_split, no need to convert
            # Use Y_train on the same device as model to avoid device mismatch
            create_standardized_scattered_plot(model, loss_list, lb_train, config, logger.get_results_dir(), model.X.__class__.__name__, config.random_seed, Y_train, model_type="original")
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
                        test_jitter = torch.eye(latent_dim).unsqueeze(0) * 1e-5
                        model.X.jitter = torch.cat([test_jitter for i in range(len(Y_test))], axis=0)
                        
                        # Call encoder directly
                        X_test_mean = model.X.mu(Y_test)
                        X_test_covar = model.X.sigma(Y_test)
                        
                        # Restore original jitter
                        model.X.jitter = original_jitter
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
            elif model_name == 'nn_gauss':
                   # For nn_gauss, X_test is the latent variable object, we need to get the mean
                   X_test_mean = X_test.q_mu.detach() # CHECK LATER HERE
       
            # Ensure tensors are on correct device and dtype (X_test_mean and X_train_mean are already tensors)
            X_test_mean_device = X_test_mean.to(dtype=torch.float32, device=config.device)
            X_train_mean_device = X_train_mean.to(dtype=torch.float32, device=config.device)


            Y_test_recon, Y_test_pred_covar = model.reconstruct_y(X_test_mean_device, Y_test, ae=ae, model_name=model_name)
            Y_train_recon, Y_train_pred_covar = model.reconstruct_y(X_train_mean_device, Y_train, ae=ae, model_name=model_name)

            # ################################
            # Compute the metrics:
            
            # 1) Reconstruction error - Train & Test
            
            mse_train = rmse(Y_train, Y_train_recon.T)
            mse_test = rmse(Y_test, Y_test_recon.T)
            
            print(f'Train Reconstruction error {model_name} = ' + str(mse_train))
            print(f'Test Reconstruction error {model_name} = ' + str(mse_test))
            
            # 2) Negative Log Predictive Density (NLPD) - using NLL-based approach inspired by movie_lens_large_expt.py

            # Train NLPD using NLL-based approach (inspired by movie_lens_large_expt.py)
            train_nlpd = compute_nlpd_nll_based(model, Y_train, X_train_mean_device, model_name, loss_list)
            
            # Test NLPD using NLL-based approach (inspired by movie_lens_large_expt.py)
            test_nlpd = compute_nlpd_nll_based(model, Y_test, X_test_mean_device, model_name, losses_test)

            print(f'Train NLPD {model_name} = ' + str(train_nlpd))
            print(f'Test NLPD {model_name} = ' + str(test_nlpd))

            # Save training results (convert tensors to floats for JSON serialization)
            # Ensure all values in loss_list and noise_trace are Python floats
            loss_list_floats = [float(x) for x in loss_list]
            noise_trace_floats = [float(x) for x in noise_trace]
            logger.save_training_results(model_name, config.random_seed, loss_list_floats, noise_trace_floats,
                                        float(mse_train.item()) if hasattr(mse_train, 'item') else float(mse_train), 
                                        float(mse_test.item()) if hasattr(mse_test, 'item') else float(mse_test))
            
            # Store results for summary (convert tensors to floats for JSON serialization)
            results_summary[model_name] = {
                'train_rmse': float(mse_train.item()) if hasattr(mse_train, 'item') else float(mse_train),
                'test_rmse': float(mse_test.item()) if hasattr(mse_test, 'item') else float(mse_test),
                'train_nlpd': float(train_nlpd),  # train_nlpd is already a float from compute_nlpd_nll_based
                'test_nlpd': float(test_nlpd),    # test_nlpd is already a float from compute_nlpd_nll_based
                'final_loss': float(loss_list[-1]),
                'final_noise': float(noise_trace[-1])
            }
            
            logger.log_training(f"Model {model_name} - Train RMSE: {mse_train:.4f}, Test RMSE: {mse_test:.4f}, Train NLPD: {train_nlpd:.4f}, Test NLPD: {test_nlpd:.4f}")
    
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
    
