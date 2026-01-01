#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results logger for Original GP LVM experiments
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional
from .original_gplvm_config import OriginalGPLVMConfig

class OriginalGPLVMResultsLogger:
    """Logger for Original GP LVM experiment results"""
    
    def __init__(self, config: OriginalGPLVMConfig, experiment_name: str = "original_gplvm"):
        self.config = config
        self.experiment_name = experiment_name
        self.results_dir = None
        
        if config.save_results:
            timestamp = datetime.now().strftime("%m_%d_%H_%M")
            self.results_dir = os.path.join(
                config.results_dir, 
                f"orig_gplvm_{experiment_name}_seed_{config.random_seed}_{timestamp}"
            )
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Create subdirectories
            self.training_results_dir = os.path.join(self.results_dir, "training_results")
            os.makedirs(self.training_results_dir, exist_ok=True)
            
            # Save configuration
            self.save_config()
            
            # Initialize log file
            self.log_file = os.path.join(self.results_dir, "training.log")
            self.log_training(f"=== Original GP LVM Experiment Started ===")
            self.log_training(f"Experiment: {experiment_name}")
            self.log_training(f"Timestamp: {timestamp}")
            self.log_training(f"Configuration: {self._config_to_dict()}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        config_dict = {
            'device': self.config.device,
            'dataset': self.config.dataset,
            'models': self.config.models,
            'latent_dim': self.config.latent_dim,
            'n_inducing': self.config.n_inducing,
            'training_steps': self.config.training_steps,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'random_seed': self.config.random_seed,
            'init_lengthscales': self.config.init_lengthscales,
            'init_output_noise_scale': self.config.init_output_noise_scale,
            'pca_init': self.config.pca_init,
            'test_size': self.config.test_size,
            'checkpoint_frequency': self.config.checkpoint_frequency,
            'metrics_frequency': self.config.metrics_frequency,
            'enable_testing': self.config.enable_testing,
            'test_iterations': self.config.test_iterations,
            # Missing data parameters
            'enable_missing_data': self.config.enable_missing_data,
            'missing_data_ratio': self.config.missing_data_ratio,
            'reconstruction_samples': self.config.reconstruction_samples,
            # Additional config parameters
            'prior_type': self.config.prior_type,
            'kernel_type': self.config.kernel_type,
            'use_ard': self.config.use_ard,
            'mean_function': self.config.mean_function,
            'variational_strategy': self.config.variational_strategy,
            'learn_inducing_locations': self.config.learn_inducing_locations
        }
        
        # Add OOD-specific fields if present
        if hasattr(self.config, 'id_dataset'):
            config_dict['id_dataset'] = self.config.id_dataset
        if hasattr(self.config, 'id_classes'):
            config_dict['id_classes'] = self.config.id_classes
        if hasattr(self.config, 'id_samples_per_class'):
            config_dict['id_samples_per_class'] = self.config.id_samples_per_class
        if hasattr(self.config, 'ood_dataset'):
            config_dict['ood_dataset'] = self.config.ood_dataset
        if hasattr(self.config, 'ood_classes'):
            config_dict['ood_classes'] = self.config.ood_classes
        if hasattr(self.config, 'ood_samples_per_class'):
            config_dict['ood_samples_per_class'] = self.config.ood_samples_per_class
        if hasattr(self.config, 'ood_score_types'):
            config_dict['ood_score_types'] = self.config.ood_score_types
        
        return config_dict
    
    def save_config(self):
        """Save configuration to JSON file"""
        if self.results_dir:
            config_path = os.path.join(self.results_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self._config_to_dict(), f, indent=2)
    
    def log_training(self, message: str):
        """Log training message to file"""
        if self.results_dir:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
    
    def save_model(self, model, model_name: str, seed: int):
        """Save model state dict (only to pre_trained_models like original code)"""
        if self.results_dir:
            # Only save to pre_trained_models (like original generalised_gplvm_code)
            pre_trained_dir = os.path.join(self.results_dir, 'pre_trained_models')
            os.makedirs(pre_trained_dir, exist_ok=True)
            
            filename = f'oilflow_{model_name}_{seed}.pkl'
            filepath = os.path.join(pre_trained_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model.state_dict(), f)
            
            self.log_training(f"Model saved: {filepath}")
    
    def save_training_results(self, model_name: str, seed: int, 
                            loss_list: list, noise_trace: list,
                            train_rmse: float, test_rmse: float):
        """Save training results for a model"""
        if self.results_dir:
            results_dir = os.path.join(self.results_dir, "training_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save loss history
            loss_file = os.path.join(results_dir, f"{model_name}_{seed}_losses.json")
            with open(loss_file, 'w') as f:
                json.dump({
                    'losses': loss_list,
                    'noise_trace': noise_trace,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'model_name': model_name,
                    'seed': seed
                }, f, indent=2)
            
            self.log_training(f"Training results saved: {loss_file}")
    
    def save_plot(self, plot_path: str, model_name: str, seed: int):
        """Save plot to results directory"""
        if self.results_dir and os.path.exists(plot_path):
            plots_dir = os.path.join(self.results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Copy plot to results directory
            import shutil
            new_filename = f"{model_name}_{seed}_{os.path.basename(plot_path)}"
            new_path = os.path.join(plots_dir, new_filename)
            shutil.copy2(plot_path, new_path)
            
            self.log_training(f"Plot saved: {new_path}")
    
    def save_final_summary(self, results_summary: Dict[str, Any]):
        """Save final experiment summary"""
        if self.results_dir:
            summary_file = os.path.join(self.results_dir, "experiment_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            self.log_training(f"Experiment summary saved: {summary_file}")
    
    def save_checkpoint_metrics(self, train_rmse_list, train_nlpd_list, metrics_iterations, 
                               metrics_losses, model_name: str, random_seed: int, effective_dims_list=None, train_missing_nlpd_list=None, train_missing_rmse_list=None):
        """Save checkpoint metrics to JSON file in training_results folder (train metrics only)"""
        if self.results_dir and train_rmse_list and train_nlpd_list:
            # Prepare checkpoint data (train metrics only)
            checkpoint_data = {
                'model_name': model_name,
                'seed': random_seed,
                'total_iterations': getattr(self.config, 'training_steps', {}).get(model_name, 0),
                'checkpoints': []
            }
            
            # Extract metrics from each checkpoint
            for i, (iteration, train_rmse, train_nlpd, loss) in enumerate(zip(
                metrics_iterations, train_rmse_list, train_nlpd_list, metrics_losses)):
                checkpoint_entry = {
                    'iteration': iteration,
                    'train_rmse': float(train_rmse),
                    'train_nlpd': float(train_nlpd),
                    'elbo_loss': float(loss)
                }
                # Add effective dimensions if available
                if effective_dims_list and i < len(effective_dims_list):
                    checkpoint_entry['effective_dims'] = int(effective_dims_list[i])
                # Add missing data NLPD if available
                if train_missing_nlpd_list and i < len(train_missing_nlpd_list):
                    checkpoint_entry['train_missing_nlpd'] = float(train_missing_nlpd_list[i])
                # Add missing data RMSE if available
                if train_missing_rmse_list and i < len(train_missing_rmse_list):
                    checkpoint_entry['train_missing_rmse'] = float(train_missing_rmse_list[i])
                checkpoint_data['checkpoints'].append(checkpoint_entry)
            
            # Save to JSON file
            checkpoint_file = os.path.join(self.training_results_dir, f"{model_name}_{random_seed}_checkpoint_metrics.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.log_training(f"Checkpoint metrics saved: training_results/{os.path.basename(checkpoint_file)}")
    
    def get_results_dir(self) -> Optional[str]:
        """Get results directory path"""
        return self.results_dir
