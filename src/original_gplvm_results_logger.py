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
            # Create results directory with proper naming convention
            timestamp = datetime.now().strftime("%m_%d_%H_%M")
            self.results_dir = os.path.join(
                config.results_dir, 
                f"oil_flow_orig_gplvm_{experiment_name}_res_{timestamp}"
            )
            os.makedirs(self.results_dir, exist_ok=True)
            
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
        return {
            'device': self.config.device,
            'dataset': self.config.dataset,
            'models': self.config.models,
            'latent_dim': self.config.latent_dim,
            'n_inducing': self.config.n_inducing,
            'training_steps': self.config.training_steps,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'random_seed': self.config.random_seed
        }
    
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
        """Save model state dict"""
        if self.results_dir:
            model_dir = os.path.join(self.results_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            filename = f"{model_name}_{seed}.pkl"
            filepath = os.path.join(model_dir, filename)
            
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
    
    def get_results_dir(self) -> Optional[str]:
        """Get results directory path"""
        return self.results_dir
