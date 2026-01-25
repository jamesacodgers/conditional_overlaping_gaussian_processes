#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Results Comparison Script for GP-LVM Experiments

This script compares results across different model types (gauss, nn_gauss, ssvi) by:
1. Matching runs based on configuration parameters and seeds
2. Collecting checkpoint metrics from JSON files
3. Creating comparative plots with mean and std across runs

Usage:
    python automatic_results_comparison.py \
        --orig_gplvm_dir "path/to/orig_gplvm_run_results" \
        --ssvi_dir "path/to/ssvi_run_results" \
        --output_dir "path/to/output"
"""

import os
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import glob


@dataclass
class RunConfig:
    """Configuration parameters for a single run"""
    n_ind: int
    arch: int
    len: float
    sig: float
    b: int
    s: int  # seed
    lr: float
    n_steps: int
    P: int
    miss_f_ratio: float
    seed: int
    timestamp: str
    
    def get_config_key(self) -> str:
        """Get a key that identifies the configuration (excluding seed and timestamp)"""
        return f"n_ind_{self.n_ind}_arch_{self.arch}_len_{self.len}_sig_{self.sig}_b_{self.b}_lr_{self.lr}_n_steps_{self.n_steps}_P_{self.P}_miss_f_ratio_{self.miss_f_ratio}"


@dataclass
class CheckpointMetrics:
    """Checkpoint metrics from a single run"""
    iteration: int
    train_rmse: float
    train_nlpd: float
    elbo_loss: float  # May be full_dataset_elbo_loss in newer checkpoints


@dataclass
class FinalMetrics:
    """Final metrics from experiment_summary.json"""
    train_rmse: Optional[float]
    test_rmse: Optional[float]
    train_nlpd: Optional[float]
    test_nlpd: Optional[float]
    train_missing_data_nlpd: Optional[float]
    test_missing_data_nlpd: Optional[float]
    train_missing_data_rmse: Optional[float]
    test_missing_data_rmse: Optional[float]
    final_full_loss: Optional[float]
    final_batch_loss: Optional[float]
    final_noise: Optional[float]

@dataclass
class RunResults:
    """Results from a single run"""
    config: RunConfig
    model_type: str  # 'gauss', 'nn_gauss', or 'ssvi'
    checkpoint_metrics: List[CheckpointMetrics]
    final_metrics: Optional[FinalMetrics] = None
    success: bool = True


class ResultsComparator:
    """Main class for comparing results across different model types"""
    
    def __init__(self, output_dir: str, experiment_name: str = None):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for results
        self.run_results: Dict[str, List[RunResults]] = defaultdict(list)  # config_key -> [RunResults]
        
    def parse_run_name(self, run_name: str) -> Optional[RunConfig]:
        """
        Parse run name to extract configuration parameters
        
        Examples:
        - orig_gplvm_n_ind_64_arch_192_len_01_sig_1_b_128_s_0_lr_0001_n_steps_60000_P_64_miss_f_ratio_00_seed_0_10_10_05_49
        - orig_gplvm_n_ind_32_arch_192_len_1_sig_1_b_32_s_0_lr_0001_n_steps_100000_P_64_miss_f_ratio_00_test_01_seed_0_10_14_11_32
        - ssvi_n_ind_64_arch_192_len_001_sig_1_b_128_s_0_lr_0001_n_steps_60000_P_64_miss_f_ratio_00_seed_0_10_10_04_39
        """
        # Pattern to match the run name structure
        # Handle both miss_f_ratio and miss_t_ratio formats, and optional test_XX parameter
        pattern = r'(?:orig_gplvm_|ssvi_)?n_ind_(\d+)_arch_(\d+)_len_([\d.]+)_sig_([\d.]+)_b_(\d+)_s_(\d+)_lr_([\d.]+)_n_steps_(\d+)_P_(\d+)_miss_[tf]_ratio_([\d.]+)(?:_test_\d+)?_seed_(\d+)_(\d+_\d+_\d+_\d+)'
        
        match = re.match(pattern, run_name)
        if not match:
            print(f"Warning: Could not parse run name: {run_name}")
            return None
            
        try:
            n_ind = int(match.group(1))
            arch = int(match.group(2))
            len_val = float(match.group(3))
            sig = float(match.group(4))
            b = int(match.group(5))
            s = int(match.group(6))
            lr = float(match.group(7))
            n_steps = int(match.group(8))
            P = int(match.group(9))
            miss_f_ratio = float(match.group(10))
            seed = int(match.group(11))
            timestamp = match.group(12)
            
            return RunConfig(n_ind, arch, len_val, sig, b, s, lr, n_steps, P, miss_f_ratio, seed, timestamp)
        except (ValueError, IndexError) as e:
            print(f"Error parsing run name {run_name}: {e}")
            return None
    
    def load_checkpoint_metrics(self, checkpoint_file: str) -> List[CheckpointMetrics]:
        """Load checkpoint metrics from JSON file"""
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            metrics = []
            for checkpoint in data.get('checkpoints', []):
                # Try full_dataset_elbo_loss first (new format), fall back to elbo_loss (old format)
                elbo_loss = checkpoint.get('full_dataset_elbo_loss', checkpoint.get('elbo_loss', 0.0))
                
                metrics.append(CheckpointMetrics(
                    iteration=checkpoint.get('iteration', 0),
                    train_rmse=checkpoint.get('train_rmse', 0.0),
                    train_nlpd=checkpoint.get('train_nlpd', 0.0),
                    elbo_loss=elbo_loss
                ))
            
            return metrics
        except Exception as e:
            print(f"Error loading checkpoint metrics from {checkpoint_file}: {e}")
            return []
    
    def load_final_metrics(self, experiment_summary_file: str, model_type: str) -> Optional[FinalMetrics]:
        """Load final metrics from experiment_summary.json file"""
        try:
            with open(experiment_summary_file, 'r') as f:
                data = json.load(f)
            
            # Check if data is nested by model type
            model_data = None
            
            # Try to find the model_type key in the data
            # Handle variations in naming (e.g., 'ssvi' might be 'oilflow_ssvi', 'mnist_ssvi', etc.)
            if model_type in data:
                model_data = data[model_type]
            else:
                # Try to find a key containing the model_type (for ssvi variants)
                matching_keys = [k for k in data.keys() if model_type in k.lower()]
                if matching_keys:
                    model_data = data[matching_keys[0]]
                elif len(data) == 1 and isinstance(list(data.values())[0], dict):
                    # Fall back to taking the first nested dictionary
                    model_data = list(data.values())[0]
                else:
                    # Use flat structure
                    model_data = data
            
            # Handle NaN values by converting them to None
            def safe_float(value):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return None
                return float(value)
            
            # Handle both new and old field names for backward compatibility
            final_metrics = FinalMetrics(
                train_rmse=safe_float(model_data.get('train_rmse', 0.0)),
                test_rmse=safe_float(model_data.get('test_rmse', 0.0)),
                train_nlpd=safe_float(model_data.get('train_nlpd', 0.0)),
                test_nlpd=safe_float(model_data.get('test_nlpd', 0.0)),
                train_missing_data_nlpd=safe_float(model_data.get('train_missing_data_nlpd', model_data.get('train_missing_nlpd', None))),
                test_missing_data_nlpd=safe_float(model_data.get('test_missing_data_nlpd', model_data.get('test_missing_nlpd', None))),
                train_missing_data_rmse=safe_float(model_data.get('train_missing_data_rmse', None)),
                test_missing_data_rmse=safe_float(model_data.get('test_missing_data_rmse', None)),
                final_full_loss=safe_float(model_data.get('final_full_loss', model_data.get('final_loss', None))),
                final_batch_loss=safe_float(model_data.get('final_batch_loss', None)),
                final_noise=safe_float(model_data.get('final_noise', 0.0))
            )
            
            return final_metrics
        except Exception as e:
            print(f"Error loading final metrics from {experiment_summary_file}: {e}")
            return None
    
    def find_checkpoint_files(self, run_dir: str, model_type: str) -> List[str]:
        """Find checkpoint metrics JSON files for a specific model type in a run directory"""
        checkpoint_files = []
        
        # Look for checkpoint files in training_results subdirectory
        training_results_dir = os.path.join(run_dir, 'training_results')
        if os.path.exists(training_results_dir):
            # Find files matching the specific model type
            if model_type == 'gauss':
                # Look for gauss_*_checkpoint_metrics.json files
                pattern = os.path.join(training_results_dir, 'gauss_*_checkpoint_metrics.json')
            elif model_type == 'nn_gauss':
                # Look for nn_gauss_*_checkpoint_metrics.json files
                pattern = os.path.join(training_results_dir, 'nn_gauss_*_checkpoint_metrics.json')
            elif model_type == 'ssvi':
                # Look for *_ssvi_*_checkpoint_metrics.json files
                pattern = os.path.join(training_results_dir, '*_ssvi_*_checkpoint_metrics.json')
            else:
                # Fallback: look for any *_checkpoint_metrics.json files
                pattern = os.path.join(training_results_dir, '*_checkpoint_metrics.json')
            
            checkpoint_files.extend(glob.glob(pattern))
        
        return checkpoint_files
    
    def load_results_from_directory(self, results_dir: str, model_type: str):
        """Load all results from a directory containing run results"""
        if not os.path.exists(results_dir):
            print(f"Warning: Directory {results_dir} does not exist")
            return
        
        print(f"Loading {model_type} results from {results_dir}")
        
        # Check if we need to look one level deeper (for *_run_results subdirectories)
        subdirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
        
        # If we find directories like 'orig_gplvm_run_results' or 'ssvi_run_results', look inside them
        run_results_subdirs = [d for d in subdirs if 'run_results' in d.lower()]
        
        if run_results_subdirs:
            # Look inside the *_run_results subdirectories
            for subdir in run_results_subdirs:
                actual_results_dir = os.path.join(results_dir, subdir)
                print(f"  Looking inside subdirectory: {subdir}")
                self._load_runs_from_specific_directory(actual_results_dir, model_type)
        else:
            # Look directly in the provided directory
            self._load_runs_from_specific_directory(results_dir, model_type)
    
    def _load_runs_from_specific_directory(self, results_dir: str, model_type: str):
        """Helper method to load runs from a specific directory"""
        # Find all run directories
        run_dirs = [d for d in os.listdir(results_dir) 
                   if os.path.isdir(os.path.join(results_dir, d))]
        
        for run_dir_name in run_dirs:
            # Parse run configuration
            config = self.parse_run_name(run_dir_name)
            if config is None:
                print(f"  Warning: Could not parse run name: {run_dir_name}")
                continue
            
            run_path = os.path.join(results_dir, run_dir_name)
            
            # Find checkpoint files for this specific model type
            checkpoint_files = self.find_checkpoint_files(run_path, model_type)
            
            if not checkpoint_files:
                print(f"  Warning: No checkpoint files found for run {run_dir_name}")
                continue
            
            # Load metrics from the first checkpoint file (should be consistent across files)
            # In practice, we might want to aggregate or choose the best one
            checkpoint_metrics = self.load_checkpoint_metrics(checkpoint_files[0])
            
            if not checkpoint_metrics:
                print(f"  Warning: No valid checkpoint metrics for run {run_dir_name}")
                continue
            
            # Load final metrics from experiment_summary.json
            experiment_summary_file = os.path.join(run_path, 'experiment_summary.json')
            final_metrics = None
            if os.path.exists(experiment_summary_file):
                final_metrics = self.load_final_metrics(experiment_summary_file, model_type)
            else:
                print(f"  Warning: No experiment_summary.json found for run {run_dir_name}")
            
            # Create run results
            run_results = RunResults(
                config=config,
                model_type=model_type,
                checkpoint_metrics=checkpoint_metrics,
                final_metrics=final_metrics,
                success=True
            )
            
            # Store by configuration key
            config_key = config.get_config_key()
            self.run_results[config_key].append(run_results)
            
            print(f"  Loaded {len(checkpoint_metrics)} checkpoints for {model_type} run: {run_dir_name}")
    
    def match_runs_across_models(self) -> Dict[str, Dict[str, List[RunResults]]]:
        """
        Match runs across different model types based on configuration parameters
        
        Returns:
            Dictionary mapping config_key -> {model_type -> [RunResults]}
        """
        matched_runs = defaultdict(lambda: defaultdict(list))
        
        for config_key, runs in self.run_results.items():
            for run in runs:
                matched_runs[config_key][run.model_type].append(run)
        
        return matched_runs
    
    def aggregate_metrics_across_runs(self, runs: List[RunResults]) -> Tuple[List[int], List[float], List[float], List[float]]:
        """
        Aggregate metrics across multiple runs with the same configuration
        
        Returns:
            (iterations, mean_train_rmse, mean_train_nlpd, mean_elbo_loss)
        """
        if not runs:
            return [], [], [], []
        
        # Get all unique iterations
        all_iterations = set()
        for run in runs:
            for checkpoint in run.checkpoint_metrics:
                all_iterations.add(checkpoint.iteration)
        
        iterations = sorted(list(all_iterations))
        
        # Aggregate metrics for each iteration
        mean_train_rmse = []
        mean_train_nlpd = []
        mean_elbo_loss = []
        
        for iteration in iterations:
            train_rmse_values = []
            train_nlpd_values = []
            elbo_loss_values = []
            
            # Collect values from all runs for this iteration
            for run in runs:
                for checkpoint in run.checkpoint_metrics:
                    if checkpoint.iteration == iteration:
                        train_rmse_values.append(checkpoint.train_rmse)
                        train_nlpd_values.append(checkpoint.train_nlpd)
                        elbo_loss_values.append(checkpoint.elbo_loss)
                        break
            
            # Compute means
            mean_train_rmse.append(np.mean(train_rmse_values) if train_rmse_values else np.nan)
            mean_train_nlpd.append(np.mean(train_nlpd_values) if train_nlpd_values else np.nan)
            mean_elbo_loss.append(np.mean(elbo_loss_values) if elbo_loss_values else np.nan)
        
        return iterations, mean_train_rmse, mean_train_nlpd, mean_elbo_loss
    
    def compute_std_across_runs(self, runs: List[RunResults]) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute standard deviation across runs for each iteration
        
        Returns:
            (std_train_rmse, std_train_nlpd, std_elbo_loss)
        """
        if not runs:
            return [], [], []
        
        # Get all unique iterations
        all_iterations = set()
        for run in runs:
            for checkpoint in run.checkpoint_metrics:
                all_iterations.add(checkpoint.iteration)
        
        iterations = sorted(list(all_iterations))
        
        # Compute std for each iteration
        std_train_rmse = []
        std_train_nlpd = []
        std_elbo_loss = []
        
        for iteration in iterations:
            train_rmse_values = []
            train_nlpd_values = []
            elbo_loss_values = []
            
            # Collect values from all runs for this iteration
            for run in runs:
                for checkpoint in run.checkpoint_metrics:
                    if checkpoint.iteration == iteration:
                        train_rmse_values.append(checkpoint.train_rmse)
                        train_nlpd_values.append(checkpoint.train_nlpd)
                        elbo_loss_values.append(checkpoint.elbo_loss)
                        break
            
            # Compute standard deviations
            std_train_rmse.append(np.std(train_rmse_values) if len(train_rmse_values) > 1 else 0.0)
            std_train_nlpd.append(np.std(train_nlpd_values) if len(train_nlpd_values) > 1 else 0.0)
            std_elbo_loss.append(np.std(elbo_loss_values) if len(elbo_loss_values) > 1 else 0.0)
        
        return std_train_rmse, std_train_nlpd, std_elbo_loss
    
    def plot_comparison(self, config_key: str, model_runs: Dict[str, List[RunResults]]):
        """Create comparison plots for a specific configuration"""
        
        # Define colors and display names for different model types
        model_colors = {
            'gauss': 'blue',
            'nn_gauss': 'red', 
            'ssvi': 'green'
        }
        
        model_display_names = {
            'gauss': 'MF',
            'nn_gauss': 'A-MF', 
            'ssvi': 'A-S'
        }
        
        # Create figure with subplots (only 3 plots, no text box)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot metrics for each model type
        for model_type, runs in model_runs.items():
            if not runs:
                continue
                
            color = model_colors.get(model_type, 'black')
            display_name = model_display_names.get(model_type, model_type)
            
            # Aggregate metrics
            iterations, mean_train_rmse, mean_train_nlpd, mean_elbo_loss = self.aggregate_metrics_across_runs(runs)
            std_train_rmse, std_train_nlpd, std_elbo_loss = self.compute_std_across_runs(runs)
            
            if not iterations:
                continue
            
            iterations = np.array(iterations)
            mean_train_rmse = np.array(mean_train_rmse)
            mean_train_nlpd = np.array(mean_train_nlpd)
            mean_elbo_loss = np.array(mean_elbo_loss)
            std_train_rmse = np.array(std_train_rmse)
            std_train_nlpd = np.array(std_train_nlpd)
            std_elbo_loss = np.array(std_elbo_loss)
            
            # Plot 1: ELBO Loss
            ax1.plot(iterations, mean_elbo_loss, color=color, label=f'{display_name}', linewidth=2)
            ax1.fill_between(iterations, 
                           mean_elbo_loss - std_elbo_loss, 
                           mean_elbo_loss + std_elbo_loss, 
                           color=color, alpha=0.2)
            
            # Plot 2: Train RMSE
            ax2.plot(iterations, mean_train_rmse, color=color, label=f'{display_name}', linewidth=2)
            ax2.fill_between(iterations, 
                           mean_train_rmse - std_train_rmse, 
                           mean_train_rmse + std_train_rmse, 
                           color=color, alpha=0.2)
            
            # Plot 3: Train NLPD
            ax3.plot(iterations, mean_train_nlpd, color=color, label=f'{display_name}', linewidth=2)
            ax3.fill_between(iterations, 
                           mean_train_nlpd - std_train_nlpd, 
                           mean_train_nlpd + std_train_nlpd, 
                           color=color, alpha=0.2)
        
        # Configure subplots with consistent font size
        ax1.set_xlabel('Iteration', fontsize=14)
        ax1.set_ylabel('ELBO Loss', fontsize=14)
        ax1.set_title('ELBO Loss Comparison', fontsize=14)
        ax1.legend(fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=14)
        
        ax2.set_xlabel('Iteration', fontsize=14)
        ax2.set_ylabel('Train RMSE', fontsize=14)
        ax2.set_title('Training RMSE Comparison', fontsize=14)
        ax2.legend(fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=14)
        
        ax3.set_xlabel('Iteration', fontsize=14)
        ax3.set_ylabel('Train NLPD', fontsize=14)
        ax3.set_title('Training NLPD Comparison', fontsize=14)
        ax3.legend(fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=14)
        
        plt.tight_layout()
        
        # Save plot with experiment name prefix (both PNG and PDF)
        safe_config_key = re.sub(r'[^\w\-_\.]', '_', config_key)
        if self.experiment_name:
            plot_filename_png = f'{self.experiment_name}_comparison_{safe_config_key}.png'
            plot_filename_pdf = f'{self.experiment_name}_comparison_{safe_config_key}.pdf'
        else:
            plot_filename_png = f'comparison_{safe_config_key}.png'
            plot_filename_pdf = f'comparison_{safe_config_key}.pdf'
        
        plot_path_png = os.path.join(self.output_dir, plot_filename_png)
        plot_path_pdf = os.path.join(self.output_dir, plot_filename_pdf)
        plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
        plt.savefig(plot_path_pdf, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {plot_path_png}")
        print(f"Saved comparison plot (PDF): {plot_path_pdf}")
    
    def plot_nlpd_comparison(self, config_key: str, model_runs: Dict[str, List[RunResults]]):
        """Create a standalone NLPD comparison plot saved as PDF"""
        
        # Define colors and display names for different model types
        model_colors = {
            'gauss': 'blue',
            'nn_gauss': 'red', 
            'ssvi': 'green'
        }
        
        model_display_names = {
            'gauss': 'MF',
            'nn_gauss': 'A-MF', 
            'ssvi': 'A-S'
        }
        
        # Create figure with single subplot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot metrics for each model type
        for model_type, runs in model_runs.items():
            if not runs:
                continue
                
            color = model_colors.get(model_type, 'black')
            display_name = model_display_names.get(model_type, model_type)
            
            # Aggregate metrics
            iterations, mean_train_rmse, mean_train_nlpd, mean_elbo_loss = self.aggregate_metrics_across_runs(runs)
            std_train_rmse, std_train_nlpd, std_elbo_loss = self.compute_std_across_runs(runs)
            
            if not iterations:
                continue
            
            iterations = np.array(iterations)
            mean_train_nlpd = np.array(mean_train_nlpd)
            std_train_nlpd = np.array(std_train_nlpd)
            
            # Plot Train NLPD
            ax.plot(iterations, mean_train_nlpd, color=color, label=f'{display_name}', linewidth=2)
            ax.fill_between(iterations, 
                           mean_train_nlpd - std_train_nlpd, 
                           mean_train_nlpd + std_train_nlpd, 
                           color=color, alpha=0.2)
        
        # Configure subplot with consistent font size
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Train NLPD', fontsize=14)
        ax.set_title('Training NLPD Comparison', fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        
        plt.tight_layout()
        
        # Save plot as PDF
        safe_config_key = re.sub(r'[^\w\-_\.]', '_', config_key)
        if self.experiment_name:
            plot_filename = f'{self.experiment_name}_nlpd_comparison_{safe_config_key}.pdf'
        else:
            plot_filename = f'nlpd_comparison_{safe_config_key}.pdf'
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"Saved NLPD comparison plot (PDF): {plot_path}")
    
    def create_final_metrics_boxplots(self, config_key: str, model_runs: Dict[str, List[RunResults]]):
        """Create boxplots for final metrics comparison"""
        
        # Define colors and display names for different model types
        model_colors = {
            'gauss': 'blue',
            'nn_gauss': 'red', 
            'ssvi': 'green'
        }
        
        model_display_names = {
            'gauss': 'MF',
            'nn_gauss': 'A-MF', 
            'ssvi': 'A-S'
        }
        
        # Collect final metrics for each model type
        final_metrics_data = {}
        
        for model_type, runs in model_runs.items():
            if not runs:
                continue
                
            # Collect metrics from runs that have final_metrics
            valid_runs = [run for run in runs if run.final_metrics is not None]
            if not valid_runs:
                continue
                
            final_metrics_data[model_type] = {
                'train_rmse': [run.final_metrics.train_rmse for run in valid_runs if run.final_metrics.train_rmse is not None],
                'test_rmse': [run.final_metrics.test_rmse for run in valid_runs if run.final_metrics.test_rmse is not None],
                'train_nlpd': [run.final_metrics.train_nlpd for run in valid_runs if run.final_metrics.train_nlpd is not None],
                'test_nlpd': [run.final_metrics.test_nlpd for run in valid_runs if run.final_metrics.test_nlpd is not None],
                'final_full_loss': [run.final_metrics.final_full_loss for run in valid_runs if run.final_metrics.final_full_loss is not None],
                'final_batch_loss': [run.final_metrics.final_batch_loss for run in valid_runs if run.final_metrics.final_batch_loss is not None],
                'final_noise': [run.final_metrics.final_noise for run in valid_runs if run.final_metrics.final_noise is not None]
            }
        
        if not final_metrics_data:
            print(f"No final metrics data available for boxplots")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Define metrics to plot
        metrics_to_plot = [
            ('train_rmse', 'Train RMSE', ax1),
            ('test_rmse', 'Test RMSE', ax2),
            ('train_nlpd', 'Train NLPD', ax3),
            ('test_nlpd', 'Test NLPD', ax4)
        ]
        
        for metric_key, metric_name, ax in metrics_to_plot:
            # Prepare data for boxplot
            boxplot_data = []
            boxplot_labels = []
            boxplot_colors = []
            
            for model_type in ['gauss', 'nn_gauss', 'ssvi']:
                if model_type in final_metrics_data and final_metrics_data[model_type][metric_key]:
                    display_name = model_display_names.get(model_type, model_type)
                    boxplot_data.append(final_metrics_data[model_type][metric_key])
                    boxplot_labels.append(f'{display_name}')
                    boxplot_colors.append(model_colors.get(model_type, 'gray'))
            
            if boxplot_data:
                # Create boxplot
                bp = ax.boxplot(boxplot_data, tick_labels=boxplot_labels, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], boxplot_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{metric_name} Final Performance', fontsize=14)
                ax.set_ylabel(metric_name, fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels if needed
                ax.tick_params(axis='x', rotation=45, labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
        
        plt.tight_layout()
        
        # Save boxplot
        safe_config_key = re.sub(r'[^\w\-_\.]', '_', config_key)
        if self.experiment_name:
            boxplot_filename = f'{self.experiment_name}_final_metrics_boxplot_{safe_config_key}.png'
        else:
            boxplot_filename = f'final_metrics_boxplot_{safe_config_key}.png'
        boxplot_path = os.path.join(self.output_dir, boxplot_filename)
        plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved final metrics boxplot: {boxplot_path}")
    
    def generate_summary_report(self, config_key: str, model_runs: Dict[str, List[RunResults]], summary_file: str):
        """Generate a summary report for a specific configuration"""
        
        with open(summary_file, 'w') as f:
            f.write("Automatic Results Comparison Summary\n")
            if self.experiment_name:
                f.write(f"Experiment: {self.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration: {config_key}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            # Add configuration summary
            f.write("CONFIGURATION SUMMARY:\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration: {config_key}\n")
            f.write("-" * 40 + "\n")
            
            total_runs = sum(len(runs) for runs in model_runs.values())
            f.write(f"Total runs: {total_runs}\n")
            
            for model_type, runs in model_runs.items():
                if runs:
                    f.write(f"{model_type}: {len(runs)} runs\n")
                    
                    # Show run details
                    for run in runs:
                        f.write(f"  - seed_{run.config.seed} ({run.config.timestamp})\n")
            
            f.write("\n")
        
        print(f"Generated summary report: {summary_file}")
    
    def generate_json_results_table(self, config_key: str, model_runs: Dict[str, List[RunResults]], json_file: str):
        """Generate JSON output with mean ± std results in table format"""
        
        # Collect final metrics for each model type
        results_table = {}
        
        for model_type, runs in model_runs.items():
            if not runs:
                continue
                
            # Collect metrics from runs that have final_metrics
            valid_runs = [run for run in runs if run.final_metrics is not None]
            if not valid_runs:
                continue
            
            # Calculate mean and std for each metric
            metrics_data = {
                'train_rmse': [run.final_metrics.train_rmse for run in valid_runs if run.final_metrics.train_rmse is not None],
                'test_rmse': [run.final_metrics.test_rmse for run in valid_runs if run.final_metrics.test_rmse is not None],
                'train_nlpd': [run.final_metrics.train_nlpd for run in valid_runs if run.final_metrics.train_nlpd is not None],
                'test_nlpd': [run.final_metrics.test_nlpd for run in valid_runs if run.final_metrics.test_nlpd is not None],
                'train_missing_data_nlpd': [run.final_metrics.train_missing_data_nlpd for run in valid_runs if run.final_metrics.train_missing_data_nlpd is not None],
                'test_missing_data_nlpd': [run.final_metrics.test_missing_data_nlpd for run in valid_runs if run.final_metrics.test_missing_data_nlpd is not None],
                'train_missing_data_rmse': [run.final_metrics.train_missing_data_rmse for run in valid_runs if run.final_metrics.train_missing_data_rmse is not None],
                'test_missing_data_rmse': [run.final_metrics.test_missing_data_rmse for run in valid_runs if run.final_metrics.test_missing_data_rmse is not None],
                'final_full_loss': [run.final_metrics.final_full_loss for run in valid_runs if run.final_metrics.final_full_loss is not None],
                'final_batch_loss': [run.final_metrics.final_batch_loss for run in valid_runs if run.final_metrics.final_batch_loss is not None],
                'final_noise': [run.final_metrics.final_noise for run in valid_runs if run.final_metrics.final_noise is not None]
            }
            
            # Calculate mean ± std for each metric
            model_results = {}
            for metric_name, values in metrics_data.items():
                if values:  # Only if we have valid values
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    model_results[metric_name] = {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'mean_plus_minus_std': f"{mean_val:.4f} ± {std_val:.4f}",
                        'n_runs': len(values)
                    }
                else:
                    model_results[metric_name] = {
                        'mean': None,
                        'std': None,
                        'mean_plus_minus_std': "-",
                        'n_runs': 0
                    }
            
            results_table[model_type] = model_results
        
        # Create the complete JSON structure
        json_output = {
            'experiment_name': self.experiment_name,
            'configuration': config_key,
            'output_directory': self.output_dir,
            'results_table': results_table,
            'summary': {
                'total_configurations': 1,
                'total_runs': sum(len(runs) for runs in model_runs.values()),
                'model_types': list(model_runs.keys()),
                'runs_per_model': {model_type: len(runs) for model_type, runs in model_runs.items()}
            }
        }
        
        # Write JSON file
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        print(f"Generated JSON results table: {json_file}")
        
        # Also print a summary table to console
        print(f"\nResults Table for {config_key}:")
        print("=" * 80)
        print(f"{'Model':<15} {'Train RMSE':<20} {'Test RMSE':<20} {'Train NLPD':<20} {'Test NLPD':<20}")
        print("-" * 80)
        
        # Define display names for console output
        model_display_names = {
            'gauss': 'MF',
            'nn_gauss': 'A-MF', 
            'ssvi': 'A-S'
        }
        
        for model_type in ['gauss', 'nn_gauss', 'ssvi']:
            if model_type in results_table:
                display_name = model_display_names.get(model_type, model_type)
                train_rmse = results_table[model_type]['train_rmse']['mean_plus_minus_std']
                test_rmse = results_table[model_type]['test_rmse']['mean_plus_minus_std']
                train_nlpd = results_table[model_type]['train_nlpd']['mean_plus_minus_std']
                test_nlpd = results_table[model_type]['test_nlpd']['mean_plus_minus_std']
                print(f"{display_name:<15} {train_rmse:<20} {test_rmse:<20} {train_nlpd:<20} {test_nlpd:<20}")
        
        print("=" * 80)
    
    def run_comparison(self, orig_gplvm_dir: str, ssvi_dir: str):
        """Main method to run the complete comparison"""
        
        print("Starting automatic results comparison...")
        
        # Load results from both directories
        self.load_results_from_directory(orig_gplvm_dir, 'gauss')
        self.load_results_from_directory(orig_gplvm_dir, 'nn_gauss')  # Both gauss and nn_gauss might be in orig_gplvm_dir
        self.load_results_from_directory(ssvi_dir, 'ssvi')
        
        # Match runs across models
        matched_runs = self.match_runs_across_models()
        
        print(f"\nFound {len(matched_runs)} unique configurations")
        
        # Generate comparison plots and summary for each configuration
        for config_key, model_runs in matched_runs.items():
            print(f"\nProcessing configuration: {config_key}")
            # Create progression plots (ELBO, RMSE, NLPD over time)
            self.plot_comparison(config_key, model_runs)
            # Create standalone NLPD plot (PDF)
            self.plot_nlpd_comparison(config_key, model_runs)
            # Create final metrics boxplots
            self.create_final_metrics_boxplots(config_key, model_runs)
            # Generate summary report for this specific configuration
            safe_config_key = re.sub(r'[^\w\-_\.]', '_', config_key)
            summary_file = os.path.join(self.output_dir, f'{self.experiment_name}_comparison_summary_{safe_config_key}.txt')
            self.generate_summary_report(config_key, model_runs, summary_file)
            # Generate JSON results table with mean ± std
            json_file = os.path.join(self.output_dir, f'{self.experiment_name}_results_table_{safe_config_key}.json')
            self.generate_json_results_table(config_key, model_runs, json_file)
        
        print(f"\nComparison complete! Results saved to: {self.output_dir}")


def detect_experiment_name(path: str) -> str:
    """Detect experiment name from directory path (e.g., 'qpcr_related', 'oil_flow_related')"""
    # Look for patterns like 'qpcr_related', 'oil_flow_related', etc.
    path_parts = path.replace('\\', '/').split('/')
    for part in path_parts:
        if '_related' in part.lower():
            return part
    return "unknown_experiment"


def main():
    # =============================================================================
    # HARDCODED DEFAULT PATHS - Modify these for your specific setup
    # =============================================================================
    DEFAULT_ORIG_GPLVM_DIR = r"C:\Users\mtret\MyStudy\conditional_overlaping_gaussian_processes\ssvi_scripts_experiments\oil_flow_related\final_v1_orig_gplvm\orig_gplvm_run_results"
    DEFAULT_SSVI_DIR = r"C:\Users\mtret\MyStudy\conditional_overlaping_gaussian_processes\ssvi_scripts_experiments\oil_flow_related\final_v1_ssvi\ssvi_run_results"
    
    # Auto-detect experiment name and create output directory in ssvi_scripts_experiments
    experiment_name = detect_experiment_name(DEFAULT_ORIG_GPLVM_DIR)
    DEFAULT_OUTPUT_DIR = r"C:\Users\mtret\MyStudy\conditional_overlaping_gaussian_processes\ssvi_scripts_experiments\comparison_results"
    
    # Hardcoded subfolder name for this comparison
    COMPARISON_SUBFOLDER = "oilflow_related_comparison_26_01"
    # =============================================================================
    
    parser = argparse.ArgumentParser(description='Automatic Results Comparison for GP-LVM Experiments')
    parser.add_argument('--orig_gplvm_dir', type=str, default=DEFAULT_ORIG_GPLVM_DIR,
                       help=f'Directory containing orig_gplvm_run_results (default: {DEFAULT_ORIG_GPLVM_DIR})')
    parser.add_argument('--ssvi_dir', type=str, default=DEFAULT_SSVI_DIR,
                       help=f'Directory containing ssvi_run_results (default: {DEFAULT_SSVI_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for comparison plots and reports (default: {DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # Auto-detect experiment name from the provided paths
    detected_experiment = detect_experiment_name(args.orig_gplvm_dir)
    
    # Create output directory with subfolder
    output_dir_with_subfolder = os.path.join(args.output_dir, COMPARISON_SUBFOLDER)
    
    print("=" * 60)
    print("Automatic GP-LVM Results Comparison")
    print("=" * 60)
    print(f"Experiment: {detected_experiment}")
    print(f"Original GP-LVM directory: {args.orig_gplvm_dir}")
    print(f"SSVI directory: {args.ssvi_dir}")
    print(f"Output directory: {output_dir_with_subfolder}")
    print("=" * 60)
    
    # Create comparator and run comparison
    comparator = ResultsComparator(output_dir_with_subfolder, detected_experiment)
    comparator.run_comparison(args.orig_gplvm_dir, args.ssvi_dir)


if __name__ == "__main__":
    main()
