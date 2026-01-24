# %%
"""
Replot qPCR SSVI GP-LVM Results from Checkpoint

This script loads a trained SSVI GP-LVM model from a checkpoint and regenerates all plots
without retraining the model.

Usage:
    python ssvi_scripts_experiments/qpcr_ssvi_replot_results.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import pickle
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Get the project root directory (parent of ssvi_scripts_experiments)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add both project root and generalised_gplvm_code to path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF
from src.ssvi_utils import SSVIResultsLogger, plot_metrics_history, plot_final_metrics_summary, \
    plot_missing_nlpd_metrics_progression, plot_missing_rmse_metrics_progression, plot_latent_slice_outer
from src.ssvi_metrics import SSVIMetricsComputer, SSVIMetrics
from src.missing_data_utils import preprocess_data_for_missing_training
from src.save_utils import print_final_model_parameters
from src.ssvi_config import OilFlowSSVIConfig
from src.ssvi_config_utils import dict_to_dataclass
from ssvi_scripts_experiments.mnist_ood_GPLVM_SSVI_AM_INF import OilFlow_SSVI_GPLVM
from src.pytorch_scalers import MeanCenter, StandardScalerPytorch, MinMaxScalerPytorch, NoTransform


def get_pseudotime():
    """Get pseudotime for qPCR data (same as qPCR_GPLVM_SSVI_AM_INF.py)"""
    # license: Copyright (c) 2014, the Open Data Science Initiative
    # license: https://www.elsevier.com/legal/elsevier-website-terms-and-conditions
    URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
    
    df = pd.read_csv(URL, index_col=0)
    data = torch.tensor(df.values, dtype=torch.float64)
    # we need to transpose data to correct its shape
    y = data.t()
    capture_time = y.new_tensor([int(cell_name.split(" ")[0]) for cell_name in df.index.values])
    # we scale the time into the interval [0, 1]
    time = capture_time.log2() / 6
    return time


def load_config_from_results_dir(results_dir: str) -> OilFlowSSVIConfig:
    """Load configuration from results directory config.json"""
    config_path = os.path.join(results_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dict to dataclass
    config = dict_to_dataclass(config_dict, OilFlowSSVIConfig)
    
    return config


def load_model_from_checkpoint(results_dir: str, model_filename: str = None) -> OilFlow_SSVI_GPLVM:
    """Load model from checkpoint pickle file"""
    models_dir = os.path.join(results_dir, "pre_trained_models")
    
    if model_filename is None:
        # Try to find the final model (not checkpoint)
        # Look for files like qpcr_ssvi_7.pkl
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'iter_' not in f]
        if not model_files:
            # Fall back to checkpoints
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        # Use the first model file found (or could sort by modification time)
        model_filename = model_files[0]
        print(f"Using model file: {model_filename}")
    
    model_path = os.path.join(models_dir, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load model - the file was saved with pickle.dump(), so we need to use pickle.load()
    # We need to handle device mapping for PyTorch tensors inside the model
    # Use pickle.load() with a custom persistent_load to map devices
    
    # Determine target device
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if target_device == 'cpu':
        print("  CUDA not available, will load model to CPU...")
    
    # Monkey-patch torch.serialization.default_restore_location to handle device mapping
    original_restore_location = torch.serialization.default_restore_location
    
    def custom_restore_location(storage, location):
        # Map CUDA devices to target device
        if isinstance(location, torch.device):
            if location.type == 'cuda' and target_device == 'cpu':
                location = torch.device('cpu')
            elif location.type == 'cpu' and target_device == 'cuda' and torch.cuda.is_available():
                location = torch.device('cuda')
        return original_restore_location(storage, location)
    
    # Temporarily replace the restore_location function
    torch.serialization.default_restore_location = custom_restore_location
    
    try:
        # Load with pickle - the custom restore_location will handle device mapping
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    finally:
        # Restore original function
        torch.serialization.default_restore_location = original_restore_location
    
    return model


def load_data_for_plotting(config: OilFlowSSVIConfig):
    """Load and preprocess qPCR data (same as training script)"""
    print("Loading qPCR data...")
    
    from utils.data import load_real_data
    
    # Load data using the same function as Original GP-LVM
    N, d, q, X, Y, labels = load_real_data('gene')
    
    # Get pseudotime for qPCR data
    pseudo_time = get_pseudotime()
    
    # Convert to tensors and move to device
    y_full = Y.to(config.device)
    y_cpu = y_full.cpu() if y_full.is_cuda else y_full
    
    # Handle labels - convert to numpy array if it's a tensor, otherwise keep as is
    if hasattr(labels, 'cpu'):
        labels_cpu = labels.cpu().numpy()
    elif hasattr(labels, 'numpy'):
        labels_cpu = labels.numpy()
    else:
        # If labels is already a list or numpy array, convert to numpy array
        labels_cpu = np.array(labels)
    
    y_np = y_cpu.numpy()
    pseudo_time_full = pseudo_time.to(config.device)
    
    # Create train/test split if metrics enabled
    print(f"Creating train/test split (test_size={config.test_size})...")
    train_indices, test_indices, y_train_np, y_test_np, labels_train_np, labels_test_np, pseudo_train_np, pseudo_test_np = train_test_split(
        np.arange(len(y_full)), y_np, labels_cpu, pseudo_time_full.cpu().numpy(),
        test_size=config.test_size,
        random_state=config.random_seed
    )
    
    # Convert back to tensors on original device
    y_train = torch.tensor(y_train_np, dtype=torch.float64, device=config.device)
    y_test = torch.tensor(y_test_np, dtype=torch.float64, device=config.device)
    labels_train = labels_train_np  # Keep as numpy array for compatibility
    labels_test = labels_test_np    # Keep as numpy array for compatibility
    pseudo_train = torch.tensor(pseudo_train_np, dtype=torch.float64, device=config.device)
    pseudo_test = torch.tensor(pseudo_test_np, dtype=torch.float64, device=config.device)
    
    # Convert indices to tensors for consistency
    train_indices_tensor = torch.tensor(train_indices, dtype=torch.int64, device=config.device)
    test_indices_tensor = torch.tensor(test_indices, dtype=torch.int64, device=config.device)
    
    print(f"Data split shapes - Train: {y_train.shape}, Test: {y_test.shape}")
    
    # === Data Standardization ===
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
            f"Unknown standardization method: {config.standardization}. Must be 'standard', 'minmax', 'mean', or 'none'"
        )
    
    # Fit scaler on training data only
    print(f"Fitting scaler on training data only ({y_train.shape[0]} samples)...")
    scaler.fit(y_train)
    
    # Transform all data
    print(f"Transforming train/test data...")
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    
    print(f"Standardization completed.")
    print("=" * 50)
    
    split_data = {
        'y_train': y_train,
        'y_test': y_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'train_indices': train_indices_tensor,
        'test_indices': test_indices_tensor,
        'pseudo_train': pseudo_train,
        'pseudo_test': pseudo_test,
        'scaler': scaler,
    }
    
    # Apply missing data preprocessing if enabled
    if config.enable_missing_data:
        print(f"\n=== Missing Data Preprocessing ===")
        
        y_train_missing, train_missing_mask, train_missing_info = preprocess_data_for_missing_training(
            split_data['y_train'], config)
        y_test_missing, test_missing_mask, test_missing_info = preprocess_data_for_missing_training(
            split_data['y_test'], config)
        
        split_data.update({
            'y_train_original': split_data['y_train'].clone(),
            'y_test_original': split_data['y_test'].clone(),
            'y_train': y_train_missing,
            'y_test': y_test_missing,
            'train_missing_mask': train_missing_mask,
            'test_missing_mask': test_missing_mask
        })
        
        print(f"Missing data preprocessing completed.")
        print("=" * 50)
    
    return split_data


def load_metrics_from_checkpoint(results_dir: str, config: OilFlowSSVIConfig) -> SSVIMetricsComputer:
    """Load metrics from checkpoint_metrics.json and reconstruct SSVIMetricsComputer"""
    training_results_dir = os.path.join(results_dir, "training_results")
    
    # Try to find checkpoint metrics file
    model_name = "qpcr_ssvi"
    seed = config.random_seed
    metrics_file = os.path.join(training_results_dir, f"{model_name}_{seed}_checkpoint_metrics.json")
    
    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found: {metrics_file}")
        return None
    
    print(f"Loading metrics from: {metrics_file}")
    
    # Handle NaN values in JSON (they might be unquoted NaN which is invalid JSON)
    # Read file as text and replace unquoted NaN with null, then parse
    with open(metrics_file, 'r') as f:
        content = f.read()
        # Replace unquoted NaN with null (valid JSON)
        content = re.sub(r'\bNaN\b', 'null', content)
        metrics_data = json.loads(content)
    
    # Reconstruct SSVIMetricsComputer with metrics history
    metrics_computer = SSVIMetricsComputer(device=config.device, metrics_batch_size=config.batch_size)
    
    # Convert JSON data to SSVIMetrics objects
    for checkpoint in metrics_data['checkpoints']:
        # Handle NaN/null values in JSON
        def safe_float(value):
            if value is None:
                return float('nan')
            if isinstance(value, str) and value.lower() == 'nan':
                return float('nan')
            try:
                return float(value)
            except (ValueError, TypeError):
                return float('nan')
        
        metrics = SSVIMetrics(
            iteration=checkpoint['iteration'],
            train_rmse=safe_float(checkpoint['train_rmse']),
            test_rmse=safe_float(checkpoint['test_rmse']),
            train_nlpd=safe_float(checkpoint['train_nlpd']),
            test_nlpd=safe_float(checkpoint['test_nlpd']),
            train_missing_nlpd=safe_float(checkpoint.get('train_missing_nlpd', None)),
            test_missing_nlpd=safe_float(checkpoint.get('test_missing_nlpd', None)),
            train_missing_rmse=safe_float(checkpoint.get('train_missing_rmse', None)),
            test_missing_rmse=safe_float(checkpoint.get('test_missing_rmse', None)),
            elbo_loss=-safe_float(checkpoint.get('full_dataset_elbo_loss', 0.0)),  # Convert loss back to ELBO
            effective_latent_dims=checkpoint.get('effective_latent_dims', 0),
            total_latent_dims=checkpoint.get('total_latent_dims', config.latent_dim)
        )
        metrics_computer.metrics_history.append(metrics)
    
    print(f"Loaded {len(metrics_computer.metrics_history)} metric checkpoints")
    return metrics_computer


def load_losses_from_checkpoint(results_dir: str, config: OilFlowSSVIConfig):
    """Load training losses from JSON files"""
    training_results_dir = os.path.join(results_dir, "training_results")
    
    model_name = "qpcr_ssvi"
    seed = config.random_seed
    
    # Load full ELBO losses
    full_losses_file = os.path.join(training_results_dir, f"{model_name}_full_loss_{seed}_losses.json")
    full_elbo_losses = []
    full_elbo_iterations = []
    
    if os.path.exists(full_losses_file):
        print(f"Loading full ELBO losses from: {full_losses_file}")
        with open(full_losses_file, 'r') as f:
            losses_data = json.load(f)
        full_elbo_losses = losses_data['losses']
        
        # Reconstruct iterations from metrics if available
        metrics_file = os.path.join(training_results_dir, f"{model_name}_{seed}_checkpoint_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                content = f.read()
                content = re.sub(r'\bNaN\b', 'null', content)
                metrics_data = json.loads(content)
            full_elbo_iterations = [cp['iteration'] for cp in metrics_data['checkpoints']]
        else:
            # Estimate iterations (assuming uniform spacing)
            if len(full_elbo_losses) > 0:
                step = config.training_iterations // (len(full_elbo_losses) - 1) if len(
                    full_elbo_losses) > 1 else config.metrics_frequency
                full_elbo_iterations = [i * step for i in range(len(full_elbo_losses))]
    else:
        print(f"Warning: Full losses file not found: {full_losses_file}")
    
    # Load batch losses
    batch_losses_file = os.path.join(training_results_dir, f"{model_name}_batch_{seed}_losses.json")
    batch_losses = []
    
    if os.path.exists(batch_losses_file):
        print(f"Loading batch losses from: {batch_losses_file}")
        with open(batch_losses_file, 'r') as f:
            losses_data = json.load(f)
        batch_losses = losses_data['losses']
    else:
        print(f"Warning: Batch losses file not found: {batch_losses_file}")
    
    return full_elbo_losses, full_elbo_iterations, batch_losses


def main(results_dir: str, model_filename: str = None):
    """Main function to reload checkpoint and regenerate plots"""
    
    # ============================================================================
    # HARDCODED CONFIGURATION - Modify these as needed
    # ============================================================================
    # Output folder name - will be created in ssvi_scripts_experiments directory
    OUTPUT_FOLDER_NAME = "qpcr_replotting_results"
    # ============================================================================
    
    # Hardcode output dir to create output folder in ssvi_scripts_experiments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_FOLDER_NAME)
    
    print("=" * 70)
    print("qPCR SSVI GP-LVM Results Replotting")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config_from_results_dir(results_dir)
    
    # Set device and dtype from config
    # Handle device setting - torch.set_default_device might not be available in all versions
    try:
        # Normalize device string (handle "auto", "cuda", etc.)
        device_str = config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_str == "cuda" and not torch.cuda.is_available():
            print("  Warning: CUDA requested but not available, using CPU")
            device_str = "cpu"
            config.device = "cpu"  # Update config to reflect actual device
        
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device(device_str)
        else:
            # Fallback for older PyTorch versions
            print(f"  Note: torch.set_default_device not available, using device={device_str} for tensors")
    except Exception as e:
        print(f"  Warning: Could not set default device: {e}")
        # Continue anyway - we'll set device explicitly for tensors
    
    torch.set_default_dtype(torch.float64)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
    
    config.print_config()
    
    # Load model
    print("\n2. Loading model checkpoint...")
    model = load_model_from_checkpoint(results_dir, model_filename)
    
    # Move model to the correct device (from config)
    # Determine target device
    target_device = config.device
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"
        print("  Warning: CUDA requested but not available, using CPU")
    
    print(f"  Moving model to device: {target_device}")
    
    # Since the model inherits from nn.Module (via Struc_WS_GPLVM -> MF_WS_GPLVM),
    # we can use .to() to move all registered parameters and buffers automatically.
    # This is the standard PyTorch way and handles all registered parameters correctly.
    model = model.to(target_device)
    
    # Also ensure the gp submodule is moved (in case it's a separate module)
    if hasattr(model, 'gp'):
        model.gp = model.gp.to(target_device)
    
    # Update model's device attribute if it exists
    if hasattr(model, 'device'):
        model.device = target_device
    
    # Patch get_epsilon to ensure it creates tensors on the correct device
    # The original get_epsilon() uses torch.randn() which might use default device
    original_get_epsilon = model.get_epsilon
    
    def patched_get_epsilon():
        return torch.randn(model.P, model.M, model.C * model.Lx, device=target_device, dtype=torch.float64)
    
    model.get_epsilon = patched_get_epsilon
    
    print(f"Model loaded successfully")
    print(f"  Mu Encoder: {model.mu_encoder}")
    print(f"  Sigma Encoder: {model.sigma_encoder}")
    print(f"  Inducing points: {model.gp.Vx.shape}")
    print(f"  Latent variables: {model.mu_x.shape}")
    
    # Load data
    print("\n3. Loading data...")
    split_data = load_data_for_plotting(config)
    y_train = split_data['y_train']
    labels_train = split_data['labels_train']
    
    # Load metrics
    print("\n4. Loading metrics...")
    metrics_computer = load_metrics_from_checkpoint(results_dir, config)
    
    # Load losses
    print("\n5. Loading losses...")
    full_elbo_losses, full_elbo_iterations, batch_losses = load_losses_from_checkpoint(results_dir, config)
    
    # Initialize logger (pointing to the output directory)
    print("\n6. Initializing logger...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_name = os.path.basename(results_dir).replace(f"_seed_{config.random_seed}_", "").split("_seed_")[0]
    logger = SSVIResultsLogger(config, experiment_name=experiment_name)
    # Override results_dir to use the output directory
    logger.results_dir = output_dir
    logger.plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(logger.plots_dir, exist_ok=True)
    logger.models_dir = os.path.join(output_dir, "pre_trained_models")
    logger.training_results_dir = os.path.join(output_dir, "training_results")
    logger.training_log_path = os.path.join(output_dir, "training_log.txt")
    logger.gradient_log_path = os.path.join(output_dir, "gradient_log.jsonl")
    
    print(f"Results will be saved to: {logger.results_dir}")
    
    # Regenerate all plots
    print("\n7. Regenerating plots...")
    
    # Main visualization plot (qPCR-specific)
    print("  Generating main visualization plot...")
    try:
        from src.qpcr_plot_helper import visualize_ssvi_results as visualize_qpcr_ssvi_results
        visualize_qpcr_ssvi_results(
            model=model,
            y_train=y_train,
            labels_train=labels_train,
            losses=full_elbo_losses,
            config=config,
            logger=logger,
            show_plot=False,
            train_indices=split_data['train_indices'],
            full_elbo_iterations=full_elbo_iterations,
            batch_losses=batch_losses
        )
        print("    Main visualization plot saved")
    except Exception as e:
        print(f"    Warning: Could not create qPCR visualization: {e}")
    
    # Metrics plots (if metrics available)
    if metrics_computer is not None and metrics_computer.metrics_history:
        print("  Generating metrics progression plots...")
        plot_metrics_history(metrics_computer, logger, show_plot=False, batch_losses=batch_losses)
        print("    Metrics progression plots saved")
        
        print("  Generating final metrics summary plots...")
        plot_final_metrics_summary(metrics_computer, model, logger, show_plot=False)
        print("    Final metrics summary plots saved")
        
        # Missing data plots if enabled
        if config.enable_missing_data:
            print("  Generating missing NLPD metrics progression plot...")
            plot_missing_nlpd_metrics_progression(metrics_computer, logger, show_plot=False)
            print("    Missing NLPD metrics progression plot saved")
            
            print("  Generating missing RMSE metrics progression plot...")
            plot_missing_rmse_metrics_progression(metrics_computer, logger, show_plot=False)
            print("    Missing RMSE metrics progression plot saved")
    else:
        print("  Warning: No metrics available, skipping metrics plots")
    
    # Latent slice visualizations
    try:
        print("  Generating latent slice visualizations...")
        
        # Plot slice centered at origin
        latent_slice_path_origin = os.path.join(logger.plots_dir, f"latent_slice_origin_seed_{config.random_seed}.png")
        plot_latent_slice_outer(
            model=model,
            y_train=y_train,
            center="zeros",
            save_path=latent_slice_path_origin,
            scaler=split_data.get('scaler')
        )
        print(f"    Saved latent slice (origin) to: {latent_slice_path_origin}")
        
        # Plot slices centered at a few interesting data points
        N = y_train.shape[0]
        representative_indices = [0, N // 2, N - 1]
        
        for idx in representative_indices:
            latent_slice_path = os.path.join(logger.plots_dir,
                                             f"latent_slice_point_{idx}_seed_{config.random_seed}.png")
            plot_latent_slice_outer(
                model=model,
                y_train=y_train,
                center="data_point",
                n_idx=idx,
                save_path=latent_slice_path,
                scaler=split_data.get('scaler')
            )
            print(f"    Saved latent slice (point {idx}) to: {latent_slice_path}")
    except Exception as e:
        print(f"    Warning: Failed to generate latent slice visualizations: {e}")
    
    # Print final model parameters
    print("\n8. Final model parameters:")
    print_final_model_parameters(model, "GP LVM SSVI")
    
    print("\n" + "=" * 70)
    print("All plots regenerated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # ============================================================================
    # HARDCODED CONFIGURATION - Modify these as needed
    # ============================================================================
    # Hardcoded results folder path
    RESULTS_FOLDER_PATH = "ssvi_scripts_experiments/qpcr_results/ssvi_results/ssvi_fast_seed_7_18_01_16_07"
    # ============================================================================
    
    # Convert to absolute path if relative
    results_dir = RESULTS_FOLDER_PATH
    if not os.path.isabs(results_dir):
        # Try relative to current working directory first
        if not os.path.exists(results_dir):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(project_root, RESULTS_FOLDER_PATH)
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    print(f"Using results directory: {results_dir}")
    
    # Auto-detect model filename (use final model, not checkpoint)
    model_filename = None
    
    main(results_dir, model_filename)

