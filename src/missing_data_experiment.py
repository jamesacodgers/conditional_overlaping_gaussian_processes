"""
Missing Data Reconstruction Experiment

This module contains the main logic for running missing data reconstruction experiments
that are shared between different GP-LVM model implementations.
"""

from src.missing_data_utils import evaluate_reconstruction_quality
from src.save_utils import (
    print_missing_data_experiment_header, 
    print_missing_data_results, 
    print_missing_data_experiment_footer,
    _save_missing_data_reconstruction_json
)


def run_missing_data_reconstruction_experiment(model, config, y, split_data, logger, model_type="GP LVM", model_description="Gaussian Process Latent Variable Model"):
    """
    Run comprehensive missing data reconstruction experiment following paper methodology.
    
    This function coordinates the entire missing data reconstruction experiment including:
    - Performing reconstruction on training and test data
    - Evaluating reconstruction quality metrics
    - Printing comprehensive results in paper format
    - Saving results to JSON and visualizations
    
    Args:
        model: Trained model with missing data capability
        config: Configuration object with missing data settings
        y: Training data
        split_data: Dictionary with train/test splits (optional)
        logger: Logger object for saving results
        model_type: Model type name for display (e.g. "GP LVM", "GP LVM SSVI")
        model_description: Detailed model description
    """
    # Print experiment header
    print_missing_data_experiment_header(model_type)
    
    # Perform training data reconstruction
    print("Performing comprehensive missing data reconstruction...")
    y_reconstructed, recon_std = model.reconstruct_missing_data(
        y, config.missing_mask, n_samples=config.reconstruction_samples
    )
    
    # Evaluate training reconstruction quality
    recon_metrics = evaluate_reconstruction_quality(
        config.y_original_train, y_reconstructed, config.missing_mask, recon_std
    )
    
    # Perform test data reconstruction if available
    test_recon_metrics = None
    y_test_reconstructed = None
    if split_data is not None and 'y_test_original' in split_data:
        y_test_reconstructed, test_recon_std = model.reconstruct_missing_data(
            split_data['y_test'], split_data['test_missing_mask'], 
            n_samples=config.reconstruction_samples
        )
        
        test_recon_metrics = evaluate_reconstruction_quality(
            split_data['y_test_original'], y_test_reconstructed, 
            split_data['test_missing_mask'], test_recon_std
        )
    
    # Print comprehensive results
    quality_assessment = print_missing_data_results(
        model_type, model_description, config, y, recon_metrics, test_recon_metrics, split_data
    )
    
    # Save JSON results if test data available
    if test_recon_metrics is not None and logger:
        _save_missing_data_reconstruction_json(
            model_type, model_description, config, y, split_data, 
            recon_metrics, test_recon_metrics, quality_assessment, logger
        )
    
    # Print experiment footer
    print_missing_data_experiment_footer(model_type)
