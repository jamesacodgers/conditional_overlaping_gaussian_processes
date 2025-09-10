import pandas as pd
import torch
from pathlib import Path
import os
import json
import datetime

def save_results_csv(file_name, path, results_dict):
    """save results in csv file"""

    results_df = pd.DataFrame(results_dict, index = [0])

    if not path.exists():
        os.makedirs(path.parent, exist_ok = True)
        results_df.to_csv(path, index = False)
    else:
        df = pd.read_csv(path)
        df = pd.concat([df, results_df])
        df.to_csv(path, index = False)


    print(f"results saved in {path}")




def save_model(path, experiment_name, seed, data_idx, model):
    """save model parameter gradients"""

    os.makedirs(path, exist_ok = True)

    file_name = f'{experiment_name}_rs={seed}_data_idx={data_idx}.pt'
    torch.save(model.state_dict(), path / file_name)
    print(f"model saved in {path}")

    

def save_elbos(path, file_appendix, elbos):
    """save elbos in csv file"""

    os.makedirs(path, exist_ok = True)

    torch.save(elbos, path / f'elbo_{file_appendix}')


def save_paper_style_metrics(model, split_data, losses, logger, config, metrics_computer, final_metrics, model_type="GP LVM", model_description="Gaussian Process Latent Variable Model"):
    """Save standard GP LVM paper metrics in JSON format for comparison
    
    Args:
        model: Trained model
        split_data: Dictionary with train/test data splits
        losses: List of training losses
        logger: Logger object with results_dir
        config: Configuration object
        metrics_computer: Metrics computer object
        final_metrics: Final metrics dictionary
        model_type: Model type name (e.g. "GP LVM", "GP LVM SSVI")
        model_description: Detailed model description
    """
    # Get final metrics summary if available
    metrics_summary = metrics_computer.get_metrics_summary() if metrics_computer else {}
    
    # Create comprehensive results for paper comparison
    paper_metrics = {
        "experiment_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_type": model_type,
            "model_description": model_description,
            "dataset": "Oil Flow",
            "paper_reference": "Generalised GPLVM with SVI",
            "missing_data_enabled": getattr(config, 'enable_missing_data', False)
        },
        "data_info": {
            "train_samples": int(split_data['y_train'].shape[0]),
            "test_samples": int(split_data['y_test'].shape[0]),
            "data_dimensions": int(split_data['y_train'].shape[1]),
            "total_samples": int(getattr(config, 'total_size', split_data['y_train'].shape[0] + split_data['y_test'].shape[0]))
        },
        "hyperparameters": {
            "latent_dim": int(config.latent_dim),
            "inducing_points": int(getattr(config, 'n_inducing_points', 0)),
            "P_samples": int(config.P),
            "training_iterations": int(config.training_iterations),
            "learning_rate": float(config.learning_rate)
        },
        "training_results": {
            "final_loss": float(losses[-1]) if losses else 0.0,
            "convergence_iterations": len(losses) if losses else 0
        },
        "paper_metrics": {
            "train_rmse": float(metrics_summary.get('final_train_rmse', 0.0)),
            "test_rmse": float(metrics_summary.get('final_test_rmse', 0.0)),
            "train_nlpd": float(metrics_summary.get('final_train_nlpd', 0.0)),
            "test_nlpd": float(metrics_summary.get('final_test_nlpd', 0.0)),
            "effective_latent_dims": int(metrics_summary.get('final_effective_dims', config.latent_dim)),
            "total_latent_dims": int(config.latent_dim),
            "overfitting_ratio": float(metrics_summary.get('overfitting_ratio', 1.0))
        },
        "paper_style_summary": {
            "dataset": "Oil Flow",
            "train_rmse": float(metrics_summary.get('final_train_rmse', 0.0)),
            "test_rmse": float(metrics_summary.get('final_test_rmse', 0.0)),
            "model_quality": "Excellent" if metrics_summary.get('final_test_rmse', 1.0) < 0.1 else 
                           "Good" if metrics_summary.get('final_test_rmse', 1.0) < 0.3 else "Fair"
        }
    }
    
    # Add missing data specific metrics if available
    if getattr(config, 'enable_missing_data', False) and hasattr(config, 'missing_info'):
        paper_metrics["missing_data_info"] = {
            "missing_pattern": config.missing_info.get('pattern_type', 'unknown'),
            "missing_percentage": float(config.missing_info.get('missing_percentage', 0.0)),
            "total_elements": int(config.missing_info.get('total_elements', 0)),
            "missing_elements": int(config.missing_info.get('n_missing', 0))
        }
    
    # Add batch size for SSVI models (if available)
    if hasattr(config, 'batch_size'):
        paper_metrics["hyperparameters"]["batch_size"] = int(config.batch_size)
    
    # Save to JSON  
    json_path = os.path.join(logger.results_dir, "paper_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(paper_metrics, f, indent=2)
    
    print(f"Paper-style metrics saved to: {json_path}")


def print_final_model_parameters(model, model_type="GP LVM"):
    """Print final model parameters in a standardized format
    
    Args:
        model: Trained model
        model_type: Type of model for display purposes
    """
    print(f"\nFinal {model_type} Model Parameters:")
    print(f"  Observation noise (sigma2): {model.sigma2:.4f}")
    if hasattr(model, 'gp'):
        print(f"  Signal variances (sigma2_s): {model.gp.sigma2_s}")
        print(f"  Lengthscales (1/beta): {(1/model.gp.beta).flatten()}")


def print_missing_data_experiment_header(model_type):
    """Print header for missing data reconstruction experiment"""
    print(f"\n=== Missing Data Reconstruction Experiment ===")
    print("Following methodology from 'Generalised GPLVM with SVI' paper")
    if "SSVI" in model_type:
        print("Using amortized inference for missing data reconstruction")
    else:
        print("Using separate latent variable optimization for missing data reconstruction")


def print_missing_data_results(model_type, model_description, config, y, recon_metrics, test_recon_metrics=None, split_data=None):
    """Print comprehensive missing data reconstruction results following paper format"""
    print(f"\n" + "="*80)
    print(f"MISSING DATA RECONSTRUCTION EXPERIMENT RESULTS")
    print(f"Following 'Generalised GPLVM with SVI' Paper Methodology")
    print(f"="*80)
    
    print(f"\nExperimental Setup:")
    print(f"  Model: {model_type} ({model_description})")
    print(f"  Dataset: Oil Flow ({y.shape[0]} samples, {y.shape[1]} dimensions)")
    print(f"  Missing Pattern: {config.missing_info['pattern_type']}")
    print(f"  Missing Percentage: {config.missing_info['missing_percentage']:.1f}%")
    print(f"  Missing Elements: {recon_metrics['n_missing']}/{config.missing_info['total_elements']}")
    print(f"  Reconstruction Samples: {config.reconstruction_samples}")
    
    print(f"\nTraining Data Reconstruction Quality:")
    print(f"  RMSE (missing dimensions):  {recon_metrics['rmse']:.4f}")
    print(f"  MAE (missing dimensions):   {recon_metrics['mae']:.4f}")
    print(f"  Relative Error:             {recon_metrics['relative_error']:.4f}")
    print(f"  Mean Uncertainty (sigma):   {recon_metrics['mean_uncertainty']:.4f}")
    
    # Test set results if available
    if test_recon_metrics is not None and split_data is not None:
        print(f"\nTest Data Reconstruction Quality:")
        print(f"  Test Set Size: {split_data['y_test'].shape[0]} samples")
        print(f"  RMSE (missing dimensions):  {test_recon_metrics['rmse']:.4f}")
        print(f"  MAE (missing dimensions):   {test_recon_metrics['mae']:.4f}")
        print(f"  Relative Error:             {test_recon_metrics['relative_error']:.4f}")
        print(f"  Mean Uncertainty (sigma):   {test_recon_metrics['mean_uncertainty']:.4f}")
        
        # Paper-style summary table
        print(f"\n" + "-"*60)
        print(f"PAPER-STYLE RECONSTRUCTION SUMMARY")
        print(f"-"*60)
        print(f"{'Dataset':<15} {'Missing%':<10} {'Train RMSE':<12} {'Test RMSE':<12}")
        print(f"-"*60)
        print(f"{'Oil Flow':<15} {config.missing_info['missing_percentage']:<10.1f} {recon_metrics['rmse']:<12.4f} {test_recon_metrics['rmse']:<12.4f}")
        print(f"-"*60)
        
        # Quality assessment
        quality_assessment = "Excellent" if test_recon_metrics['rmse'] < 0.1 else \
                           "Good" if test_recon_metrics['rmse'] < 0.3 else \
                           "Fair" if test_recon_metrics['rmse'] < 0.5 else "Poor"
        
        print(f"\nReconstruction Quality Assessment: {quality_assessment}")
        print(f"(Based on Test RMSE: {test_recon_metrics['rmse']:.4f})")
        
        return quality_assessment
    else:
        print(f"\nNo test data available for evaluation.")
        quality_assessment = "Excellent" if recon_metrics['rmse'] < 0.1 else \
                           "Good" if recon_metrics['rmse'] < 0.3 else "Fair"
        print(f"Training reconstruction quality: {quality_assessment}")
        return quality_assessment


def print_missing_data_experiment_footer(model_type):
    """Print footer for missing data reconstruction experiment"""
    print(f"\nMissing data reconstruction experiment completed.")
    if "SSVI" in model_type:
        print("Results demonstrate model's ability to recover missing dimensions")
        print("using amortized inference and GP-based reconstruction.")
    else:
        print("Results demonstrate GP-LVM's ability to recover missing dimensions")
        print("using separate latent variable optimization for each data point.")
    print("=" * 50)


def save_missing_data_visualizations(model_type, config, y_reconstructed, test_reconstructed=None, split_data=None, logger=None):
    """Save missing data reconstruction visualizations"""
    from src.missing_data_utils import visualize_reconstruction_results
    
    if logger:
        print("\nSaving reconstruction visualizations...")
        # Training data visualization
        visualize_reconstruction_results(
            config.y_original, config.y_original, y_reconstructed, config.missing_mask,
            save_path=os.path.join(logger.results_dir, "missing_data_reconstruction_train.png")
        )
        
        if test_reconstructed is not None and split_data is not None:
            # Test data visualization
            visualize_reconstruction_results(
                split_data['y_test_original'], split_data['y_test'], test_reconstructed, split_data['test_missing_mask'],
                save_path=os.path.join(logger.results_dir, "missing_data_reconstruction_test.png")
            )


def _save_missing_data_reconstruction_json(model_type, model_description, config, y, split_data, recon_metrics, test_recon_metrics, quality_assessment, logger):
    """Helper function to save missing data reconstruction results to JSON"""
    json_results = {
        "experiment_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_type": model_type,
            "model_description": model_description,
            "dataset": "Oil Flow",
            "paper_reference": "Generalised GPLVM with SVI"
        },
        "data_info": {
            "train_samples": int(y.shape[0]),
            "test_samples": int(split_data['y_test'].shape[0]) if split_data else 0,
            "data_dimensions": int(y.shape[1]),
            "missing_pattern": config.missing_info['pattern_type'],
            "missing_percentage": float(config.missing_info['missing_percentage']),
            "total_elements": int(config.missing_info['total_elements']),
            "missing_elements": int(config.missing_info['n_missing'])
        },
        "hyperparameters": {
            "reconstruction_samples": int(config.reconstruction_samples),
            "latent_dim": int(config.latent_dim),
            "inducing_points": int(getattr(config, 'n_inducing_points', 0)),
            "P_samples": int(config.P)
        },
        "reconstruction_metrics": {
            "train": {
                "rmse": float(recon_metrics['rmse']),
                "mae": float(recon_metrics['mae']),
                "relative_error": float(recon_metrics['relative_error']),
                "mean_uncertainty": float(recon_metrics['mean_uncertainty'])
            },
            "test": {
                "rmse": float(test_recon_metrics['rmse']),
                "mae": float(test_recon_metrics['mae']),
                "relative_error": float(test_recon_metrics['relative_error']),
                "mean_uncertainty": float(test_recon_metrics['mean_uncertainty'])
            }
        },
        "paper_style_summary": {
            "dataset": "Oil Flow",
            "missing_percentage": float(config.missing_info['missing_percentage']),
            "train_rmse": float(recon_metrics['rmse']),
            "test_rmse": float(test_recon_metrics['rmse']),
            "quality_assessment": quality_assessment
        }
    }
    
    json_path = os.path.join(logger.results_dir, "missing_data_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Missing data results saved to: {json_path}")    
