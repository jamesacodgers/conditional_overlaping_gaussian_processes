"""
Shared utilities for entry-wise cross-validation experiments on missing data.

This module contains common functions used across GPLVM, SSVI GPLVM, and KNN
cross-validation scripts for missing data imputation.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


class StandardScalerPytorch:
    """Standard Scaler for PyTorch tensors that handles NaN values."""

    def __init__(self, device="cpu"):
        self.device = device

    def fit(self, X):
        self.mean = torch.nanmean(X, dim=0)
        self.std = np.nanstd(X.cpu().numpy(), axis=0, ddof=0)
        self.std = torch.tensor(self.std, dtype=torch.float64, device=self.device)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# Score columns used in all CV experiments
SCORE_COLUMNS = [
    "Sort1_mean_score",
    "Sort2_mean_score",
    "Sort3_mean_score",
    "Sort4_mean_score",
    "Sort5_mean_score",
    "Sort6_mean_score",
    "Sort7_mean_score",
    "Sort8_mean_score",
    "Sort9_mean_score",
    "Sort10_mean_score",
]


def load_data(data_path=None):
    """Load the protein assay data.

    Args:
        data_path: Path to CSV file. If None, uses default location.

    Returns:
        Tuple of (data DataFrame, numpy array of scores)
    """
    if data_path is None:
        data_path = Path(__file__).parent / "assay_to_yield_training_sequences.csv"

    data = pd.read_csv(data_path)
    data_np = data[SCORE_COLUMNS].to_numpy().astype(float)

    return data, data_np


def _to_torch(x):
    """Convert numpy array to torch tensor if needed."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def compute_rmse(predictions, targets, mask):
    """Compute RMSE on masked entries.

    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: True values (numpy array or torch tensor)
        mask: Boolean mask where True = compute metric (i.e., held-out entries)

    Returns:
        Dict with 'rmse_by_dim' and 'total_rmse'
    """
    predictions = _to_torch(predictions)
    targets = _to_torch(targets)
    mask = _to_torch(mask)

    squared_errors = (predictions - targets) ** 2
    squared_errors_masked = squared_errors.clone()
    squared_errors_masked[~mask] = float("nan")
    rmse_by_dim = squared_errors_masked.nanmean(dim=0).sqrt()
    total_rmse = rmse_by_dim.nanmean().item()
    rmse_by_dim = (
        rmse_by_dim.cpu().numpy() if rmse_by_dim.is_cuda else rmse_by_dim.numpy()
    )

    return {"rmse_by_dim": rmse_by_dim, "total_rmse": total_rmse}


def compute_nlpd(predictions, targets, variances, mask):
    """Compute Negative Log Predictive Density on masked entries.

    NLPD = 0.5 * log(2 * pi * var) + 0.5 * (error^2 / var)

    Args:
        predictions: Predicted means (numpy array or torch tensor)
        targets: True values (numpy array or torch tensor)
        variances: Predicted variances (numpy array or torch tensor)
        mask: Boolean mask where True = compute metric (i.e., held-out entries)

    Returns:
        Dict with 'nlpd_by_dim' and 'total_nlpd'
    """
    predictions = _to_torch(predictions)
    targets = _to_torch(targets)
    variances = _to_torch(variances)
    mask = _to_torch(mask)

    nlpd = (
        0.5 * torch.log(2 * np.pi * variances)
        + 0.5 * ((targets - predictions) ** 2) / variances
    )
    nlpd_masked = nlpd.clone()
    nlpd_masked[~mask] = float("nan")
    nlpd_by_dim = nlpd_masked.nanmean(dim=0)
    total_nlpd = nlpd_by_dim.nanmean().item()
    nlpd_by_dim = (
        nlpd_by_dim.cpu().numpy() if nlpd_by_dim.is_cuda else nlpd_by_dim.numpy()
    )

    return {"nlpd_by_dim": nlpd_by_dim, "total_nlpd": total_nlpd}


def compute_metrics(predictions, targets, variances, mask):
    """Compute both RMSE and NLPD metrics.

    Args:
        predictions: Predicted means
        targets: True values
        variances: Predicted variances (can be None, in which case only RMSE is computed)
        mask: Boolean mask where True = compute metric (i.e., held-out entries)

    Returns:
        Dict with 'rmse_by_dim', 'total_rmse', and optionally 'nlpd_by_dim', 'total_nlpd'
    """
    results = compute_rmse(predictions, targets, mask)

    if variances is not None:
        nlpd_results = compute_nlpd(predictions, targets, variances, mask)
        results.update(nlpd_results)

    return results


def save_cv_results(
    all_results, all_results_dict, output_dir, filename, include_nlpd=True
):
    """Save cross-validation results to CSV.

    Args:
        all_results: List of dicts with fold results
        output_dir: Directory to save results
        filename: Name of output file
        include_nlpd: Whether to include NLPD columns

    Returns:
        DataFrame of results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results.to_csv(output_dir / filename, index=False)

    # also dump the full results dict
    full_results_file = output_dir / (filename.split(".")[0] + ".pkl")
    with open(full_results_file, "wb") as f:
        pickle.dump(all_results_dict, f)

    return all_results


def print_cv_header(data_shape, **kwargs):
    """Print standard CV experiment header.

    Args:
        data_shape: Shape of data array
        **kwargs: Additional parameters to print
    """
    print(f"Data shape: {data_shape}")
    for key, value in kwargs.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 60)


def print_fold_header(fold, n_folds, fold_indices):
    """Print standard fold header.

    Args:
        fold: Current fold number (0-indexed)
        n_folds: Total number of folds
        fold_indices: Array of fold indices
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{n_folds}")
    print(f"{'='*60}")

    n_test = (fold_indices == fold).sum()
    n_total = fold_indices.size
    print(f"Test entries: {n_test}, Train entries: {n_total - n_test}")


def print_fold_results(fold, results, include_nlpd=True):
    """Print results for a single fold.

    Args:
        fold: Fold number (0-indexed)
        results: Dict with 'total_rmse' and optionally 'total_nlpd'
        include_nlpd: Whether to print NLPD
    """
    print(f"\nFold {fold + 1} Results:")
    print(f"  Total RMSE: {results['total_rmse_test']:.4f}")
    if include_nlpd and "total_nlpd" in results:
        print(f"  Total NLPD: {results['total_nlpd']:.4f}")


def print_cv_summary(all_results, include_nlpd=True):
    """Print cross-validation summary.

    Args:
        all_results: List of dicts with fold results
        include_nlpd: Whether to print NLPD summary
    """
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    rmse_values = [r["rmse"] for r in all_results]
    print(f"\nRMSE across folds: {rmse_values}")
    print(f"Mean RMSE: {np.mean(rmse_values):.4f} (+/- {np.std(rmse_values):.4f})")

    if include_nlpd and "nlpd" in all_results[0]:
        nlpd_values = [r["nlpd"] for r in all_results]
        print(f"\nNLPD across folds: {nlpd_values}")
        print(f"Mean NLPD: {np.mean(nlpd_values):.4f} (+/- {np.std(nlpd_values):.4f})")


def plot_inverse_lengthscales(lengthscales, output_path, title="Inverse Lengthscales"):
    """Plot inverse lengthscales as a bar chart.

    Args:
        lengthscales: Array of lengthscales (shape: [latent_dim] or [n_outputs, latent_dim])
        output_path: Path to save the figure
        title: Plot title
    """
    lengthscales = np.atleast_1d(lengthscales)
    if lengthscales.ndim == 1:
        inv_lengthscales = 1.0 / lengthscales
    else:
        inv_lengthscales = 1.0 / lengthscales.mean(axis=0)

    latent_dim = len(inv_lengthscales)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(latent_dim)
    bars = ax.bar(x, inv_lengthscales, color="steelblue", edgecolor="black")

    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel("Inverse Lengthscale (1/l)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in range(latent_dim)])

    for bar, val in zip(bars, inv_lengthscales):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_space(
    latent_coords,
    output_path,
    title="Latent Space",
    dim1=0,
    dim2=1,
    color_values=None,
    color_label=None,
):
    """Plot 2D projection of latent space.

    Args:
        latent_coords: Array of latent coordinates (shape: [n_samples, latent_dim])
        output_path: Path to save the figure
        title: Plot title
        dim1: First dimension to plot (default: 0)
        dim2: Second dimension to plot (default: 1)
        color_values: Optional array of values to color points by (default: sample index)
        color_label: Optional label for colorbar (default: 'Sample Index' or inferred)
    """
    latent_coords = np.atleast_2d(latent_coords)

    fig, ax = plt.subplots(figsize=(8, 6))

    if color_values is None:
        color_values = np.arange(len(latent_coords))
        if color_label is None:
            color_label = "Sample Index"
    elif len(color_values) != len(latent_coords):
        # Fall back to sample index if sizes don't match
        color_values = np.arange(len(latent_coords))
        if color_label is None:
            color_label = "Sample Index"
    elif color_label is None:
        color_label = "Color"

    scatter = ax.scatter(
        latent_coords[:, dim1],
        latent_coords[:, dim2],
        c=color_values,
        cmap="viridis",
        alpha=0.7,
        s=30,
        edgecolors="white",
        linewidths=0.5,
    )

    ax.set_xlabel(f"Latent Dimension {dim1 + 1}", fontsize=12)
    ax.set_ylabel(f"Latent Dimension {dim2 + 1}", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def ensure_figures_dir(base_dir, seed):
    """Create and return the figures directory path.

    Args:
        base_dir: Base directory (e.g., gplvm_missing_data)
        seed: Random seed used for the experiment

    Returns:
        Path to figures directory
    """
    figures_dir = Path(base_dir) / "figures" / f"seed{seed}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir
