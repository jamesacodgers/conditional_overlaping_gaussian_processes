"""
SSVI GPLVM for Missing Data with Entry-wise Cross Validation

This script trains an SSVI GPLVM on protein developability assay data,
using entry-wise cross validation to evaluate missing data imputation performance.
"""

import os
import numpy as np
import torch
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.ssvi_metrics import SSVIMetricsComputer
from src.ssvi_utils import SSVIResultsLogger


from src.knn_clustering_metrics import compute_knn_accuracies_ssvi
from src.mnist_missing_reconstruction import (
    visualize_prior_samples,
    visualize_prior_samples_averaged,
    visualize_test_image_predictions,
)
from ssvi_scripts_experiments.mnist_ood_GPLVM_SSVI_AM_INF import (
    OilFlow_SSVI_GPLVM,
    OODSSVIConfig,
)
import time
from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF

from cv_utils import (
    load_data,
    compute_metrics,
    save_cv_results,
    print_cv_header,
    print_fold_header,
    print_fold_results,
    StandardScalerPytorch,
    plot_inverse_lengthscales,
    plot_latent_space,
    ensure_figures_dir,
)


def generate_random_missing_mask(
    data: torch.Tensor, missing_ratio: float, device: str = None
) -> torch.Tensor:
    """
    Generate a random missing data mask for the given data.
    Uses numpy random number generation (already seeded globally) for reproducibility.

    Args:
        data: Data tensor [N, M]
        missing_ratio: Fraction of dimensions to mask per data point
        device: Device for the mask tensor

    Returns:
        missing_mask: Boolean mask [N, M] where True=missing
    """
    if device is None:
        device = data.device

    N, M = data.shape

    missing_mask = torch.rand(N, M, device=device) < missing_ratio

    return missing_mask


class OilFlow_SSVI_GPLVM(GPLVM_SSVI_AM_INF):
    """MNIST/FashionMNIST SSVI GP-LVM with Amortized Inference"""

    def __init__(self, config: OODSSVIConfig, Vx, mu_x, M):
        # Use init_lengthscales to create beta values
        beta = [[config.init_lengthscales] * config.latent_dim] * 1  # C=1 for GP-LVM

        # Use init_output_noise_scale for sigma2 initialization
        sigma2 = config.init_output_noise_scale

        # Initialize underlying SSVI model
        super().__init__(
            Vx=Vx,
            M=M,  # Observation dimension
            mu_x=mu_x,
            sigma2=sigma2,
            beta=beta,
            sigma2_s=config.sigma2_s,
            P=config.P,
            trainable_V=config.trainable_V,
            encoder_hidden_dims=config.encoder_hidden_dims,
            local_lr=config.local_lr,
            global_lr=config.global_lr,
            device=config.device,
            enable_missing_data=config.enable_missing_data,
        )

        # Store config
        self.config = config

    def train_step(
        self,
        Y_batch: torch.Tensor,
        batch_indices: torch.Tensor,
        prior_weight: float,
        U_samples: torch.Tensor = None,
        iteration: int = None,
        missing_mask: torch.Tensor = None,
    ):
        """
        Modified train_step: proper SSVI separation of local/global optimization
        Uses provided missing mask if available.

        Args:
            iteration: Current training iteration (for reproducible mask generation)
            missing_mask: Optional missing data mask for this batch [batch_size, M]
        """
        # Zero gradients for local optimizer (encoder) only initially
        self.optimizer_local.zero_grad()

        # Sample U before amortized inference step
        epsilon = self.get_epsilon()
        U_samples = self.get_u_samples(epsilon)

        # Get local parameters via amortized inference (encoder handles NaN replacement internally)
        mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
            Y_batch, epsilon, missing_mask=missing_mask
        )

        # Now zero gradients for global optimizer before global computation
        self.optimizer_global.zero_grad()

        # Compute global ELBO loss (gradients flow to all parameters)
        loss = -self.ssvi_elbo(
            Y_batch,
            mu_n_batch,
            sigma_n_batch,
            U_samples=U_samples,
            batch_size=self.config.batch_size,
            total_size=self.config.total_size,
            missing_mask=missing_mask,
            prior_weight=prior_weight,
        )

        # Backward pass (gradients computed for all parameters)
        loss.backward()

        # Update parameters with different learning rates
        self.optimizer_local.step()
        self.optimizer_global.step()

        return loss.item()

    @property
    def mu_x(self):
        return self.mu_x_train

    @property
    def Sigma_x(self):
        return self.Sigma_x_train

    @torch.no_grad()
    def compute_full_elbo(
        self, y_full: torch.Tensor, missing_mask: torch.Tensor = None
    ) -> float:
        """
        Compute ELBO on full dataset using minibatches (for metrics and plots, not optimization).
        Processes data in minibatches to reduce memory usage, using the same parameters as training.
        Uses provided missing mask if available.

        Args:
            y_full: Full dataset [N, M] (original data without missing values)
            missing_mask: Optional missing data mask [N, M] (if provided, uses this mask; otherwise generates new mask per batch)

        Returns:
            Full dataset ELBO value (scalar)
        """
        # Sample U for ELBO computation (same U for all batches)
        epsilon = self.get_epsilon()
        U_samples = self.get_u_samples(epsilon)

        # Use same batch size as training
        batch_size = self.config.batch_size
        N = y_full.shape[0]

        # Process data in minibatches and collect ELBOs (same as training loop)
        batch_elbos = []

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            y_batch = y_full[
                start_idx:end_idx
            ].clone()  # Clone to avoid modifying original

            # Use provided mask or generate new one for this batch
            mask_batch = None
            if self.enable_missing_data:
                if missing_mask is not None:
                    # Use provided mask (slice for this batch)
                    mask_batch = missing_mask[start_idx:end_idx]
                else:
                    # Generate new random mask for this batch (fallback if no mask provided)
                    mask_batch = generate_random_missing_mask(
                        y_batch,
                        missing_ratio=self.config.missing_data_ratio,
                        device=y_batch.device,
                    )
                # Apply mask: set missing values to NaN
                y_batch[mask_batch] = float("nan")

            # Get latent parameters via amortized inference for this batch
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                y_batch, epsilon, missing_mask=mask_batch
            )

            # Compute ELBO for this batch (EXACTLY same as in training: batch_size=config.batch_size, total_size=config.total_size)
            batch_elbo = self.ssvi_elbo(
                y_batch,
                mu_n_batch,
                sigma_n_batch,
                U_samples,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                missing_mask=mask_batch,
                prior_weight=1,
            )

            batch_elbos.append(batch_elbo)

        # Average the batch ELBOs (same approach as would be used in training)
        mean_elbo = sum(batch_elbos) / len(batch_elbos)

        return mean_elbo.item()


def train_oil_flow_ssvi(
    model: OilFlow_SSVI_GPLVM,
    y_train,
    config: OODSSVIConfig,
    logger: SSVIResultsLogger = None,
    split_data=None,
    labels_train=None,
):
    """Training function for SSVI GP-LVM with optional logging and metrics"""
    losses = []  # Batch ELBO losses (for training monitoring)
    full_elbo_losses = []  # Full dataset ELBO (for metrics and plots)
    full_elbo_iterations = []  # Iterations at which full ELBO was computed

    print(f"Training {config.id_dataset} SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")

    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(
            f"Metrics computation enabled (frequency: every {config.metrics_frequency} iterations)"
        )
        metrics_computer = SSVIMetricsComputer(
            device=config.device, metrics_batch_size=config.batch_size
        )

    # Initialize k-NN metrics tracking
    knn_metrics_history = []
    knn_metrics_iterations = []

    start_time = time.time()
    init_prior_weight = 10
    for i in range(config.training_iterations):
        # Generate random missing masks for this iteration (same mask used for training step, full ELBO, and metrics)
        train_missing_mask = None
        test_missing_mask = None
        if config.enable_missing_data:
            train_missing_mask = generate_random_missing_mask(
                split_data["y_train_original"],
                missing_ratio=config.missing_data_ratio,
                device=config.device,
            )
            test_missing_mask = generate_random_missing_mask(
                split_data["y_test_original"],
                missing_ratio=config.missing_data_ratio,
                device=config.device,
            )

        # Sample batch indices
        N = y_train.shape[0]
        batch_indices = torch.randperm(N, device=y_train.device)[: config.batch_size]
        y_batch = y_train[batch_indices]

        # Get batch mask if missing data is enabled
        batch_missing_mask = None
        if config.enable_missing_data and train_missing_mask is not None:
            batch_missing_mask = train_missing_mask[batch_indices]
            # Apply mask to batch
            y_batch = y_batch.clone()
            y_batch[batch_missing_mask] = float("nan")

        frac = i / config.training_iterations
        weight = (1 - frac) ** 2
        # Training step - U sampling happens inside automatically each step
        # prior_weight = init_prior_weight
        # prior_weight = init_prior_weight + (i/config.training_iterations)*(1-init_prior_weight)
        # prior_weight = (init_prior_weight-1)/2 + (init_prior_weight-1)/2*torch.cos(torch.tensor(torch.pi*i/config.training_iterations)) + 1
        loss = model.train_step(
            y_batch,
            batch_indices,
            prior_weight=weight * init_prior_weight + (1 - weight),
            iteration=i,
            missing_mask=batch_missing_mask,
        )
        losses.append(loss)

        # Metrics computation (periodic)
        if (
            metrics_computer is not None
            and config.enable_metrics
            and i % config.metrics_frequency == 0
        ):

            print(f"  Computing metrics at iteration {i}...")

            # Compute full dataset loss (negative ELBO, for metrics and plots)
            # Use the same mask generated at the start of this iteration
            full_elbo = model.compute_full_elbo(
                split_data["y_train_original"], missing_mask=train_missing_mask
            )

            # Store as negative ELBO (loss) for consistency with batch losses
            full_elbo_losses.append(-full_elbo)
            full_elbo_iterations.append(i)
            print(f"    Full Dataset Loss: {-full_elbo:.4f} (batch loss: {loss:.4f})")

            # Use full datasets for proper evaluation
            # Use the same masks generated at the start of this iteration
            if config.enable_missing_data:
                # Create masked versions for metrics computation
                y_train_masked = split_data["y_train_original"].clone()
                y_train_masked[train_missing_mask] = float("nan")
                y_test_masked = split_data["y_test_original"].clone()
                y_test_masked[test_missing_mask] = float("nan")
            else:
                y_train_masked = split_data["y_train_original"]
                y_test_masked = split_data["y_test_original"]

            # Compute metrics with missing data info if enabled
            if config.enable_missing_data:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=y_train_masked,
                    y_test=y_test_masked,
                    iteration=i,
                    elbo_loss=full_elbo,  # Use full dataset ELBO
                    n_samples=32,  # Fixed number of samples for metrics computation
                    y_train_original=split_data["y_train_original"],
                    y_test_original=split_data["y_test_original"],
                    train_missing_mask=train_missing_mask,
                    test_missing_mask=test_missing_mask,
                )
            else:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=y_train_masked,
                    y_test=y_test_masked,
                    iteration=i,
                    elbo_loss=full_elbo,  # Use full dataset ELBO
                    n_samples=32,  # Fixed number of samples for metrics computation
                )

            print(
                f"    Train RMSE: {metrics.train_rmse:.4f}, Test RMSE: {metrics.test_rmse:.4f} "
            )
            print(
                f"    Train NLPD: {metrics.train_nlpd:.3f}, Test NLPD: {metrics.test_nlpd:.3f}"
            )
            if config.enable_missing_data and not np.isnan(metrics.train_missing_nlpd):
                print(f"    Train Missing Data NLPD: {metrics.train_missing_nlpd:.3f}")
                print(f"    Test Missing Data NLPD: {metrics.test_missing_nlpd:.3f}")
            if config.enable_missing_data and not np.isnan(metrics.train_missing_rmse):
                print(f"    Train Missing Data RMSE: {metrics.train_missing_rmse:.4f}")
                print(f"    Test Missing Data RMSE: {metrics.test_missing_rmse:.4f}")
            print(
                f"    Effective dims: {metrics.effective_latent_dims}/{metrics.total_latent_dims}"
            )
            print(
                f"    Full datasets: Train={split_data['y_train'].shape[0]}, Test={split_data['y_test'].shape[0]} samples"
            )

            # Compute k-NN clustering quality metrics
            print(f"    Computing k-NN clustering metrics...")
            # knn_results = compute_knn_accuracies_ssvi(
            #     model=model,
            #     y_train=(
            #         split_data["y_test_original"]
            #         if config.enable_missing_data
            #         else split_data["y_test"]
            #     ),
            #     labels_train=split_data["labels_test"],
            #     k_values=[1, 3, 7, 12],
            #     batch_size=config.batch_size,
            # )
            # print(f"      k=1: {knn_results['knn_1']:.4f}")
            # print(f"      k=3: {knn_results['knn_3']:.4f}")
            # print(f"      k=7: {knn_results['knn_7']:.4f}")
            # print(f"      k=12: {knn_results['knn_12']:.4f}")
            # knn_metrics_history.append(knn_results)
            # knn_metrics_iterations.append(i)

            # Log metrics
            if logger:
                logger.log_training(f"Metrics at iteration {i}: {metrics.to_dict()}")
                # logger.log_training(f"k-NN metrics at iteration {i}: {knn_results}")

        # Checkpoint saving (models and plots together)
        if logger and i > 0 and i % config.checkpoint_frequency == 0:
            print(f"  Saving checkpoint at iteration {i}...")
            # Save checkpoint model
            checkpoint_filename = f"ood_ssvi_iter_{i}_seed_{config.random_seed}.pkl"
            logger.save_model(model, checkpoint_filename, is_checkpoint=True)
            print(f"    Checkpoint model saved: {checkpoint_filename}")

            # Save checkpoint plot (if enabled)
            if config.save_training_plots:
                plot_filename = f"training_results_iter_{i}.png"
                from src.mnist_ood_plot_helper import visualize_ssvi_results

                # Use full ELBO losses for checkpoint plots
                visualize_ssvi_results(
                    model=model,
                    y_train=y_train,
                    labels_train=labels_train,
                    losses=full_elbo_losses,  # Use full dataset ELBO for plots
                    config=config,
                    logger=logger,
                    save_filename=plot_filename,
                    show_plot=False,
                    is_checkpoint=True,
                    train_indices=split_data["train_indices"],
                    dataset_name=config.id_dataset,
                    is_ood=False,
                    full_elbo_iterations=full_elbo_iterations,
                    batch_losses=losses,  # Batch losses up to current iteration
                )
                print(f"    Checkpoint plot saved: {plot_filename}")

                # Save prior samples plots at checkpoint

                plots_checkpoints_dir = os.path.join(logger.plots_dir, "checkpoints")
                os.makedirs(plots_checkpoints_dir, exist_ok=True)

                prior_samples_path = os.path.join(
                    plots_checkpoints_dir,
                    f"prior_samples_iter_{i}_{config.random_seed}.png",
                )
                visualize_prior_samples(
                    model=model,
                    n_images=24,
                    save_path=prior_samples_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get("scaler"),
                )
                print(
                    f"    Prior samples checkpoint saved: prior_samples_iter_{i}_{config.random_seed}.png"
                )

                prior_samples_avg_path = os.path.join(
                    plots_checkpoints_dir,
                    f"prior_samples_averaged_iter_{i}_{config.random_seed}.png",
                )
                visualize_prior_samples_averaged(
                    model=model,
                    n_images=8,
                    save_path=prior_samples_avg_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get("scaler"),
                )
                print(
                    f"    Prior samples averaged checkpoint saved: prior_samples_averaged_iter_{i}_{config.random_seed}.png"
                )

                test_image_predictions_checkpoint_path = os.path.join(
                    plots_checkpoints_dir,
                    f"test_image_predictions_iter_{i}_{config.random_seed}.png",
                )
                visualize_test_image_predictions(
                    model=model,
                    y_train=(
                        split_data["y_test_original"]
                        if config.enable_missing_data
                        else split_data["y_test"]
                    ),
                    config=config,
                    n_images=24,
                    save_path=test_image_predictions_checkpoint_path,
                    show_plot=False,
                    dataset_name=config.id_dataset,
                    image_shape=(28, 28),
                    scaler=split_data.get("scaler"),
                )
                print(
                    f"    Test image predictions checkpoint saved: test_image_predictions_iter_{i}_{config.random_seed}.png"
                )

        # Gradient logging (only if logger provided and enabled)
        if (
            logger
            and config.enable_gradient_logging
            and i % config.gradient_log_frequency == 0
        ):
            gradient_info = model.get_gradient_info()
            logger.log_gradients(i, loss, gradient_info)

        # Progress reporting
        if i % config.log_frequency == 0:
            elapsed = time.time() - start_time
            message = f"Iteration {i}/{config.training_iterations}: Loss={loss:.4f}, Time={elapsed:.1f}s"
            print(f"Iteration {i}/{config.training_iterations}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            # Log to file
            if logger:
                logger.log_training(message)


def get_config(
    training_iterations=1000, n_inducing_points=20, batch_size=50, latent_dim=6, seed=42
):
    """Get SSVI GPLVM configuration."""
    return OODSSVIConfig(
        training_iterations=training_iterations,
        batch_size=batch_size,
        n_inducing_points=n_inducing_points,
        latent_dim=latent_dim,
        encoder_hidden_dims=[5, 12, 256],
        local_lr=1e-3,
        global_lr=1e-3,
        P=16,
        log_frequency=100,
        checkpoint_frequency=100,
        enable_gradient_logging=False,
        enable_metrics=True,
        metrics_frequency=100,
        save_training_plots=False,
        plot_training=False,
        plot_latent_space=False,
        plot_hyperparams=False,
        init_lengthscales=1.0,
        init_output_noise_scale=1.0,
        enable_missing_data=True,
        missing_data_ratio=0.0,
        reconstruction_samples=50,
        random_seed=seed,
        test_size=0.2,
        id_dataset="FashionMNIST",
        id_classes=[1, 2],
        id_samples_per_class=100,
        ood_dataset="MNIST",
        ood_classes=[1, 2],
        ood_samples_per_class=100,
        ood_n_samples=16,
    )


def evaluate_ssvi_model(
    model, data_norm_all, scaler, missing_mask, config, batch_size=20
):
    """Compute RMSE and NLPD metrics for missing data predictions."""
    model.eval()

    y_true = data_norm_all
    N = y_true.shape[0]

    f_means = torch.empty_like(y_true)
    f_vars = torch.empty_like(y_true)

    with torch.no_grad():
        epsilon = model.get_epsilon()
        U_samples = model.get_u_samples(epsilon)

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            y_batch = y_true[start_idx:end_idx]
            mask_batch = missing_mask[start_idx:end_idx]

            y_encoder_input = y_batch.clone()
            y_encoder_input[mask_batch] = float("nan")

            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(
                y_encoder_input, epsilon, missing_mask=mask_batch
            )
            x_dist = torch.distributions.Normal(mu_x, sigma_x)
            x_samples = x_dist.rsample()

            f_mean = model.gp.get_gp_expectation_given_U(x_samples, U_samples)
            gp_cov = model.gp.get_gp_covariance_given_U(x_samples, U_samples)

            if gp_cov.shape[-1] == 1:
                f_var = gp_cov[..., 0, 0] + model.sigma2
                f_mean = f_mean[..., 0]
            else:
                f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2

            f_mean = f_mean.mean(dim=0)
            f_var = f_var.mean(dim=0)

            f_means[start_idx:end_idx] = f_mean
            f_vars[start_idx:end_idx] = f_var

    results = {}
    results_test = compute_metrics(f_means, data_norm_all, f_vars, missing_mask)
    for key, value in results_test.items():
        results[key + "_test"] = value
    # train results
    mask = ~missing_mask
    results_train = compute_metrics(f_means, data_norm_all, f_vars, mask)
    for key, value in results_train.items():
        results[key + "_train"] = value

    # results = {**results, **results_train}
    results["pred_means"] = f_means
    results["pred_vars"] = f_vars

    return results


def run_fold(
    all_data, entry_mask, train_indices, test_indices, fold, config, verbose=True
):
    """
    Run training and evaluation for a single fold (Case 2: row-wise split).

    Args:
        all_data: Full dataset [N, M]
        entry_mask: Entry-wise mask for full dataset [N, M], True = observed
        train_indices: Indices of training rows
        test_indices: Indices of test rows
        fold: Fold number
        config: Configuration object
        verbose: Whether to print progress
    """
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # Split data by row
    y_train_original = all_data[train_indices]  # Training rows (fully observed)
    y_test_original = all_data[test_indices]  # Test rows (will have entry masking)

    # Get entry masks for train and test
    train_entry_mask = entry_mask[train_indices]  # Should be all True for train
    test_entry_mask = entry_mask[test_indices]  # Has some False for evaluation

    # Missing mask is inverse of entry mask (True = missing)
    train_missing_mask = ~train_entry_mask
    test_missing_mask = ~test_entry_mask

    if verbose:
        print(
            f"  Train: {len(train_indices)} rows, {train_missing_mask.sum().item()} missing entries"
        )
        print(
            f"  Test:  {len(test_indices)} rows, {test_missing_mask.sum().item()} missing entries"
        )

    # Fit scaler on training data only (fully observed)
    scaler = StandardScalerPytorch(device=config.device)
    y_train_scaled = scaler.fit_transform(y_train_original)
    y_test_scaled = scaler.transform(y_test_original)

    # Create masked versions for training/inference
    # Train data is fully observed (no masking needed during training)
    y_train_masked = y_train_scaled.clone()
    # Test data has entry-wise masking
    y_test_masked = y_test_scaled.clone()
    y_test_masked[test_missing_mask] = float("nan")

    # Initialize model
    n_inducing_points = config.n_inducing_points
    Vx = torch.randn(
        n_inducing_points,
        config.latent_dim,
        dtype=torch.float64,
        device=config.device,
    )
    mu_x = torch.nn.Parameter(
        torch.randn(len(y_train_scaled), config.latent_dim, dtype=torch.float64) * 0.1
    )

    config.total_size = y_train_scaled.shape[0]

    model = OilFlow_SSVI_GPLVM(
        config=config, Vx=Vx, mu_x=mu_x, M=y_train_scaled.shape[1]
    )

    # Create split_data dict for training function
    split_data = {
        "y_train": y_train_masked,
        "y_test": y_test_masked,
        "y_train_original": scaler.transform(y_train_original),
        "y_test_original": scaler.transform(y_test_original),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "labels_train": None,  # No labels for protein data
        "labels_test": None,
        "scaler": scaler,
    }

    if verbose:
        print("  Training SSVI GPLVM...")

    # Train on training data only (fully observed, no masking during training)
    train_oil_flow_ssvi(
        model,
        y_train_masked,  # Train on training rows only
        config,
        logger=None,
        split_data=split_data,
        labels_train=None,
    )

    # Evaluate on test set with entry-wise masking
    results = evaluate_ssvi_model_case2(
        model,
        y_test_scaled,  # Original test data (for ground truth)
        scaler,
        test_missing_mask,  # Which entries are missing in test
        config,
    )

    with torch.no_grad():
        lengthscales = model.gp.beta.detach().cpu().numpy()
        # Get latent coords for test data
        epsilon = model.get_epsilon()
        y_test_for_encoding = y_test_scaled.clone()
        y_test_for_encoding[test_missing_mask] = float("nan")
        mu_x_test, _ = model.get_mu_sigma_x_train_samples(
            y_test_for_encoding, epsilon, missing_mask=test_missing_mask
        )
        latent_coords = mu_x_test.mean(dim=0).detach().cpu().numpy()

    results["lengthscales"] = lengthscales
    results["latent_coords"] = latent_coords
    results["train_indices"] = train_indices
    results["test_indices"] = test_indices

    return results


def evaluate_ssvi_model_case2(
    model, y_test_original, scaler, missing_mask, config, batch_size=20
):
    """
    Compute RMSE and NLPD metrics for missing data predictions (Case 2).

    Evaluates only on test set, predicting the masked entries.

    Args:
        model: Trained model
        y_test_original: Original test data (scaled, no NaN) [N_test, M]
        scaler: Scaler used for data
        missing_mask: Boolean mask [N_test, M] where True = missing (to predict)
        config: Configuration
        batch_size: Batch size for evaluation
    """
    model.eval()

    y_true = y_test_original
    N = y_true.shape[0]

    f_means = torch.empty_like(y_true)
    f_vars = torch.empty_like(y_true)

    with torch.no_grad():
        epsilon = model.get_epsilon()
        U_samples = model.get_u_samples(epsilon)

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            y_batch = y_true[start_idx:end_idx]
            mask_batch = missing_mask[start_idx:end_idx]

            # Create input with missing values as NaN
            y_encoder_input = y_batch.clone()
            y_encoder_input[mask_batch] = float("nan")

            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(
                y_encoder_input, epsilon, missing_mask=mask_batch
            )
            x_dist = torch.distributions.Normal(mu_x, sigma_x)
            x_samples = x_dist.rsample()

            f_mean = model.gp.get_gp_expectation_given_U(x_samples, U_samples)
            gp_cov = model.gp.get_gp_covariance_given_U(x_samples, U_samples)

            if gp_cov.shape[-1] == 1:
                f_var = gp_cov[..., 0, 0] + model.sigma2
                f_mean = f_mean[..., 0]
            else:
                f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + model.sigma2

            f_mean = f_mean.mean(dim=0)
            f_var = f_var.mean(dim=0)

            f_means[start_idx:end_idx] = f_mean
            f_vars[start_idx:end_idx] = f_var

    results = {}
    # Metrics on missing entries (the ones we're trying to predict)
    results_missing = compute_metrics(f_means, y_true, f_vars, missing_mask)
    for key, value in results_missing.items():
        results[key + "_test"] = value

    # Metrics on observed entries (sanity check)
    observed_mask = ~missing_mask
    results_observed = compute_metrics(f_means, y_true, f_vars, observed_mask)
    for key, value in results_observed.items():
        results[key + "_train"] = value

    results["pred_means"] = f_means
    results["pred_vars"] = f_vars

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SSVI GPLVM for Missing Data - Case 2: Row-wise CV with entry masking on test"
    )
    parser.add_argument("--latent_dim", type=int, default=6, help="Latent dimension")
    parser.add_argument(
        "--n_inducing", type=int, default=50, help="Number of inducing points"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=2000, help="Training iterations"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="masks",
        help="Path to CV masks",
    )
    parser.add_argument(
        "--mask_seed", type=int, default=42, help="Random seed for CV masks"
    )
    args = parser.parse_args()

    data_df, data_np = load_data()
    cc_values = data_df["CC"].values

    all_data = torch.tensor(data_np, dtype=torch.float64)

    # Load Case 2 masks and train/test splits
    work_dir = Path(__file__).parent
    masks = np.load(
        work_dir / args.mask_path / f"masks_case_2_seed{int(args.mask_seed)}.npy"
    )
    masks = masks.astype(bool)  # True = observed, False = missing

    train_test_splits = np.load(
        work_dir
        / args.mask_path
        / f"train_test_splits_case_2_seed{int(args.mask_seed)}.npy",
        allow_pickle=True,
    )

    n_folds = len(masks)
    print(f"\n{'='*60}")
    print(f"SSVI GPLVM - Case 2: Row-wise CV with entry masking on test")
    print(f"{'='*60}")
    print(f"Data shape: {data_np.shape}")
    print(f"Number of folds: {n_folds}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Inducing points: {args.n_inducing}")
    print(f"Training iterations: {args.num_iterations}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")

    all_results = []
    all_results_dict = {}
    figures_dir = ensure_figures_dir(Path(__file__).parent, args.seed)

    save_name = f"case2_seed_{args.seed}_latentdim_{args.latent_dim}_inducing_{args.n_inducing}_maskseed_{args.mask_seed}_batchsize_{args.batch_size}"

    for fold in range(n_folds):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*40}")

        # Get train/test indices for this fold
        train_indices = train_test_splits[fold]["train_indices"]
        test_indices = train_test_splits[fold]["test_indices"]

        # Convert to torch tensors
        train_indices_t = torch.tensor(train_indices, dtype=torch.long)
        test_indices_t = torch.tensor(test_indices, dtype=torch.long)

        # Get entry mask for this fold (True = observed)
        entry_mask = torch.tensor(masks[fold], dtype=torch.bool)

        config = get_config(
            training_iterations=args.num_iterations,
            n_inducing_points=args.n_inducing,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            seed=args.seed + fold,
        )

        fold_results = run_fold(
            all_data,
            entry_mask=entry_mask,
            train_indices=train_indices_t,
            test_indices=test_indices_t,
            fold=fold,
            config=config,
        )

        print_fold_results(fold, fold_results)

        plot_inverse_lengthscales(
            fold_results["lengthscales"],
            figures_dir
            / f"gplvm_ssvi_case2_inverse_lengthscales_fold{fold+1}_{save_name}.png",
            title=f"GPLVM-SSVI Case 2 Inverse Lengthscales - Fold {fold+1}",
        )

        # Get CC values for test indices only (for latent space plot)
        test_cc_values = cc_values[test_indices]
        plot_latent_space(
            fold_results["latent_coords"],
            figures_dir / f"gplvm_ssvi_case2_latent_space_fold{fold+1}_{save_name}.png",
            title=f"GPLVM-SSVI Case 2 Latent Space (Test Set) - Fold {fold+1}",
            color_values=test_cc_values,
            color_label="CC",
        )

        res_df = pd.DataFrame(index=[0])
        res_df["fold"] = fold
        res_df["n_train"] = len(train_indices)
        res_df["n_test"] = len(test_indices)
        for dataset in ["train", "test"]:
            for i, v in enumerate(fold_results[f"rmse_by_dim_{dataset}"]):
                res_df[f"rmse_dim_{dataset}_{i+1}"] = v
            for i, v in enumerate(fold_results[f"nlpd_by_dim_{dataset}"]):
                res_df[f"nlpd_dim_{dataset}_{i+1}"] = v
            res_df[f"total_rmse_{dataset}"] = fold_results[f"total_rmse_{dataset}"]
            res_df[f"total_nlpd_{dataset}"] = fold_results[f"total_nlpd_{dataset}"]

        all_results.append(res_df)
        all_results_dict[fold] = fold_results

    all_results = pd.concat(all_results, ignore_index=True)

    output_dir = Path(__file__).parent / args.output_dir
    results_df = save_cv_results(
        all_results,
        all_results_dict,
        output_dir,
        f"gplvm_ssvi_case2_cv_results_{save_name}.csv",
    )

    print(f"\n{'='*60}")
    print(
        f"Results saved to: {output_dir / f'gplvm_ssvi_case2_cv_results_{save_name}.csv'}"
    )
    print(f"Figures saved to: {figures_dir}")
    print(f"{'='*60}")

    return results_df


if __name__ == "__main__":
    main()
