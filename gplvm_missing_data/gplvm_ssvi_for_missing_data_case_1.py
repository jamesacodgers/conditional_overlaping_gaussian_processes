"""
SSVI GPLVM for Missing Data with Entry-wise Cross Validation

This script trains an SSVI GPLVM on protein developability assay data,
using entry-wise cross validation to evaluate missing data imputation performance.
"""

import numpy as np
import torch
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.ssvi_config import OilFlowSSVIConfig
from src.ssvi_models_am_inf_sep_opt_old import GPLVM_SSVI_AM_INF
from src.ssvi_utils import SSVIResultsLogger


from src.ssvi_metrics import SSVIMetricsComputer
from ssvi_scripts_experiments.mnist_ood_GPLVM_SSVI_AM_INF import (
    OODSSVIConfig,
)
import time

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


class OilFlow_SSVI_GPLVM(GPLVM_SSVI_AM_INF):
    """SSVI GP-LVM with Amortized Inference for missing data imputation"""

    def __init__(self, config: OilFlowSSVIConfig, Vx, mu_x, M):
        # Use init_lengthscales to create beta values
        # beta = lengthscales, so for init_lengthscales=1.0, beta=[1,1,1,1,1,1]
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
        U_samples: torch.Tensor = None,
    ):
        """
        Modified train_step: proper SSVI separation of local/global optimization
        """
        # Zero gradients for local optimizer (encoder) only initially
        self.optimizer_local.zero_grad()

        # Sample U before amortized inference step
        epsilon = self.get_epsilon()
        U_samples = self.get_u_samples(epsilon)

        # Handle missing data preprocessing for encoder input
        if self.enable_missing_data:
            # Replace NaN values with -9.0 for encoder input (encoder expects clean data)
            Y_batch_clean = torch.where(
                torch.isnan(Y_batch),
                torch.full_like(Y_batch, -9.0, dtype=torch.float64),
                Y_batch,
            )
            # Get local parameters via amortized inference (gradients only to encoder)
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                Y_batch_clean, epsilon
            )
        else:
            # Standard case: no missing data preprocessing needed
            mu_n_batch, sigma_n_batch = self.get_mu_sigma_x_train_samples(
                Y_batch, epsilon
            )

        # Now zero gradients for global optimizer before global computation
        self.optimizer_global.zero_grad()

        # Compute global ELBO loss (gradients flow to all parameters)
        if self.enable_missing_data:
            # Use missing data aware ELBO (coordinator handles the missing mask automatically)
            missing_mask = torch.isnan(Y_batch)
            # Use the same clean data for ELBO computation to maintain consistency with encoder
            loss = -self.ssvi_elbo(
                Y_batch_clean,
                mu_n_batch,
                sigma_n_batch,
                U_samples=U_samples,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                missing_mask=missing_mask,
                prior_weight=1,
            )
        else:
            # Standard ELBO computation
            loss = -self.ssvi_elbo(
                Y_batch,
                mu_n_batch,
                sigma_n_batch,
                U_samples=U_samples,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                missing_mask=None,
                prior_weight=1,
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


def train_oil_flow_ssvi(
    model: OilFlow_SSVI_GPLVM,
    y_train,
    config: OilFlowSSVIConfig,
    logger: SSVIResultsLogger = None,
    split_data=None,
    labels_train=None,
):
    """Training function for SSVI GP-LVM with optional logging and metrics"""
    losses = []

    print("Training SSVI GP-LVM...")
    print(f"Iterations: {config.training_iterations}, Batch size: {config.batch_size}")

    # Initialize metrics computer if enabled
    metrics_computer = None
    if config.enable_metrics and split_data is not None:
        print(
            f"Metrics computation enabled (frequency: every {config.metrics_frequency} iterations)"
        )
        metrics_computer = SSVIMetricsComputer(device=config.device)

    start_time = time.time()

    for i in range(config.training_iterations):
        # Sample batch indices
        N = y_train.shape[0]
        batch_indices = torch.randperm(N, device=y_train.device)[: config.batch_size]
        y_batch = y_train[batch_indices]

        # Training step - U sampling happens inside automatically each step
        loss = model.train_step(y_batch, batch_indices)
        losses.append(loss)

        # Metrics computation (periodic)
        if (
            metrics_computer is not None
            and config.enable_metrics
            and i % config.metrics_frequency == 0
        ):

            print(f"  Computing metrics at iteration {i}...")

            # Use full datasets for proper evaluation
            # Compute metrics with missing data info if enabled
            if config.enable_missing_data:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data["y_train"],
                    y_test=split_data["y_test"],
                    iteration=i,
                    elbo_loss=loss,
                    n_samples=32,  # Fixed number of samples for metrics computation
                    y_train_original=split_data["y_train_original"],
                    y_test_original=split_data["y_test_original"],
                    train_missing_mask=split_data["train_missing_mask"],
                    test_missing_mask=split_data["test_missing_mask"],
                )
            else:
                metrics = metrics_computer.compute_full_metrics(
                    model=model,
                    y_train=split_data["y_train"],
                    y_test=split_data["y_test"],
                    iteration=i,
                    elbo_loss=loss,
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

            # Log metrics
            if logger:
                logger.log_training(f"Metrics at iteration {i}: {metrics.to_dict()}")

        # Checkpoint saving (models and plots together)
        if logger and i > 0 and i % config.checkpoint_frequency == 0:
            print(f"  Saving checkpoint at iteration {i}...")
            try:
                # Save checkpoint model
                checkpoint_filename = (
                    f"mnist_ssvi_iter_{i}_seed_{config.random_seed}.pkl"
                )
                logger.save_model(model, checkpoint_filename, is_checkpoint=True)
                print(f"    Checkpoint model saved: {checkpoint_filename}")

                # Save checkpoint plot (if enabled)
                if config.save_training_plots:
                    plot_filename = f"training_results_iter_{i}.png"
                    try:
                        from src.mnist_plot_helper import visualize_ssvi_results

                        visualize_ssvi_results(
                            model=model,
                            y_train=y_train,
                            labels_train=labels_train,
                            losses=losses,
                            config=config,
                            logger=logger,
                            save_filename=plot_filename,
                            show_plot=False,
                            is_checkpoint=True,
                            train_indices=split_data["train_indices"],
                        )
                        print(f"    Checkpoint plot saved: {plot_filename}")
                    except Exception as e:
                        print(
                            f"    Warning: Could not create checkpoint visualization at iteration {i}: {e}"
                        )

            except Exception as e:
                print(f"    Warning: Error saving checkpoint at iteration {i}: {e}")

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

    total_time = time.time() - start_time
    final_message = f"Training completed! Final Loss: {losses[-1]:.4f}, Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    print(f"Training completed!")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(
        f"Total time: {total_time:.1f}s ({config.training_iterations/total_time:.1f} iter/s)"
    )
    # Log to file
    if logger:
        logger.log_training(final_message)

    return losses, metrics_computer


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
        P=32,
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

    # Ensure missing_mask is a tensor
    if not isinstance(missing_mask, torch.Tensor):
        missing_mask = torch.tensor(
            missing_mask, dtype=torch.bool, device=config.device
        )

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

            # Replace missing values with -9.0 (matching training behavior)
            y_encoder_input = torch.where(
                mask_batch,
                torch.full_like(y_batch, -9.0, dtype=torch.float64),
                y_batch,
            )

            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(y_encoder_input, epsilon)
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


def run_fold(all_data, mask, fold, config, verbose=True):
    """Run training and evaluation for a single fold."""
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    missing_mask = ~mask

    data_with_missing = all_data.clone()
    data_with_missing[missing_mask] = float("nan")

    scaler = StandardScalerPytorch(device=config.device)
    y_train = scaler.fit_transform(data_with_missing)

    # data_missing_for_test = all_data.clone()
    # data_missing_for_test[missing_mask] = float("nan")
    # y_test = scaler.transform(data_missing_for_test)

    n_inducing_points = config.n_inducing_points
    Vx = torch.randn(
        n_inducing_points,
        config.latent_dim,
        dtype=torch.float64,
        device=config.device,
    )
    mu_x = torch.nn.Parameter(
        torch.randn(len(data_with_missing), config.latent_dim, dtype=torch.float64)
        * 0.1
    )

    config.total_size = y_train.shape[0]

    model = OilFlow_SSVI_GPLVM(config=config, Vx=Vx, mu_x=mu_x, M=y_train.shape[1])

    if verbose:
        print(f"  Training SSVI GPLVM...")

    training_result, _ = train_oil_flow_ssvi(
        model, y_train, config, logger=None, split_data=None, labels_train=None
    )

    data_norm_all = scaler.transform(all_data)

    results = evaluate_ssvi_model(model, data_norm_all, scaler, missing_mask, config)

    with torch.no_grad():
        lengthscales = model.gp.beta.detach().cpu().numpy()
        # mu_x has shape [P, N, latent_dim], average over P dimension
        latent_coords = model.mu_x.detach().cpu().numpy().mean(axis=0)

    results["lengthscales"] = lengthscales
    results["latent_coords"] = latent_coords

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SSVI GPLVM for Missing Data with Entry-wise CV"
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

    # load the cv masks
    work_dir = Path(__file__).parent
    masks = np.load(
        work_dir / args.mask_path / f"cv_binary_masks_seed{int(args.mask_seed)}.npy"
    )
    # convert to boolean
    masks = masks.astype(bool)

    print_cv_header(
        data_np.shape,
        latent_dim=args.latent_dim,
        inducing_points=args.n_inducing,
        training_iterations=args.num_iterations,
        batch_size=args.batch_size,
    )

    all_results = []
    all_results_dict = {}
    figures_dir = ensure_figures_dir(Path(__file__).parent, args.seed)

    save_name = f"seed_{args.seed}_latentdim_{args.latent_dim}_inducing_{args.n_inducing}_maskseed_{args.mask_seed}_batchsize_{args.batch_size}"

    for fold, mask in enumerate(masks):
        print_fold_header(fold, len(masks), masks)

        config = get_config(
            training_iterations=args.num_iterations,
            n_inducing_points=args.n_inducing,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            seed=args.seed + fold,
        )

        fold_results = run_fold(all_data, mask=mask, fold=fold, config=config)

        print_fold_results(fold, fold_results)

        plot_inverse_lengthscales(
            fold_results["lengthscales"],
            figures_dir
            / f"gplvm_ssvi_case_1_inverse_lengthscales_fold{fold+1}_{save_name}.png",
            title=f"GPLVM-SSVI Inverse Lengthscales - Fold {fold+1}",
        )
        plot_latent_space(
            fold_results["latent_coords"],
            figures_dir
            / f"gplvm_ssvi_case_1_latent_space_fold{fold+1}_{save_name}.png",
            title=f"GPLVM-SSVI Latent Space - Fold {fold+1}",
            color_values=cc_values,
            color_label="CC",
        )

        res_df = pd.DataFrame(index=[0])
        res_df["fold"] = fold
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
        f"gplvm_ssvi_case_1_cv_results_{save_name}.csv",
    )

    print(
        f"\nResults saved to: {output_dir / f'gplvm_ssvi_case_1_cv_results_{save_name}.csv'}"
    )
    print(f"Figures saved to: {figures_dir}")

    return results_df


if __name__ == "__main__":
    main()
