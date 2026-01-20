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

from ssvi_scripts_experiments.mnist_ood_GPLVM_SSVI_AM_INF import (
    OilFlow_SSVI_GPLVM,
    OODSSVIConfig,
    train_oil_flow_ssvi,
)

from cv_utils import (
    load_data,
    compute_metrics,
    save_cv_results,
    print_cv_header,
    print_fold_header,
    print_fold_results,
    print_cv_summary,
    StandardScalerPytorch,
    plot_inverse_lengthscales,
    plot_latent_space,
    ensure_figures_dir,
)


def get_config(
    training_iterations=1000, n_inducing_points=20, batch_size=50, latent_dim=6, seed=42
):
    """Get SSVI GPLVM configuration."""
    return OODSSVIConfig(
        training_iterations=training_iterations,
        batch_size=batch_size,
        n_inducing_points=n_inducing_points,
        latent_dim=latent_dim,
        encoder_hidden_dims=[128, 64],
        local_lr=2e-3,
        global_lr=1e-2,
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
        U_samples = model.get_u_samples()

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            y_batch = y_true[start_idx:end_idx]
            mask_batch = missing_mask[start_idx:end_idx]

            y_encoder_input = y_batch.clone()
            y_encoder_input[mask_batch] = -9.0

            mu_x, sigma_x = model.get_mu_sigma_x_train_samples(
                y_encoder_input, U_samples
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
    parser.add_argument("--batch_size", type=int, default=195, help="Batch size")
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
            / f"gplvm_ssvi_inverse_lengthscales_fold{fold+1}_{save_name}.png",
            title=f"GPLVM-SSVI Inverse Lengthscales - Fold {fold+1}",
        )
        plot_latent_space(
            fold_results["latent_coords"],
            figures_dir / f"gplvm_ssvi_latent_space_fold{fold+1}_{save_name}.png",
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
        f"gplvm_ssvi_cv_results_{save_name}.csv",
    )

    print(
        f"\nResults saved to: {output_dir / f'gplvm_ssvi_cv_results_{save_name}.csv'}"
    )
    print(f"Figures saved to: {figures_dir}")

    return results_df


if __name__ == "__main__":
    main()
