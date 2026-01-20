"""
GPLVM for Missing Data with Entry-wise Cross Validation

This script trains a Bayesian GPLVM on protein developability assay data,
using entry-wise cross validation to evaluate missing data imputation performance.
"""

import numpy as np
import torch
import argparse
from pathlib import Path

from gpytorch.models.gplvm.latent_variable import VariationalLatentVariable
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.priors import NormalPrior
import pandas as pd

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


def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q=latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:, :latent_dim]))


class GaussianLikelihoodWithMissingObs(GaussianLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_masked_obs(x):
        missing_idx = x.isnan()
        x_masked = x.masked_fill(missing_idx, -999.0)
        return missing_idx, x_masked

    def expected_log_prob(self, target, input, *params, **kwargs):
        missing_idx, target = self._get_masked_obs(target)
        res = super().expected_log_prob(target, input, *params, **kwargs)
        return res * ~missing_idx

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        missing_idx, observations = self._get_masked_obs(observations)
        res = super().log_marginal(observations, function_dist, *params, **kwargs)
        return res * ~missing_idx


class BayesianGPLVMModel(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=True):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        self.inducing_inputs = torch.randn(
            data_dim, n_inducing, latent_dim, dtype=torch.float64
        )

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(
            self, self.inducing_inputs, q_u, learn_inducing_locations=True
        )

        X_prior_mean = torch.zeros(n, latent_dim, dtype=torch.float64)
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        # if pca:
        #     X_init = _init_pca(Y, latent_dim)

        # else:
        X_init = torch.nn.Parameter(
            torch.randn(n, latent_dim, dtype=torch.float64) * 0.1
        )
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        self.mean_module = ZeroMean(batch_shape=self.batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=latent_dim), batch_shape=self.batch_shape
        )

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)


def train_gplvm(
    data_norm,
    mask,
    latent_dim,
    n_inducing,
    num_iterations,
    lr=0.01,
    verbose=True,
    pca=True,
):
    """Train a Bayesian GPLVM model."""
    num_data, data_dim = data_norm.shape

    model = BayesianGPLVMModel(num_data, data_dim, latent_dim, n_inducing, pca=pca)
    # likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    likelihood = GaussianLikelihoodWithMissingObs(batch_shape=model.batch_shape)
    mll = VariationalELBO(likelihood, model, num_data)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )

    model.to(torch.float64)
    model.train()
    likelihood.train()

    for i in range(num_iterations):
        optimizer.zero_grad()
        sample = model.sample_latent_variable()
        output = model(sample)
        loss = -mll(output, data_norm.t()).sum()
        loss.backward()
        optimizer.step()

        if verbose and i % 100 == 0:
            print(f"  Iteration {i}/{num_iterations} - Loss: {loss.item():.4f}")

    return model, likelihood


def evaluate_model(model, likelihood, data_norm_all, mask, data_dim):
    """Evaluate model on missing values, computing RMSE and NLPD."""
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        sample = model.sample_latent_variable()
        output = model(sample)

        pred_means = []
        pred_vars = []
        for j in range(data_dim):
            pred_dist = likelihood(output[j])
            pred_means.append(pred_dist.mean)
            var = pred_dist.variance
            pred_vars.append(var[j])

        pred_means = torch.stack(pred_means, dim=1)
        pred_vars = torch.stack(pred_vars, dim=1)

    missing_mask = ~mask

    # test results
    results = {}
    results_test = compute_metrics(pred_means, data_norm_all, pred_vars, missing_mask)
    for key, value in results_test.items():
        results[key + "_test"] = value
    # train results
    results_train = compute_metrics(pred_means, data_norm_all, pred_vars, mask)
    for key, value in results_train.items():
        results[key + "_train"] = value

    # results = {**results, **results_train}
    results["pred_means"] = pred_means
    results["pred_vars"] = pred_vars

    return results


def run_fold(data_np, mask, latent_dim, n_inducing, num_iterations, seed, pca):
    """Run training and evaluation for a single fold."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    missing_mask = ~mask

    all_data = torch.tensor(data_np, dtype=torch.float64)
    data_with_missing = all_data.clone()
    data_with_missing[missing_mask] = float("nan")

    scaler = StandardScalerPytorch()
    data_norm = scaler.fit_transform(data_with_missing)
    # data_norm[missing_mask] = 0.0

    data_norm_all = scaler.transform(all_data)

    num_data, data_dim = all_data.shape

    model, likelihood = train_gplvm(
        data_norm=data_norm,
        mask=mask,
        latent_dim=latent_dim,
        n_inducing=n_inducing,
        num_iterations=num_iterations,
        pca=pca,
    )

    results = evaluate_model(model, likelihood, data_norm_all, mask, data_dim)

    with torch.no_grad():
        lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        latent_coords = model.X.q_mu.detach().cpu().numpy()

    results["lengthscales"] = lengthscales
    results["latent_coords"] = latent_coords

    return results


def main():
    parser = argparse.ArgumentParser(
        description="GPLVM for Missing Data with Entry-wise CV"
    )
    parser.add_argument("--latent_dim", type=int, default=5, help="Latent dimension")
    parser.add_argument(
        "--n_inducing", type=int, default=50, help="Number of inducing points"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1000, help="Training iterations"
    )
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
    parser.add_argument(
        "--pca", action="store_true", help="Initialize latent variables with PCA"
    )
    args = parser.parse_args()

    data_df, data_np = load_data()
    cc_values = data_df["CC"].values

    # fold_indices = create_fold_indices(data_np.shape, args.n_folds, args.seed)

    work_dir = Path(__file__).parent

    # load the cv masks
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
    )

    all_results = []
    all_results_dict = {}
    figures_dir = ensure_figures_dir(Path(__file__).parent, args.seed)

    for fold, mask in enumerate(masks):
        print_fold_header(fold, len(masks), masks)

        fold_results = run_fold(
            data_np=data_np,
            mask=mask,
            latent_dim=args.latent_dim,
            n_inducing=args.n_inducing,
            num_iterations=args.num_iterations,
            seed=args.seed + fold,
            pca=args.pca,
        )

        print_fold_results(fold, fold_results)

        save_name = f"seed_{args.seed}_latentdim_{args.latent_dim}_inducing_{args.n_inducing}_maskseed_{args.mask_seed}"
        if args.pca:
            save_name += "_pcainit"

        plot_inverse_lengthscales(
            fold_results["lengthscales"],
            figures_dir / f"gplvm_inverse_lengthscales_fold{fold+1}_{save_name}.png",
            title=f"GPLVM Inverse Lengthscales - Fold {fold+1}",
        )
        plot_latent_space(
            fold_results["latent_coords"],
            figures_dir / f"gplvm_latent_space_fold{fold+1}_{save_name}.png",
            title=f"GPLVM Latent Space - Fold {fold+1}",
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

    # print_cv_summary(all_results)

    output_dir = Path(__file__).parent / args.output_dir
    results_df = save_cv_results(
        all_results, all_results_dict, output_dir, f"gplvm_cv_results_{save_name}.csv"
    )
    print(f"\nResults saved to: {output_dir / f'gplvm_cv_results_{save_name}.csv'}")
    print(f"Figures saved to: {figures_dir}")

    return results_df


if __name__ == "__main__":
    main()
