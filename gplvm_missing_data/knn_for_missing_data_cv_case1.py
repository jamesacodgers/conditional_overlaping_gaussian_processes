"""
K-Nearest Neighbors for Missing Data Imputation with Entry-wise Cross Validation

This script uses scikit-learn's KNNImputer on protein developability assay data,
using entry-wise cross validation to evaluate missing data imputation performance.
"""

import numpy as np
import torch
import argparse
from pathlib import Path
from sklearn.impute import KNNImputer
import pandas as pd
from cv_utils import (
    load_data,
    compute_rmse,
    save_cv_results,
    print_cv_header,
    print_fold_header,
    print_fold_results,
    print_cv_summary,
    StandardScalerPytorch,
)


def run_fold(data_np, mask, test_fold, n_neighbors, seed):
    """Run KNN imputation and evaluation for a single fold."""
    np.random.seed(seed)

    missing_mask = ~mask
    all_data = torch.tensor(data_np, dtype=torch.float64)
    data_with_missing = all_data.clone()
    data_with_missing[missing_mask] = float("nan")

    scaler = StandardScalerPytorch()
    data_norm_with_missing = scaler.fit_transform(data_with_missing)

    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    data_imputed = imputer.fit_transform(data_norm_with_missing.numpy())

    data_norm_all = scaler.transform(all_data)

    results = {}
    results_test = compute_rmse(data_imputed, data_norm_all, missing_mask)
    for key, value in results_test.items():
        results[key + "_test"] = value
    results_train = compute_rmse(data_imputed, data_norm_all, ~mask)
    for key, value in results_train.items():
        results[key + "_train"] = value

    results["pred_values"] = data_imputed

    return results


def main():
    parser = argparse.ArgumentParser(
        description="KNN for Missing Data with Entry-wise CV"
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=5, help="Number of neighbors for KNN"
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

    args = parser.parse_args()

    data_df, data_np = load_data()
    cc_values = data_df["CC"].values

    work_dir = Path(__file__).parent

    # load the cv masks
    masks = np.load(
        work_dir / args.mask_path / f"cv_binary_masks_seed{int(args.mask_seed)}.npy"
    )
    # convert to boolean
    masks = masks.astype(bool)

    print_cv_header(data_np.shape, n_neighbors=args.n_neighbors)

    all_results = []
    all_results_dict = {}

    save_name = (
        f"seed_{args.seed}_neighbours_{args.n_neighbors}_maskseed_{args.mask_seed}"
    )

    for fold, mask in enumerate(masks):
        print_fold_header(fold, len(masks), masks)

        fold_results = run_fold(
            data_np, mask, fold, args.n_neighbors, seed=args.seed + fold
        )

        print_fold_results(fold, fold_results, include_nlpd=False)

        res_df = pd.DataFrame(index=[0])
        res_df["fold"] = fold
        for dataset in ["train", "test"]:
            for i, v in enumerate(fold_results[f"rmse_by_dim_{dataset}"]):
                res_df[f"rmse_dim_{dataset}_{i+1}"] = v
            res_df[f"total_rmse_{dataset}"] = fold_results[f"total_rmse_{dataset}"]

        all_results.append(res_df)
        all_results_dict[fold] = fold_results

    all_results = pd.concat(all_results, ignore_index=True)

    output_dir = Path(__file__).parent / args.output_dir
    results_df = save_cv_results(
        all_results,
        all_results_dict,
        output_dir,
        f"knn_cv_results_{save_name}.csv",
        include_nlpd=False,
    )

    print(f"\nResults saved to: {output_dir / f'knn_cv_results_{save_name}.csv'}")

    return results_df


if __name__ == "__main__":
    main()
