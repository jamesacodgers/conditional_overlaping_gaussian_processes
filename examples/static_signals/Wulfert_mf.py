# %%

import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from itertools import chain
import copy
import numpy as np
import pandas as pd
from src.save_utils import save_parameters, save_results_csv


from sklearn.cross_decomposition import PLSRegression

from src.mf_methods import MFStaticSignals
from src.utils import train_lbfgs, plot_preds

torch.set_default_dtype(torch.float64)
# %%

def main(args):
    experiment_name = f"Wulfertd_test_size={args.test_size}_num_unobs={args.num_unobs}_num_samples={args.num_samples}_covar_rank={args.covar_rank}"
    work_dir = Path(args.work_dir)
    torch.manual_seed(args.random_seed)
    d_test = torch.load(f"data/Wulfert/splits/d_test_seed={args.data_idx}_test_size={args.test_size}.pkl")
    d_train = torch.load(f"data/Wulfert/splits/d_train_seed={args.data_idx}_test_size={args.test_size}.pkl")

    r_train = torch.load(f"data/Wulfert/splits/r_train_seed={args.data_idx}_test_size={args.test_size}.pkl")
    r_test = torch.load(f"data/Wulfert/splits/r_test_seed={args.data_idx}_test_size={args.test_size}.pkl")

    mf_sto_eiot = MFStaticSignals(M=d_test.shape[1], C_obs=3, C_unobs=args.num_unobs, sigma2=1)

    mf_sto_ll = train_lbfgs(mf_sto_eiot, D_test=d_test, D_train=d_train, R_train=r_train, n_steps=100, plot=args.plot, convergence_level=1)

    mf_sto_preds = mf_sto_eiot.predict_weights(D_test=d_test)

    mf_sto_log_prob = torch.distributions.MultivariateNormal(mf_sto_preds.mean[:,:3], mf_sto_preds.covariance_matrix[:, :3, :3]).log_prob(r_test).sum()

    mf_sto_rmse = torch.sqrt(((r_test - mf_sto_preds.mean[:, :3])**2).mean())

    if args.plot == True: 
        plot_preds(
            R_test=r_test,  
            mf_preds=mf_sto_preds,
        )

    mf_sto_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "rmsep": mf_sto_rmse.item(), "log_prob": mf_sto_log_prob.item(), "elbo": mf_sto_ll[-1]}

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "Wulfert" / "mf_static" /"csvs" / f'{experiment_name}.csv', 
                    results_dict=mf_sto_results_dict)
    
    
    save_parameters(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "Wulfert" / "mf_static" /"models", 
                    model=mf_sto_eiot)


if __name__ == "__main__":
    # Log in to your W&B account
    import os

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        "--random_seed",
        "-rs",
        type=int,
        default=0,
        help="Random seed.",
    )

    argparser.add_argument(
        "--test_size",
        "-ts",
        type=float,
        default=0.1,
        help="Test size.",
    )

    argparser.add_argument(
        "--data_idx",
        "-d",
        type=int,
        default=0,
        help="Random seed.",
    )

    argparser.add_argument(
        "--plot",
        "-p",
        type=bool,
        default=False,
        help="plots figures",
    )

    argparser.add_argument(
        "--work_dir",
        "-wd",
	type=str,
        default=".",
        help="working directory",
    )
    
    argparser.add_argument(
        "--num_unobs",
        "-nu",
	type=int,
        default=0,
        help="number of unobserved components",
    )

    argparser.add_argument(
        "--num_samples",
        "-ns",
	type=int,
        default=16,
        help="number of MC samples.",
    )

    argparser.add_argument(
        "--covar_rank",
        "-cr",
	type=int,
        default=1,
        help="rank of low rank covariance matrix.",
    )

    args = argparser.parse_args()

    main(args)
