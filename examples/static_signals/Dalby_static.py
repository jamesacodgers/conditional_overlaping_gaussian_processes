# %%

import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from itertools import chain
import copy
import numpy as np
import pandas as pd
from src.save_utils import save_model, save_results_csv

from sklearn.cross_decomposition import PLSRegression

from src.eiot import  ConditionalEIOT
from src.utils import train_lbfgs, plot_preds
from src.mf_methods import MFStaticSignals


torch.set_default_dtype(torch.float64)
# %%

def main(args):
    experiment_name = f"Dalby_test_size={args.test_size}_num_unobs={args.num_unobs}"
    work_dir = Path(args.work_dir)
    torch.manual_seed(args.random_seed)
    d_test = torch.load(f"data/Dalby/splits/d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    d_train = torch.load(f"data/Dalby/splits/d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    r_train = torch.load(f"data/Dalby/splits/r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    R_test = torch.load(f"data/Dalby/splits/r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    model = MFStaticSignals(M=d_test.shape[1], C_obs=r_train.shape[1], C_unobs=args.num_unobs, sigma2=1, weight_distribution="sum_to_one")
    model = torch.compile(model)

    ll = []
    # mf_ll = train_eiot_sgd(mf_eiot, D_test=d_test, D_train=d_train, R_train=r_train, n_steps=1, plot=args.plot)
    ll.extend(train_lbfgs(model, D_test=d_test, D_train=d_train, R_train=r_train, n_steps=25, plot=args.plot, convergence_level=5))


    preds = model.predict_weights(D=d_test)
    # cond_preds = cond_eiot.predict_weights(D_test=d_test)

    mf_rmse = torch.sqrt(((R_test - preds.mean[:, :2])**2).mean())
    # cond_rmse = torch.sqrt(((r_test - cond_preds.mean.mean(dim=[0])[:, :2])**2).mean())



    mf_log_prob = torch.distributions.MultivariateNormal(preds.mean[:,:r_train.shape[1]], preds.covariance_matrix[:, :r_train.shape[1], :r_train.shape[1]]).log_prob(R_test).sum()

    print(mf_rmse)

    print(mf_log_prob)
    if args.plot: 
        plot_preds(
            R_test=R_test,  
            mf_preds=preds,
            # cond_preds=cond_preds
        )

    mean_interval_length = (1.96*2*preds.stddev[:,0]).mean()


    coverage = 0
    for c in range(model.C_obs):
        lb = preds.mean[:,c] - 1.96*preds.stddev[:,c]
        ub = preds.mean[:,c] + 1.96*preds.stddev[:,c]
        coverage += torch.logical_and(lb < R_test[:,c], ub > R_test[:,c]).type(torch.float32).sum()
    coverage = coverage/(R_test.shape[0]*R_test.shape[1])

    mf_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "rmsep": mf_rmse.item(), "log_prob": mf_log_prob.item(), "elbo": ll[-1], "mean_interval_length": mean_interval_length.item(), "coverage": coverage.item() }
    print(mf_results_dict)

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "Dalby"/ "mf_static_signals" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=mf_results_dict)
    
    save_model(experiment_name=experiment_name, seed=args.random_seed, data_idx=args.data_idx, 
                    path=work_dir / "results" / "Dalby"/ "mf_static_signals" / "models", 
                    model=model)
    

    


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
        default=5,
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
        default=1,
        help="number of unobserved components",
    )

    argparser.add_argument(
        "--sum_to_one",
        "-sto",
	type=bool,
        default=True,
        help="should the components sum to one",
    )

    argparser.add_argument(
        "--min_wavelength",
        "-minwl",
	type=int,
        default=150,
        help="where to trim spectra below",
    )
    argparser.add_argument(
        "--max_wavelength",
        "-maxwl",
	type=int,
        default=250,
        help="trim spectra above",
    )

    args = argparser.parse_args()

    main(args)
