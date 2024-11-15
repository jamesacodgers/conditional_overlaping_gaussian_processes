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

from src.eiot import  ConditionalEIOT
from src.utils import train_lbfgs, plot_preds
from src.mf_methods import MFStaticSignals


torch.set_default_dtype(torch.float64)
# %%

def main(args):
    experiment_name = f"Lovastain_test_size={args.test_size}_num_unobs={args.num_unobs}_num_samples={args.num_samples}_covar_rank={args.covar_rank}_sum_to_one={args.sum_to_one}"
    work_dir = Path(args.work_dir)
    torch.manual_seed(args.random_seed)
    d_test = torch.load(f"data/Dalby/splits/d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    d_train = torch.load(f"data/Dalby/splits/d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    r_train = torch.load(f"data/Dalby/splits/r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    r_test = torch.load(f"data/Dalby/splits/r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    mf_static_signals = MFStaticSignals(M=d_test.shape[1], C_obs=r_train.shape[1], C_unobs=args.num_unobs, sigma2=1, sum_to_one=args.sum_to_one)
    cond_eiot = ConditionalEIOT(M=d_test.shape[1], C_obs=r_train.shape[1], C_unobs=args.num_unobs, sigma2=1, P=args.num_samples, sum_to_one=args.sum_to_one)

    mf_ll = []
    # mf_ll = train_eiot_sgd(mf_eiot, D_test=d_test, D_train=d_train, R_train=r_train, n_steps=1, plot=args.plot)
    mf_ll.append(train_lbfgs(mf_static_signals, D_test=d_test, D_train=d_train, R_train=r_train, steps=25, plot=args.plot, convergence_level=5))
    # mf_ll = train_eiot_sgd(mf_eiot, d_train, r_train, batch_size=32, n_steps=10000, plot=True)
    # print("IMPORTANT: ", mf_eiot._log_priors_T)
    # cond_ll = train_eiot_sgd(cond_eiot, d_train, r_train, batch_size=8, n_steps=10, plot=True)
    # cond_ll = train_eiot_gd(cond_eiot, D_test=d_test, D_train=d_train, R_train=r_train, n_steps=500, plot=args.plot, )
    # cond_ll = train_eiot_lbfgs(cond_eiot, D_test=d_test, D_train=d_train, R_train=r_train, n_steps=2, plot=args.plot, to_convergence=True)
    # cond_ll = train_eiot_sgd(cond_eiot, D_test=d_test, D_train=d_train, R_train=r_train, n_steps=1000, plot=args.plot)

    mf_preds = mf_static_signals.predict_weights(D_test=d_test)
    # cond_preds = cond_eiot.predict_weights(D_test=d_test)

    mf_rmse = torch.sqrt(((r_test - mf_preds.mean[:, :2])**2).mean())
    # cond_rmse = torch.sqrt(((r_test - cond_preds.mean.mean(dim=[0])[:, :2])**2).mean())



    mf_log_prob = torch.distributions.MultivariateNormal(mf_preds.mean[:,:r_train.shape[1]], mf_preds.covariance_matrix[:, :r_train.shape[1], :r_train.shape[1]]).log_prob(r_test).sum()
    # cond_log_prob = torch.log(torch.exp(torch.distributions.MultivariateNormal(cond_preds.mean[:, :,:r_train.shape[1]], cond_preds.covariance_matrix[:,:, :r_train.shape[1], :r_train.shape[1]]).log_prob(r_test)).mean(dim=0)).sum()
    # cond_log_prob = torch.log(torch.exp(torch.distributions.MultivariateNormal(cond_preds.mean[:, :,:r_train.shape[1]], cond_preds.covariance_matrix[:,:, :r_train.shape[1], :r_train.shape[1]]).log_prob(r_test)).mean(dim=0)).sum()

    print(mf_rmse)
    # print(cond_rmse) 

    print(mf_log_prob)
    # print(cond_log_prob)
    if args.plot: 
        plot_preds(
            R_test=r_test,  
            mf_preds=mf_preds,
            # cond_preds=cond_preds
        )

    mf_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "msep": mf_rmse.item(), "log_prob": mf_log_prob.item(), "elbo": mf_ll[-1].item()}
    cond_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "msep": cond_rmse.item(), "log_prob": cond_log_prob.item(), "elbo": cond_ll[-1].item()}

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "Dalby"/ "mf_eiot" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=mf_results_dict)
    
    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "Dalby"/ "cond_eiot" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=cond_results_dict)


    save_parameters(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "Dalby"/ "mf_eiot" / "models", 
                    model=mf_static_signals)
    
    save_parameters(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "Dalby"/ "cond_eiot" / "models", 
                    model=cond_eiot)
    


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
        type=int,
        default=0.5,
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
        default=True,
        help="plots figures",
    )

    argparser.add_argument(
        "--work_dir",
        "-wd",
	    type=str,
        default="/Users/jo816/projects/conditional_overlaping_gaussian_processes",
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
        "--num_samples",
        "-ns",
	    type=int,
        default=1024,
        help="number of MC samples.",
    )

    argparser.add_argument(
        "--covar_rank",
        "-cr",
	type=int,
        default=1,
        help="rank of low rank covariance matrix.",
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
