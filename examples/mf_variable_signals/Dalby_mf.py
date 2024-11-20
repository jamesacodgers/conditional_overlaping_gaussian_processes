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
from src.utils import plot_preds
from src.utils import get_both_init_values_for_latent_variables, train_gd, train_lbfgs, get_both_pca_values

from sklearn.cross_decomposition import PLSRegression

from src.mf_methods import MFVariableSignals


torch.set_default_dtype(torch.float64)
# %%

def main(args):
    print(args.test_size, args.num_inducing, args.random_seed, args.data_idx)


    torch.manual_seed(args.random_seed)

    if args.no_snv:
        experiment_name = f"Dalby_test_size={args.test_size}_c_unobs={args.num_unobs}_L={args.num_inducing}_A={args.num_latent_dims}_long_lengthscale"
        work_dir = Path(args.work_dir)
        D_test = torch.load(f"data/Dalby/splits/no_snv_d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
        D_train = torch.load(f"data/Dalby/splits/no_snv_d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

        R_train = torch.load(f"data/Dalby/splits/no_snv_r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
        R_test = torch.load(f"data/Dalby/splits/no_snv_r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    else:
        experiment_name = f"Dalby_test_size={args.test_size}_c_unobs={args.num_unobs}_L={args.num_inducing}_A={args.num_latent_dims}_long_lengthscale"
        work_dir = Path(args.work_dir)
        D_test = torch.load(f"data/Dalby/splits/d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
        D_train = torch.load(f"data/Dalby/splits/d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

        R_train = torch.load(f"data/Dalby/splits/r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
        R_test = torch.load(f"data/Dalby/splits/r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")


    V = torch.randn(args.num_inducing,args.num_latent_dims)

    mu_x_train, mu_x_test = get_both_init_values_for_latent_variables(D_train=D_train, R_train=R_train, D_test=D_test, num_latent_dimensions=args.num_latent_dims)
 
    if args.plot:
        with torch.no_grad():
            plt.figure()
            plt.plot(D_train.T, c="orange", alpha = 0.1)
            plt.plot(D_test.T, c="blue", alpha=0.1)
            plt.show()


    model = MFVariableSignals(mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=D_test.shape[1], V=V, C_obs=R_train.shape[1], C_unobs=args.num_unobs, sigma2=1, weight_distribution="sum_to_one")
    ll = []

    model = torch.compile(model)

    ll.extend(train_gd(model=model, D_train=D_train, R_train=R_train, D_test=D_test, n_steps=1000, plot=False,lr = 0.01))
    ll.extend(train_lbfgs(model=model, D_train=D_train, R_train=R_train, D_test=D_test, n_steps=5, plot=False, convergence_level=1))

    mf_preds = model.predict_weights(D_test)

    mf_log_prob = torch.distributions.MultivariateNormal(mf_preds.mean[:,:R_train.shape[1]], mf_preds.covariance_matrix[:, :R_train.shape[1], :R_train.shape[1]]).log_prob(R_test).sum()
    mf_rmse = torch.sqrt(((R_test - mf_preds.mean[:, :2])**2).mean())
    
    print("beta",model.beta)
    print("sigma2s",model.sigma2_s)
    # for name,param in model.named_parameters():
    #     print(name, param)

    print(mf_log_prob)

    if args.plot: 
        plot_preds(
            R_test=R_test,  
            mf_preds=mf_preds,
            # cond_preds=cond_preds
        )

        S = model.get_s_expectation(model.mu_x_train,model.Sigma_x_train)


        with torch.no_grad():
            # for p in range(16):
            fig, axs = plt.subplots(model.C, figsize = (6, 1.5*model.C))
            for c in range(model.C):
                for i in range(R_train.shape[0]):
                    axs[c].plot(S[i,:,c], alpha = 0.5)
            plt.ylabel("pure component mean")
            plt.xlabel("wavelength")

            plt.show()
        argmin_beta = torch.argsort(model.beta)
        print(argmin_beta)
        with torch.no_grad():
            plt.figure()
            plt.scatter(model.mu_x_train[:,0], model.mu_x_train[:,1])
            plt.scatter(model.mu_x_test[:,0], model.mu_x_test[:,1])
            plt.show()


    mf_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "rmsep": mf_rmse.item(), "log_prob": mf_log_prob.item(), "elbo": ll[-1]}

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "Dalby"/ "mf_var_signals" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=mf_results_dict)
    

    save_model(experiment_name=experiment_name, seed=args.random_seed, data_idx=args.data_idx, 
                    path=work_dir / "results" / "Dalby"/ "mf_var_signals" / "models", 
                    model=model)
    

    


if __name__ == "__main__":
    # Log in to your W&B account
    import os

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        "--random_seed",
        "-rs",
        type=int,
        default=4,
        help="Random seed.",
    )

    argparser.add_argument(
        "--data_idx",
        "-d",
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
        "--plot",
        "-p",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        # default=1,
        default=1,
        help="number of unobserved components",
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

    argparser.add_argument(
        "--num_latent_dims",
        "-a",
	type=int,
        # default=2,
        default=2,
        help="number of latent dimensions",
    )

    argparser.add_argument(
        "--num_inducing",
        "-ni",
        type=int,
        # default=16,
        # default=8,
        default=8,
        help="number of inducing dimensions"
    )

    argparser.add_argument(
        "--no_snv",
        "-nsnv",
        type=bool,
        default=False,
        help="number of inducing dimensions"
    )

    args = argparser.parse_args()

    main(args)
