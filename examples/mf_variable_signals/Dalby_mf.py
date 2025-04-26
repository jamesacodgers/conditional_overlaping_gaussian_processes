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

from src.mf_methods import MF_WS_GPLVM


torch.set_default_dtype(torch.float64)
# %%

def main(args):
    print(args.test_size, args.num_inducing, args.random_seed, args.data_idx)


    torch.manual_seed(args.random_seed)


    experiment_name = f"Dalby_test_size={args.test_size}_c_unobs={args.num_unobs}_Lx={args.num_inducing}_Ll={args.num_wl_inducing}_A={args.num_latent_dims}"
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


    # model = MFVariableSignals(mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=D_test.shape[1], V=V, C_obs=R_train.shape[1], C_unobs=args.num_unobs, sigma2=1, weight_distribution="sum_to_one")
    model = MF_WS_GPLVM(mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=D_test.shape[1], Vx=V, Vl = torch.linspace(0,1,args.num_wl_inducing).reshape(-1,1), C_obs=R_train.shape[1], C_unobs=args.num_unobs, sigma2=1, weight_distribution="sum_to_one")
    ll = []

    model = torch.compile(model)

    ll.extend(train_gd(model=model, D_train=D_train, R_train=R_train, D_test=D_test, n_steps=1000, plot=False,lr = 0.01))
    ll.extend(train_lbfgs(model=model, D_train=D_train, R_train=R_train, D_test=D_test, n_steps=5, plot=False, convergence_level=10))

    preds = model.predict_weights(D_test)

    log_prob = torch.distributions.MultivariateNormal(preds.mean[:,:R_train.shape[1]], preds.covariance_matrix[:, :R_train.shape[1], :R_train.shape[1]]).log_prob(R_test).sum()
    rmse = torch.sqrt(((R_test - preds.mean[:, :2])**2).mean())
    
    print("beta",model.gp.beta)
    print("sigma2s",model.gp.sigma2_s)
    # for name,param in model.named_parameters():
    #     print(name, param)

    print(log_prob)

    if args.plot: 
        plot_preds(
            R_test=R_test,  
            mf_preds=preds,
            # cond_preds=cond_preds
        )

        # S = model.gp.get_gp_expectation(model.mu_x_train)
        # with torch.no_grad():
        #     # for p in range(16):
        #     fig, axs = plt.subplots(nrows=model.C_obs, ncols=model.C_unobs, figsize = (6, 6))
        #     for c in range(model.C_obs):
        #         for d in range(model.C_unobs):
        #             for i in range(R_train.shape[0]):
        #                 axs[c,d].plot(S[i,:,c*2+d], alpha = 0.5)
        #     plt.ylabel("pure component mean")
        #     plt.xlabel("wavelength")

        #     plt.show()

        # argmin_beta = torch.argsort(model.gp.beta)
        # print(argmin_beta)
        # with torch.no_grad():
        #     plt.figure()
        #     plt.scatter(model.mu_x_train[:,0], model.mu_x_train[:,1])
        #     plt.scatter(model.mu_x_test[:,0], model.mu_x_test[:,1])
        #     plt.show()

    coverage = 0
    for c in range(model.C_obs):
        lb = preds.mean[:,c] - 1.96*preds.stddev[:,c]
        ub = preds.mean[:,c] + 1.96*preds.stddev[:,c]
        coverage += torch.logical_and(lb < R_test[:,c], ub > R_test[:,c]).type(torch.float32).sum()
    coverage = coverage/(R_test.shape[0]*R_test.shape[1])

    mean_interval_length = (1.96*2*preds.stddev[:,0]).mean()



    results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "rmsep": rmse.item(), "log_prob": log_prob.item(), "elbo": ll[-1], "mean_interval_length": mean_interval_length.item(), "coverage": coverage.item() }

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "Dalby"/ "mf_var_signals" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)
    

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
        default=0.5,
        help="Test size.",
    )



    argparser.add_argument(
        "--plot",
        "-p",
        action=argparse.BooleanOptionalAction,
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
        "--num_unobs",
        "-nu",
	    type=int,
        # default=1,
        default=0,
        help="number of unobserved components",
    )


    argparser.add_argument(
        "--num_latent_dims",
        "-a",
	    type=int,
        default=4,
        help="number of latent dimensions",
    )

    argparser.add_argument(
        "--num_inducing",
        "-ni",
        type=int,
        default=8,
        help="number of inducing dimensions"
    )

    argparser.add_argument(
        "--num_wl_inducing",
        "-nil",
        type=int,
        default=32,
        help="number of inducing dimensions"
    )

    args = argparser.parse_args()

    main(args)
