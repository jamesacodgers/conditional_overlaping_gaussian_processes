# %%

import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from itertools import chain
import copy
import numpy as np
import pandas as pd
from src.prob_utils import get_struc_ci
from src.save_utils import save_model, save_results_csv
from src.utils import plot_preds
from src.utils import get_both_init_values_for_latent_variables, train_gd, train_lbfgs

from sklearn.cross_decomposition import PLSRegression

from src.struc_models import StrucVariableSignals


torch.set_default_dtype(torch.float64)
# %%

def main(args):
    print(args.plot, args.data_idx, args.random_seed)


    torch.manual_seed(args.random_seed)

    # if args.no_snv:
    #     experiment_name = f"Structured_Dalby_test_size={args.test_size}_c_unobs={args.num_unobs}_L={args.num_inducing}_A={args.num_latent_dims}_nosnv"
    #     work_dir = Path(args.work_dir)
    #     D_test = torch.load(f"data/Dalby/splits/no_snv_d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    #     D_train = torch.load(f"data/Dalby/splits/no_snv_d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    #     R_train = torch.load(f"data/Dalby/splits/no_snv_r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    #     R_test = torch.load(f"data/Dalby/splits/no_snv_r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    # else:
    experiment_name = f"Dalby_test_size={args.test_size}_c_unobs={args.num_unobs}_L={args.num_inducing}_A={args.num_latent_dims}_P={args.num_samples}_an={args.add_noise_to_lvs}_iz={args.init_zero}"
    work_dir = Path(args.work_dir)
    D_test = torch.load(f"data/Dalby/splits/d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin=150_trimmax=250.pkl")
    D_train = torch.load(f"data/Dalby/splits/d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin=150_trimmax=250.pkl")

    R_train = torch.load(f"data/Dalby/splits/r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin=150_trimmax=250.pkl")
    R_test = torch.load(f"data/Dalby/splits/r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin=150_trimmax=250.pkl")

    V = torch.randn(args.num_inducing,args.num_latent_dims)

    if args.init_zero:
        mu_x_train = torch.zeros(R_train.shape[0], args.num_latent_dims)
        mu_x_test = torch.zeros(R_test.shape[0], args.num_latent_dims)
    else: 
        mu_x_train,mu_x_test = get_both_init_values_for_latent_variables(D_train=D_train, D_test=D_test, R_train=R_train, num_latent_dimensions=args.num_latent_dims)

    model = StrucVariableSignals(P=args.num_samples,mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=D_test.shape[1], V=V, C_obs=R_train.shape[1], C_unobs=args.num_unobs, sigma2=1, weight_distribution="sum_to_one", add_randn_to_lvs=args.add_noise_to_lvs)
    model = torch.compile(model)
    ll = []


    ll.extend(train_gd(model, D_train, R_train, D_test=D_test, n_steps=1000, plot=False, lr = 2e-2))
    ll.extend(train_lbfgs(model, D_train, R_train, D_test=D_test, n_steps=10, plot=False, convergence_level=5))

    preds = model.predict_weights(D_test)

    log_prob = torch.distributions.MultivariateNormal(preds.mean[...,:R_train.shape[1]], preds.covariance_matrix[..., :R_train.shape[1], :R_train.shape[1]]).log_prob(R_test).sum()
    rmse = torch.sqrt(((R_test - preds.mean[..., :2])**2).mean())
    
    coverage = 0
    interval_length = 0
    for c in range(model.C_obs):
        lb, ub = get_struc_ci(preds=preds, dim=c)
        coverage += np.logical_and(lb < R_test[:,c].numpy(), ub > R_test[:,c].numpy()).sum()
        interval_length = (ub - lb).sum()
    coverage = coverage/(R_test.shape[0]*R_test.shape[1])

    mean_interval_length = interval_length/(R_test.shape[0]*R_test.shape[1])



    results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "rmsep": rmse.item(), "log_prob": log_prob.item(), "elbo": ll[-1], "mean_interval_length": mean_interval_length.item(), "coverage": coverage.item() }


    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "Dalby"/ "struc_var_signals" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)
    

    save_model(experiment_name=experiment_name, seed=args.random_seed, data_idx=args.data_idx, 
                    path=work_dir / "results" / "Dalby"/ "struc_var_signals" / "models", 
                    model=model)
    



    print("beta",model.beta)
    print("sigma2s",model.sigma2_s)
    # for name,param in model.named_parameters():
    #     print(name, param)

    print(log_prob)
    print(rmse)
    if args.plot: 
        

        S = model.get_s_expectation(model.mu_x_train,model.Sigma_x_train)


        with torch.no_grad():
            # for p in range(16):
            fig, axs = plt.subplots(model.C, figsize = (6, 1.5*model.C))
            for c in range(model.C):
                for i in range(R_train.shape[0]):
                    for p in range(model.P):
                        axs[c].plot(S[p,i,:,c], alpha = 0.5)
            plt.ylabel("pure component mean")
            plt.xlabel("wavelength")

            plt.show()
        argmin_beta = torch.argsort(model.beta)
        print(argmin_beta)
        with torch.no_grad():
            plt.figure()
            for i in range(R_train.shape[0]):
                plt.scatter(model.mu_x_train[:,i,0], model.mu_x_train[:,i,1])
            for i in range(R_test.shape[0]):
                plt.scatter(model.mu_x_test[:,i,0], model.mu_x_test[:,i,1], marker = "x")
            plt.show()

        plot_preds(
            R_test=R_test,  
            # mf_preds=preds,
            cond_preds=preds
        )
    
    



    


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
        default=5,
        help="number of unobserved components",
    )

    argparser.add_argument(
        "--num_latent_dims",
        "-a",
	type=int,
        default=2,
        help="number of latent dimensions",
    )

    argparser.add_argument(
        "--num_inducing",
        "-ni",
        type=int,
        default=4,
        help="number of inducing dimensions"
    )

    argparser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=8,
        help="number of samples"
    )
    
    argparser.add_argument(
        "--init_zero",
        "-iz",
        type = bool,
        default = False,
        help = "lvs initialized with zero"
    )

    argparser.add_argument(
        "--add_noise_to_lvs",
        "-an",
        type = bool,
        default = True,
        help = "should noise be added to the latent variables on init"
    )

    # argparser.add_argument(
    #     "--no_snv",
    #     "-nsnv",
    #     type=bool,
    #     default=False,
    #     help="number of inducing dimensions"
    # )

    args = argparser.parse_args()

    main(args)
