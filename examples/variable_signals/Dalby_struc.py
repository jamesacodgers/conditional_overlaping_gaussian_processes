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
from src.utils import plot_preds
from src.utils import get_both_init_values_for_latent_variables, train_gd, train_lbfgs

from sklearn.cross_decomposition import PLSRegression

from src.struc_models import StrucVariableSignals


torch.set_default_dtype(torch.float64)
# %%

def main(args):
    experiment_name = f"Dalby_test_size={args.test_size}_num_unobs={args.num_unobs}_sum_to_one={args.sum_to_one}_num_latent_dims={args.num_latent_dims}_num_inducing_points={args.num_inducing}_num_samples={args.num_samples}"
    work_dir = Path(args.work_dir)
    torch.manual_seed(args.random_seed)
    D_test = torch.load(f"data/Dalby/splits/no_snv_d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    D_train = torch.load(f"data/Dalby/splits/no_snv_d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    R_train = torch.load(f"data/Dalby/splits/no_snv_r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    R_test = torch.load(f"data/Dalby/splits/no_snv_r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")


    V = torch.randn(args.num_inducing,args.num_latent_dims)

    mu_x_train, mu_x_test = get_both_init_values_for_latent_variables(D_train=D_train, R_train=R_train, D_test=D_test, num_latent_dimensions=args.num_latent_dims)
 

    model = StrucVariableSignals(P=args.num_samples,mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=D_test.shape[1], V=V, C_obs=R_train.shape[1], C_unobs=args.num_unobs, sigma2=1, sum_to_one=args.sum_to_one)
    model = torch.compile(model)
    ll = []


    ll.extend(train_gd(model, D_train, R_train, D_test, 400, plot=args.plot, lr = 2e-2))
    ll.extend(train_lbfgs(model, D_train, R_train, D_test, 10, plot=args.plot, convergence_level=5))

    preds = model.predict_weights(D_test)

    log_prob = torch.distributions.MultivariateNormal(preds.mean[...,:R_train.shape[1]], preds.covariance_matrix[..., :R_train.shape[1], :R_train.shape[1]]).log_prob(R_test).sum()
    rmse = torch.sqrt(((R_test - preds.mean[..., :2])**2).mean())
    
    print("beta",model.beta)
    print("sigma2s",model.sigma2_s)
    # for name,param in model.named_parameters():
    #     print(name, param)

    print(log_prob)
    if args.plot: 
        plot_preds(
            R_test=R_test,  
            cond_preds=preds,
            # cond_preds=cond_preds
        )

    results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "msep": rmse.item(), "log_prob": log_prob.item(), "elbo": ll[-1]}

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "Dalby"/ "struc_var_signals" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=results_dict)
    

    save_parameters(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "Dalby"/ "struc_var_signals" / "models", 
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
        type=int,
        default=0.5,
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
        default="/Users/jo816/projects/conditional_overlaping_gaussian_processes",
        help="working directory",
    )
    
    argparser.add_argument(
        "--num_unobs",
        "-nu",
	    type=int,
        default=4,
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
        default=16,
        help="number of inducing dimensions"
    )

    argparser.add_argument(
        "--num_samples",
        "-ns",
        type=int,
        default=32,
        help="number of samples"
    )

    args = argparser.parse_args()

    main(args)
