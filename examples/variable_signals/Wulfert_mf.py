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

from src.mf_methods import MFVariableSignals


torch.set_default_dtype(torch.float64)
# %%

def main(args):
    experiment_name = f"Wulfert_test_size={args.test_size}_num_unobs={args.num_unobs}_num_inducing={args.num_inducing}_num_latent_dims={args.num_latent_dims}"
    work_dir = Path(args.work_dir)
    torch.manual_seed(args.random_seed)
    D_test = torch.load(f"data/Wulfert/splits/d_test_seed={args.data_idx}_test_size={args.test_size}.pkl")
    D_train = torch.load(f"data/Wulfert/splits/d_train_seed={args.data_idx}_test_size={args.test_size}.pkl")

    R_train = torch.load(f"data/Wulfert/splits/r_train_seed={args.data_idx}_test_size={args.test_size}.pkl")
    R_test = torch.load(f"data/Wulfert/splits/r_test_seed={args.data_idx}_test_size={args.test_size}.pkl")

    V = torch.randn(args.num_inducing,args.num_latent_dims)

    mu_x_train, mu_x_test = get_both_init_values_for_latent_variables(D_train=D_train, R_train=R_train, D_test=D_test, num_latent_dimensions=args.num_latent_dims)
 

    model = MFVariableSignals(mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=D_test.shape[1], V=V, C_obs=R_train.shape[1], C_unobs=args.num_unobs, sigma2=1, sum_to_one=args.sum_to_one)

    ll = []
    ll.extend(train_lbfgs(model, D_train, R_train, D_test, 20, plot=args.plot, convergence_level=1000))

    mf_preds = model.predict_weights(D_test)

    
    mf_rmse = torch.sqrt(((R_test - mf_preds.mean[:, :3])**2).mean())

    mf_log_prob = torch.distributions.MultivariateNormal(mf_preds.mean[:,:R_train.shape[1]], mf_preds.covariance_matrix[:, :R_train.shape[1], :R_train.shape[1]]).log_prob(R_test).sum()



    print(mf_rmse)
    print("scale",model.sigma2_s)
    print("beta",model.beta)

    if args.plot == True: 
        plot_preds(
            R_test=R_test,  
            mf_preds=mf_preds,
        )

        # with torch.no_grad():
        #     plt.figure()
        #     plt.scatter(mu_x_train[:,0], mu_x_train[:,1])
        #     plt.scatter(mu_x_test[:,0], mu_x_test[:,1])
        #     plt.show()
        

    mf_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "msep": mf_rmse.item(), "log_prob": mf_log_prob.item(), "elbo": ll[-1]}


    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "mf_eiot" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=mf_results_dict)

    
    save_parameters(file_name= f'{experiment_name}.csv', 
                    path=work_dir / "results" / "Wulfert"/ "mf_eiot", 
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
        default=3,
        help="number of unobserved components",
    )



    # argparser.add_argument(
    #     "--covar_rank",
    #     "-cr",
	# type=int,
    #     default=1,
    #     help="rank of low rank covariance matrix.",
    # )

    argparser.add_argument(
        "--sum_to_one",
        "-sto",
	type=bool,
        default=True,
        help="should the components sum to one",
    )

    # argparser.add_argument(
    #     "--min_wavelength",
    #     "-minwl",
	# type=int,
    #     default=150,
    #     help="where to trim spectra below",
    # )
    # argparser.add_argument(
    #     "--max_wavelength",
    #     "-maxwl",
	# type=int,
    #     default=250,
    #     help="trim spectra above",
    # )

    argparser.add_argument(
        "--num_latent_dims",
        "-nld",
	type=int,
        default=3,
        help="number of latent dimensions",
    )

    argparser.add_argument(
        "--num_inducing",
        "-ni",
        type=int,
        default=12,
        help="number of inducing dimensions"
    )

    args = argparser.parse_args()

    main(args)
