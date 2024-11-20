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

from sklearn.model_selection import KFold


torch.set_default_dtype(torch.float64)
# %%

def main(args):
    experiment_name = f"Dalby_test_size={args.test_size}"
    work_dir = Path(args.work_dir)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    d_test = torch.load(f"data/Dalby/splits/d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    d_train = torch.load(f"data/Dalby/splits/d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    r_train = torch.load(f"data/Dalby/splits/r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    r_test = torch.load(f"data/Dalby/splits/r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    mf_static_signals = MFStaticSignals(M=d_test.shape[1], C_obs=r_train.shape[1], C_unobs=args.num_unobs, sigma2=1, weight_distribution="sum_to_one")


    train_x = d_train
    train_y = r_train

    test_x = d_test
    test_y = r_test

    nA = 10
    msep = np.zeros([nA,3])

    kf = KFold(n_splits=10, shuffle=True)

    for i in range(nA):
        for train_idx, test_idx in kf.split(train_x):
            pls_reg = PLSRegression(i+1)
            pls_reg.fit(train_x[train_idx], train_y[train_idx])
            # y_hat = pls_reg.predict(test_spectra.spectra)
            y_hat = pls_reg.predict(train_x[test_idx])
            msep[i] += ((y_hat - train_y.numpy()[test_idx])**2).sum()

    min_err_components = np.argmin(msep.sum(axis=1))

    pls_reg = PLSRegression(min_err_components.item()+1)
    pls_reg.fit(train_x, train_y)
    y_hat = pls_reg.predict(test_x)

    if args.plot:
        fig, ax = plt.subplots(3)
        for i in range(3):
            ax[i].scatter(test_y, y_hat)
        plt.show()
    rmsep = np.sqrt(((y_hat - test_y.numpy())**2).mean())

    print("msep:", )
    print("cross val error:", min_err_components)

    






    mf_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "rmsep": rmsep}

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "Dalby"/ "pls" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=mf_results_dict)
    

    


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
