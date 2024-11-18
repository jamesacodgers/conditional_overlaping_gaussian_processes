# %%

import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from src.save_utils import save_parameters, save_results_csv

from sklearn.cross_decomposition import PLSRegression



torch.set_default_dtype(torch.float64)
# %%

def main(args):
    print(args.plot, args.data_idx, args.random_seed)


    experiment_name = f"Dalby_test_size={args.test_size}_c_unobs={args.num_unobs}_sto={args.sum_to_one}_L={args.num_inducing}_A={args.num_latent_dims}"
    work_dir = Path(args.work_dir)
    torch.manual_seed(args.random_seed)


    D_test = torch.load(f"data/Dalby/splits/no_snv_d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    D_train = torch.load(f"data/Dalby/splits/no_snv_d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    R_train = torch.load(f"data/Dalby/splits/no_snv_r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    R_test = torch.load(f"data/Dalby/splits/no_snv_r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    # D_test = torch.load(f"data/Dalby/splits/d_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    # D_train = torch.load(f"data/Dalby/splits/d_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")

    # R_train = torch.load(f"data/Dalby/splits/r_train_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")
    # R_test = torch.load(f"data/Dalby/splits/r_test_seed={args.data_idx}_test_size={args.test_size}_trimmin={args.min_wavelength}_trimmax={args.max_wavelength}.pkl")


   
    # for name,param in model.named_parameters():
    #     print(name, param)
    pls = PLSRegression(8)
    pls.fit(D_train, R_train)
    y_hat = pls.predict(D_test)
    rmsep = torch.sqrt(((R_test - y_hat)**2)).mean()
    print(rmsep)
    if args.plot: 
        plt.figure()
        plt.scatter(R_test[:,0], y_hat[:,0])
        plt.show()
        

    mf_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "msep": mf_rmse.item(), }

    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "Dalby"/ "mf_var_signals" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=mf_results_dict)
    

    save_parameters(file_name= f'{experiment_name}.csv', 
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
        default=0,
        help="Random seed.",
    )

    argparser.add_argument(
        "--test_size",
        "-ts",
        type=int,
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
        action=argparse.BooleanOptionalAction,
        default=False,
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
        # default=1,
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

    args = argparser.parse_args()

    main(args)
