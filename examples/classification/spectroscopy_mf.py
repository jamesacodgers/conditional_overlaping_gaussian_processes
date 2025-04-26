# %%
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.mf_methods import MF_WS_GPLVM
from src.save_utils import save_model, save_results_csv
from src.utils import get_both_init_values_for_latent_variables, train_gd, train_lbfgs, snv_correction

from torcheval.metrics import BinaryAUROC
from pathlib import Path

import argparse


def main(args):


    torch.set_default_dtype(torch.float64)
    work_dir = Path(args.work_dir)

    experiment_name = f"spectroscopy_seed_L={args.num_inducing}_A={args.num_latent_dims}"


    training_df = pd.read_csv(f"data/Hyperspectral/Rock_TRAIN_diff_fold_{args.data_idx}.tsv", delimiter="\t", header=None)
    test_df = pd.read_csv(f"data/Hyperspectral/Rock_TEST_diff_fold_{args.data_idx}.tsv", delimiter="\t",header=None)


    R_train = torch.tensor(training_df[0])-1
    R_test= torch.tensor(test_df[0])-1

    D_train=torch.tensor(training_df.drop(0,axis=1).to_numpy())
    D_test=torch.tensor(test_df.drop(0,axis=1).to_numpy())
    D_test=snv_correction(D_test)
    D_train=snv_correction(D_train)

    # %%
    A = 2

    mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train,torch.nn.functional.one_hot(R_train), D_test, A)

    V = torch.randn(args.num_inducing,A)

    R_train_obs = R_train
    model = MF_WS_GPLVM(C_obs=4, C_unobs=0, M = D_test.shape[1],V=V,weight_distribution="categorical", mu_x_test=mu_x_test, mu_x_train=mu_x_train)

    ll = []

    ll.extend(
        train_gd(model=model, 
            D_train=D_train, 
            R_train=R_train_obs, 
            D_test=D_test, 
            n_steps=1000, 
            lr = 1e-2,
            plot=False
            )
    )

    # train_lbfgs(model=model, 
    #             D_train=D_train, 
    #             R_train=R_train_obs, 
    #             D_test=D_test, 
    #             n_steps=10, 
    #             plot=False,
    #             convergence_level=5)


    # %%
    preds = model.predict_weights(D_test, )
    print(torch.argmax(preds) - R_test)

    # %%
    # plot_preds(R_test, mf_preds=preds, save_path=f"figs/toy_example/mf_preds_i={independent_latents}.pdf" )
    acc = torch.mean((torch.argmax(preds, dim=1) == R_test).type(torch.float))


    log_prob = torch.log(torch.gather(preds, dim=1, index=R_test[:,None])).sum()
    auc = BinaryAUROC()
    for c in range(model.C):
        class_vals = (R_test == c).type(torch.int)
    auc.update(preds[:,c], class_vals)
    auc.compute()
        

    mf_results_dict = {"random_seed": args.random_seed, "data_set": args.data_idx, "acc": acc.item(), "log_prob": log_prob.item(), "elbo": ll[-1], "auc": auc.compute().item()}


    model.elbo(
            D_train=D_train,
            R_train=R_train_obs,
            #    D_test=D_test
            )
    save_results_csv(file_name= f'{experiment_name}.csv', 
                    path= work_dir / "results" / "classification"/ "mf_var_signals" / "csvs" / f'{experiment_name}.csv', 
                    results_dict=mf_results_dict)
    

    save_model(experiment_name=experiment_name, seed=args.random_seed, data_idx=args.data_idx, 
                    path=work_dir / "results" / "classification"/ "mf_var_signals" / "models", 
                    model=model)



    # %%





if __name__ == "__main__":

    import os

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        "--random_seed",
        "-rs",
        type=int,
        default=0,
        help="Random seed.",
    )

    # argparser.add_argument(
    #     "--test_size",
    #     "-ts",
    #     type=int,
    #     default=0.1,
    #     help="Test size.",
    # )

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
        default=8,
        help="number of inducing dimensions"
    )



    args = argparser.parse_args()

    main(args)
