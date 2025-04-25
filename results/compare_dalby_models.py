# %%

import pandas as pd
import matplotlib.pyplot as plt
import torch

# from src.prob_utils import get_struc_ci
# from src.struc_models import StrucVariableSignals

torch.set_default_dtype(torch.float64)


ts = 0.5


paths = {
    # "static": f"results/Dalby/mf_static_signals/csvs/Dalby_test_size={ts}_num_unobs=1.csv",
    # "mf": f"results/Dalby/mf_var_signals/csvs/Dalby_test_size={ts}_c_unobs=2_L=4_A=2.csv",
    # "struc":f"results/Dalby/struc_var_signals/csvs/Dalby_test_size={ts}_c_unobs=2_L=4_A=2_P=8_an=False_iz=False.csv", 
    "mf": "results/Dalby/mf_var_signals/csvs/Dalby_test_size=0.5_c_unobs=0_Lx=8_Ll=16_A=4.csv"
    }



for k,p in paths.items():

    df = pd.read_csv(p)
    df["msep"] = df["rmsep"]**2
    datas= df.groupby("data_set")
    elbo_idx = datas["elbo"].idxmax()
    max_elbo = df[["log_prob","msep", "coverage", "mean_interval_length"]].loc[elbo_idx]

    if k == "struc":
        for i in range(10):
            D_test = torch.load(f"data/Dalby/splits/d_test_seed={i}_test_size={ts}_trimmin=150_trimmax=250.pkl", weights_only=False)
            R_test = torch.load(f"data/Dalby/splits/r_test_seed={i}_test_size={ts}_trimmin=150_trimmax=250.pkl", weights_only=False)
            R_train = torch.load(f"data/Dalby/splits/r_train_seed={i}_test_size={ts}_trimmin=150_trimmax=250.pkl", weights_only=False)
            V = torch.randn(4,2)
            mu_x_train = torch.zeros(R_train.shape[0], 2)
            mu_x_test = torch.zeros(R_test.shape[0], 2)
            model = StrucVariableSignals(P=8,mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=D_test.shape[1], V=V, C_obs=R_train.shape[1], C_unobs=2, sigma2=1, weight_distribution="sum_to_one", add_randn_to_lvs=False)
            state_dict = torch.load(f"results/Dalby/struc_var_signals/models/Dalby_test_size={ts}_c_unobs=2_L=4_A=2_P=8_an=False_iz=False_rs={df["random_seed"].loc[elbo_idx.loc[i]]}_data_idx={i}.pt", weights_only=True)
            new_dic = {}
            for key, item in state_dict.items():
                new_dic[key[10:]] = item
            model.load_state_dict(new_dic)
            preds = model.predict_weights(D_test)
            rmse = torch.sqrt(((R_test - preds.mean[:,:, :2].mean(dim=0))**2).mean())
            log_prob = torch.log(torch.distributions.MultivariateNormal(preds.mean[...,:R_train.shape[1]], preds.covariance_matrix[..., :R_train.shape[1], :R_train.shape[1]]).log_prob(R_test).exp().mean(dim=0)).sum()
            # rmse = torch.sqrt(((R_test - preds.mean[..., :2])**2).mean())

            # print(rmse.item()**2, max_elbo["msep"].iloc[i])
            interval_length = 0
            for c in range(model.C_obs):
                lb, ub = get_struc_ci(preds=preds, dim=c)
                interval_length += (ub - lb).sum() # Note: this is incorrectly calculated here, but corrected in the analysis

            mean_interval_length = interval_length/(R_test.shape[0]*R_test.shape[1])
            max_elbo.loc[i, "mean_interval_length"] = mean_interval_length
            max_elbo.loc[i, "msep"] = rmse.item()**2
            max_elbo.loc[i, "log_prob"] = log_prob.item()

    print("\n\n",k)
    print("MEAN")
    print(max_elbo.mean())
    print(max_elbo["msep"].mean())
    print("STD")
    print(max_elbo.std())
    print(max_elbo["msep"].std())


# %%
plt.figure()
df = pd.read_csv(f"results/Dalby/mf_var_signals/csvs/Dalby_test_size=0.5_c_unobs=0_Lx=8_Ll=16_A=4.csv")
df["msep"] = df["rmsep"]**2

datas= df.groupby("data_set")
elbo_idx = datas["elbo"].idxmax()
max_elbo = df[["log_prob","msep", "coverage", "mean_interval_length"]].loc[elbo_idx]

plt.scatter(df["elbo"].loc[elbo_idx], df["coverage"].loc[elbo_idx], marker="o",s=64)

plt.ylabel("Coverage")
plt.xlabel("ELBO")
plt.legend(loc="lower left")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()


