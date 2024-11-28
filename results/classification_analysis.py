# %%

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv("results/classification/mf_var_signals/csvs/spectroscopy_seed_L=12_A=3.csv")

datas = df.groupby("data_set")
elbo_idx = datas["elbo"].idxmax()
acc_idx = datas["acc"].idxmax()


plt.figure()

plt.scatter(df["elbo"], df["acc"], marker="x")
plt.scatter(df["elbo"].loc[elbo_idx], df["acc"].loc[elbo_idx], marker="o", s=64)
plt.scatter(df["elbo"].loc[acc_idx], df["acc"].loc[acc_idx], marker="o", s=64)
plt.ylabel("MSEP")
plt.xlabel("ELBO")
plt.legend(loc="upper left")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()