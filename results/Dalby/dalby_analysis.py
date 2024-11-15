# %%

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("results/Dalby/mf_var_signals/csvs/Dalby_test_size=0.5_c_unobs=1_sto=True_L=8_A=2.csv")


# %%
datas= data.groupby("data_set")
# %%
datas.max("elbo")
# %%


plt.figure()
plt.scatter(data["elbo"], data["msep"], c = data["data_set"])
plt.show()

plt.figure()
plt.scatter(data["elbo"], data["log_prob"], c=data["random_seed"])
plt.show()

# %%
