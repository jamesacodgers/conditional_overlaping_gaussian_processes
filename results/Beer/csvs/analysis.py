# %%

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("results/Beer/csvs/MF_Beer_test_size=0.5_c_unobs=1_L=8_A=2_nosnv.csv")


# %%
datas= data.groupby("data_set")
# %%
datas.max("elbo")["rmsep"].mean()

# %%
data["rmsep"].mean()

# %%


plt.figure()
plt.scatter(data["elbo"], data["rmsep"], c = data["data_set"])
plt.show()

plt.figure()
plt.scatter(data["elbo"], data["log_prob"], c=data["data_set"])
plt.show()

# %%


plt