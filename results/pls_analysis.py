# %%

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


ts = "0.5"





# data = pd.read_csv(f"results/Dalby/pls/csvs/Dalby_test_size={ts}.csv")
data = pd.read_csv(f"results/Wulfert/pls/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=32.csv")


# %%
data["msep"] = data["rmsep"]**2 # This is bad, but is the easiest way to do this 

# %%
print(data["msep"].mean())
print(data["msep"].std())
# %%
# elbo_idx = datas["elbo"].idxmax()
# rmse_idx = datas["rmsep"].idxmin()
# nll_idx = datas["log_prob"].idxmax()
# %%

plt.scatter(data["data_set"], data["msep"])