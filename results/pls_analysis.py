# %%

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


ts = "0.1"



name = "Wulfert"
ts = "0.1"


data = pd.read_csv(f"results/Wulfert/pls/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=32.csv")


# %%
data["msep"] = data["rmsep"]**2 # This is bad, but is the easiest way to do this 

# %%
print(data["msep"].mean()*1e4)
print(data["msep"].std())
# %%
# elbo_idx = datas["elbo"].idxmax()
# rmse_idx = datas["rmsep"].idxmin()
# nll_idx = datas["log_prob"].idxmax()
# %%

