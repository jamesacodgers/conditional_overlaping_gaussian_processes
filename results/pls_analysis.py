# %%

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


ts = "0.1"



data = pd.read_csv(f"results/Dalby/pls/csvs/Dalby_test_size={ts}.csv")
    


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

