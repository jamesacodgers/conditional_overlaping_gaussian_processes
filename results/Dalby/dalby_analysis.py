# %%

import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("results/Dalby/mf_var_signals/csvs/Dalby_test_size=0.5_num_unobs=1_covar_rank=1_sum_to_one=True.csv")


# %%
datas= data.groupby("data_set")
# %%
datas.max("elbo")
# %%
