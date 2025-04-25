# %%

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

name = "Wulfert"
ts = "0.5"
l = 8
c_unobs = 2

# if name=="Dalby":
#     data = pd.read_csv(f"results/Dalby/mf_var_signals/csvs/Dalby_test_size={ts}_c_unobs={c_unobs}_L={l}_A=4.csv")
    
# else: 
#     data = pd.read_csv(f"results/Wulfert/mf_variable/csvs/archive/MF_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4.csv")

data = pd.read_csv("results/Wulfert/mf_variable/csvs/MF_Wulfert_test_size=0.5_c_unobs=0_Lx=8_Ll=32_A=4.csv")
# data = pd.read_csv("results/Wulfert/mf_variable/csvs/archive/MF_Wulfert_test_size=0.5_c_unobs=0_L=8_A=4.csv")
# data = pd.read_csv("results/Dalby/mf_var_signals/csvs/Dalby_test_size=0.5_c_unobs=0_Lx=8_Ll=16_A=4.csv")

# %%
data["msep"] = data["rmsep"]**2 # This is bad, but is the easiest way to do this 

# %%
datas= data.groupby("data_set")
# %%
elbo_idx = datas["elbo"].idxmax()
# elbo_idx = datas["elbo"].idxmax()
rmse_idx = datas["rmsep"].idxmin()
nll_idx = datas["log_prob"].idxmax()

# %%
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",  # Use LaTeX default font (Computer Modern)
#     "font.serif": ["Computer Modern"],  # Alternatively, specify another font
#     "axes.labelsize": 12,  # Font size for axis labels
#     "font.size": 12,       # General font size
#     "legend.fontsize": 10, # Font size for legends
#     "xtick.labelsize": 10, # Font size for x-axis tick labels
#     "ytick.labelsize": 10  # Font size for y-axis tick labels
# })





# %%
plt.figure()
plt.scatter(data["elbo"], data["msep"], marker="x", label="all runs")
plt.scatter(data["elbo"].loc[rmse_idx], data["msep"].loc[rmse_idx], marker="v", label="best MSE")
plt.scatter(data["elbo"].loc[nll_idx], data["msep"].loc[nll_idx], marker="^", label = "best NLL")
plt.scatter(data["elbo"].loc[elbo_idx], data["msep"].loc[elbo_idx], marker="o", label = "best ELBO", s=64)
plt.ylabel("MSEP")
plt.xlabel("ELBO")
plt.legend(loc="upper left")
# plt.savefig(f"figs/{name}/elbo_vs_msep_ts={ts}_l={l}.pdf", bbox_inches="tight")
plt.show()



# %%
print("log prob")
print(data["log_prob"].loc[elbo_idx].mean())
# print(data["msep"].loc[elbo_idx].min())
print(data["log_prob"].loc[elbo_idx].std())

print("msep")
print(data["msep"].loc[elbo_idx].mean())
# print(data["msep"].loc[elbo_idx].min())
print(data["msep"].loc[elbo_idx].std())

print("coverage")
print(data["coverage"].loc[elbo_idx].mean())
# print(data["msep"].loc[elbo_idx].min())
print(data["coverage"].loc[elbo_idx].std())

print("interval")
print(data["mean_interval_length"].loc[elbo_idx].mean())
# print(data["msep"].loc[elbo_idx].min())
print(data["mean_interval_length"].loc[elbo_idx].std())


