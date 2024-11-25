# %%

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

name = "Dalby"
ts = "0.5"
# l = 8

if name == "Dalby":
    data = pd.read_csv(f"results/Dalby/struc_var_signals/csvs/Dalby_test_size={ts}_c_unobs=1_L=4_A=2_P=8_an=True_iz=False.csv")
else: 
    data = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size={ts}_c_unobs=1_L=4_A=4_P=8_an=True_iz=False.csv")

# %%
data["msep"] = data["rmsep"]**2 # This is bad, but is the easiest way to do this 

# %%
datas= data.groupby("data_set")
# %%
elbo_idx = datas["elbo"].idxmax()
rmse_idx = datas["rmsep"].idxmin()
nll_idx = datas["log_prob"].idxmax()
# %%
print("msep")
print(data["msep"].loc[elbo_idx].mean()*1e4)
# print(data["msep"].loc[elbo_idx].min())
print(data["msep"].loc[elbo_idx].std()*1e4)
print("nll")
print(data["log_prob"].loc[elbo_idx].mean())
# print(data["msep"].loc[elbo_idx].min())
print(data["log_prob"].loc[elbo_idx].std())
# %%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Use LaTeX default font (Computer Modern)
    "font.serif": ["Computer Modern"],  # Alternatively, specify another font
    "axes.labelsize": 12,  # Font size for axis labels
    "font.size": 12,       # General font size
    "legend.fontsize": 10, # Font size for legends
    "xtick.labelsize": 10, # Font size for x-axis tick labels
    "ytick.labelsize": 10  # Font size for y-axis tick labels
})



# %%
plt.figure()
plt.scatter(data["elbo"], data["msep"], marker="x", label="all runs")
plt.scatter(data["elbo"].loc[rmse_idx], data["msep"].loc[rmse_idx], marker="v", label="best MSE")
plt.scatter(data["elbo"].loc[nll_idx], data["msep"].loc[nll_idx], marker="^", label = "best NLL")
plt.scatter(data["elbo"].loc[elbo_idx], data["msep"].loc[elbo_idx], marker="o", label = "best ELBO", s=64)
plt.ylabel("MSEP")
plt.xlabel("ELBO")
plt.legend(loc="upper left")
plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()


# plt.figure()
# plt.scatter(data["elbo"], -data["log_prob"], marker="x")
# plt.scatter(data["elbo"].loc[rmse_idx], data["log_prob"].loc[rmse_idx], marker="v")
# plt.scatter(data["elbo"].loc[nll_idx], data["log_prob"].loc[nll_idx], marker="^")
# plt.scatter(data["elbo"].loc[elbo_idx], -data["log_prob"].loc[elbo_idx], marker="o", s=64)
# plt.savefig(f"figs/{name}/elbo_vs_nll_ts={ts}_l={l}.pdf")
# plt.ylabel("NLL")
# plt.xlabel("ELBO")

# plt.show()

