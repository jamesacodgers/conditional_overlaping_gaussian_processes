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

plt.figure()
plt.scatter(data["elbo"], data["msep"], c = data["data_set"])
plt.ylabel("MSEP")
plt.xlabel("elbo")
plt.show()

plt.figure()
plt.scatter(data["elbo"], -data["log_prob"], c=data["data_set"])
plt.ylabel("NLL")
plt.xlabel("elbo")
plt.show()

# %%
plt.scatter(np.log(data["log_prob"]), data["msep"])
# %%
