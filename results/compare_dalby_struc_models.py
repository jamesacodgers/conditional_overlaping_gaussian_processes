# %%

import pandas as pd
import matplotlib.pyplot as plt

ts = 0.1

df_an_iz = pd.read_csv(f"results/Dalby/struc_var_signals/csvs/Dalby_test_size=0.1_c_unobs=2_L=8_A=4_P=8_an=True_iz=False.csv")
# df_an = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=8_an=True_iz=False.csv")
# df_iz = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=8_an=False_iz=True.csv")
# df = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=8_an=False_iz=False.csv")

# %%
# datas= df.groupby("data_set")
# datas_an= df_an.groupby("data_set")
# datas_iz= df_iz.groupby("data_set")
datas_an_iz= df_an_iz.groupby("data_set")
# %%
# elbo_idx = datas["elbo"].idxmax()
# elbo_idx_an = datas_an["elbo"].idxmax()
# elbo_idx_iz = datas_iz["elbo"].idxmax()
elbo_idx_an_iz = datas_an_iz["elbo"].idxmax()




plt.figure()
plt.scatter(df_an_iz["elbo"], df_an_iz["rmsep"], marker="x", label="AN IZ")
# plt.scatter(df_an["elbo"], df_an["rmsep"], marker="x", label="Add Noise")
# plt.scatter(df_iz["elbo"], df_iz["rmsep"], marker="x", label="Init zero")
# plt.scatter(df["elbo"], df["rmsep"], marker="x", label="None")
plt.scatter(df_an_iz["elbo"].loc[elbo_idx_an_iz], df_an_iz["rmsep"].loc[elbo_idx_an_iz], marker="o", label="AN IZ",s=64)
# plt.scatter(df_an["elbo"].loc[elbo_idx_an], df_an["rmsep"].loc[elbo_idx_an], marker="o", label="Add Noise",s=64)
# plt.scatter(df_iz["elbo"].loc[elbo_idx_iz], df_iz["rmsep"].loc[elbo_idx_iz], marker="o", label="Init zero",s=64)
# plt.scatter(df["elbo"].loc[elbo_idx], df["rmsep"].loc[elbo_idx], marker="o", label="None",s=64)



plt.ylabel("MSEP")
plt.xlabel("ELBO")
plt.legend(loc="upper right")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()

# %%
(df_an_iz["rmsep"].loc[elbo_idx_an_iz]**2).mean()
(df_an_iz["coverage"].loc[elbo_idx_an_iz]).mean()