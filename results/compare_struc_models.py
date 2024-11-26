# %%

import pandas as pd
import matplotlib.pyplot as plt

df_an_iz = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size=0.1_c_unobs=0_L=12_A=4_P=8_an=True_iz=True.csv")
df_an = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size=0.1_c_unobs=0_L=12_A=4_P=8_an=True_iz=False.csv")
df_iz = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size=0.1_c_unobs=0_L=12_A=4_P=8_an=False_iz=True.csv")
df = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size=0.1_c_unobs=0_L=12_A=4_P=8_an=False_iz=False.csv")




plt.figure()
plt.scatter(df_an_iz["elbo"], df_an_iz["rmsep"], marker="x", label="AN IZ")
plt.scatter(df_an["elbo"], df_an["rmsep"], marker="x", label="Add Noise")
plt.scatter(df_iz["elbo"], df_iz["rmsep"], marker="x", label="Init zero")
plt.scatter(df["elbo"], df["rmsep"], marker="x", label="None")

plt.ylabel("MSEP")
plt.xlabel("ELBO")
plt.legend(loc="upper left")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()
