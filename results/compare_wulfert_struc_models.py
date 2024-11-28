# %%

import pandas as pd
import matplotlib.pyplot as plt

ts = 0.1

df_an_iz = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=8_an=True_iz=True.csv")
df_an = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=8_an=True_iz=False.csv")
df_iz = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=8_an=False_iz=True.csv")
df = pd.read_csv(f"results/Wulfert/struct_variable/csvs/Struc_Wulfert_test_size={ts}_c_unobs=0_L=8_A=4_P=8_an=False_iz=False.csv")

# %%
datas= df.groupby("data_set")
datas_an= df_an.groupby("data_set")
datas_iz= df_iz.groupby("data_set")
datas_an_iz= df_an_iz.groupby("data_set")
# %%
elbo_idx = datas["elbo"].idxmax()
elbo_idx_an = datas_an["elbo"].idxmax()
elbo_idx_iz = datas_iz["elbo"].idxmax()
elbo_idx_an_iz = datas_an_iz["elbo"].idxmax()




plt.figure()
plt.scatter(df_an_iz["elbo"], df_an_iz["rmsep"], marker="x", label="AN IZ")
plt.scatter(df_an["elbo"], df_an["rmsep"], marker="x", label="Add Noise")
plt.scatter(df_iz["elbo"], df_iz["rmsep"], marker="x", label="Init zero")
plt.scatter(df["elbo"], df["rmsep"], marker="x", label="None")
plt.scatter(df_an_iz["elbo"].loc[elbo_idx_an_iz], df_an_iz["rmsep"].loc[elbo_idx_an_iz], marker="o", label="AN IZ",s=64)
plt.scatter(df_an["elbo"].loc[elbo_idx_an], df_an["rmsep"].loc[elbo_idx_an], marker="o", label="Add Noise",s=64)
plt.scatter(df_iz["elbo"].loc[elbo_idx_iz], df_iz["rmsep"].loc[elbo_idx_iz], marker="o", label="Init zero",s=64)
plt.scatter(df["elbo"].loc[elbo_idx], df["rmsep"].loc[elbo_idx], marker="o", label="None",s=64)



plt.ylabel("MSEP")
plt.xlabel("ELBO")
plt.legend(loc="upper right")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()

plt.figure()
plt.scatter(df_an_iz["elbo"], df_an_iz["coverage"], marker="x", label="AN IZ")
plt.scatter(df_an["elbo"], df_an["coverage"], marker="x", label="Add Noise")
plt.scatter(df_iz["elbo"], df_iz["coverage"], marker="x", label="Init zero")
plt.scatter(df["elbo"], df["coverage"], marker="x", label="None")
# plt.scatter(df["elbo"], df["rmsep"], marker="x", label="None")
plt.scatter(df_an_iz["elbo"].loc[elbo_idx_an_iz], df_an_iz["coverage"].loc[elbo_idx_an_iz], marker="o", label="AN IZ",s=64)
plt.scatter(df_an["elbo"].loc[elbo_idx_an], df_an["coverage"].loc[elbo_idx_an], marker="o", label="Add Noise",s=64)
plt.scatter(df_iz["elbo"].loc[elbo_idx_iz], df_iz["coverage"].loc[elbo_idx_iz], marker="o", label="Init zero",s=64)
plt.scatter(df["elbo"].loc[elbo_idx], df["coverage"].loc[elbo_idx], marker="o", label="None",s=64)

plt.ylabel("Coverage")
plt.xlabel("ELBO")
plt.legend(loc="lower left")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()


plt.figure()
plt.scatter(df_an_iz["rmsep"], df_an_iz["coverage"], marker="x", label="AN IZ")
plt.scatter(df_an["rmsep"], df_an["coverage"], marker="x", label="Add Noise")
plt.scatter(df_iz["rmsep"], df_iz["coverage"], marker="x", label="Init zero")
plt.scatter(df["rmsep"], df["coverage"], marker="x", label="None")
# plt.scatter(df["elbo"], df["rmsep"], marker="x", label="None")
plt.scatter(df_an_iz["rmsep"].loc[elbo_idx_an_iz], df_an_iz["coverage"].loc[elbo_idx_an_iz], marker="o", label="AN IZ",s=64)
plt.scatter(df_an["rmsep"].loc[elbo_idx_an], df_an["coverage"].loc[elbo_idx_an], marker="o", label="Add Noise",s=64)
plt.scatter(df_iz["rmsep"].loc[elbo_idx_iz], df_iz["coverage"].loc[elbo_idx_iz], marker="o", label="Init zero",s=64)
plt.scatter(df["rmsep"].loc[elbo_idx], df["coverage"].loc[elbo_idx], marker="o", label="None",s=64)

plt.ylabel("Coverage")
plt.xlabel("ELBO")
plt.legend(loc="lower left")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()


# plt.figure()
# plt.scatter(df_an_iz["elbo"], df_an_iz["log_prob"], marker="x", label="AN IZ")
# plt.scatter(df_an["elbo"], df_an["log_prob"], marker="x", label="Add Noise")
# plt.scatter(df_iz["elbo"], df_iz["log_prob"], marker="x", label="Init zero")
# plt.scatter(df["elbo"], df["log_prob"], marker="x", label="None")

# plt.ylabel("Coverage")
# plt.xlabel("ELBO")
# plt.legend(loc="lower left")
# # plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
# plt.show()

# %%

print(datas["rmsep"].mean())
print(datas_an["rmsep"].mean())
print(datas_iz["rmsep"].mean())
print(datas_an_iz["rmsep"].mean())

# %%
print(df["rmsep"].mean())
print(df_an["rmsep"].mean())
print(df_iz["rmsep"].mean())
print(df_an_iz["rmsep"].mean())

print(df["rmsep"].std())
print(df_an["rmsep"].std())
print(df_iz["rmsep"].std())
print(df_an_iz["rmsep"].std())