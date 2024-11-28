# %%

import pandas as pd
import matplotlib.pyplot as plt

ts = 0.1


results = {"struc_l=4": "results/Dalby/struc_var_signals/csvs/Dalby_test_size=0.1_c_unobs=2_L=8_A=4_P=8_an=True_iz=False.csv", "struc_l8":"results/Dalby/struc_var_signals/csvs/Dalby_test_size=0.1_c_unobs=2_L=4_A=2_P=8_an=False_iz=False.csv", "mf_L4": "results/Dalby/mf_var_signals/csvs/Dalby_test_size=0.1_c_unobs=2_L=4_A=2.csv", "mf_L8": "results/Dalby/mf_var_signals/csvs/Dalby_test_size=0.1_c_unobs=2_L=8_A=4.csv"}






plt.figure()
for k,r in results.items():
    df = pd.read_csv(r)
    datas= df.groupby("data_set")
    elbo_idx = datas["elbo"].idxmax()
    plt.scatter(df["elbo"], df["rmsep"], marker="x", label=k)
    plt.scatter(df["elbo"].loc[elbo_idx], df["rmsep"].loc[elbo_idx], marker="o", s=64, label=k)
    print(k,(df["rmsep"].loc[elbo_idx]**2).mean())
    print(k,(df["coverage"].loc[elbo_idx]).mean())
plt.ylabel("MSEP")
plt.xlabel("ELBO")
plt.legend(loc="upper left")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()


# %%

plt.figure()
for k,r in results.items():
    df = pd.read_csv(r)
    datas= df.groupby("data_set")
    elbo_idx = datas["elbo"].idxmax()
    plt.scatter(df["elbo"], df["rmsep"], marker="x", label=k)
    plt.scatter(df["elbo"].loc[elbo_idx], df["rmsep"].loc[elbo_idx], marker="o", s=64, label=k)
    print(k,(df["rmsep"].loc[elbo_idx]**2).mean())
    # print(k,(df["coverage"].loc[elbo_idx]).mean())
plt.ylabel("MSEP")
plt.xlabel("ELBO")
plt.legend(loc="upper right")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()

# %%
plt.figure()
for k,r in results.items():
    df = pd.read_csv(r)
    datas= df.groupby("data_set")
    elbo_idx = datas["elbo"].idxmax()
    plt.scatter(df["coverage"], df["rmsep"], marker="x", label=k)
    plt.scatter(df["coverage"].loc[elbo_idx], df["rmsep"].loc[elbo_idx], marker="o", s=64, label=k)
    print(k,(df["rmsep"].loc[elbo_idx]**2).mean())
    print(k,(df["coverage"].loc[elbo_idx]).mean())
plt.ylabel("MSEP")
plt.xlabel("coverage")
plt.legend(loc="upper left")
# plt.savefig(f"figs/{name}/struc_elbo_vs_msep_ts={ts}_l={l}_p=.pdf", bbox_inches="tight")
plt.show()

