# %%
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.struc_models import Struc_WS_GPLVM
from src.utils import get_both_init_values_for_latent_variables, train_gd, train_lbfgs, snv_correction

torch.set_default_dtype(torch.float64)


training_df = pd.read_csv("data/Hyperspectral/Rock_TRAIN_diff_fold_1.tsv", delimiter="\t", header=None)
test_df = pd.read_csv("data/Hyperspectral/Rock_TEST_diff_fold_1.tsv", delimiter="\t",header=None)


R_train = torch.tensor(training_df[0])-1
R_test= torch.tensor(test_df[0])-1

D_train=torch.tensor(training_df.drop(0,axis=1).to_numpy())
D_test=torch.tensor(test_df.drop(0,axis=1).to_numpy())
D_test=snv_correction(D_test)
D_train=snv_correction(D_train)

# %%



A = 2

mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train,torch.nn.functional.one_hot(R_train), D_test, A)

V = torch.randn(8,A)

R_train_obs = R_train
model = Struc_WS_GPLVM(P=8,C_obs=4, C_unobs=0, M = D_test.shape[1],V=V,weight_distribution="categorical", mu_x_test=mu_x_test, mu_x_train=mu_x_train)



# %%
train_gd(model=model, 
         D_train=D_train, 
         R_train=R_train_obs, 
         D_test=D_test, 
         n_steps=1000, 
         lr = 1e-2,
         plot=False
         )

train_lbfgs(model=model, 
            D_train=D_train, 
            R_train=R_train_obs, 
            D_test=D_test, 
            n_steps=10, 
            plot=False,
            convergence_level=5)


# %%
preds = model.predict_weights(D_test, )
print(torch.argmax(preds) - R_test)

with torch.no_grad():
    plt.scatter
    for p in preds:
        plt.scatter(R_test, p[:,0],alpha = 0.1)
    plt.show()

# %%
# plot_preds(R_test, mf_preds=preds, save_path=f"figs/toy_example/mf_preds_i={independent_latents}.pdf" )

model.elbo(
        D_train=D_train,
        R_train=R_train_obs,
        #    D_test=D_test
           )
# %%

# %%
with torch.no_grad():
    # Plot R_train points
    plt.scatter(
        model.mu_x_train[:, 0],  # X-coordinates
        model.mu_x_train[:, 1],  # Y-coordinates
        # c=x_train[:, 0],  # Color based on R_train values
        cmap="plasma",  # Choose a colormap
        vmin=0,
        vmax=1
    )
    # Plot R_test points
    plt.scatter(
        model.mu_x_test[:, 0],  # X-coordinates
        model.mu_x_test[:, 1],  # Y-coordinates
        marker="x",  # Use a different marker for test points
        # c=x_test[:, 0],  # Color based on R_test values
        cmap="plasma",  # Choose a colormap
        vmin=0,
        vmax=1
    )
    # Add a colorbar to provide a scale for the colors
    plt.colorbar(label="Component 1 fraction")
    # plt.savefig(f"figs/toy_example/mf_latent_space_i={independent_latents}.pdf")
    # Display the plot
    plt.show()


# %%
S = model.get_s_expectation(model.mu_x_train,model.Sigma_x_train)


with torch.no_grad():
    fig, axs = plt.subplots(3, figsize=(6,9))
    axs[0].bar(["1", "2"], model.sigma2_s, label = "$\sigma^2_s$", width=0.4)
    axs[0].xaxis.set_visible(False)
    axs[0].set_ylabel("$\\sigma^{\; s\;2}_c$")

    for j in range(2):
        axs[j+1].bar(["1", "2"], 1/model.beta[:,j], width=0.4)
        axs[j+1].set_ylim(0,(1/model.beta).max()*1.1)
        axs[j+1].set_ylabel("$1/\\beta_{c"+str(j+1)+"}$")

        axs[j+1].xaxis.set_visible(False)
    axs[-1].xaxis.set_visible(True)

    axs[-1].set_xlabel("Component")
    # plt.savefig(f"figs/toy_example/mf_GP_hyperparameters_i={independent_latents}.pdf")
# %%

S = model.get_gp_expectation(torch.zeros(1, model.A))
S_cov = model.get_gp_covariance(torch.zeros(1, model.A))
S_std = torch.sqrt(S_cov[:,:, range(model.C), range(model.C)])
colours = ["orange", "blue", "green"]
with torch.no_grad():
    plt.figure()
    for c in range(model.C):
        plt.plot( S[0,:,c], c = colours[c])
        # plt.fill_between(wavelengths,S[0,:,c] + 1.96*S_std[0,:,c], S[0,:,c] - 1.96*S_std[0,:,c], color = colours[c], alpha=0.5)
        # plt.savefig(f"figs/toy_example/mf_pure_component_estimates_i={independent_latents}.pdf")
    plt.show()
# %%
print(S_std.mean())
