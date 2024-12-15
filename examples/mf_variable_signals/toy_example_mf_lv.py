# Toy example from the paper

# %%
import copy

from matplotlib.lines import Line2D
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import cauchy
from itertools import chain
from src.utils import get_init_values_for_latent_variables, train_gd, get_both_init_values_for_latent_variables, train_lbfgs, plot_preds


from src.mf_methods import  MFVariableSignals



# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)


# %%
# Define the training set parameters

independent_latents=False
num_training_samples = 100
num_test_samples=100

wavelengths = torch.linspace(-15, 15, 101)
sigma2 = torch.Tensor([1e-2])
# %%
# Define model parameters

def get_pure_components(latent_variable):
    centroids = torch.vstack((latent_variable[:,0]*2 - 7.5,  latent_variable[:,1]*2 + 7.5))
    amplitudes = torch.vstack((2*latent_variable[:,0] + 15, 2 * latent_variable[:,1] + 10))
    return pure_component(centroids, amplitudes).permute(2,1,0).to(torch.float32)

def pure_component(mu, a, wavelengths=wavelengths):
    return a[:, None, : ] * cauchy.pdf((wavelengths[None, :, None] - mu[:, None, :]) / 3)

def get_observations(components, pure_component_spectra):
    noise_free_observations = (components[:, None, :] * pure_component_spectra[:, :, :]).sum(axis=-1) 
    return noise_free_observations + torch.randn_like(noise_free_observations)*torch.sqrt(sigma2)


choices = [[0.2,0.8]]*int(num_training_samples/4) + [[0.4,0.6]]*int(num_training_samples/4) +  [[0.6,0.4]]*int(num_training_samples/4) + [[0.8,0.2]]*int(num_training_samples/4)

if independent_latents: 
    x_train = torch.randn(num_training_samples,2)
    x_test = torch.randn(num_test_samples,2) 
else:
    x_train = torch.randn(num_training_samples).reshape(-1,1).repeat(1,2)
    x_test = torch.randn(num_test_samples).reshape(-1,1).repeat(1,2)

R_train = torch.Tensor(choices)
R_test = torch.distributions.Dirichlet(torch.ones(2)).sample((num_test_samples,))

S_train = get_pure_components(x_train)
S_test = get_pure_components(x_test)

D_test = get_observations(R_test, S_test)
D_train = get_observations(R_train, S_train)


# %%

with torch.no_grad():
    for i,c in enumerate(["red","blue","green"]):
        plt.plot(S_test[i,:,0], c= c )
        plt.plot(S_test[i,:,1], c = c)
    plt.show()

# %%
A = 2

mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train, R_train, D_test, A)

V = torch.randn(8,A)

R_train_obs = R_train[:,0].reshape(-1,1)
model = MFVariableSignals(C_obs=1, C_unobs=2, M = wavelengths.shape[0],V=V,weight_distribution="continuos", mu_x_test=mu_x_test, mu_x_train=mu_x_train, )




train_gd(model=model, D_train=D_train, R_train=R_train_obs, D_test=D_test, n_steps=1000, lr = 1e-1)

# %%

train_lbfgs(model=model, D_train=D_train, R_train=R_train_obs, D_test=D_test, n_steps=10, convergence_level=5)


preds = model.predict_weights(D_test, )

# %%
plot_preds(R_test, mf_preds=preds, save_path=f"figs/toy_example/mf_preds_i={independent_latents}_v2.pdf" )

model.elbo(D_train=D_train,R_train=R_train_obs,D_test=D_test)
# %%

# %%
with torch.no_grad():
    # Plot R_train points
    dist = torch.distributions.Normal(model.mu_x_train, model.Sigma_x_train)
    samples = dist.sample((16**2,))
    # for i in range(16**2):
    #     plt.scatter(
    #         # model.mu_x_train[:, 0],  # X-coordinates
    #         # model.mu_x_train[:, 1],  # Y-coordinates
    #         samples[i,:, 0],  # X-coordinates
    #         samples[i,:, 1],  # Y-coordinates
    #         c=R_train[:, 0],  # Color based on R_train values
    #         cmap="plasma",  # Choose a colormap
    #         vmin=0,
    #         vmax=1,
    #         alpha=1
    #     )
    dist = torch.distributions.Normal(model.mu_x_test, model.Sigma_x_test)
    samples = dist.sample((16*32,))
    for i in range(16**2):
        plt.scatter(
            # model.mu_x_train[:, 0],  # X-coordinates
            # model.mu_x_train[:, 1],  # Y-coordinates
            samples[i,:, 0],  # X-coordinates
            samples[i,:, 1],  # Y-coordinates
            c=R_test[:, 0],  # Color based on R_train values
            # marker="x",
            cmap="plasma",  # Choose a colormap
            vmin=0,
            vmax=1,
            alpha=1
        )
    # Plot R_test points
    # plt.scatter(
    #     model.mu_x_test[:, 0],  # X-coordinates
    #     model.mu_x_test[:, 1],  # Y-coordinates
    #     marker="x",  # Use a different marker for test points
    #     c=R_test[:, 0],  # Color based on R_test values
    #     cmap="plasma",  # Choose a colormap
    #     vmin=0,
    #     vmax=1
    # )
    # Add a colorbar to provide a scale for the colors
    plt.colorbar(label="Component 1 fraction")
    plt.savefig(f"figs/toy_example/mf_latent_space_i={independent_latents}_v2.pdf")
    # Display the plot
    plt.show()


# %%
S = model.get_s_expectation(model.mu_x_train,model.Sigma_x_train)


with torch.no_grad():
    fig, axs = plt.subplots(3, figsize=(6,9))
    axs[0].bar(["observed", "unobserved 1", "unobserved 2"], model.sigma2_s, label = "$\sigma^2_s$", width=0.4)
    axs[0].xaxis.set_visible(False)
    axs[0].set_ylabel("$\\sigma^{\; s\;2}_c$")

    for j in range(2):
        axs[j+1].bar(["observed", "unobserved 1", "unobserved 2"], 1/model.beta[:,j], width=0.4)
        axs[j+1].set_ylim(0,(1/model.beta).max()*1.1)
        axs[j+1].set_ylabel("$1/\\beta_{c"+str(j+1)+"}$")

        axs[j+1].xaxis.set_visible(False)
    axs[-1].xaxis.set_visible(True)

    axs[-1].set_xlabel("Component")
    plt.savefig(f"figs/toy_example/mf_GP_hyperparameters_i={independent_latents}_v2.pdf")
# %%

S = model.get_gp_expectation(torch.zeros(1, model.A))
S_cov = model.get_gp_covariance(torch.zeros(1, model.A))
S_std = torch.sqrt(S_cov[:,:, range(model.C), range(model.C)])
colours = ["orange", "blue", "green"]
with torch.no_grad():
    for c in range(model.C):
        plt.plot(wavelengths, S[0,:,c], c = colours[c])
        plt.fill_between(wavelengths.numpy(),S[0,:,c] + 1.96*S_std[0,:,c], S[0,:,c] - 1.96*S_std[0,:,c], color = colours[c], alpha=0.5)
        plt.savefig(f"figs/toy_example/mf_pure_component_estimates_i={independent_latents}_v2.pdf")
# %%
print(S_std.mean())

# %%

(1.96*2*preds.stddev[:,0]).mean()

lb = preds.mean[:,0] - 1.96*preds.stddev[:,0]
ub = preds.mean[:,0] + 1.96*preds.stddev[:,0]

torch.logical_and(lb < R_test[:,0], ub > R_test[:,0]).type(torch.float32).mean()

