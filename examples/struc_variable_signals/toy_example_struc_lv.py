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
from src.utils import get_init_values_for_latent_variables, train_gd, get_both_init_values_for_latent_variables, train_lbfgs, plot_preds, get_both_pca_values, train_conditional_model_gd, train_conditional_model_lbfgs
from src.prob_utils import get_struc_ci

# import multiprocessing
# multiprocessing.set_start_method('spawn')

from src.struc_models import  StrucVariableSignals

import tqdm

import pandas as pd



# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)


# %%
# Define the training set parameters


independent_latents = True


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
A = 2



mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train, R_train, D_test, A)
# mu_x_train = torch.zeros_like(mu_x_train)
# mu_x_test = torch.zeros_like(mu_x_test)


V = torch.randn(6,A)
# V = torch.linspace(-2,2,).reshape(-1,1)

R_train = R_train[:,0].reshape(-1,1) # To test unobserved 
model = StrucVariableSignals(P=16,C_obs=1, C_unobs=2, M = wavelengths.shape[0],V=V,weight_distribution="continuos", mu_x_test=mu_x_test, mu_x_train=mu_x_train, add_randn_to_lvs=False)

opt_model = torch.compile(model)




# %%

train_gd(opt_model, 
         D_train, 
         R_train,
         D_test=D_test, 
         n_steps=500, 
         lr = 1e-1
        )




# %%

train_lbfgs(model, D_train, R_train, D_test, 10, convergence_level=1)



# %%
preds = model.predict_weights(D_test)

lb,ub = get_struc_ci(preds)
print(ub, lb)
with torch.no_grad():
    plot_preds(R_test=R_test, cond_preds=preds, save_path=f"figs/toy_example/structured_preds_i={independent_latents}.pdf")

model.elbo(D_train=D_train,R_train=R_train,D_test=D_test)

print(model.beta, model.sigma2_s)

# %%
with torch.no_grad():
    # Plot R_train points
    for j in range(R_train.shape[0]):
        plt.scatter(
            model.mu_x_train[:, j, 0],  # X-coordinates
            model.mu_x_train[:, j, 1],  # Y-coordinates
            c=R_train[j, 0].repeat(model.P),  # Color based on R_train values
            cmap="plasma",  # Choose a colormap
            vmin=0,
            vmax=1
        )
    # Plot R_test points
    for j in range(R_test.shape[0]):
        plt.scatter(
            model.mu_x_test[:, j, 0],  # X-coordinates
            model.mu_x_test[:, j, 1],  # Y-coordinates
            marker="x",  # Use a different marker for test points
            c=R_test[j, 0].repeat(model.P),  # Color based on R_test values
            cmap="plasma",  # Choose a colormap
            vmin=0,
            vmax=1
        )
    # Add a colorbar to provide a scale for the colors
    plt.colorbar(label="Component 1 fraction")
    plt.savefig(f"figs/toy_example/structured_latent_space_i={independent_latents}.pdf")
    # Display the plot
    plt.show()


# %%
with torch.no_grad():
    for j in range(1):
        plt.scatter(model.get_u_vec_samples()[:,j,0], model.get_u_vec_samples()[:,j,1])
    plt.show()
# %%
a = torch.randn(2,model.L*model.C)
b = model._samples@a.T


with torch.no_grad():
    plt.figure()
    for j in range(10):
        plt.scatter(b[:,j+10,0], b[:,j+10,1])
    plt.show()

# %%


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
    plt.savefig(f"figs/toy_example/structured_GP_hyperparameters_i={independent_latents}.pdf")
# %%

model.elbo(D_train, R_train, D_test)
# %%


S = model.get_gp_expectation(torch.zeros(1, model.A))
S_cov = model.get_gp_covariance(torch.zeros(1, model.A))
S_std = torch.sqrt(S_cov[:,:, range(model.C), range(model.C)])
colours = ["orange", "blue", "green"]
with torch.no_grad():
    for i in range(1):
        for c in range(model.C):
            plt.plot(wavelengths, S[i,:,c], c = colours[c])
            plt.fill_between(wavelengths,S[i,:,c] + 1.96*S_std[i,:,c], S[i,:,c] - 1.96*S_std[i,:,c], color = colours[c], alpha=0.5)

    plt.savefig(f"figs/toy_example/structured_pure_component_estimates_i={independent_latents}.pdf")

# %%
print(S_std.mean())
# %%
