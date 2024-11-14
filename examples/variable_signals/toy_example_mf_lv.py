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


# import multiprocessing
# multiprocessing.set_start_method('spawn')

from src.mf_methods import  MFVariableSignals

import tqdm

import pandas as pd



# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)


# %%
# Define the training set parameters
num_training_samples = 100

num_test_samples=100

wavelengths = torch.linspace(-15, 15, 101)
sigma2 = torch.Tensor([1e-2])
# %%
# Define model parameters

def get_pure_components(latent_variable):
    centroids = torch.vstack((latent_variable*2 - 7.5,  latent_variable*2 + 7.5))
    amplitudes = torch.vstack((2*latent_variable + 15, 2 * latent_variable + 10))
    return pure_component(centroids, amplitudes).permute(2,1,0).to(torch.float32)

def pure_component(mu, a, wavelengths=wavelengths):
    return a[:, None, : ] * cauchy.pdf((wavelengths[None, :, None] - mu[:, None, :]) / 3)

def get_observations(components, pure_component_spectra):
    noise_free_observations = (components[:, None, :] * pure_component_spectra[:, :, :]).sum(axis=-1) 
    return noise_free_observations + torch.randn_like(noise_free_observations)*torch.sqrt(sigma2)


choices = [[0.2,0.8]]*int(num_training_samples/4) + [[0.4,0.6]]*int(num_training_samples/4) +  [[0.6,0.4]]*int(num_training_samples/4) + [[0.8,0.2]]*int(num_training_samples/4)

R_train = torch.Tensor(choices)
x_train = torch.randn(num_training_samples)

S_train = get_pure_components(x_train)

D_train = get_observations(R_train, S_train)
R_train = R_train[:,0].reshape(-1,1) # To test unobserved components



R_test = torch.distributions.Dirichlet(torch.ones(2)).sample((num_test_samples,))
x_test = torch.randn(num_test_samples) 


S_test = get_pure_components(x_test)
D_test = get_observations(R_test, S_test)

x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

# %%
A = 3


# mu_x_train = get_init_values_for_latent_variables(D_train, R_train, D_train, A)

mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train, R_train, D_test, A)

V = torch.randn(8,A)
# V = torch.linspace(-2,2,).reshape(-1,1)

# model = MFVariableSignals(C_obs=2, C_unobs=0, M = wavelengths.shape[0],V=V,sum_to_one=False, mu_x_test=mu_x_test, mu_x_train=mu_x_train, )
model = MFVariableSignals(C_obs=1, C_unobs=2, M = wavelengths.shape[0],V=V,sum_to_one=False, mu_x_test=mu_x_test, mu_x_train=mu_x_train, )



# print(model.elbo(D_train=D_train,R_train=R_train,D_test=D_test))



train_gd(model, D_train, R_train, D_test, 1000, lr = 1e-1)
# %%
# train_lbfgs(model, D_train, R_train, D_test, 10, convergence_level=5)


preds = model.predict_weights(D_test)

# %%
plot_preds(R_test, mf_preds=preds)

model.elbo(D_train=D_train,R_train=R_train,D_test=D_test)
# %%

# %%
with torch.no_grad():
    fig, axs = plt.subplots(3, figsize=(6,9))
    axs[0].bar(["observed", "unobserved 1", "unobserved 2"], model.sigma2_s, label = "$\sigma^2_s$", width=0.4)
    axs[0].xaxis.set_visible(False)
    # axs[0].set_ylabel("$\\sigma^{\; s\;2}_c$")

    for i in range(2):
        axs[i+1].bar(["observed", "unobserved 1", "unobserved 2"], 1/model.beta[:,i], width=0.4)
        axs[i+1].set_ylim(0,(1/model.beta).max()*1.1)
        axs[i+1].set_ylabel("$1/\\beta_{c"+str(i+1)+"}$")

        axs[i+1].xaxis.set_visible(False)
    axs[-1].xaxis.set_visible(True)

    axs[-1].set_xlabel("Component")
    plt.savefig("figs/toy_example/mf_GP_hyperparameters.pdf")


# %%
S = model.get_s_expectation(model.mu_x_train,model.Sigma_x_train)


with torch.no_grad():
    # for p in range(16):
    for i in range(100):
        plt.plot(wavelengths,S[i,:,0], c = "orange", alpha = 0.5)
        plt.plot(wavelengths,S[i,:,1], c = "blue", alpha=0.5)
        plt.plot(wavelengths,S[i,:,2], c = "green", alpha=0.5)
    plt.ylabel("pure component mean")
    plt.xlabel("wavelength")
    plt.savefig("figs/toy_example/mf_pure_components.pdf")
    plt.show()
# %%
