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


# import multiprocessing
# multiprocessing.set_start_method('spawn')

from src.struc_models import  StrucVariableSignals

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


R_test = torch.distributions.Dirichlet(torch.ones(2)).sample((num_test_samples,))
x_test = torch.randn(num_test_samples) 


S_test = get_pure_components(x_test)
D_test = get_observations(R_test, S_test)

x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

# %%
A = 2



mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train, R_train, D_test, A)

# mu_x_train = torch.zeros(num_training_samples, A)
# mu_x_test = torch.zeros(num_test_samples, A) 

V = torch.randn(8,A)
# V = torch.linspace(-2,2,).reshape(-1,1)

R_train = R_train[:,0].reshape(-1,1) # To test unobserved components
# model = MFVariableSignals(C_obs=2, C_unobs=0, M = wavelengths.shape[0],V=V,sum_to_one=False, mu_x_test=mu_x_test, mu_x_train=mu_x_train, )
model = StrucVariableSignals(P=16,C_obs=1, C_unobs=2, M = wavelengths.shape[0],V=V,weight_distribution="continuos", mu_x_test=mu_x_test, mu_x_train=mu_x_train, )

opt_model = torch.compile(model)




# %%

train_gd(opt_model, 
         D_train, 
         R_train,
         D_test=D_test, 
         n_steps=500, 
         lr = 1e-1
        )

# train_conditional_model_gd(opt_model, 
#          D_train, 
#          R_train,
#          D_test=D_test, 
#          n_steps=1000, 
#         )



# # %%

train_lbfgs(model, D_train, R_train, D_test, 10, convergence_level=5)



# %%
preds = model.predict_weights(D_test)


with torch.no_grad():
    plot_preds(R_test=R_test, cond_preds=preds)

model.elbo(D_train=D_train,R_train=R_train,D_test=D_test)

print(model.beta, model.sigma2_s)
# %%


S = model.get_s_expectation(model.mu_x_train,model.Sigma_x_train)

with torch.no_grad():
    for p in range(model.P):
        for j in range(100):
            plt.plot(S[p,j,:,0], c = "orange", alpha = 0.02)
            plt.plot(S[p,j,:,1], c = "blue", alpha=0.02)
            plt.plot(S[p,j,:,2], c = "green", alpha=0.02)
    plt.xlabel("wavelength")
    plt.savefig("figs/toy_example/struc_pure_components.pdf")
    plt.show()
# %%

with torch.no_grad():
    for j in range(R_train.shape[0]):
        plt.scatter( model.mu_x_train[:,j,0], model.mu_x_train[:,j,1])
    for j in range(R_test.shape[0]):
        plt.scatter( model.mu_x_test[:,j,0], model.mu_x_test[:,j,1], marker="x")

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
    plt.savefig("figs/toy_example/structured_GP_hyperparameters.pdf")
# %%

model.elbo(D_train, R_train, D_test)
# %%
