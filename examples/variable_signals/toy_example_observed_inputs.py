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

from src.mf_methods import MFVariableSignalsObservedInputs

import tqdm

import pandas as pd

from src.eiot import plot_preds


# %%
torch.set_default_dtype(torch.float64)
# torch.manual_seed(1234567890)
# np.random.seed(1234567890)


# %%
# Define the training set parameters
num_training_samples = 100
wavelengths = torch.linspace(-15, 15, 101)
sigma2 = torch.Tensor([1e-2])
# %%
# Define model parameters

def get_pure_components(latent_variable):
    centroids = torch.vstack((latent_variable*2 - 7.5,  latent_variable*2 + 7.5))
    amplitudes = torch.vstack((2*latent_variable + 15, 2 * latent_variable + 10))
    return pure_component(centroids, amplitudes).permute(2,1,0)

def pure_component(mu, a, wavelengths=wavelengths):
    return a[:, None, : ] * cauchy.pdf((wavelengths[None, :, None] - mu[:, None, :]) / 3)

def get_observations(components, pure_component_spectra):
    noise_free_observations = (components[:, None, :] * pure_component_spectra[:, :, :]).sum(axis=-1) 
    return noise_free_observations + torch.randn_like(noise_free_observations)*torch.sqrt(sigma2)


choices = [[0.2,0.8]]*int(num_training_samples/4) + [[0.4,0.6]]*int(num_training_samples/4) +  [[0.6,0.4]]*int(num_training_samples/4) + [[0.8,0.2]]*int(num_training_samples/4)

# training_components = torch.distributions.Dirichlet(torch.ones(2)).sample((num_training_samples,))
R_train = torch.Tensor(choices)
x_train = torch.randn(num_training_samples)

S_train = get_pure_components(x_train)

D_train = get_observations(R_train, S_train)

num_test_samples=100

R_test = torch.distributions.Dirichlet(torch.ones(2)).sample((num_test_samples,))
x_test = torch.randn(num_test_samples) 


S_test = get_pure_components(x_test)
D_test = get_observations(R_test, S_test)

x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

# %%

# cmap = plt.get_cmap("plasma")
# with torch.no_grad():
#     fig, ax = plt.subplots(2,figsize = (8,12))
#     for i in range(training_observed_spectra.shape[0]):
#         ax[0].plot(training_observed_spectra[i], c = cmap(training_weights[i,0]))
#     legend_elements = [
#     Line2D([0], [0], color=cmap(0.2),  label='$r_1 = 0.2$, $r_2 = 0.8$'),
#     Line2D([0], [0], color=cmap(0.4),  label='$r_1 = 0.4$, $r_2 = 0.6$'),
#     Line2D([0], [0], color=cmap(0.6),  label='$r_1 = 0.6$, $r_2 = 0.4$'),
#     Line2D([0], [0], color=cmap(0.8),  label='$r_1 = 0.8$, $r_2 = 0.2$'),
#     ]
#     ax[0].legend(handles = legend_elements)
#     # plt.savefig("examples/figs/toy_example/training_data.pdf")



    
#     for i in range(test_observed_spectra.shape[0]):
#         ax[1].plot(test_observed_spectra[i], c = "grey", alpha = 0.5)
#     legend_elements = [
#     Line2D([0], [0], color="grey",  label= "Unknown component fractions"), 
#     ]
#     ax[1].legend(handles = legend_elements)
#     ax[0].set_ylabel("$y$")
#     ax[1].set_ylabel("$y^*$")
#     ax[1].set_xlabel("$\lambda$")
#     fig.subplots_adjust(hspace = 0.1)
#     # plt.savefig("/Users/jo816/github/submission/examples/data.pdf", bbox_inches="tight")
#     plt.show()


# %%
V = torch.randn(8,1)

model = MFVariableSignalsObservedInputs(C_obs=2, C_unobs=0, M = wavelengths.shape[0], input_dims=1,V=V,sum_to_one=False)
model_sum = MFVariableSignalsObservedInputs(C_obs=2, C_unobs=0, M = wavelengths.shape[0], input_dims=1,V=V,sum_to_one=True)

def train_lbfgs(model, D_train, R_train, D_test, x_train, x_test, n_steps = None, plot = False, to_convergence=False):


    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    ll = []
    # Define a closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        loss = -model.elbo( D_train, R_train, D_test,x_train, x_test)
        loss.backward()
        
        return loss
    if n_steps:
        for epoch in tqdm.tqdm(range(n_steps)):  
            # print(epoch)
            optimizer.step(closure)  # Update model
            ll.append(model.elbo( D_train, R_train, D_test,x_train, x_test))
    if to_convergence:
        while ll[-1] - ll[-2] >1e-3:  
            print(f"\r{ll[-1]}")
            optimizer.step(closure)  # Update model
            ll.append(model.elbo( D_train, R_train, D_test,x_train, x_test))

    if plot:
        with torch.no_grad():
            plt.figure()
            plt.plot(ll)
            plt.show()
    return ll 

print(model.sigma2)

loss_list = train_lbfgs(model, D_train=D_train, R_train=R_train, D_test=D_test, x_train=x_train, x_test=x_test, n_steps=10, to_convergence=True)
loss_list = train_lbfgs(model_sum, D_train=D_train, R_train=R_train, D_test=D_test, x_train=x_train, x_test=x_test, n_steps=10, to_convergence=True)

preds_sum = model_sum.predict_weights(D_test, x_test)
preds = model.predict_weights(D_test, x_test)

plot_preds(R_test=R_test, mf_preds=preds_sum)
plot_preds(R_test=R_test, mf_preds=preds)


print(loss_list)
# %%
with torch.no_grad():
    plt.plot(model.get_s_expectation(x_test)[...,0].T)