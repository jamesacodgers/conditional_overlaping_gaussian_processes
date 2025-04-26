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


from src.mf_methods import  MF_WS_GPLVM
from torcheval.metrics import BinaryAUROC



# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(1234)
np.random.seed(1234)


# %%
# Define the training set parameters

independent_latents=True
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


choices = [[0,1]]*int(num_training_samples/2) +  [[1,0]]*int(num_training_samples/2)

if independent_latents: 
    x_train = torch.randn(num_training_samples,2)
    x_test = torch.randn(num_test_samples,2) 
else:
    x_train = torch.randn(num_training_samples).reshape(-1,1).repeat(1,2)
    x_test = torch.randn(num_test_samples).reshape(-1,1).repeat(1,2)

# R_train = torch.tensor([1]*int(num_training_samples/2) + [0]*int(num_training_samples/2))
# R_test = torch.tensor([1]*int(num_training_samples/2) + [0]*int(num_training_samples/2))
R_sample_generator = torch.tensor(choices, dtype=torch.float64)
R_train = R_sample_generator.argmax(dim=1)
R_test = R_sample_generator.argmax(dim=1)
S_train = get_pure_components(x_train)
S_test = get_pure_components(x_test)

D_test = get_observations(R_sample_generator, S_test)
D_train = get_observations(R_sample_generator, S_train)


# %%

# with torch.no_grad():
#     for i,c in enumerate(["red","blue","green"]):
#         plt.plot(S_test[i,:,0], c= c )
#         plt.plot(S_test[i,:,1], c = c)
#     plt.show()

# %%
A = 2

mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train,R_sample_generator, D_test, A)

V = torch.randn(6,A)

R_train_obs = R_train
model = MF_WS_GPLVM(C_obs=2, C_unobs=0, M = wavelengths.shape[0],V=V,weight_distribution="categorical", mu_x_test=mu_x_test, mu_x_train=mu_x_train)




train_gd(model=model, 
         D_train=D_train, 
         R_train=R_train_obs, 
         D_test=D_test, 
         n_steps=100, 
         lr = 1e-1,
         plot=False
         )

# # %%

train_lbfgs(model=model, 
            D_train=D_train, 
            R_train=R_train_obs, 
            D_test=D_test, 
            n_steps=10, 
            plot=False,
            convergence_level=5)


preds = model.predict_weights(D_test, )

print(preds)
# %%
with torch.no_grad():
    plt.scatter(R_test, torch.argmax(preds,dim=1))
    plt.show()

# %%

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
        c=x_train[:, 0],  # Color based on R_train values
        cmap="plasma",  # Choose a colormap
        vmin=0,
        vmax=1
    )
    # Plot R_test points
    plt.scatter(
        model.mu_x_test[:, 0],  # X-coordinates
        model.mu_x_test[:, 1],  # Y-coordinates
        marker="x",  # Use a different marker for test points
        c=x_test[:, 0],  # Color based on R_test values
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

S = model.get_gp_expectation(torch.zeros(1, model.A))
S_cov = model.get_gp_covariance(torch.zeros(1, model.A))
S_std = torch.sqrt(S_cov[:,:, range(model.C), range(model.C)])
colours = ["orange", "blue", "green"]
with torch.no_grad():
    plt.figure()
    for c in range(model.C):
        plt.plot(wavelengths, S[0,:,c], c = colours[c])
        # plt.fill_between(wavelengths,S[0,:,c] + 1.96*S_std[0,:,c], S[0,:,c] - 1.96*S_std[0,:,c], color = colours[c], alpha=0.5)
        # plt.savefig(f"figs/toy_example/mf_pure_component_estimates_i={independent_latents}.pdf")
    plt.show()
# %%

acc = torch.mean((torch.argmax(preds, dim=1) == R_test).type(torch.float))


ll = torch.gather(preds, dim=1, index=R_test[:,None]).sum()

auc = BinaryAUROC()

auc.update(preds[:,1], R_test)
auc.compute()