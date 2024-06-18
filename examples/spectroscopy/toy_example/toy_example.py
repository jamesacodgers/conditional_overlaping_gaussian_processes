# %%

import torch
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from src.cogp import COGP
import time
import cProfile

from utils import train
# import os

from torch.distributions.constraints import positive

import cProfile

# %%

torch.set_default_dtype(torch.float64)


wavelengths = torch.linspace(0,2*torch.pi, 101)

temp_train = torch.linspace(0,2*torch.pi,100).reshape(-1,1)
temp_test = torch.rand(50,1)*2*torch.pi

# S_train = torch.stack([torch.sin((wavelengths[None, :] - temp_train[:, None]))], dim=-1)
S_train = torch.stack([torch.sin((wavelengths[None, :] - 3*temp_train)), torch.cos((3*wavelengths[None, :] - temp_train))], dim=-1)
S_test = torch.stack([torch.sin(wavelengths[None, :] - 3*temp_test), torch.cos(3*wavelengths[None, :] - temp_test)], dim=-1)

R_train = torch.randn(100,2)
# R_train = torch.ones(100,2)
# R_train = torch.ones(100,1)
R_test = torch.randn(50,2)
# R_test = torch.ones(50,2)

# D_train = (S_train[:,:,0]) + torch.randn(100,101)/10
D_train = (S_train[:,:,:]*R_train[:, None, :]).sum(dim=-1) + torch.randn(100,101)/10
D_test = (S_test[:,:,:]*R_test[:, None, :]).sum(dim=-1) + torch.randn(50,101)/10


# %%

def make_grid(x,y):
    grid_x, grid_y = torch.meshgrid(x, y)
        # Combine grid_x and grid_y if needed
    return torch.stack([grid_x, grid_y], dim=-1) 


# %%
V_temp = torch.linspace(0, 2*torch.pi, 8).reshape(-1,1)
# model = OverlappingGP(2, V_temp=V_temp, V_wavelengths=V_wavelenghs)
model = COGP(C_obs=2, C_unobs=0, M = wavelengths.shape[0], V = V_temp, )

r = R_train#.unsqueeze(1).repeat([1,101,1]).reshape(-1,2)
D_train = D_train
# D_test = D_test

# cProfile.run("model.elbo(training_targets=D_train, test_targets=D_test, training_inputs=temp_train, test_inputs=temp_test, training_weights=R_train)", sort="cumtime")


loss = train(model=model, training_targets=D_train, test_targets=D_test, training_inputs=temp_train, test_inputs=temp_test, training_weights=R_train, steps=200)

print("beta", model.beta)



# %%
with torch.no_grad():
    plt.plot(loss)
    plt.show()
# %%
with torch.no_grad():
    f_hat = model.get_mu_s(temp_train[:,None,:])
    sigma = model.get_Sigma_s(temp_train[:,None,:])
    print("stuff",sigma.min(), sigma.max())

# %%
with torch.no_grad():
    
    plt.matshow(f_hat[:,:,0])
    plt.matshow(f_hat[:,:,1])
    plt.show()
    plt.matshow(S_train[:,:,0])
    plt.matshow(S_train[:,:,1])
    plt.show()

    plt.figure()
    plt.plot(f_hat[50,:,0])
    plt.plot(S_train[50,:,0])
    plt.plot(D_train[50,:])
    plt.show()

# %%

# r_samples = gp.get_r_samples(D_test, temp_test, wavelengths, num_samples=num_samples)

# plt.figure()
# for i in range(num_samples):
#     plt.scatter(R_test[:,0], r_samples[i,:,0], c= temp_test, alpha=0.5)
# plt.plot(R_test[:,0],R_test[:,0])
# plt.show()

# %%

num_samples = 1000
R_dists = model.get_R_dist_samples(D_test, temp_test[:,None,:],num_samples)
R_samples = R_dists.sample()




with torch.no_grad():
    plt.figure()
    for i in range(num_samples):
        plt.scatter(R_test[:,0], R_dists.loc[i,:,0], c= temp_test, alpha=0.5)
    plt.plot(R_test[:,0],R_test[:,0])
    plt.show()
with torch.no_grad():
    plt.figure()
    for i in range(num_samples):
        plt.scatter(R_test[:,0], R_samples[i,:,0], c= temp_test, alpha=0.5)
    plt.plot(R_test[:,0],R_test[:,0])
    plt.show()

