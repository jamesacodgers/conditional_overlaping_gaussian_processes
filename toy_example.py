# %%

import torch
import tqdm
import numpy as np
from matplotlib import pyplot as plt
from overlapping_gp import OverlappingGP
import time
# import os

from torch.distributions.constraints import positive

import cProfile

# %%
torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

wavelengths = torch.linspace(0,2*torch.pi, 101)

temp_train = torch.linspace(0,2*torch.pi,100)
temp_test = torch.rand(50)*2*torch.pi

# S_train = torch.stack([torch.sin((wavelengths[None, :] - temp_train[:, None]))], dim=-1)
S_train = torch.stack([torch.sin((wavelengths[None, :] - 3*temp_train[:, None])), torch.cos((wavelengths[None, :] - temp_train[:, None]))], dim=-1)
S_test = torch.stack([torch.sin(wavelengths[None, :] - 3*temp_test[:, None]), torch.cos(wavelengths[None, :] - temp_test[:, None])], dim=-1)

R_train = torch.randn(100,2)
# R_train = torch.ones(100,1)
R_test = torch.randn(50,2)

# D_train = (S_train[:,:,0]) + torch.randn(100,101)/10
D_train = (S_train[:,:,:]*R_train[:, None, :]).sum(dim=-1) + torch.randn(100,101)/10
D_test = (S_test[:,:,:]*R_test[:, None, :]).sum(dim=-1) + torch.randn(50,101)/10


# %%

def make_grid(x,y):
    grid_x, grid_y = torch.meshgrid(x, y)
        # Combine grid_x and grid_y if needed
    return torch.stack([grid_x, grid_y], dim=-1) 


# %%
V_temp = torch.linspace(0, 2*torch.pi, 8)
V_wavelenghs = torch.linspace(0, 2*torch.pi, 8)
# model = OverlappingGP(2, V_temp=V_temp, V_wavelengths=V_wavelenghs)
model = OverlappingGP(2, wavelengths.shape[0], V_temp=V_temp)

r = R_train#.unsqueeze(1).repeat([1,101,1]).reshape(-1,2)
D_train = D_train
# D_test = D_test




def train():
    loss_list = []
    optimizer = torch.optim.Adam(lr=1e-1, params=model.parameters())
    iterator = tqdm.tqdm(range(2000))
    for _ in iterator:
        loss = -model.elbo(
            training_spectra=D_train, 
            training_weights=R_train, 
            training_temperatures=temp_train,
            test_spectra=D_test,
            test_temperatures=temp_test, 
            wavelengths=wavelengths
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterator.set_postfix(loss=loss[-1])
        loss_list.append(loss)
    return loss_list

def train_lbfgs():
    def closure():
        optimizer.zero_grad()
        loss = -model.elbo(
            training_spectra=D_train,
            # training_spectra=D_train,
            training_weights=R_train, 
            training_temperatures=temp_train,
            test_spectra=D_test,
            # test_spectra=D_test[::frac_of_train],
            test_temperatures=temp_test, 
            wavelengths=wavelengths
            # wavelengths=wavelengths
            )
        loss.backward()
        return loss
    loss_list = []
    optimizer = torch.optim.LBFGS(params=model.parameters(), history_size= 10)
    iterator = tqdm.tqdm(range(20))
    for _ in iterator:
        loss = -model.elbo(
            training_spectra=D_train,
            # training_spectra=D_train,
            training_weights=R_train, 
            training_temperatures=temp_train,
            test_spectra=D_test,
            # test_spectra=D_test[::frac_of_train],
            test_temperatures=temp_test, 
            wavelengths=wavelengths
            # wavelengths=wavelengths
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)
        iterator.set_postfix(loss=loss[-1])
        loss_list.append(loss)
    return loss_list
print("beta", model.beta)
cProfile.run("""-model.elbo(
            training_spectra=D_train, 
            training_weights=R_train, 
            training_temperatures=temp_train,
            test_spectra=D_test,
            test_temperatures=temp_test, 
            wavelengths=wavelengths
            )""", sort="cumtime")
loss = train()
# loss = train_lbfgs()


# %%
with torch.no_grad():
    plt.plot(-np.array(loss))
    plt.show()
# %%
with torch.no_grad():
    f_hat = model.get_mu_s(temp_train, wavelengths)
    sigma = model.get_Sigma_s_cov_C(temp_train, wavelengths)
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
with torch.no_grad():
    S_samples = model.get_S_samples(temp_test, wavelengths)
    plt.figure()
    plt.plot(wavelengths, S_samples[0,0,:,0])
    plt.show()
# r_samples = gp.get_r_samples(D_test, temp_test, wavelengths, num_samples=num_samples)

# plt.figure()
# for i in range(num_samples):
#     plt.scatter(R_test[:,0], r_samples[i,:,0], c= temp_test, alpha=0.5)
# plt.plot(R_test[:,0],R_test[:,0])
# plt.show()

# %%

num_samples = 1000
R_samples = model.get_R_estimate_samples(D_test, temp_test, wavelengths,num_samples)
R_dists = model.get_R_dist_samples(D_test, temp_test, wavelengths, num_samples)

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

# %%
