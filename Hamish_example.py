# %%

import pandas as pd
import matplotlib.pyplot as plt
import torch
import arrow
import numpy as np
import tqdm
from overlapping_gp import OverlappingGP


torch.set_default_dtype(torch.float32)

def make_grid(x,y):
    grid_x, grid_y = torch.meshgrid(x, y)
        # Combine grid_x and grid_y if needed
    return torch.stack([grid_x, grid_y], dim=-1) 

temp = pd.read_excel("Triglycine_Tsat30_Tcrys21_Collated (2).xlsx", sheet_name="Temperature")
temp = temp.drop(index=0)
spectra = pd.read_excel("Triglycine_Tsat30_Tcrys21_Collated (2).xlsx", sheet_name="IR")
wavelengths = torch.Tensor(spectra["Wavenumbercm-1"].to_numpy())
spectra = spectra.drop(columns="Wavenumbercm-1").to_numpy().T
high_conc = 70.6e-3
low_conc = 56.0e-3
# high_conc = 70.6e-2
# low_conc = 56.0e-2

# %%
temp_train = torch.concat([
                torch.tensor(temp["Tr"].to_numpy().astype(np.float32)[150:550]),
                torch.tensor(temp["Tr"].to_numpy().astype(np.float32)[1500:1650])
                ])
temp_train = temp_train[350:]
temp_test = torch.Tensor(temp["Tr"].to_numpy().astype(np.float32)[550:1500])
D_train = torch.concat([
                torch.Tensor(spectra[150:550]),
                torch.Tensor(spectra[1500:1650])
                ])
D_train = D_train[350:]
D_test = torch.Tensor(spectra[550:1500])
R_train = torch.vstack(
    [torch.concat([
    torch.ones(400)*high_conc,
    torch.ones(150)*low_conc
    ]), 
    torch.concat([
    torch.ones(400) *(1-high_conc),
    torch.ones(150) *(1-low_conc)
    # torch.ones(400) *(100-high_conc),
    # torch.ones(150) *(100-low_conc)
])]
).T
# R_train = R_train - R_train.mean(dim=0)
# R_train = R_train/ torch.sqrt(R_train.var(dim=0))
R_train = R_train[350:]
# %%
def trim_wavelengths(min_wavelengths, max_wavelengths, wavelengths, spectra):
    filter = torch.logical_and(wavelengths<max_wavelengths , min_wavelengths<wavelengths)
    return  spectra[:,filter], wavelengths[filter]

min_wavelengths = 1200
max_wavelengths = 1350

# spectra = spectra - spectra.mean()
# spectra = spectra/np.sqrt(spectra.var())

D_train_filtered, wavelengths_filtered = trim_wavelengths(min_wavelengths,max_wavelengths,wavelengths, D_train)
# D_train_filtered = D_train_filtered - D_train_filtered.mean()
D_test_filtered, wavelengths_filtered = trim_wavelengths(min_wavelengths,max_wavelengths,wavelengths, D_test)
# D_test_filtered = D_test_filtered - D_test_filtered.mean()
# D_test_filtered = D_test_filtered/torch.sqrt(D_train_filtered.var())
# D_train_filtered = D_train_filtered/torch.sqrt(D_train_filtered.var())

plt.figure()
# for t,d in zip(test_tau,D_test_filtered):
plt.plot(wavelengths_filtered, D_test_filtered.T, c="blue")
plt.plot(wavelengths_filtered, D_train_filtered.T, c="orange")
plt.plot()
# %%
    
# model = OverlappingGP(
#                     2, 
#                     wavelengths_filtered.shape[0],
#                     # wavelengths.shape[0],
#                     torch.linspace(wavelengths_filtered.min(), wavelengths_filtered.max(),16), 
#                     beta=1e10, 
#                     sigma2=1e-5, 
#                     sigma2_s=1e6
#                     )
model = OverlappingGP(
                    2, 
                    wavelengths_filtered.shape[0],
                    # wavelengths.shape[0],
                    torch.linspace(wavelengths_filtered.min(), wavelengths_filtered.max(),2), 
                    beta=1e6, 
                    sigma2=1e-5, 
                    sigma2_s=1
                    )

# %%


frac_of_test = 1

def train():
    loss_list = []
    optimizer = torch.optim.Adam(lr=1e-1, params=model.parameters())
    iterator = tqdm.tqdm(range(500))
    for _ in iterator:
        loss = -model.elbo(
            training_spectra=D_train_filtered,
            # training_spectra=D_train,
            training_weights=R_train, 
            training_temperatures=temp_train,
            test_spectra=D_test_filtered[::frac_of_test],
            # test_spectra=D_test[::frac_of_train],
            test_temperatures=temp_test[::frac_of_test], 
            wavelengths=wavelengths_filtered
            # wavelengths=wavelengths
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
            training_spectra=D_train_filtered,
            # training_spectra=D_train,
            training_weights=R_train, 
            training_temperatures=temp_train,
            test_spectra=D_test_filtered[::frac_of_test],
            # test_spectra=D_test[::frac_of_train],
            test_temperatures=temp_test[::frac_of_test], 
            wavelengths=wavelengths_filtered
            # wavelengths=wavelengths
            )
        loss.backward()
        return loss
    loss_list = []
    optimizer = torch.optim.LBFGS(params=model.parameters())
    iterator = tqdm.tqdm(range(10))
    for _ in iterator:
        loss = -model.elbo(
            training_spectra=D_train_filtered,
            # training_spectra=D_train,
            training_weights=R_train, 
            training_temperatures=temp_train,
            test_spectra=D_test_filtered[::frac_of_test],
            # test_spectra=D_test[::frac_of_train],
            test_temperatures=temp_test[::frac_of_test], 
            wavelengths=wavelengths_filtered
            # wavelengths=wavelengths
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)
        iterator.set_postfix(loss=loss[-1])
        loss_list.append(loss)
    return loss_list
print("beta", model.beta)
# %%
loss = train()
# loss = train_lbfgs()
# loss = model.get_test_fit(D_test_filtered, temp_test, wavelengths_filtered)
# loss.backward()
# for name, param in model.named_parameters():
#     print(name, param.grad)
# loss = model.get_kl_u()
# loss.backward()
# for name, param in model.named_parameters():
#     print(name, param.grad)


# # %%
# with torch.no_grad():
#     plt.figure()
#     plt.plot(-np.array(loss))
#     plt.show()
# # %%
# with torch.no_grad():
#     # f_hat = model.get_mu_s(temp_train, wavelengths)
#     f_hat = model.get_mu_s(temp_train, wavelengths_filtered)
#     # sigma = model.get_Sigma_s_cov_C(temp_train, wavelengths)
#     sigma = model.get_Sigma_s_cov_C(temp_train, wavelengths_filtered)

# # %%
# with torch.no_grad():
    
#     plt.figure()
#     plt.matshow(f_hat[:,:,0])
#     plt.matshow(f_hat[:,:,1])
#     plt.show()



# # %%

# with torch.no_grad():
#     S_samples = model.get_S_samples(temp_test, wavelengths_filtered)
#     plt.figure()
#     plt.plot(wavelengths_filtered, S_samples[0,0,:,0])
#     plt.show()

# # %%

# num_samples = 100
# R_samples = model.get_R_estimate_samples(D_test_filtered, temp_test, wavelengths_filtered,num_samples)
# R_dists = model.get_R_dist_samples(D_test_filtered, temp_test, wavelengths_filtered, num_samples)

# %%
low_temp_spectra = D_train_filtered[350:]
low_temp_conc = R_train[350:]
low_temp_spectra = D_train_filtered
low_temp_conc = R_train

S_hat = torch.inverse(low_temp_conc.T @ low_temp_conc) @ low_temp_conc.T @ low_temp_spectra
with torch.no_grad():
    plt.figure()
    plt.plot(wavelengths_filtered,S_hat[0])
    plt.plot(wavelengths_filtered,S_hat[1])
    plt.show()
# # %%

# for name, params in model.named_parameters():
#     print(name, params)


# %%

S_samples = model.get_mu_s(temp_test, wavelengths_filtered)

with torch.no_grad():
    plt.plot(wavelengths_filtered, S_samples[0, :, 0])
    plt.plot(wavelengths_filtered, S_samples[0, :, 1])
    plt.show()

# %%
with torch.no_grad():
    plt.plot(loss)

# %%

with torch.no_grad():
    num_samples = 100
    R_est = model.get_R_estimate_samples(D_test_filtered, temp_test, wavelengths_filtered)
    plt.figure()
    for i in range(num_samples):
        plt.scatter(range(R_est.shape[1]), R_est[i,:,0 ], c = "orange", alpha=0.01 )
    plt.hlines(R_train[0,0], 0, 900, )
    plt.hlines(R_train[-1,0],0, 900, )

    plt.show()
# %%
