# %%

import pandas as pd
import matplotlib.pyplot as plt
import torch
import arrow
import numpy as np
import tqdm
from overlapping_gp import OverlappingGP
from utils import trim_wavelengths


torch.set_default_dtype(torch.float64)

def make_grid(x,y):
    grid_x, grid_y = torch.meshgrid(x, y)
        # Combine grid_x and grid_y if needed
    return torch.stack([grid_x, grid_y], dim=-1) 

temp = pd.read_excel("Triglycine_Tsat30_Tcrys21_Collated (2).xlsx", sheet_name="Temperature")
temp = temp.drop(index=0)
spectra = pd.read_excel("Triglycine_Tsat30_Tcrys21_Collated (2).xlsx", sheet_name="IR")
wavelengths = torch.Tensor(spectra["Wavenumbercm-1"].to_numpy())
spectra = spectra.drop(columns="Wavenumbercm-1").to_numpy().T
spectra = spectra - spectra.mean(axis=0)
high_conc = 70.6e-3
low_conc = 56.0e-3
# high_conc = 70.6e-2
# low_conc = 56.0e-2

# %%
train_start_index = 150
train_end_index = 1650

test_start_index = 475
test_end_index = 1575

temp_train = torch.concat([
                torch.tensor(temp["Tr"].to_numpy().astype(np.float32)[train_start_index:test_start_index]),
                torch.tensor(temp["Tr"].to_numpy().astype(np.float32)[test_end_index:train_end_index])
                ])
# temp_train = temp_train[350:]
temp_test = torch.Tensor(temp["Tr"].to_numpy().astype(np.float32)[test_start_index:test_end_index])
D_train = torch.concat([
                torch.Tensor(spectra[train_start_index:test_start_index]),
                torch.Tensor(spectra[test_end_index:train_end_index])
                ])
# D_train = D_train[350:]
D_test = torch.Tensor(spectra[test_start_index:test_end_index])
R_train = torch.vstack(
    [torch.concat([
    torch.ones(test_start_index - train_start_index)*high_conc,
    torch.ones(train_end_index - test_end_index)*low_conc
    ]), 
    torch.concat([
    torch.ones(test_start_index - train_start_index) *(1-high_conc),
    torch.ones(train_end_index - test_end_index) *(1-low_conc)
    # torch.ones(400) *(100-high_conc),
    # torch.ones(150) *(100-low_conc)
])]
).T
R_train = R_train - R_train.mean(dim=0)
# R_train = R_train/ torch.sqrt(R_train.var(dim=0))
# R_train = R_train[350:]
# %%


min_wavelengths = 1250
max_wavelengths = 1750

# spectra = spectra - spectra.mean()
# spectra = spectra/np.sqrt(spectra.var())

D_train, _ = trim_wavelengths(min_wavelengths,max_wavelengths,wavelengths, D_train)
# D_train_filtered = D_train_filtered - D_train_filtered.mean()
D_test, wavelengths = trim_wavelengths(min_wavelengths,max_wavelengths,wavelengths, D_test)
# D_test_filtered = D_test_filtered - D_test_filtered.mean()
# D_test_filtered = D_test_filtered/torch.sqrt(D_train_filtered.var())
# D_train_filtered = D_train_filtered/torch.sqrt(D_train_filtered.var())

plt.figure()
# for t,d in zip(test_tau,D_test_filtered):
plt.plot(wavelengths, D_test.T, c="blue")
plt.plot(wavelengths, D_train.T, c="orange")
plt.show()
# %%
    
# model = OverlappingGP(
#                     2, 
#                     wavelengths.shape[0],
#                     # wavelengths.shape[0],
#                     torch.linspace(wavelengths_filtered.min(), wavelengths_filtered.max(),1), 
#                     beta=[1e6, 1e6], 
#                     sigma2=1e-5, 
#                     # sigma2_s=1
#                     )
model = OverlappingGP(
                    2, 
                    wavelengths.shape[0],
                    # wavelengths.shape[0],
                    torch.linspace(temp_train.min(), temp_train.max(),2), 
                    # beta=[1e2, 1e2], 
                    sigma2=1, 
                    # sigma2_s=1
                    )

# %%


frac_of_test = 1


print("beta", model.beta)
# %%
loss = train()
loss = train_lbfgs()

# %%
low_temp_spectra = D_train[350:]
low_temp_conc = R_train[350:]
low_temp_spectra = D_train
low_temp_conc = R_train

S_hat = torch.inverse(low_temp_conc.T @ low_temp_conc) @ low_temp_conc.T @ low_temp_spectra
with torch.no_grad():
    plt.figure()
    plt.plot(wavelengths,S_hat[0])
    plt.plot(wavelengths,S_hat[1])
    plt.show()
# # %%

# for name, params in model.named_parameters():
#     print(name, params)


# %%

S_samples = model.get_mu_s(temp_test)

with torch.no_grad():
    plt.plot(wavelengths, S_samples[0, :, 0])
    plt.plot(wavelengths, S_samples[0, :, 1])
    plt.show()

# %%
with torch.no_grad():
    plt.plot(loss)

# %%

with torch.no_grad():
    num_samples = 100
    R_est = model.get_R_estimate_samples(D_test, temp_test)
    plt.figure()
    for i in range(num_samples-1):
        plt.scatter(range(R_est.shape[1]), R_est[i,:,0 ], c = "orange", alpha=0.01,  )
    plt.scatter(range(R_est.shape[1]), R_est[-1,:,0 ], c = "orange", alpha=0.5,  label = "estimates for concentration")
        # plt.plot(range(R_est.shape[1]), temp_test, c = "red", alpha=1 )
    plt.hlines(R_train[0,0], -250, 0, label = "initial conentration", ls = "--", color = "green", lw = 3)
    plt.hlines(R_train[-1,0], R_est.shape[1],  R_est.shape[1] +250, label="final concentration", color = "red", ls = "--", lw = 3 )
    plt.legend()

    plt.show()
# %%
