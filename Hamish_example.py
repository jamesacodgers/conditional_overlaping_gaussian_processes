# %%

import pandas as pd
import matplotlib.pyplot as plt
import torch
import arrow
import numpy as np
import tqdm

from overlapping_gp import OverlappingGP




def make_grid(x,y):
    grid_x, grid_y = torch.meshgrid(x, y)
        # Combine grid_x and grid_y if needed
    return torch.stack([grid_x, grid_y], dim=-1) 

temp = pd.read_excel("Triglycine_Tsat30_Tcrys21_Collated (2).xlsx", sheet_name="Temperature")
temp = temp.drop(index=0)
spectra = pd.read_excel("Triglycine_Tsat30_Tcrys21_Collated (2).xlsx", sheet_name="IR")
high_conc = 70.6e-3
low_conc = 56.0e-3

# %%
wavelengths = torch.Tensor(spectra["Wavenumbercm-1"].to_numpy())
temp_train = torch.concat([
                torch.tensor(temp["Tr"].to_numpy().astype(np.float32)[150:550]),
                torch.tensor(temp["Tr"].to_numpy().astype(np.float32)[1500:1650])
                ])
temp_test = torch.Tensor(temp["Tr"].to_numpy().astype(np.float32)[550:1500])
D_train = torch.concat([
                torch.Tensor(spectra.drop(columns="Wavenumbercm-1").to_numpy()).T[150:550],
                torch.Tensor(spectra.drop(columns="Wavenumbercm-1").to_numpy()).T[1500:1650]
                ])
D_test = torch.Tensor(spectra.drop(columns="Wavenumbercm-1").to_numpy()).T[550:1500]
R_train = torch.vstack(
    [torch.concat([
    torch.ones(400)*high_conc,
    torch.ones(150)*low_conc
    ]), 
    torch.concat([
    torch.ones(400) *(1-high_conc),
    torch.ones(150) *(1-low_conc)
])]
).T
# %%
def trim_wavelengths(min_wavelengths, max_wavelengths, wavelengths, spectra):
    filter = torch.logical_and(wavelengths<max_wavelengths , min_wavelengths<wavelengths)
    return  spectra[:,filter], wavelengths[filter]

min_wavelengths = 1200
max_wavelengths = 1400

D_train_filtered, wavelengths_filtered = trim_wavelengths(min_wavelengths,max_wavelengths,wavelengths, D_train)
D_test_filtered, wavelengths_filtered = trim_wavelengths(min_wavelengths,max_wavelengths,wavelengths, D_test)

plt.figure()
# for t,d in zip(test_tau,D_test_filtered):
plt.plot(wavelengths_filtered, D_test_filtered.T, c="blue")
plt.plot(wavelengths_filtered, D_train_filtered.T, c="orange")
plt.plot()
# %%
    
model = OverlappingGP(2, torch.linspace(temp_train.min(), temp_train.max(), 4), torch.linspace(wavelengths_filtered.min(), wavelengths_filtered.max(),4))

# %%




def train():
    loss_list = []
    optimizer = torch.optim.Adam(lr=1e-2, params=model.parameters())
    iterator = tqdm.tqdm(range(1000))
    for _ in iterator:
        loss = -model.elbo(
            training_spectra=D_train_filtered,
            training_weights=R_train, 
            training_temperatures=temp_train,
            test_spectra=D_test_filtered,
            test_temperatures=temp_test, 
            wavelengths=wavelengths_filtered
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iterator.set_postfix(loss=loss[-1])
        loss_list.append(loss)
    return loss_list
print("beta", model.beta)
print("gamma", model.gamma)
loss = train()
print("beta", model.beta)
print("gamma", model.gamma)
# %%
with torch.no_grad():
    plt.figure()
    plt.plot(-np.array(loss))
    plt.show()
    # %%

    f_hat = model.get_mu_s(temp_train, wavelengths_filtered)
    sigma = model.get_Sigma_s_cov_C(temp_train, wavelengths_filtered)
    print("stuff",sigma.min(), sigma.max())

    # %%
    plt.figure()
    plt.matshow(f_hat[:,:,0])
    plt.matshow(f_hat[:,:,1])
    plt.show()



# %%

with torch.no_grad():
    
    S_samples = model.get_S_samples(temp_test, wavelengths_filtered)
    plt.figure()
    plt.plot(wavelengths_filtered, S_samples[0,0,:,0])
    plt.show()
# r_samples = gp.get_r_samples(D_test, temp_test, wavelengths, num_samples=num_samples)

# plt.figure()
# for i in range(num_samples):
#     plt.scatter(R_test[:,0], r_samples[i,:,0], c= temp_test, alpha=0.5)
# plt.plot(R_test[:,0],R_test[:,0])
# plt.show()

# %%

num_samples = 1000
R_samples = model.get_R_estimate_samples(D_test_filtered, temp_test, wavelengths_filtered,num_samples)
R_dists = model.get_R_dist_samples(D_test_filtered, temp_test, wavelengths_filtered, num_samples)

# with torch.no_grad():
#     plt.figure()
#     for i in range(num_samples):
#         plt.scatter(R_dists.loc[i,:,0], c= temp_test, alpha=0.5)
#     plt.plot(R_test[:,0],R_test[:,0])
#     plt.show()
# with torch.no_grad():
#     plt.figure()
#     for i in range(num_samples):
#         plt.scatter(R_test[:,0], R_samples[i,:,0], c= temp_test, alpha=0.5)
#     plt.plot(R_test[:,0],R_test[:,0])
#     plt.show()
