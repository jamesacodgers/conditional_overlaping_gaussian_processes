# %%

import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from utils import trim_wavelengths, train, train_lbfgs
from overlapping_gp import OverlappingGP
import numpy as np

torch.set_default_dtype(torch.float64)

df_temp_conc = pd.read_excel("Lovastatin_PLS_data.xlsx", sheet_name="Temp_Conc" )
df_spectra = pd.read_excel("Lovastatin_PLS_data.xlsx", sheet_name="Spectra" )

# %%
temps = torch.tensor(df_temp_conc["X: temp"].to_numpy(dtype = np.float64))
# concs = torch.tensor([df_temp_conc["Y: conc"].to_numpy()/1000, 1- df_temp_conc["Y: conc"].to_numpy()/1000]).T
concs = torch.tensor([df_temp_conc["Y: conc"].to_numpy()/1000, [1]*df_temp_conc["Y: conc"].shape[0]]).T
concs = torch.tensor([df_temp_conc["Y: conc"].to_numpy()/1000, [1]*df_temp_conc["Y: conc"].shape[0]]).T
concs = torch.tensor(df_temp_conc["Y: conc"].to_numpy()/1000).reshape(-1,1)
wavelengths = torch.tensor(df_spectra["Wavenumber (cm-1)"])
spectra = torch.tensor(df_spectra.drop(["Wavenumber (cm-1)", "Unnamed: 32"], axis =1).to_numpy().T)
# %%
spectra = spectra - spectra.mean(dim=0)
# spectra = spectra/torch.sqrt(spectra.var(dim=0))
concs = concs - concs.mean(dim=0)
# concs = concs/torch.sqrt(concs.var(dim=0))
# temps = temps - temps.mean()
# temps = temps/torch.sqrt(temps.var())

# %%
fig, ax = plt.subplots()
ax.plot(wavelengths, spectra.T, c = "orange", alpha = 0.5)
fig.show()

spectra_filtered, wavelengths_filtered  = trim_wavelengths(950, 1500, wavelengths, spectra)

fig, ax = plt.subplots()
ax.plot(wavelengths_filtered, spectra_filtered.T, c = "orange", alpha = 0.5)
fig.show()

# %%

train_temps, test_temps, train_concs, test_concs, train_spectra, test_spectra, = train_test_split(temps, concs, spectra_filtered,test_size=0.75)
# %%

model = OverlappingGP(concs.shape[1], wavelengths_filtered.shape[0], torch.linspace(temps.min(), temps.max(), 16), 
                    #   beta=[10,10], sigma2=1e-4, sigma2_s=[0.1, 0.1]
                      )
# %%
loss = train(model, training_spectra=train_spectra, test_spectra=test_spectra, training_temps=train_temps, test_temps=test_temps, training_weights= train_concs, steps=500, lr = 2.5e-2)
# %%
loss.extend(train_lbfgs(model, training_spectra=train_spectra, test_spectra=test_spectra, training_temps=train_temps, test_temps=test_temps, training_weights= train_concs, steps=3))

# %%

with torch.no_grad():
    plt.figure()
    plt.plot(loss)
    plt.show()

    num_samples = 1000
    R_est = model.get_R_estimate_samples(test_spectra, test_temps, num_samples)
    plt.figure()
    for i in range(num_samples-1):
        plt.scatter(test_concs[:,0], R_est[i,:,0 ], c = test_temps, alpha=0.01,  )
    plt.scatter(test_concs[:,0], R_est[-1,:,0 ], c = test_temps, alpha=0.5,  label = "estimates for concentration")
        # plt.plot(range(R_est.shape[1]), temp_test, c = "red", alpha=1 )
    plt.plot(test_concs[:,0], test_concs[:,0])
    plt.legend()

    plt.show()
# %%

pls = PLSRegression(3)
pls.fit(train_spectra, train_concs)
y_hat = pls.predict(test_spectra)

plt.figure()
plt.scatter(test_concs[:,0], y_hat[:,0])
plt.plot(test_concs[:,0], test_concs[:,0])
plt.show()
# %%
