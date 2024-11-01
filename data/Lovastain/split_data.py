import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import trim_wavelengths


df_temp_conc = pd.read_excel("data/Lovastain/Lovastatin_PLS_data.xlsx", sheet_name="Temp_Conc" )
df_spectra = pd.read_excel("data/Lovastain/Lovastatin_PLS_data.xlsx", sheet_name="Spectra" )

# %%
temps = torch.tensor(df_temp_conc["X: temp"].to_numpy(dtype = np.float64)).reshape(-1,1)
# concs = torch.tensor([df_temp_conc["Y: conc"].to_numpy()/1000, 1- df_temp_conc["Y: conc"].to_numpy()/1000]).T
# concs = torch.tensor([df_temp_conc["Y: conc"].to_numpy()/1000, [1]*df_temp_conc["Y: conc"].shape[0]]).T
# concs = torch.tensor([df_temp_conc["Y: conc"].to_numpy()/1000, [1]*df_temp_conc["Y: conc"].shape[0]]).T
concs = torch.tensor(df_temp_conc["Y: conc"].to_numpy(np.float64)/1000).reshape(-1,1)
wavelengths = torch.tensor(df_spectra["Wavenumber (cm-1)"])
spectra = torch.tensor(df_spectra.drop(["Wavenumber (cm-1)", "Unnamed: 32"], axis =1).to_numpy(np.float64).T)
# %%
# spectra = spectra[2:]
# concs = concs[2:]
# temps = temps[2:]

spectra = spectra - spectra.mean(dim=0)
# spectra = spectra/spectra.std()
concs = concs - concs.mean(dim=0)
# concs = concs/concs.std()
temps = temps - temps.mean()
temps = temps/temps.std()

# %%
fig, ax = plt.subplots()
ax.plot(wavelengths, spectra.T, c = "orange", alpha = 0.5)
fig.show()

# spectra_filtered, wavelengths_filtered  = trim_wavelengths(950, 1500, wavelengths, spectra)
spectra_filtered, wavelengths_filtered  = trim_wavelengths(950, 1800, wavelengths, spectra)

fig, ax = plt.subplots()
ax.plot(wavelengths_filtered, spectra_filtered.T, c = "orange", alpha = 0.5)
fig.show()

# %%

def random_train_test_split(tensors, test_size=0.2, random_seed=None):
    """
    Randomly splits a list of tensors into training and testing sets.
    
    Parameters:
    - tensors (list of torch.Tensor): List of tensors to split.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_seed (int): Random seed for reproducibility.
    
    Returns:
    - train_tensors (list of torch.Tensor): List of training tensors.
    - test_tensors (list of torch.Tensor): List of testing tensors.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate the number of samples for the split
    total_samples = tensors[0].size(0)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    split_index = int(total_samples * (1 - test_size))
    train_indices, test_indices = indices[:split_index], indices[split_index:]

    # Split the tensors using the calculated indices
    train_tensors = [tensor[train_indices] for tensor in tensors]
    test_tensors = [tensor[test_indices] for tensor in tensors]

    return train_tensors, test_tensors

# %%

test_size=0.5
for i in range(10):
    np.random.seed(i)
    torch.manual_seed(i)
    (r_train, d_train, temps_train), (r_test, d_test, temps_test) = random_train_test_split([concs,spectra_filtered, temps], test_size=test_size, random_seed=i)
    torch.save(r_train, f"data/Lovastain/splits/r_train_seed={i}_test_size={test_size}.pkl")
    torch.save(r_test, f"data/Lovastain/splits/r_test_seed={i}_test_size={test_size}.pkl")
    torch.save(d_train, f"data/Lovastain/splits/d_train_seed={i}_test_size={test_size}.pkl")
    torch.save(d_test, f"data/Lovastain/splits/d_test_seed={i}_test_size={test_size}.pkl")
    torch.save(temps_train, f"data/Lovastain/splits/temps_train_seed={i}_test_size={test_size}.pkl")
    torch.save(temps_test, f"data/Lovastain/splits/temps_test_seed={i}_test_size={test_size}.pkl")