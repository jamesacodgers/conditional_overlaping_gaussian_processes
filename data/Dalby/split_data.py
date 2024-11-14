# %%
from scipy.io import loadmat
import pandas as pd
import torch
import torch.utils
import numpy as np

data = loadmat("data/Dalby/NIR_EIOT_Dyrby_et_al (1).mat")

import matplotlib.pyplot as plt

# plt.figure()
# plt.scatter(data["Ck"][:,0], data["Ck"][:,1], c = data["dose_source"][:,3])
# plt.show()

# %%
trim_min = 150
trim_max = 250
r = torch.tensor(data["Ck"])
d = torch.tensor(data["spectra"])
d = d[:,trim_min:trim_max]
# %%

def snv_correction(spectra):
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)

    # Apply SNV correction
    snv_corrected = (spectra - mean) / std

    return snv_corrected
# d = snv_correction(d)
plt.plot(d.T)
plt.show()
# %%

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
test_size = 0.25
for i in range(10):
    (r_train, d_train), (r_test, d_test) = random_train_test_split([r,d], test_size=test_size, random_seed=i)
    torch.save(r_train, f"data/Dalby/splits/no_snv_r_train_seed={i}_test_size={test_size}_trimmin={trim_min}_trimmax={trim_max}.pkl")
    torch.save(r_test, f"data/Dalby/splits/no_snv_r_test_seed={i}_test_size={test_size}_trimmin={trim_min}_trimmax={trim_max}.pkl")
    torch.save(d_train, f"data/Dalby/splits/no_snv_d_train_seed={i}_test_size={test_size}_trimmin={trim_min}_trimmax={trim_max}.pkl")
    torch.save(d_test, f"data/Dalby/splits/no_snv_d_test_seed={i}_test_size={test_size}_trimmin={trim_min}_trimmax={trim_max}.pkl")
# %%



# %%
