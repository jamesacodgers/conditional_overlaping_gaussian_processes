# %%
from scipy.io import loadmat
import pandas as pd
import torch
import torch.utils
import numpy as np
import matplotlib.pyplot as plt

data = loadmat("data/Wulfert/flodat.mat")
# %%
r = torch.tensor(data["conc"]).repeat(5,1)
# %%

def snv_correction(spectra):
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)

    # Apply SNV correction
    snv_corrected = (spectra - mean) / std

    return snv_corrected

spec_keys = 'spec70', 'spec30', 'spec40', 'spec50', 'spec60'
tensors = [torch.tensor(data[key]) for key in spec_keys]
d = torch.hstack(tensors).T
d = d[:,300:440]
# d = snv_correction(d)




# %%

plt.figure()
plt.plot(d.T)
plt.show()

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
test_size = 0.5
for i in range(10):
    (r_train, d_train), (r_test, d_test) = random_train_test_split([r,d], test_size=test_size, random_seed=i)
    torch.save(r_train, f"data/Wulfert/splits/no_snv_r_train_seed={i}_test_size={test_size}.pkl")
    torch.save(r_test, f"data/Wulfert/splits/no_snv_r_test_seed={i}_test_size={test_size}.pkl")
    torch.save(d_train, f"data/Wulfert/splits/no_snv_d_train_seed={i}_test_size={test_size}.pkl")
    torch.save(d_test, f"data/Wulfert/splits/no_snv_d_test_seed={i}_test_size={test_size}.pkl")
# %%


# from sklearn.model_selection import KFold
# k=10
# kf = KFold(n_splits=k, shuffle=True, random_state=42)

# # Loop over each fold
# for i, (train_index, test_index) in enumerate(kf.split(d)):
#     print(f"Fold {i + 1}")
#     print(f"Train indices: {train_index}")
#     print(f"Test indices: {test_index}\n")
    
#     # Split the data into training and testing sets for this fold
#     d_train, d_test = d[train_index], d[test_index]
#     r_train, r_test = r[train_index], r[test_index]


#     torch.save(r_train, f"data/Wulfert/splits/no_snv_r_train_seed={i}_test_size=0.1.pkl")
#     torch.save(r_test, f"data/Wulfert/splits/no_snv_r_test_seed={i}_test_size=0.1.pkl")
#     torch.save(d_train, f"data/Wulfert/splits/no_snv_d_train_seed={i}_test_size=0.1.pkl")
#     torch.save(d_test, f"data/Wulfert/splits/no_snv_d_test_seed={i}_test_size=0.1.pkl")
#     # torch.save(r_train, f"data/Wulfert/splits/r_train_seed={i}_test_size=0.1.pkl")
#     # torch.save(r_test, f"data/Wulfert/splits/r_test_seed={i}_test_size=0.1.pkl")
#     # torch.save(d_train, f"data/Wulfert/splits/d_train_seed={i}_test_size=0.1.pkl")
#     # torch.save(d_test, f"data/Wulfert/splits/d_test_seed={i}_test_size=0.1.pkl")