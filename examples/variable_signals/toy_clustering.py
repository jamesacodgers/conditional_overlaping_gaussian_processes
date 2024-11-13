# %%

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from src.mf_methods import MFVariableSignals
from src.utils import get_both_init_values_for_latent_variables

import torch

D_train, R_train = make_moons(250, noise=0.1)
D_test, R_test = make_moons(250, noise=0.1)


A=1

plt.figure()
plt.scatter(R_train[:,0], R_train[:,1], c = R_test)

D_train = torch.tensor(D_train)
D_test = torch.tensor(D_test)
R_train = torch.nn.functional.one_hot(torch.tensor(R_train))
R_test = torch.nn.functional.one_hot(torch.tensor(R_test))


mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train, R_train, D_test, A)


V = torch.randn(8, A)