# %%

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

from src.mf_methods import MFVariableSignals
from src.struc_models import StrucVariableSignals
from src.utils import get_both_init_values_for_latent_variables

from src.utils import train_lbfgs, train_gd, train_conditional_model_gd

import torch
torch.set_default_dtype(torch.float64)

D_train, R_train = make_moons(200, noise=0.1)
D_test, R_test = make_moons(250, noise=0.1)

A=2
 
# plt.figure()
# plt.scatter(D_train[:,0], D_train[:,1], c = R_train)
# plt.show()

D_train = torch.tensor(D_train)
D_test = torch.tensor(D_test)
R_train = torch.tensor(R_train)
R_test = torch.tensor(R_test)

R_train_one_hot = torch.nn.functional.one_hot(R_train).type(torch.float64)

mu_x_train, mu_x_test =  get_both_init_values_for_latent_variables(D_train, R_train_one_hot, D_test, A)
# mu_x_train = torch.zeros_like(mu_x_train)
# mu_x_test = torch.zeros_like(mu_x_test)

V = torch.randn(8, A)

# model = MFVariableSignals(V, C_obs=2,mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=2, weight_distribution="categorical",C_unobs=0)
model = StrucVariableSignals(V=V, P=8, C_obs=2,mu_x_train=mu_x_train, mu_x_test=mu_x_test,M=2, weight_distribution="categorical",C_unobs=0)
model = torch.compile(model)

# %%
train_conditional_model_gd(model, D_train,R_train, n_steps=5000, lr = 2e-3)
# train_gd(model, D_train,R_train, n_steps=1000, lr = 0.02)

# %%
train_lbfgs(model, D_train,R_train, convergence_level=5)

# %%

x_test = torch.linspace(-3,3,50).reshape(-1,1)

mu_s = model.get_gp_expectation(x_test)
print(mu_s.shape)
with torch.no_grad():
    plt.figure()
    plt.scatter(D_train[:,0], D_train[:,1], c = R_train)
    mu_u = model.get_mu_u()
    for i in range(2):
        plt.plot(mu_s[:,0,i], mu_s[:,1,i])
        # u_samples = model.get_u_samples()
        # plt.scatter(u_samples[:,0, i, :].flatten(), u_samples[:,1,i,:].flatten(), c="black", marker="x")
        # plt.scatter(u_samples[:,0, i, :].flatten(), u_samples[:,1,i,:].flatten(), marker="x")
        plt.scatter(mu_u[0, i, :], mu_u[1,i,:],marker="*")
    plt.show()
# %%
# with torch.no_grad():
#     plt.scatter(model._samples[:,0,:], model._samples[:,1,:])
#     plt.show()
# u_samples = model.get_u_samples()
# with torch.no_grad():
#     fig, axs = plt.subplots(2,2)
#     for c in range(2):
#         for m in range(2):
#             axs[m,c].scatter(model.V.flatten(), model.get_mu_u()[m,c], color="k")
#             axs[m,c].plot(x_test.flatten(),mu_s[:,m,c])
#             for p in range(model.P):
#                 axs[m,c].scatter(model.V.flatten(), u_samples[p,m,c,:], color="orange", alpha = 0.1)
#     plt.show()

#%%

model.elbo(D_train,R_train)


# %%

with torch.no_grad():
    plt.scatter(model.V[:,0], model.V[:,1])
    plt.scatter(model.mu_x_train[:,0], model.mu_x_train[:,1])
    # plt.scatter(model.mu_x_test[:,0], model.mu_x_test[:,1])
    plt.show()