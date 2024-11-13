# %%
import tqdm
import numpy as np
import torch 

import matplotlib.pyplot as plt
from src.mf_methods import MFStaticSignals

from src.utils import train_lbfgs, plot_preds

import numpy as np


# torch.manual_seed(1)
# np.random.seed(1)
torch.set_default_dtype(torch.float64)


sum_to_one_err = []
no_sum_err = []
sum_to_one_S_err = []
no_sum_S_err = []
# %%
N_train = 16
N_test = 64
M = 100
C_obs = 2
C_unobs = 2
covar_rank = 3
sigma = np.sqrt(1e-2)
sigma2= sigma**2

S_true = torch.randn(M,C_obs)
theta=np.pi*1/2
# S_true = 1*torch.tensor([[np.cos(theta),np.cos(theta)],[np.sin(theta),np.sin(theta)]])
A=2
# S_true = torch.tensor([[A*np.cos(theta),1.],[A*np.sin(theta),0.]])
# S_true = torch.tensor([[1.,0.],[0.,1.]])
T_true = torch.randn(M,C_unobs)

# R_train = torch.distributions.Dirichlet(torch.ones(C_obs)).sample((N,)) 
# R_test = torch.distributions.Dirichlet(torch.ones(C_obs)).sample((N_test,)) 

# R_train = torch.randn((N_train,C_obs)) 
R_train = torch.distributions.Dirichlet(torch.ones(C_obs)).sample((N_train,))

# R_test = torch.randn((N_test,C_obs)) 
R_test = torch.distributions.Dirichlet(1*torch.ones(C_obs)).sample((N_test,))
U_train = torch.randn(N_train, C_unobs)
U_test = torch.randn(N_test, C_unobs)
D_train = R_train@S_true.T + U_train@T_true.T
D_train = D_train + torch.randn_like(D_train)*sigma

D_test = R_test @ S_true.T + U_test@T_true.T
D_test = D_test + torch.randn_like(D_test)*sigma




# mf_eiot = MFEIOT(M, C_obs, C_unobs, 1)
mf_eiot_sum = MFStaticSignals(M, C_obs, C_unobs+2, 1, sum_to_one=True)
mf_eiot = MFStaticSignals(M, C_obs, C_unobs+2, 1, sum_to_one=False)

train_lbfgs(model=mf_eiot_sum, D_train=D_train, R_train=R_train, D_test=D_test, steps=25, convergence_level=1)
train_lbfgs(model=mf_eiot, D_train=D_train, R_train=R_train, D_test=D_test, steps=25, convergence_level=1)


mf_preds = mf_eiot.predict_weights(D_test)
mf_preds_sum = mf_eiot_sum.predict_weights(D_test)


# cond_preds = conditional_eiot.predict_weights(D_test)

plot_preds(
    R_test, 
    # cond_preds = cond_preds_sum, 
    mf_preds=mf_preds_sum, 
    title="sum_to_one"
    )
plot_preds(
    R_test, 
    # cond_preds=cond_preds, 
    mf_preds=mf_preds
    )
print(((mf_preds_sum.mean[:,0] - R_test[:,0])**2).mean())
print(((mf_preds.mean[:,0] - R_test[:,0])**2).mean())
# %%
print(mf_eiot.prior_Sigma_S)
print(mf_eiot_sum.prior_Sigma_S)

with torch.no_grad():
    plt.figure()
    for i in range(C_obs+C_unobs):
        plt.scatter(mf_eiot.mu_S[:,i], mf_eiot_sum.mu_S[:,i])
    plt.show()


with torch.no_grad():
    plt.figure()
    for i in range(C_obs):
        plt.scatter(S_true[:,i], S_true[:,i])
    plt.show()