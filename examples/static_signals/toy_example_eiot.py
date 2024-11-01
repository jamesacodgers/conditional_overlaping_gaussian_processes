# %%
import tqdm
import numpy as np
import torch 
from scipy.stats import cauchy, norm, beta
from scipy.optimize import fsolve


from src.eiot import ConditionalEIOT, train_eiot_lbfgs, train_eiot_gd, plot_preds
from src.mf_methods import MFStaticSignals


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from src.utils import get_percentile

# torch.manual_seed(1)
# np.random.seed(1)
torch.set_default_dtype(torch.float64)


sum_to_one_err = []
no_sum_err = []
sum_to_one_S_err = []
no_sum_S_err = []
# %%
for i in range(1):
    N_train = 2**4
    N_test = 2**6
    M = 100
    P = 2**8
    C_obs = 2
    C_unobs = 4
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
    mf_eiot_sum = MFStaticSignals(M, C_obs, C_unobs, 1, sum_to_one=True)
    mf_eiot = MFStaticSignals(M, C_obs, C_unobs, 1, sum_to_one=False)
    cond_eiot_sum = ConditionalEIOT(M, C_obs, C_unobs, 1, P = P, sum_to_one=True)
    cond_eiot = ConditionalEIOT(M, C_obs, C_unobs=4, P = P, covar_rank=covar_rank)
    train_eiot_lbfgs(mf_eiot_sum, D_train, R_train, D_test, n_steps=25, to_convergence=True)
    train_eiot_lbfgs(mf_eiot, D_train, R_train, D_test, n_steps=25, to_convergence=True)

    # train_eiot_lbfgs(cond_eiot, D_train, R_train, D_test, n_steps=10, to_convergence=True)
    # train_eiot_lbfgs(cond_eiot_sum, D_train, R_train, D_test, n_steps=10, to_convergence=True)
    # train_eiot_sgd(mf_eiot, D_train, R_train, D_test, n_steps=5000,)
    # train_eiot_sgd(mf_eiot_sum, D_train, R_train, D_test, n_steps=5000,)
    # train_eiot_sgd(cond_eiot_sum, D_train, R_train, D_test, n_steps=5000,)
    # train_eiot_lbfgs(conditional_eiot, D_train, R_train, D_test, n_steps=3, to_convergence=True)

    mf_preds = mf_eiot.predict_weights(D_test)
    mf_preds_sum = mf_eiot_sum.predict_weights(D_test)
    cond_preds_sum = cond_eiot_sum.predict_weights(D_test)
    cond_preds = cond_eiot.predict_weights(D_test)

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
    # print(mf_preds.mean.sum(dim=1))
    # with torch.no_grad():
    #     sum_to_one_err.append((mf_preds_sum.mean[:,0] - R_test[:,0]).numpy())
    #     no_sum_err.append((mf_preds.mean[:,0] - R_test[:,0]).numpy())
    #     sum_to_one_S_err.append((mf_eiot_sum.mu_S_T[:,0] - S_true[:,0]).numpy())
    #     no_sum_S_err.append((mf_eiot.mu_S_T[:,0] - S_true[:,0]).numpy())

    # with torch.no_grad():
    #     plt.figure()
    #     plt.title("signal preds")
    #     # plt.errorbar(S_true[:,0], conditional_eiot.mu_S[:M], yerr=1.96*torch.sqrt(conditional_eiot.Sigma_S[range(M),range(M)]), color = "blue", ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
    #     plt.errorbar(S_true[:,0], cond_eiot_sum.mu_S[:M], yerr=1.96*torch.sqrt(cond_eiot_sum.Sigma_S[range(M),range(M)]), color = "blue", ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
    #     # plt.errorbar(S_true[:,0], mf_eiot.mu_S[:,0], yerr=1.96*torch.sqrt(mf_eiot.Sigma_S[0,0]), color = "orange", ls = "None", marker = ".", markersize = 10, capsize=5, label = "MF VI ")
    #     # plt.scatter(S_true[:,0], mf_eiot.mu_S[:,0])
    #     # plt.scatter(S_true[:,0], mf_eiot.mu_S[:,0])
    #     plt.plot(S_true[:,0],S_true[:,0], c = "orange")
    #     plt.show()
# %%

# %%
# from sklearn.cross_decomposition import PLSRegression

# pls = PLSRegression(4)
# pls.fit(D_train, R_train)
# # y_hat = pls.predict(D_test)


# # %%
# with torch.no_grad():
#     plt.plot(conditional_eiot.mu_S_T)
#     plt.plot(torch.diagonal(conditional_eiot.prior_Sigma_S_T))
#     plt.show()
# with torch.no_grad():
#     dist = torch.distributions.MultivariateNormal(conditional_eiot.mu_S_T, conditional_eiot.Sigma_S_T)
#     for i in range(10):
#         plt.scatter(range(M*(C_obs + 4)),dist.sample())
#     plt.show()
#     # plt.plot(torch.diagonal(mf_eiot.prior_Sigma_S_T))
#     # plt.plot(mf_eiot.mu_S_T.flatten())


# # %%

# # print("mf priors: \n ", torch.exp(mf_eiot.))
# conditional_eiot.elbo(D_train, R_train, D_test)
# print("conditional priors: \n", conditional_eiot.prior_vars_S_T)

# %%

# fig, axs = plt.subplots(2, figsize=(6,6))
# for s,e in zip(sum_to_one_err, no_sum_err):
#     axs[0].hist(s)
#     axs[1].hist(e)
# plt.show()
# # %%
# with torch.no_grad():
#     plt.figure(figsize=(6,6))
#     plt.plot([0, mf_eiot_sum.mu_S[0,0]], [0, mf_eiot_sum.mu_S[1,0]], color = "red")
#     plt.plot([0, mf_eiot_sum.mu_S[0,1]], [0, mf_eiot_sum.mu_S[1,1]], color = "red")
#     plt.plot([0, mf_eiot.mu_S[0,0]], [0, mf_eiot.mu_S[1,0]], color = "purple")
#     plt.plot([0, mf_eiot.mu_S[0,1]], [0, mf_eiot.mu_S[1,1]], color = "purple")
#     plt.plot([0, S_true.T[0,0]], [0, S_true.T[0,1]])
#     plt.plot([0, S_true.T[1,0]], [0, S_true.T[1,1]])
#     plt.scatter(D_test[:,0],D_test[:,1])
#     recon = mf_preds.mean@mf_eiot.mu_S.T
#     recon_sum = mf_preds_sum.mean@mf_eiot_sum.mu_S.T
#     plt.scatter(recon[:,0], recon[:,1])
#     plt.scatter(recon_sum[:,0], recon_sum[:,1])
#     # plt.ylim(-1,1)
#     plt.show()
# %%
