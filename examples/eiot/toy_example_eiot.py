# %%
import numpy as np
import torch 
from scipy.stats import cauchy, norm, beta
from scipy.optimize import fsolve
from torch.nn.parameter import Parameter
from src.inner_optimisers import MultivariateNormalSolver

from src.inner_optimisers import MultivariateNormalSolver

torch.set_default_dtype(torch.float64)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from src.utils import get_percentile

N = 1
N_test = 30
M = 1000
P = 100
C_obs = 1
C_unobs = 1
sigma = 1
sigma2= sigma**2

S_true = torch.randn(M,C_obs)
T_true = torch.randn(M,C_unobs)

# R_train = torch.distributions.Dirichlet(torch.ones(C_obs)).sample((N,)) 
# R_test = torch.distributions.Dirichlet(torch.ones(C_obs)).sample((N_test,)) 
R_train = torch.randn((N,C_obs)) 
R_test = torch.randn((N_test,C_obs)) 
U_train = torch.randn(N, C_unobs)
U_test = torch.randn(N_test, C_unobs)
D_train = R_train@S_true.T + U_train@T_true.T
D_train = D_train + torch.randn_like(D_train)*sigma

D_test = R_test @ S_true.T + U_test@T_true.T
D_test = D_test + torch.randn_like(D_test)*sigma


# class TrainingDataset(torch.nn.Module):
#     def __init__(self, D, R):
#         super(Dataset, self).__init__()

class MFEIOT(torch.nn.Module):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1):
        super(MFEIOT, self).__init__()
        self.M = M 
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.C = self.C_obs + self.C_unobs
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        self.mu_S_T = torch.nn.Parameter(torch.randn(M, self.C))
        # self.mu_T = torch.nn.Parameter(torch.eye(M, C_obs))
        self._raw_chol_S_T = torch.nn.Parameter(torch.eye(self.C))
        self.inner_train_solver = MultivariateNormalSolver(self.C_unobs)
        self.inner_test_solver = MultivariateNormalSolver(self.C)
        # self._raw_chol_T = torch.nn.Parameter(torch.eye(M, C_unobs))


    def forward(self):
        ...

    def elbo(self, D_train, R_train, D_test):
        A_train = 1/(self.sigma2)*(self.mu_T.T@ self.mu_T + self.M*self.Sigma_T) # C_unobs x C_unobs
        b_train = 1/(self.sigma2)*(D_train@self.mu_T - R_train@self.mu_S.T@self.mu_T -  M*R_train@self.Sigma_cross.T) # N x C_unobs
        const_train = -(self.M * D_train.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_train**2).sum() - 2 * torch.einsum("nm, mc, cn", D_train, self.mu_S, R_train.T)+ torch.einsum("nc, cd, dn",R_train,(self.mu_S.T@self.mu_S + M*self.Sigma_S),R_train.T))
        training_inner_elbo = self.inner_train_solver(A_train, b_train, const_train)
        A_test = 1/(self.sigma2)*(self.mu_S_T.T@ self.mu_S_T + self.M*self.Sigma_S_T) # C x C 
        b_test = 1/(self.sigma2)*D_test@self.mu_S_T # N x C 
        const_test = -(self.M * D_test.shape[0] )/2*torch.log(self.sigma2)- 1/(2*self.sigma2)*(D_test**2).sum()
        test_inner_elbo = self.inner_test_solver(A_test, b_test, const_test)
        kl = self.get_kl()
        return (
                training_inner_elbo 
                + test_inner_elbo 
                - kl
                )
    
    def get_unobs_train_weights(self, D_train, R_train):
        b_train = 1/(self.sigma2)*(D_train - R_train@self.mu_S.T)@self.mu_T # N x C 
        A_train = torch.linalg.inv(1/(self.sigma2)*(self.mu_T.T@ self.mu_T + self.M*self.Sigma_T)) # C x C 
        return self.inner_train_solver.get_weight_distribution(A_train, b_train)
    
    def predict_weights(self, D_test):
        A_test = 1/(self.sigma2)*(self.mu_S_T.T@ self.mu_S_T + self.M*self.Sigma_S_T) # C x C 
        b_test = 1/(self.sigma2)*D_test@self.mu_S_T # N x C 
        return self.inner_test_solver.get_weight_distribution(A_test, b_test)
    
    def get_kl(self):
        variational_dist = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T)
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), torch.eye(self.C))
        return torch.distributions.kl_divergence(variational_dist, prior).sum()

    @property 
    def mu_S(self): 
        return self.mu_S_T[:,:self.C_obs]
    @property
    def mu_T(self): 
        return self.mu_S_T[:,self.C_obs:]
    @property 
    def Sigma_S_T(self): 
        return self._raw_chol_S_T.T @ self._raw_chol_S_T
    @property 
    def Sigma_S(self): 
        return self.Sigma_S_T[:self.C_obs, :self.C_obs]
    @property 
    def Sigma_T(self): 
        return  self.Sigma_S_T[self.C_obs:, self.C_obs:]
    @property
    def Sigma_cross(self):
        return self.Sigma_S_T[self.C_obs:, :self.C_obs]
    @property 
    def sigma2(self): 
        return torch.exp(self._log_sigma2)
    
class ConditionalEIOT(torch.nn.Module):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, P = 50):
        super(ConditionalEIOT, self).__init__()
        self.M = M 
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.C = self.C_obs + self.C_unobs
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        self.mu_S_T = torch.nn.Parameter(torch.randn(M, self.C))
        # self.mu_T = torch.nn.Parameter(torch.eye(M, C_obs))
        self._raw_chol_S_T = torch.nn.Parameter(torch.eye(self.C))
        self.inner_train_solver = MultivariateNormalSolver(self.C_unobs)
        self.inner_test_solver = MultivariateNormalSolver(self.C)
        self.P=P
        # self._raw_chol_T = torch.nn.Parameter(torch.eye(M, C_unobs))


    def forward(self):
        ...

    def elbo(self, D_train, R_train, D_test):
        samples = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T).rsample((self.P,))
        A_train = 1/(self.sigma2)*(samples[:,:, self.C_obs:].transpose(-2,-1)@samples[:,:, self.C_obs:])[:, None, :, :] # P x 1 x C_unobs x C_unobs
        b_train = 1/(self.sigma2)*(D_train - R_train@samples[:,:, :self.C_obs].transpose(-2,-1))@samples[:,:, self.C_obs:] # P x N x C_unobs
        const_train = -(self.M * D_train.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_train - R_train@samples[:,:,0:self.C_obs].transpose(-2,-1))**2).sum(dim=[-2,-1]).mean()
        training_inner_elbo = self.inner_train_solver(A_train, b_train, const_train).mean()
        A_test = 1/(self.sigma2)*(samples.transpose(-2,-1)@samples)[:, None, :, :] #P x 1 x C x C 
        b_test = 1/(self.sigma2)*D_test@samples # N x C 
        const_test = -(self.M * D_test.shape[0] )/2*torch.log(self.sigma2)- 1/(2*self.sigma2)*(D_test**2).sum()
        test_inner_elbo = self.inner_test_solver(A_test, b_test, const_test).mean()
        kl = self.get_kl()
        return (
                training_inner_elbo 
                + test_inner_elbo 
                - kl
                )
        
    def predict_weights(self, D_test):
        samples = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T).rsample((self.P,))
        A_test = 1/(self.sigma2)*(samples.transpose(-2,-1)@samples)[:, None, :, :] #P x 1 x C x C 
        b_test = 1/(self.sigma2)*D_test@samples # N x C 
        return self.inner_test_solver.get_weight_distribution(A_test, b_test)
    
    def get_kl(self):
        variational_dist = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T)
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), torch.eye(self.C))
        return torch.distributions.kl_divergence(variational_dist, prior).sum()

    @property 
    def mu_S(self): 
        return self.mu_S_T[:,:self.C_obs]
    @property
    def mu_T(self): 
        return self.mu_S_T[:,self.C_obs:]
    @property 
    def Sigma_S_T(self): 
        return self._raw_chol_S_T.T @ self._raw_chol_S_T
    @property 
    def Sigma_S(self): 
        return self.Sigma_S_T[:self.C_obs, :self.C_obs]
    @property 
    def Sigma_T(self): 
        return  self.Sigma_S_T[self.C_obs:, self.C_obs:]
    @property
    def Sigma_cross(self):
        return self.Sigma_S_T[self.C_obs:, :self.C_obs]
    @property 
    def sigma2(self): 
        return torch.exp(self._log_sigma2)

def train(model, D_train, R_train, D_test):


    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    ll = []
    # Define a closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        loss = -model.elbo( D_train, R_train, D_test)
        loss.backward()
        
        return loss

    for epoch in range(30):  # Runs for 10 epochs
        optimizer.step(closure)  # Update model
        ll.append(model.elbo( D_train, R_train, D_test))
    with torch.no_grad():
        plt.figure()
        plt.plot(ll)
        plt.show()


mf_eiot = MFEIOT(M, C_obs, C_unobs, 1)
conditional_eiot = ConditionalEIOT(M, C_obs, C_unobs, 1, P = P)
train(mf_eiot, D_train, R_train, D_test)
train(conditional_eiot, D_train, R_train, D_test)

# unobs_weig = mf_eiot.get_unobs_train_weights(D_train, R_train)
mf_preds = mf_eiot.predict_weights(D_test)
cond_preds = conditional_eiot.predict_weights(D_test)
cond_preds_1d = norm(cond_preds.mean[:,:,0].detach().numpy(), np.sqrt(cond_preds.covariance_matrix[:,:,0,0].detach().numpy()))

# # %%
# with torch.no_grad():
#     plt.figure()
#     plt.title("weights preds")
#     plt.errorbar(R_test[:,0], mf_preds.mean[:,0], yerr=1.96*mf_preds.covariance_matrix[:,0,0], color = "blue", ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
#     plt.scatter(R_test[:,0], mf_preds.mean[:,0])
#     plt.plot(R_test[:,0], R_test[:,0])
#     plt.show()
# %%
with torch.no_grad():
    plt.figure()
    plt.title("signal preds")
    plt.errorbar(S_true[:,0], conditional_eiot.mu_S[:,0], yerr=1.96*torch.sqrt(conditional_eiot.Sigma_S[0,0]), color = "blue", ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
    plt.errorbar(S_true[:,0], mf_eiot.mu_S[:,0], yerr=1.96*torch.sqrt(mf_eiot.Sigma_S[0,0]), color = "orange", ls = "None", marker = ".", markersize = 10, capsize=5, label = "MF VI ")
    # plt.scatter(S_true[:,0], mf_eiot.mu_S[:,0])
    # plt.scatter(S_true[:,0], mf_eiot.mu_S[:,0])
    plt.plot(S_true[:,0],S_true[:,0], c = "orange")
    plt.show()
# %%
with torch.no_grad():
        # samples = preds.sample((1000, )).reshape((1000*P, 150,  2))
        plt.figure()
        # plt.title(f"C = {C}")
        plt.xlabel("True weights")
        plt.ylabel("predicted weights")

        plt.plot(R_test[:,0], R_test[:,0])
        mf_means = mf_preds.mean[:,0]
        mf_var = mf_preds.covariance_matrix[:,0,0]
        means = cond_preds.mean[:,:,0]
        vars = cond_preds.covariance_matrix[:,:,0,0]
        err = np.array([get_percentile(cond_preds_1d, 0.5) - get_percentile(cond_preds_1d, 0.025), get_percentile(cond_preds_1d, 0.975) - get_percentile(cond_preds_1d, 0.5)])
        plt.errorbar(R_test[:,0], get_percentile(cond_preds_1d, 0.5), yerr=err, color = "blue", ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
        plt.errorbar(R_test[:,0], mf_means, 1.96*torch.sqrt(mf_var), marker = ".", markersize = 10, ls = "None", capsize=5, color = "orange", label = "mean field prediction")
        plt.legend()
        plt.show()