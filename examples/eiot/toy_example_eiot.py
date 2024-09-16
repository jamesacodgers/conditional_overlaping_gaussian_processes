# %%
import tqdm
import numpy as np
import torch 
from scipy.stats import cauchy, norm, beta
from scipy.optimize import fsolve
from torch.nn.parameter import Parameter
from src.inner_optimisers import MultivariateNormalSolver

from src.inner_optimisers import MultivariateNormalSolver

torch.set_default_dtype(torch.float32)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from src.utils import get_percentile

# torch.manual_seed(1)
# np.random.seed(1)

N_train = 2**6
N_test = 2**3
M = 2**7
P = 2**11
C_obs = 2
C_unobs = 3
sigma = np.sqrt(0.01)
sigma2= sigma**2

S_true = torch.randn(M,C_obs)
T_true = torch.randn(M,C_unobs)

# R_train = torch.distributions.Dirichlet(torch.ones(C_obs)).sample((N,)) 
# R_test = torch.distributions.Dirichlet(torch.ones(C_obs)).sample((N_test,)) 
R_train = torch.randn((N_train,C_obs)) 
R_test = torch.randn((N_test,C_obs)) 
U_train = torch.randn(N_train, C_unobs)
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
        self._log_priors_T = torch.nn.Parameter(torch.zeros(self.C_unobs))
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
        Sigma_prior = torch.diag(torch.concat((torch.ones(self.C_obs), torch.exp(self._log_priors_T))))
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), Sigma_prior)
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
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, P = 64):
        super(ConditionalEIOT, self).__init__()
        self.M = M 
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.C = self.C_obs + self.C_unobs
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        self.mu_S_T = torch.nn.Parameter(torch.randn(M*self.C))
        # self.mu_T = torch.nn.Parameter(torch.eye(M, C_obs))
        # self._raw_chol_S_T = torch.nn.Parameter(torch.eye(self.M*self.C))
        self.covar_transform = torch.nn.Parameter(torch.zeros(self.M*self.C, self.C))
        self._log_var_diag = torch.nn.Parameter(torch.zeros(self.M*self.C))
        self.inner_train_solver = MultivariateNormalSolver(self.C_unobs)
        self.inner_test_solver = MultivariateNormalSolver(self.C)
        self.P=P
        # self._log_prior_diag_Sigma_S = torch.log(1000*torch.ones(self.C_obs))
        self._log_prior_diag_Sigma_S = torch.nn.Parameter(torch.zeros(self.C_obs))
        # self._log_prior_diag_Sigma_T = torch.log(1000*torch.ones(self.C_unobs))
        self._log_prior_diag_Sigma_T = torch.nn.Parameter(torch.zeros(self.C_unobs))

        self._samples = (torch.quasirandom.SobolEngine(self.M*self.C).draw(P)+1e-9)*(1-1e-9)
        self._covar_samples = (torch.quasirandom.SobolEngine(self.C).draw(P)+1e-9)*(1-1e-9)



    def forward(self):
        ...

    def elbo(self, D_train, R_train, D_test):
        samples = self.get_samples()
        A_train = 1/(self.sigma2)*(samples[:,:, self.C_obs:].transpose(-2,-1)@samples[:,:, self.C_obs:])[:, None, :, :] # P x 1 x C_unobs x C_unobs
        b_train = 1/(self.sigma2)*(D_train - R_train@samples[:,:, :self.C_obs].transpose(-2,-1))@samples[:,:, self.C_obs:] # P x N x C_unobs
        const_train = -(self.M * D_train.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_train - R_train@samples[:,:,0:self.C_obs].transpose(-2,-1))**2).sum(dim=[-2,-1]).mean()
        training_inner_elbo = self.inner_train_solver(A_train, b_train, const_train).mean()
        A_test = 1/(self.sigma2)*(samples.transpose(-2,-1)@samples)[:, None, :, :] # P x 1 x C x C 
        b_test = 1/(self.sigma2)*D_test@samples # N x C 
        const_test = -(self.M * D_test.shape[0] )/2*torch.log(self.sigma2)- 1/(2*self.sigma2)*(D_test**2).sum()
        test_inner_elbo = self.inner_test_solver(A_test, b_test, const_test).mean()
        kl = self.get_kl()
        log_r = torch.distributions.Normal(torch.zeros(self.C_obs), torch.exp(self._log_prior_diag_Sigma_S)).log_prob(R_train).sum()
        return (
                training_inner_elbo 
                + test_inner_elbo 
                - kl
                + log_r
                )
        
    def predict_weights(self, D_test):
        samples = self.get_samples()
        A_test = 1/(self.sigma2)*(samples.transpose(-2,-1)@samples)[:, None, :, :] #P x 1 x C x C 
        b_test = 1/(self.sigma2)*D_test@samples # N x C 
        return self.inner_test_solver.get_weight_distribution(A_test, b_test)
    
    def get_kl(self):
        variational_dist = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T)
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.C*self.M), self.prior_Sigma_S_T)
        return torch.distributions.kl_divergence(variational_dist, prior)
    
    def get_samples(self):
        # samples = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T).rsample((self.P,))

        # n = self.P//2
        # # samples = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T).rsample((n,))
        # raw_samples = torch.randn(n, self.M*self.C)
        # raw_covar_samples = torch.randn(n, self.C)
        # anti_covar_samples = -raw_covar_samples
        # covar_samples = torch.vstack((raw_covar_samples, anti_covar_samples))
        # anti_samples = - raw_samples
        # samples = torch.vstack((raw_samples, anti_samples))
        samples = self.mu_S_T + self._samples*self.std_diag + self._covar_samples@self.covar_transform.T
        # samples = self.mu_S_T + samples@self._raw_chol_S_T.T

        return samples.reshape(self.P, self.C, self.M).transpose(-2,-1)

    @property 
    def mu_S(self): 
        return self.mu_S_T[:self.M*self.C_obs]
    @property
    def mu_T(self): 
        return self.mu_S_T[self.M*self.C_obs:]
    @property 
    def Sigma_S_T(self): 
        # return self._raw_chol_S_T.T @ self._raw_chol_S_T
        return torch.diag(self.std_diag**2)+self.covar_transform@self.covar_transform.T
    @property 
    def Sigma_S(self): 
        return self.Sigma_S_T[:self.M*self.C_obs, :self.M*self.C_obs]
    @property 
    def Sigma_T(self): 
        return  self.Sigma_S_T[:self.M*self.C_obs:, :self.M*self.C_obs:]
    @property
    def Sigma_cross(self):
        return self.Sigma_S_T[:self.M*self.C_obs:, ::self.M*self.C_obs]
    @property 
    def sigma2(self): 
        return torch.exp(self._log_sigma2)
    @property 
    def std_diag(self): 
        return  torch.exp(self._log_var_diag)
    @property 
    def prior_Sigma_S_T(self): 
        return  torch.diag(torch.exp(torch.concat((self._log_prior_diag_Sigma_S, self._log_prior_diag_Sigma_T)).repeat_interleave(self.M)))
    @property 
    def prior_vars_S_T(self): 
        return  torch.exp(torch.concat((self._log_prior_diag_Sigma_S, self._log_prior_diag_Sigma_T)))

def train_lbfgs(model, D_train, R_train, D_test, n_steps = 10):


    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    ll = []
    # Define a closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        loss = -model.elbo( D_train, R_train, D_test)
        loss.backward()
        
        return loss

    for epoch in range(n_steps):  # Runs for 10 epochs
        print(epoch)
        optimizer.step(closure)  # Update model
        ll.append(model.elbo( D_train, R_train, D_test))
    with torch.no_grad():
        plt.figure()
        plt.plot(ll)
        plt.show()
def train_sgd(model, D_train, R_train, D_test, n_steps = 200):
    optimizer = torch.optim.Rprop(model.parameters())
    ll = []
    # Define a closure function for the optimizer

    for epoch in tqdm.tqdm(range(n_steps)):  # Runs for 10 epochs
        optimizer.zero_grad()
        loss = -model.elbo( D_train, R_train, D_test)
        loss.backward()
        optimizer.step()  # Update model
        ll.append(-loss)
    with torch.no_grad():
        plt.figure()
        plt.plot(np.log(- np.array(ll) - max(ll)+1e9))
        plt.show()


# mf_eiot = MFEIOT(M, C_obs, C_unobs, 1)
mf_eiot = MFEIOT(M, C_obs, 5, 1)
# conditional_eiot = ConditionalEIOT(M, C_obs, C_unobs, 1, P = P)
conditional_eiot = ConditionalEIOT(M, C_obs, 5, 1, P = P)
# %%
# train_lbfgs(mf_eiot, D_train, R_train, D_test, n_steps=25)
train_lbfgs(conditional_eiot, D_train, R_train, D_test, n_steps=20)
# train_sgd(conditional_eiot, D_train, R_train, D_test)
# train_lbfgs(conditional_eiot, D_train, R_train, D_test, n_steps=200)

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
    plt.errorbar(S_true[:,0], conditional_eiot.mu_S[:M], yerr=1.96*torch.sqrt(conditional_eiot.Sigma_S[range(M),range(M)]), color = "blue", ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
    plt.errorbar(S_true[:,0], mf_eiot.mu_S[:,0], yerr=1.96*torch.sqrt(mf_eiot.Sigma_S[0,0]), color = "orange", ls = "None", marker = ".", markersize = 10, capsize=5, label = "MF VI ")
    # plt.scatter(S_true[:,0], mf_eiot.mu_S[:,0])
    # plt.scatter(S_true[:,0], mf_eiot.mu_S[:,0])
    plt.plot(S_true[:,0],S_true[:,0], c = "orange")
    plt.show()
# %%
# with torch.no_grad():
#         # samples = preds.sample((1000, )).reshape((1000*P, 150,  2))
#         plt.figure()
#         # plt.title(f"C = {C}")
#         plt.xlabel("True weights")
#         plt.ylabel("predicted weights")

#         plt.plot(R_test[:,0], R_test[:,0])
#         mf_means = mf_preds.mean[:,0]
#         mf_var = mf_preds.covariance_matrix[:,0,0]
#         means = cond_preds.mean[:,:,0]
#         vars = cond_preds.covariance_matrix[:,:,0,0]
#         err = np.array([get_percentile(cond_preds_1d, 0.5) - get_percentile(cond_preds_1d, 0.025), get_percentile(cond_preds_1d, 0.975) - get_percentile(cond_preds_1d, 0.5)])
#         plt.errorbar(R_test[:,0], get_percentile(cond_preds_1d, 0.5), yerr=err, color = "blue", ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
#         plt.errorbar(R_test[:,0], mf_means, 1.96*torch.sqrt(mf_var), marker = ".", markersize = 10, ls = "None", capsize=5, color = "orange", label = "mean field prediction")
#         plt.legend()
#         plt.savefig(f"examples/eiot/figures/toy_eiot_M={M}_Ntrain={N_train}_Ntest={N_test}_Cobs={C_obs}_Cunobs={C_unobs}__P={P}_sigma={sigma}.pdf")
#         plt.show()
# %%
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(4)
pls.fit(D_train, R_train)
y_hat = pls.predict(D_test)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

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
        plt.errorbar(R_test[:,0], get_percentile(cond_preds_1d, 0.5), yerr=err, color = CB_color_cycle[0], ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
        plt.errorbar(R_test[:,0], mf_means, 1.96*torch.sqrt(mf_var), marker = ".", markersize = 10, fillstyle = "none", ls = "None", capsize=5, color = CB_color_cycle[1], label = "mean field prediction")
        plt.scatter(R_test[:,0], y_hat[:,0], marker = "x",  color = CB_color_cycle[2], label = "PLS")        
        plt.legend()
        plt.savefig(f"examples/eiot/figures/toy_eiot_M={M}_Ntrain={N_train}_Ntest={N_test}_Cobs={C_obs}_Cunobs={C_unobs}__P={P}_sigma={sigma}.pdf")
        plt.show()

# %%

# print("mf priors: \n ", torch.exp(mf_eiot.))
print("conditional priors: \n", conditional_eiot.prior_vars_S_T)
print("conditional variational posterior: \n", conditional_eiot.Sigma_S_T)
print("mf kl: \n ", mf_eiot.get_kl())
print("conditional kl: \n", conditional_eiot.get_kl())
print("mf max mu: \n ", mf_eiot.mu_S_T.max())
print("conditional max mu: \n", conditional_eiot.mu_S_T.max())

# with torch.no_grad():
#     plt.hist(conditional_eiot.mu_T)
# print("mf sigma2: \n", mf_eiot.sigma2)
# print("conditional sigma2:\n", conditional_eiot.sigma2)

# unobs_cond_preds = norm(cond_preds.mean[:,:,-1].detach().numpy(), np.sqrt(cond_preds.covariance_matrix[:,:,-1,-1].detach().numpy()))
# with torch.no_grad():
#     plt.plot(T_true, T_true)
#     plt.scatter(T_true.flatten(), mf_eiot.mu_T.flatten(),)
#     plt.scatter(T_true.flatten(), conditional_eiot.mu_T.flatten(),)
# %%
with torch.no_grad():
    plt.plot(conditional_eiot.mu_S_T)
    plt.plot(torch.diagonal(conditional_eiot.prior_Sigma_S_T))
    plt.show()
    # plt.plot(torch.diagonal(mf_eiot.prior_Sigma_S_T))
    # plt.plot(mf_eiot.mu_S_T.flatten())

# %%

print(((cond_preds_1d.mean().mean(axis=0) - R_test[:,0].detach().numpy())**2).sum())
print(((mf_preds.mean[:,0].detach().numpy() - R_test[:,0].detach().numpy())**2).sum())
print(((y_hat[:,0] - R_test[:,0].detach().numpy())**2).sum())




# with open("toy_example_hierachical_VI", 'w') as file:
#     file.write(... + '\n')
# with open("toy_example_mf_VI", 'w') as file:
#     file.write(... + '\n')
# with open("toy_example_PLS", 'a') as file:
#     file.write(... + '\n')
# %%
