from matplotlib import pyplot as plt
import torch
import tqdm

from src.inner_optimisers import MultivariateNormalSolver
from src.utils import get_percentile
from scipy.stats import norm
import numpy as np

from src.utils import generate_regular_simplex

import matplotlib.pyplot as plt



class ConditionalEIOT(torch.nn.Module):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, P = 64, covar_rank = 1, sum_to_one=False):
        super(ConditionalEIOT, self).__init__()
        self.M = M 
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.C = self.C_obs + self.C_unobs
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        self.mu_S_T = torch.nn.Parameter(torch.zeros(M*self.C))
        self.covar_rank = covar_rank
        self.covar_transform = torch.nn.Parameter(torch.zeros(self.M*self.C, covar_rank))
        self._log_std_diag = torch.nn.Parameter(torch.zeros(self.M*self.C))
        self.P=P
        self._log_prior_diag_Sigma_S = torch.nn.Parameter(torch.zeros(self.C_obs))
        self._log_prior_diag_Sigma_T = torch.nn.Parameter(torch.zeros(self.C_unobs))
        self.inner_train_solver = MultivariateNormalSolver(self.C_unobs)
        self._samples = torch.distributions.Normal(0,1).icdf((torch.quasirandom.SobolEngine(self.M*self.C).draw(P)+1/P)*(1-2/P))
        # self._covar_samples = torch.distributions.Normal(0,1).icdf((torch.quasirandom.SobolEngine(covar_rank).draw(P)+1/P)*(1-2/P))

        if sum_to_one:
            self.linear_map = torch.vstack([torch.zeros(self.C-1), torch.eye(self.C-1)])
            self.linear_map[:self.C_obs, :self.C_obs-1] = generate_regular_simplex(self.C_obs-1)
            self.h = torch.zeros(self.C)
            self.h[:self.C_obs] = torch.ones(self.C_obs)/self.C_obs
            self.inner_test_solver = MultivariateNormalSolver(self.C -1)
        else:
            self.linear_map = None
            self.inner_test_solver = MultivariateNormalSolver(self.C)



    def forward(self):
        ...

    def elbo(self, D_train, R_train, D_test=None):
        samples = self.get_samples()
        A_train = 1/(self.sigma2)*(samples[:,:, self.C_obs:].transpose(-2,-1)@samples[:,:, self.C_obs:])[:, None, :, :] # P x 1 x C_unobs x C_unobs
        b_train = 1/(self.sigma2)*(D_train - R_train@samples[:,:, :self.C_obs].transpose(-2,-1))@samples[:,:, self.C_obs:] # P x N x C_unobs
        const_train = -(self.M * D_train.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_train - R_train@samples[:,:,0:self.C_obs].transpose(-2,-1))**2).sum(dim=[-2,-1]).mean()
        train_fit = self.inner_train_solver(A_train, b_train).mean()
        training_inner_elbo = train_fit + const_train
        kl = self.get_kl()
        if D_test is None:
            return training_inner_elbo - kl
        else: 
            if self.linear_map is None:
                A_test = 1/(self.sigma2)*(samples.transpose(-2,-1)@samples)[:, None, :, :] # P x 1 x C x C 
                b_test = 1/(self.sigma2)*D_test@samples # N x C 
                const_test = -(self.M * D_test.shape[0] )/2*torch.log(self.sigma2)- 1/(2*self.sigma2)*(D_test**2).sum()
                test_fit = self.inner_test_solver(A_test, b_test).mean()
                test_inner_elbo = test_fit + const_test
            else: 
                A_test = 1/(self.sigma2)*(self.linear_map.transpose(-2,-1)@(samples.transpose(-2,-1)@samples)@self.linear_map)[:, None, :, :] # P x 1 x C-1 x C-1
                b_test = (1/(self.sigma2)*(D_test[None,:,:] - (self.h[None,None,None,:]*samples[:,None,:,:]).sum(dim=-1))[:,:,:,None,None]*samples[:,None,:,:,None]*self.linear_map[None,None,None,:,:]).sum(dim=[-3,-2])
                const_test = -(self.M * D_test.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_test[None,:,:] - (self.h[None,None,None,:]*samples[:,None,:,:]).sum(dim=-1))**2).sum(dim=[-2,-1]).mean()
                test_fit = self.inner_test_solver(A_test, b_test).mean()
                test_inner_elbo = test_fit + const_test

            return (
                    training_inner_elbo 
                    + test_inner_elbo 
                    - kl
                    # + log_r
                    )
        
    def predict_weights(self, D_test):
        samples = self.get_samples()
        if self.linear_map is None:
            A_test = 1/(self.sigma2)*(samples.transpose(-2,-1)@samples)[:, None, :, :] #P x 1 x C x C 
            b_test = 1/(self.sigma2)*D_test@samples # N x C 
            mu_r, Sigma_r = self.inner_test_solver.get_weight_params(A_test, b_test)
        else:
            A_test = 1/(self.sigma2)*(self.linear_map.transpose(-2,-1)@(samples.transpose(-2,-1)@samples)@self.linear_map)[:, None, :, :] # P x 1 x C-1 x C-1
            b_test = (1/(self.sigma2)*(D_test[None,:,:] - (self.h[None,None,None,:]*samples[:,None,:,:]).sum(dim=-1))[:,:,:,None,None]*samples[:,None,:,:,None]*self.linear_map[None,None,None,:,:]).sum(dim=[-3,-2])
            mu_rho, Sigma_rho = self.inner_test_solver.get_weight_params(A_test, b_test)
            mu_r = mu_rho @ self.linear_map.T + self.h
            Sigma_r = self.linear_map@Sigma_rho@self.linear_map.T + 1e-8

        return torch.distributions.MultivariateNormal(mu_r, Sigma_r)
    
    def get_kl(self):
        variational_dist = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T, validate_args=False)
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.C*self.M), self.prior_Sigma_S_T, validate_args=False)
        return torch.distributions.kl_divergence(variational_dist, prior)
    
    def get_samples(self): 
        chol = torch.linalg.cholesky(self.Sigma_S_T)
        samples = self.mu_S_T + self._samples@chol
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
        return  torch.exp(self._log_std_diag) + 1e-3
    @property 
    def prior_Sigma_S_T(self): 
        return  torch.diag(torch.exp(torch.concat((self._log_prior_diag_Sigma_S, self._log_prior_diag_Sigma_T)).repeat_interleave(self.M)))
    @property 
    def prior_vars_S_T(self): 
        return  torch.exp(torch.concat((self._log_prior_diag_Sigma_S, self._log_prior_diag_Sigma_T)))
    
def train_eiot_lbfgs(model, D_train, R_train, D_test, n_steps = None, plot = False, to_convergence=False):


    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    ll = []
    # Define a closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        loss = -model.elbo( D_train, R_train, D_test)
        loss.backward()
        
        return loss
    if n_steps:
        for epoch in tqdm.tqdm(range(n_steps)):  
            # print(epoch)
            optimizer.step(closure)  # Update model
            ll.append(model.elbo( D_train, R_train, D_test))
    if to_convergence:
        while ll[-1] - ll[-2] > 1:  
            print(f"\r{ll[-1]}")
            optimizer.step(closure)  # Update model
            ll.append(model.elbo( D_train, R_train, D_test))

    if plot:
        with torch.no_grad():
            plt.figure()
            plt.plot(ll)
            plt.show()
    return ll 
def train_eiot_gd(model, D_train, R_train, D_test, n_steps = 200, plot = False):
    optimizer = torch.optim.Rprop(model.parameters())
    ll = []
    # Define a closure function for the optimizer

    for epoch in tqdm.tqdm(range(n_steps)):  # Runs for 10 epochs
        optimizer.zero_grad()
        loss = -model.elbo( D_train, R_train, D_test)
        loss.backward()
        optimizer.step()  # Update model
        ll.append(-loss)
    if plot:
        with torch.no_grad():
            plt.figure()
            plt.plot(ll)
            plt.show()
    return ll 


def train_eiot_sgd(model, D_train, R_train, batch_size=32, n_steps=200, plot=False):
    optimizer = torch.optim.Adam(model.parameters())
    ll = []

    # Create DataLoader for mini-batching
    train_dataset = torch.utils.data.TensorDataset(D_train, R_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in tqdm.tqdm(range(n_steps)):  # Runs for n_steps epochs
        epoch_loss = 0
        
        for batch_D, batch_R in train_loader:
            optimizer.zero_grad()
            loss = -model.elbo(batch_D, batch_R)
            loss.backward()
            optimizer.step()  # Update model
            epoch_loss += -loss.item()  # Accumulate the loss for the epoch

        ll.append(epoch_loss / len(train_loader))  # Store average loss per epoch

    if plot:
        with torch.no_grad():
            plt.figure()
            plt.plot(ll)
            plt.title("ELBO Loss over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.show()
    
    return ll


def plot_preds(R_test, cond_preds: torch.distributions.Distribution = None, mf_preds = None, title=None) -> None: 
    with torch.no_grad():
        # samples = preds.sample((1000, )).reshape((1000*P, 150,  2))
        plt.figure()
        # plt.title(f"C = {C}")
        plt.xlabel("True weights")
        plt.ylabel("predicted weights")

        plt.plot(R_test[:,0], R_test[:,0])
        CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
        if title:
            plt.title(title)
        if cond_preds: 
            means = cond_preds.mean[:,:,0]
            vars = cond_preds.covariance_matrix[:,:,0,0]
            cond_preds_1d = norm(cond_preds.mean[:,:,0].detach().numpy(), np.sqrt(cond_preds.covariance_matrix[:,:,0,0].detach().numpy()))
            err = np.array([get_percentile(cond_preds_1d, 0.5) - get_percentile(cond_preds_1d, 0.025), get_percentile(cond_preds_1d, 0.975) - get_percentile(cond_preds_1d, 0.5)])
            plt.errorbar(R_test[:,0], get_percentile(cond_preds_1d, 0.5), yerr=err, color = CB_color_cycle[0], ls = "None", marker = ".", markersize = 10, capsize=5, label = "hierarchical VI ")
        if mf_preds: 
            mf_means = mf_preds.mean[:,0]
            mf_var = mf_preds.covariance_matrix[:,0,0]
            plt.errorbar(R_test[:,0], mf_means, 1.96*torch.sqrt(mf_var), marker = ".", markersize = 10, fillstyle = "none", ls = "None", capsize=5, color = CB_color_cycle[1], label = "mean field prediction")
        # plt.scatter(R_test[:,0], y_hat[:,0], marker = "x",  color = CB_color_cycle[2], label = "PLS")        
        plt.legend()
        # plt.savefig(f"examples/eiot/figures/toy_eiot_M={M}_Ntrain={N_train}_Ntest={N_test}_Cobs={C_obs}_Cunobs={C_unobs}__P={P}_sigma={sigma}.pdf")
        plt.show()



