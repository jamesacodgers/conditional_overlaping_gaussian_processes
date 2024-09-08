# %%
import numpy as np
import torch 
from scipy.stats import cauchy, norm, beta
from scipy.optimize import fsolve


from src.inner_optimisers import MultivariateNormalSolver

torch.set_default_dtype(torch.float64)

def bayesian_linear_regression(X, Y, mu_0, Sigma_0, sigma2):
    """
    Perform Bayesian linear regression using conjugate priors.
    
    Parameters:
        X (numpy.ndarray): Design matrix (n_samples, n_features)
        Y (numpy.ndarray): Response vector (n_samples,)
        mu_0 (numpy.ndarray): Prior mean vector of beta (n_features,)
        Sigma_0 (numpy.ndarray): Prior covariance matrix of beta (n_features, n_features)
        sigma2 (float): Variance of the observation noise
    
    Returns:
        tuple: Posterior mean and covariance matrix of beta
    """
    # Inverting the prior covariance matrix
    # Sigma_0_inv = torch.linalg.inv(Sigma_0)
    Lambda_0 = torch.linalg.inv(1/sigma2*Sigma_0)
    
    # Computing the posterior covariance matrix
    # Sigma_n_inv = Sigma_0_inv + (1/sigma2) * X.T @ X
    Lambda_n = Lambda_0 + X.T @ X
    Lambda_n_inv = torch.linalg.inv(Lambda_n)
    
    # Computing the posterior mean
    mu_n = Lambda_n_inv @ (Lambda_0 @ mu_0.T + X.T @ Y)
    Sigma_n = sigma2*Lambda_n_inv
    
    return mu_n.T, Sigma_n


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy



N = 100
N_test = 150
M = 100
P = 100
C = 20
sigma = 0.1

S_true = torch.randn(M,C)

R_train = torch.distributions.Dirichlet(torch.ones(C)).sample((N,)) 
R_test = torch.distributions.Dirichlet(torch.ones(C)).sample((N_test,)) 
D_train = R_train@S_true.T 
D_train = D_train + torch.randn_like(D_train)*sigma

D_test = R_test @ S_true.T
D_test = D_test + torch.randn_like(D_test)*sigma

# %%
mu_0 = torch.zeros(M, C)
Sigma_0 = torch.eye(C)*1

mu_n, Sigma_n = bayesian_linear_regression(R_train, D_train, mu_0, Sigma_0, sigma2 = sigma**2)

dist = torch.distributions.MultivariateNormal(mu_n, Sigma_n)


S_samples = dist.sample((P,))[:, None, :, :]

def optimize(params, obj, **kwargs):
    params = torch.nn.Parameter(params)
    optimizer = torch.optim.LBFGS([params], line_search_fn="strong_wolfe")
    ll = []
    # Define a closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        loss = -obj(params, **kwargs)
        loss.backward()
        
        return loss
    optim = torch.optim.SGD([params], lr=1e-3, )
    # for _ in range(10000):
    #     loss = -obj(params, **kwargs)
    #     ll.append(loss)
    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()

    # Training loop
    for epoch in range(10):  # Runs for 10 epochs
        optimizer.step(closure)  # Update model
        ll.append(-obj(params, **kwargs))
    with torch.no_grad():
        plt.figure()
        plt.plot(ll)
        plt.show()
    return params



# %%

def dirichlet_mean(
        alpha: torch.tensor, 
) -> torch.tensor:
    alpha_0 = alpha.sum(axis=-1)
    return alpha/alpha_0[..., None]

def dirichlet_cov(
       alpha: torch.tensor, 
) -> torch.Tensor:
    alpha_0 = alpha.sum(axis=-1)
    alpha_tilde = dirichlet_mean(alpha)
    outer_alpha = alpha_tilde[...,None, :]*alpha_tilde[...,None]
    diag_alpha = torch.diag_embed(alpha_tilde)
    return (diag_alpha - outer_alpha)/(alpha_0[..., None, None] + 1)

# def get_weight_distribution(samples, targets,sigma2) -> torch.distributions.Dirichlet:
#         prior = torch.distributions.Dirichlet(torch.ones(samples.shape[0],targets.shape[0]))
#         def obj(log_alpha, **kwargs): 
#             variational_dist = torch.distributions.Dirichlet(torch.exp(log_alpha) + 1e-9)
#             fit = -1/(2*sigma2)*(-2*(torch.einsum("nm,pmc,pnc->pn",targets,samples,variational_dist.mean)) + torch.einsum("pncc -> pn", torch.einsum("pmc,pmd,pnc",samples, samples, variational_dist.mean@variational_dist.mean) + dirichlet_cov(torch.exp(log_alpha))))
#             return fit.sum() - torch.distributions.kl_divergence(variational_dist, prior).sum()
#         kwargs = {"S_outer_expectation": S_outer_expectation, "mu_n": mu_n, "targets": targets, "sigma2": sigma2, "prior":prior}
#         params = optimize(torch.zeros(targets.shape[0], S_outer_expectation.shape[0]), obj, kwargs=kwargs)
#         return torch.distributions.Dirichlet(torch.exp(params))



def get_mf_dirichlet_distribution(mu_n: torch.tensor, Sigma_n: torch.tensor, targets: torch.Tensor, sigma2: torch.Tensor) -> torch.distributions.Dirichlet:
    S_outer_expectation =  (Sigma_n*M + mu_n.T @ mu_n)*torch.ones(targets.shape[0],1,1)
    S_expectation = mu_n *torch.ones(targets.shape[0],1,1)
    return get_variational_posterior_dirichlet(S_expectation, S_outer_expectation, targets, sigma2)

def get_conditional_dirichlet_distribution(mu_n: torch.tensor, Sigma_n: torch.tensor, targets: torch.Tensor, sigma2: torch.Tensor, P) -> torch.distributions.Dirichlet:
    S_dist = torch.distributions.MultivariateNormal(mu_n, Sigma_n)
    S = S_dist.sample((P,))[:,None,:]*torch.ones(1,targets.shape[0],1,1)
    S_outer =  S.transpose(-2,-1)@S*torch.ones(1,targets.shape[0],1,1)
    targets_expanded = targets*torch.ones(P,1,1)
    return get_variational_posterior_dirichlet(S, S_outer, targets_expanded, sigma2)


def get_variational_posterior_dirichlet(S: torch.Tensor,S_outer: torch.Tensor,targets: torch.Tensor,sigma2: torch.Tensor) -> torch.distributions.Dirichlet:
        C = S.shape[-1]
        M = S.shape[-2]
        batch = list(S.shape[:-2])
        if S.shape != torch.Size(batch+[M,C]):
             raise ValueError("Incorrect S shape")
        if S_outer.shape != torch.Size(batch+[C,C]):
             raise ValueError("Incorrect S_outer Shape")
        if targets.shape != torch.Size(batch+[M]):
             raise ValueError("Incorrect target Shape")
        prior = torch.distributions.Dirichlet(torch.ones(S_outer.shape[-1]))
        softplus = torch.nn.Softplus()
        def obj(raw_alpha, **kwargs): 
            variational_dist = torch.distributions.Dirichlet(softplus(raw_alpha) + 1e-9)
            fit = -1/(2*sigma2)*(-2*(targets[...,None,:]@S@variational_dist.mean[...,None])[...,0,0] + torch.diagonal(S_outer@(variational_dist.mean[..., None]@variational_dist.mean[..., None, :] + dirichlet_cov(softplus(raw_alpha))),dim1=-2, dim2=-1).sum(dim=-1))
            return fit.sum() - torch.distributions.kl_divergence(variational_dist, prior).sum()
        kwargs = {"S_outer_expectation": S_outer, "mu_n": S, "targets": targets, "sigma2": sigma2, "prior":prior}
        params = optimize(torch.ones(tuple(batch+[C])), obj, kwargs=kwargs)
        return torch.distributions.Dirichlet(softplus(params) + 1e-9)

def mf_get_percentile(conc_1, conc_0, alpha):
    dists = beta(conc_1, conc_0 - conc_1)
    def f(a):
        return dists.cdf(a) - alpha
    return fsolve(f,conc_1/conc_0)

def get_percentile(conc_1, conc_0, alpha):
    dists = beta(conc_1.detach().numpy(), (conc_0 - conc_1).detach().numpy())
    def f(a):
        return dists.cdf(a).mean(axis=0) - alpha
    return fsolve(f,conc_1.sum(dim=0)/conc_0.sum(dim=0))


# samples = dist.sample((P,))
preds = get_conditional_dirichlet_distribution(mu_n, Sigma_n, D_test, sigma**2, 100)
mf_preds = get_mf_dirichlet_distribution(mu_n, Sigma_n, D_test, sigma**2)
with torch.no_grad():
    plt.figure()
    plt.title(f"C = {C}")
    plt.xlabel("True weights")
    plt.ylabel("predicted weights")

    plt.plot(R_test[:,0], R_test[:,0])
    mf_conc_1 = mf_preds.concentration[:, 0]
    mf_conc_0 = mf_preds.concentration.sum(dim=1)
    conc_1 =preds.concentration[:, :,0]
    conc_0 = preds.concentration.sum(dim=-1)
    plt.errorbar(R_test[:,0], get_percentile(conc_1, conc_0, 0.5), [get_percentile(conc_1, conc_0, 0.5) - get_percentile(conc_1, conc_0, 0.025), get_percentile(conc_1, conc_0, 0.975) - get_percentile(conc_1, conc_0, 0.5)], marker = ".", markersize = 10, ls = "None", capsize=5, color = "blue", label = "conditional prediction")
    plt.errorbar(R_test[:,0], mf_get_percentile(mf_conc_1, mf_conc_0, 0.5), [mf_get_percentile(mf_conc_1, mf_conc_0, 0.5) - mf_get_percentile(mf_conc_1, mf_conc_0, 0.025), mf_get_percentile(mf_conc_1, mf_conc_0, 0.975) - mf_get_percentile(mf_conc_1, mf_conc_0, 0.5)], marker = ".", markersize = 10, ls = "None", capsize=5, color = "orange", label = "mean field prediction")
    plt.legend()
    # plt.ylim(-3.5, 3.5)
    plt.savefig(f"examples/eiot/figures/toy_C={C}.pdf")
    
    # %%
plt.show()
