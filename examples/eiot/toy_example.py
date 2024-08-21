# %%
import numpy as np
import torch 
from scipy.stats import cauchy, norm
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
    
    return mu_n, Sigma_n


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

def get_pure_components(center1, center2, gamma, x_range, M):
    """
    Plot two Cauchy distribution PDFs with given centers and a shared width.
    
    Parameters:
        center1 (float): Center of the first Cauchy distribution.
        center2 (float): Center of the second Cauchy distribution.
        gamma (float): Width (scale parameter) of the Cauchy distributions.
        x_range (tuple): The range of x values for plotting (min, max).
    """
    x = np.linspace(x_range[0], x_range[1], M)
    y1 = cauchy.pdf(x, loc=center1, scale=gamma)
    y2 = cauchy.pdf(x, loc=center2, scale=gamma)
    return torch.tensor([y1,y2]).T
np.random.seed(1234)
torch.manual_seed(1234)

N = 50
M = 100
C = 25
P = 2000


# Example usage
# S_true = get_pure_components(-5, 5, 10, (-10, 10), M)
S_true = torch.randn(M,C)
sigma = 1

R_train = torch.randn(N, C) 
R_test = torch.randn(150,C) 
D_train = R_train@S_true.T 
D_train = D_train + torch.randn_like(D_train)*sigma
# D_train = D_train + torch.randn_like(D_train)*0.1**2

D_test = R_test @ S_true.T
D_test = D_test + torch.randn_like(D_test)*sigma
# D_test = D_test + torch.randn_like(D_test)*0.1**2
# %%
with torch.no_grad():
    plt.plot(D_test.T)

# %%
mu_0 = torch.zeros(M, C)
Sigma_0 = torch.eye(C)*1

mu_n, Sigma_n = bayesian_linear_regression(R_train, D_train, mu_0, Sigma_0, sigma2 = sigma**2)
print("Posterior Mean of Beta:", mu_n)
print("Posterior Covariance of Beta:", Sigma_n)

dist = torch.distributions.MultivariateNormal(mu_n.T, Sigma_n)
with torch.no_grad():
    plt.figure()
    plt.plot(mu_n.T)
    plt.plot(S_true[:,0])
    plt.plot(S_true[:,1])
    plt.show()

S_samples = dist.sample((P,))[:, None, :, :]

def get_weight_distribution(samples,targets,sigma2):
    Ma = (1/sigma2 * samples.transpose(-2,-1) @ samples + torch.eye(samples.shape[-1]))
    Ma_chol = torch.linalg.cholesky(Ma)
    Ma_inv = torch.cholesky_inverse(Ma_chol)
    mu_r = 1/sigma2*(Ma_inv@samples.transpose(-2,-1)@targets[...,None])[...,0]
    dists = torch.distributions.MultivariateNormal(mu_r, Ma_inv, validate_args=False)
    return dists

preds = get_weight_distribution(S_samples, D_test, sigma**2)
print(preds.event_shape, preds.batch_shape)
# %%
with torch.no_grad():
    plt.figure()
    for s_sample in S_samples:
        plt.plot(s_sample[0, :, 0], c = "orange" , alpha = 0.1)
    plt.plot(S_true[:,0])
    plt.show()

# %%
cov = Sigma_n*M + mu_n @ mu_n.T

def mf_get_weight_distribution(cov, mu_n, targets,sigma2):
        Ma = (1/(sigma2) * cov + torch.eye(cov.shape[-1]))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        mu_r = 1/sigma2*Ma_inv@mu_n@targets.T
        dists = torch.distributions.MultivariateNormal(mu_r.T, Ma_inv)
        return dists

def get_percentile(means, vars, alpha):
    dists = norm(means, np.sqrt(vars))
    f = lambda a: dists.cdf(a).mean(axis = 0) - alpha
    return fsolve(f,means.mean(axis=0))



mf_preds = mf_get_weight_distribution(cov, mu_n, D_test, sigma**2)
with torch.no_grad():
    # samples = preds.sample((1000, )).reshape((1000*P, 150,  2))
    # plt.figure()
    plt.plot(R_test[:,0], R_test[:,0])
    mf_means = mf_preds.mean[:,0]
    mf_var = mf_preds.covariance_matrix[:,0,0]
    plt.errorbar(R_test[:,0], mf_means, 1.96*torch.sqrt(mf_var), marker = ".", markersize = 10, ls = "None", capsize=5, color = "orange")
    means = preds.mean[:,:,0]
    vars = preds.covariance_matrix[:,:,0,0]
    plt.scatter(R_test[:,0], means.mean(axis = 0), color = "blue")
    plt.vlines(R_test[:,0],  get_percentile(means, vars, 0.025), get_percentile(means, vars, 0.975),  color = "blue")
    plt.show()
# %%
