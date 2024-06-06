import torch
from abc import ABC, abstractmethod

class InnerTrainSolver(ABC):


    @abstractmethod
    def get_inner_elbo(self, targets, samples, weights):
        ...

class AllComponentsObserved(InnerTrainSolver):
    def get_inner_elbo(self, targets, mu_s, cov_s,  weights, sigma2, num_samples=None):
        outer_observed_spectra = (targets**2).sum() # 1
        cross_term =(targets[:, :, None] * mu_s [:,:,:] *weights[:, None, :]).sum() # 1
        mu_s_outer = (mu_s[:, :, :, None] * mu_s[:, :, None, :]) # N x M x C x C 
        gp_term = (weights[:, None, None, :] * weights[:, None, :, None] * (mu_s_outer + cov_s) ).sum()
        return -1/(2*sigma2) * (outer_observed_spectra - 2*cross_term + gp_term)

class UnobservedComponents(InnerTrainSolver):
    def __init__(self, test_solver, C_obs, C_unobs, num_samples_for_elbo = 4):
        self.test_solver = test_solver
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.num_samples_for_elbo = num_samples_for_elbo
    
    def get_inner_elbo(self, targets, mu_s, cov_s,  observed_weigts, sigma2):
        gp_dist = torch.distributions.MultivariateNormal(mu_s, cov_s)
        S = gp_dist.rsample((self.num_samples_for_elbo,))
        S_obs = S[..., :self.C_obs]
        S_unobs = S[..., self.C_obs:] # P x N x M x C_unobs
        D_unexplained = targets -  (S_obs @ observed_weigts[..., None])[...,0] # P x N x M training_weights 
        Ma = (1/sigma2 * S_unobs.transpose(-2,-1) @ S_unobs + torch.eye(self.C_unobs)) # P x N x C_unobs x C_unobs
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        sample_quadratic_terms =  - 1/(2*sigma2)*(D_unexplained[..., None, :]@(torch.eye(targets.shape[1]) - 1/sigma2*S_unobs@Ma_inv@S_unobs.transpose(-2,-1))@D_unexplained[...,:, None]).mean(dim=0)
        log_det_term = torch.log(torch.linalg.det(Ma_inv)).mean(dim=0)
        return log_det_term.sum() + sample_quadratic_terms.sum()
    
    


class InnerTestSolver(ABC):
    @abstractmethod
    def get_inner_elbo(self, samples: torch.tensor, target:torch.tensor) -> torch.tensor:
        ...
    @abstractmethod
    def get_weight_distribution(self, samples: torch.tensor, target: torch.tensor) -> torch.tensor: 
        ...


class MultivariateNormal(torch.nn.Module):
    def get_inner_elbo(self, samples, targets, sigma2):
        Ma = (1/sigma2 * samples.transpose(-2,-1) @ samples + torch.eye(samples.shape[-1]))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        sample_quadratic_terms =  - 1/(2*sigma2)*(targets[..., None, :]@(torch.eye(targets.shape[1]) - 1/sigma2*samples@Ma_inv@samples.transpose(-2,-1))@targets[...,:, None]).mean(dim=0)
        log_det_term = torch.log(torch.linalg.det(Ma_inv)).mean(dim=0)
        return log_det_term.sum() + sample_quadratic_terms.sum()
    
    def get_weight_distribution(self, samples,targets,sigma2):
        Ma = (1/sigma2 * samples.transpose(-2,-1) @ samples + torch.eye(samples.shape[-1]))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        mu_r = 1/sigma2*(Ma_inv@samples.transpose(-2,-1)@targets[...,None])[...,0]
        dists = torch.distributions.MultivariateNormal(mu_r, Ma_inv, validate_args=False)
        return dists
    
