import torch
from abc import ABC, abstractmethod

class InnerTrainSolver(ABC):


    @abstractmethod
    def get_inner_elbo(self, targets, samples, weights):
        ...

class AllComponentsObserved(InnerTrainSolver):
    def get_inner_elbo(self, targets, mu_s, cov_s,  observed_weights, sigma2, num_samples=None):
        outer_observed_spectra = (targets**2).sum() # 1
        cross_term =(targets[:, :, None] * mu_s [:,:,:] *observed_weights[:, None, :]) # 1

        mu_s_outer = (mu_s[...,:, :, :, None] * mu_s[...,:, :, None, :]) # N x M x C x C 
        gp_term = (observed_weights[:, None, None, :] * observed_weights[:, None, :, None] * (mu_s_outer + cov_s) )
        if mu_s.shape[:2]==targets.shape:
            cross_term = cross_term.sum()
            gp_term = gp_term.sum()
        else:
            cross_term = cross_term.mean(dim=0).sum()
            gp_term = gp_term.mean(dim=0).sum()
        
        return -1/(2*sigma2) * (outer_observed_spectra - 2*cross_term + gp_term)

class UnobservedComponents(InnerTrainSolver):
    def __init__(self, test_solver, C_obs, C_unobs, num_samples_for_elbo = 4):
        self.test_solver = test_solver
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.num_samples_for_elbo = num_samples_for_elbo
    
    def get_inner_elbo(self, targets, mu_s, cov_s,  observed_weights, sigma2):
        gp_dist = torch.distributions.MultivariateNormal(mu_s, cov_s)
        S = gp_dist.rsample((self.num_samples_for_elbo,))
        S_obs = S[..., :self.C_obs]
        S_unobs = S[..., self.C_obs:] # P x N x M x C_unobs
        D_unexplained = targets -  (S_obs @ observed_weights[..., None])[...,0] # P x N x M training_weights 
        Ma = (1/sigma2 * S_unobs.transpose(-2,-1) @ S_unobs + torch.eye(self.C_unobs)) # P x N x C_unobs x C_unobs
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        sample_quadratic_terms =  - 1/(2*sigma2)*(D_unexplained[..., None, :]@(torch.eye(targets.shape[1]) - 1/sigma2*S_unobs@Ma_inv@S_unobs.transpose(-2,-1))@D_unexplained[...,:, None]).mean(dim=0)
        log_det_term = torch.log(torch.linalg.det(Ma_inv)).mean(dim=0)
        return log_det_term.sum() + sample_quadratic_terms.sum()
    
    


class InnerTestSolver(ABC):
    @abstractmethod
    def get_inner_elbo(self, samples: torch.tensor, target:torch.tensor, sigma2) -> torch.tensor:
        ...
    @abstractmethod
    def get_weight_distribution(self, samples: torch.tensor, target: torch.tensor) -> torch.tensor: 
        ...


class MultivariateNormalSolver(torch.nn.Module, InnerTestSolver):
    def __init__(self, C):
        super(MultivariateNormalSolver,self).__init__()
        self._log_prior = torch.zeros(C)
        # self._log_prior = torch.nn.Parameter(torch.zeros(C))

    def __call__(self, A, b, const):
        return self.get_inner_elbo(A,b, const)

    def get_inner_elbo(self, A, b, const):
        Ma = (A + torch.linalg.inv(self.prior))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        sample_quadratic_terms =  1/2*b[..., None, :]@Ma_inv@b[..., None]
        log_det_term = torch.log(torch.linalg.det(Ma_inv)).flatten()
        return log_det_term*b.shape[-2] + sample_quadratic_terms.sum(dim=[-3,-2,-1]) + const
        # return log_det_term + sample_quadratic_terms.sum(dim=[-3,-2,-1]) + const
    
    def get_weight_distribution(self,A,b):
        Ma = (A + torch.linalg.inv(self.prior))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        mu_r = (Ma_inv@b[...,None])[...,0]
        dists = torch.distributions.MultivariateNormal(mu_r, Ma_inv, validate_args=False)
        return dists
    
    @property 
    def prior(self): 
        return torch.diag(torch.exp(self._log_prior))
    
class Categorical(InnerTestSolver):
    def get_inner_elbo(self, samples: torch.tensor, targets: torch.tensor, sigma2: torch.tensor) -> torch.tensor:
        C = samples.shape[-1]
        dists = ((targets[...,None] - samples)**2).sum(dim=-2)

        min_dists,_ = dists.min(dim=-1)
        return (-1/(2*sigma2)*min_dists/C + torch.log(torch.exp(-1/(2*sigma2)*(dists-min_dists[..., None])).sum(dim=-1)/C)).sum()

    def get_weight_distribution(self, samples: torch.tensor, targets: torch.tensor, sigma2) -> torch.tensor:
        exp_dist_sq = torch.exp(-1/(2*sigma2)*(targets[...,None] - samples)**2).sum(dim=-2)
        probs = exp_dist_sq/exp_dist_sq.sum(dim=-1, keepdim=True)
        return torch.distributions.Categorical(probs=probs)
    
