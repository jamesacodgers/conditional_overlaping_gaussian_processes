import torch
import time

from src.inner_optimisers import MultivariateNormal, Categorical, AllComponentsObserved, UnobservedComponents
from src.inducing_points import FullRankCovariance, MeanField

problem_type_dict = {"continuos": MultivariateNormal, "categorical": Categorical}
inducing_point_variational_family_dict = {"full_rank": FullRankCovariance, "mean_field": MeanField}

# Precidence of dimensions: 

class COGP(torch.nn.Module):
    def __init__(self, C_obs, C_unobs, M,  V,  sigma2 = 1, beta = None, sigma2_s = None,  trainable_V = False, problem_type = "continuos", variational_family = "full_rank"):
        # torch.set_default_dtype(torch.float64)
        super(COGP, self).__init__()
        # Store dimensions
        # store variational parameters 
        if trainable_V:
            self.V = torch.nn.Parameter(V)          # L x dims
        else:
            self.V = V
        self.A = self.V.shape[-1]
        self.C = C_obs + C_unobs
        self.C_obs = C_obs 
        self.C_unobs = C_unobs
        self.M = M
        self.L = self.V.shape[0]

        if beta is None:
            beta = [[1]*self.A]*self.C
        if sigma2_s is None:
            sigma2_s = [1]*self.C

         # store GP parameters
        self._log_beta = torch.nn.Parameter(torch.log(torch.tensor(beta)))
        # self._log_beta = torch.log(torch.tensor([beta]))
        self._log_sigma_s2 = torch.nn.Parameter( torch.log(torch.tensor(sigma2_s)))
        # self._log_sigma_s2 =  torch.log(torch.tensor([sigma2_s]))
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2)))
        # self._log_sigma2 = torch.log(torch.tensor(sigma2))

        K_VV = self.get_K_VV()


        if variational_family in inducing_point_variational_family_dict:
            self._variational_family = inducing_point_variational_family_dict[variational_family](C=self.C, M=self.M, L=self.L, kernel_matrix=K_VV)
        else: 
            raise ValueError(f"problem type must be one of the implemented versions: {print(inducing_point_variational_family_dict.keys())}")

        if problem_type in problem_type_dict:
            self._test_solver = problem_type_dict[problem_type]()
        else: 
            raise ValueError(f"problem type must be one of the implemented versions: {print(problem_type_dict.keys())}")

        if C_unobs == 0: 
            self._train_solver = AllComponentsObserved()
        else:
            self._train_solver = UnobservedComponents(self._test_solver, self.C_obs, self.C_unobs)

        if V.shape[-1] != self.beta.shape[-1]:
            raise ValueError("Inducing points and beta must agree on number of dimensions")

    def forward(self):
        ...

    def get_S_dist(self, ):
        ...
    
    def elbo(self, training_targets, training_weights, training_inputs, 
             test_targets, test_inputs, n_samples = 4
             ):
        if training_targets.shape[0] != training_weights.shape[0]:
            raise ValueError("Number of training weights needs to match number of training spectra")
        if training_inputs.shape[1:] != test_inputs.shape[1:]:
            raise ValueError("input dims must mach for training and test")
        if len(training_inputs.shape) == 2:
            training_inputs = training_inputs.unsqueeze(-2)
            test_inputs = test_inputs.unsqueeze(-2)
        training_fit = self.get_training_fit(
                                targets=training_targets, 
                                training_weights=training_weights,
                                inputs=training_inputs,
                                n_samples = n_samples
                                )
        test_fit = self.get_test_fit(
            targets=test_targets,
            inputs=test_inputs,
            n_samples = n_samples
        )
        KL_u = self.get_kl_u()
        const = self.get_const_term(training_targets.shape[0] + test_targets.shape[0])
        return (training_fit + test_fit - KL_u - const).squeeze()
    
    def get_const_term(self, N):
        return self.M * (N) * (torch.log(torch.tensor([2*torch.pi])) + self._log_sigma2)
        
    def get_test_fit(self, targets, inputs, n_samples):
        samples = self.get_S_samples(inputs, num_samples=n_samples).reshape(-1, targets.shape[0], targets.shape[1], self.C)
        fit = self._test_solver.get_inner_elbo(samples=samples, targets=targets, sigma2=self.sigma2)
        return fit

    def get_training_fit(self, targets: torch.tensor, training_weights: torch.tensor, inputs: torch.tensor, n_samples: torch.tensor) -> torch.tensor:
        mu_s = self.get_mu_s(inputs)
        Sigma_s = self.get_Sigma_s(inputs)
        return self._train_solver.get_inner_elbo(targets=targets, observed_weights=training_weights, mu_s=mu_s, cov_s=Sigma_s, sigma2=self.sigma2, num_samples=n_samples)
    
    def get_S_samples(self, inputs, num_samples=4):
        K_obsobs = self.sigma_s2 
        K_obsV = self.get_K_obsV(inputs) # N x C x L 
        K_VV_inv = self.get_K_VV() # L x L 
        return self._variational_family.get_S_samples(K_obsobs=K_obsobs, K_obsV=K_obsV, K_VV_inv=K_VV_inv, num_samples=num_samples)

    
    def _kernel(self, x, y):
        if x.shape[-1] != y.shape[-1]:
            raise ValueError(f"input to kernel have differing dimensionalies: {x.shape[-1]} and {y.shape[-1]}")
        return self.sigma_s2[:, None, None]*torch.exp(-1/2*((x[..., :,None,:]-y[...,:,:])**2/self.beta[:, None, None,:]**2).sum(dim=-1)) 
    
    def get_K_VV(self):
        """
        Returns shape C x L x L 
        """
        return self._kernel(self.V, self.V)
    
    def get_K_VV_matrix(self):
        """
        Returns shape C x L x L 
        """
        K_VV_matrix  = torch.zeros(self.C, self.C, self.L, self.L)
        K_VV = self.get_K_VV()
        K_VV_matrix[range(self.C), range(self.C), :, :] = K_VV[range(self.C),:,:]
        return K_VV_matrix.transpose(1,2).reshape(self.C*self.L, self.C*self.L)
    
    def get_K_VV_inv(self):
        """
        Returns shape C x L x L
        """
        K_Vtemp_Vtemp = self._kernel(self.V, self.V)
        chol_K_Vtemp_Vtemp = torch.linalg.cholesky(K_Vtemp_Vtemp)
        return torch.cholesky_inverse(chol_K_Vtemp_Vtemp)
    
    def get_K_VV_inv_matrix(self):
        K_VV_inv_matrix  = torch.zeros(self.C, self.C, self.L, self.L)
        K_VV = self.get_K_VV_inv()
        K_VV_inv_matrix[range(self.C), range(self.C), :, :] = K_VV[range(self.C),:,:]
        return K_VV_inv_matrix.transpose(1,2).reshape(self.C*self.L, self.C*self.L)

    def get_K_obsV(self, inputs):
        """
        inputs either of shape N x A or N x C x A 
        Returns shape  N x C x L
        """
        K = self._kernel(inputs.transpose(-3,-2), self.V)
        return K.transpose(-3,-2)
    

    def get_mu_u(self):
        """
        Returns shape C x L 
        """
        return self._variational_family.mu_u
        
    def get_Sigma_u(self):
        """
        Returns shape C x C x L x L
        """
        return self._variational_family.Sigma_u
    
    def get_u_prior(self):
        K_vv = self.get_K_VV()
        block_kernel_matrix = torch.zeros(self.C, self.C, self.L, self.L)
        block_kernel_matrix[range(self.C), range(self.C),:,:] = K_vv[range(self.C),:,:]
        block_kernel_matrix = block_kernel_matrix.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
        return torch.distributions.MultivariateNormal(torch.zeros(self.C * self.L), block_kernel_matrix)
    
    def get_kl_u(self):
        prior = self.get_u_prior()
        q_u = self._variational_family.get_q_u_dist()
        return torch.distributions.kl_divergence(q_u, prior).sum()
    
    def get_mu_s(self, inputs):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        K_obsV = self.get_K_obsV(inputs) # N x C x L 
        return self._variational_family.get_mu_s(K_obsV = K_obsV, K_VV_inv=K_VV_inv)

    def get_Sigma_s(self, inputs):
        K_obsobs = self.sigma_s2 
        K_obsV = self.get_K_obsV(inputs) # N x C x L 
        K_VV_inv = self.get_K_VV() # L x L 
        return self._variational_family.get_Sigma_s(K_obsobs=K_obsobs, K_obsV=K_obsV, K_VV_inv=K_VV_inv)
 
    def get_R_dist_samples(self, targets, inputs, num_samples=100):
        samples = self.get_S_samples(inputs, num_samples=num_samples)
        return self._test_solver.get_weight_distribution(targets=targets, samples=samples, sigma2=self.sigma2)

    @property 
    def sigma2(self): 
        return  torch.exp(self._log_sigma2)
    @property 
    def sigma_s2(self): 
        return  torch.exp(self._log_sigma_s2)
    @property 
    def beta(self): 
        return  torch.exp(self._log_beta)
    @property 
    def gamma(self): 
        return  torch.exp(self._log_gamma)
    