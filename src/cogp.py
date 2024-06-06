import torch
import time

from src.inner_optimisers import MultivariateNormal, AllComponentsObserved, UnobservedComponents

problem_type_dict = {"continuos": MultivariateNormal}

class COGP(torch.nn.Module):
    def __init__(self, C_obs, C_unobs, M,  V,  sigma2 = 1, beta = None, sigma2_s = None,  trainable_V = False, problem_type = "continuos"):
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

        if problem_type in problem_type_dict:
            self._test_solver = problem_type_dict[problem_type]()
        else: 
            raise ValueError(f"problem type must be one of the implemented versions: {print(problem_type_dict.keys())}")

        if C_unobs == 0: 
            self._train_solver = AllComponentsObserved()
        else:
            self._train_solver = UnobservedComponents(self._test_solver, self.C_obs, self.C_unobs)

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

        self.L = self.V.shape[0]
        # self._mu_u = torch.nn.Parameter(torch.randn(self.M, self.C*self.L ))
        self._mu_u = torch.nn.Parameter(torch.zeros(self.M, self.C*self.L ))

        K_VV = self.get_K_VV()
        chol = torch.linalg.cholesky(K_VV)
        new_tensor = torch.zeros(self.C, self.C, self.L, self.L)
        new_tensor[range(self.C), range(self.C), :, :] = chol[range(self.C), :, :]
        new_tensor = new_tensor.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
        self._raw_chol = torch.nn.Parameter(new_tensor[None,:,:].repeat(self.M,1,1))

        if V.shape[-1] != self.beta.shape[-1]:
            raise ValueError("Inducing points and beta must agree on number of dimensions")

    def forward(self):
        ...
    
    def elbo(self, training_targets, training_weights, training_inputs, 
             test_targets, test_inputs
             ):
        if training_targets.shape[0] != training_weights.shape[0]:
            raise ValueError("Number of training weights needs to match number of training spectra")
        training_fit = self.get_training_fit(
                                targets=training_targets, 
                                training_weights=training_weights,
                                inputs=training_inputs
                                )
        test_fit = self.get_test_fit(
            targets=test_targets,
            inputs=test_inputs,
        )
        KL_u = self.get_kl_u()
        const = self.M * (training_targets.shape[0] 
                                        ) * (torch.log(torch.tensor([2*torch.pi])) + self._log_sigma2)
        return training_fit + test_fit - KL_u - const
        
    def get_test_fit(self, targets, inputs):
        samples = self.get_S_samples(inputs)
        return self._test_solver.get_inner_elbo(samples=samples, targets=targets, sigma2=self.sigma2)

    def get_S_samples(self, temperatures, num_samples=4):
        mu_s = self.get_mu_s(inputs = temperatures, )
        Sigma_s = self.get_Sigma_s_cov_C(inputs=temperatures) 
        s_dist = torch.distributions.MultivariateNormal(mu_s, Sigma_s, validate_args=False)
        return s_dist.rsample((num_samples,))

    def get_training_fit(self, targets, training_weights, inputs):
        mu_s = self.get_mu_s(inputs)
        Sigma_s = self.get_Sigma_s_cov_C(inputs)
        return self._train_solver.get_inner_elbo(targets=targets, observed_weigts=training_weights, mu_s=mu_s, cov_s=Sigma_s, sigma2=self.sigma2, )
        # S = self.get_S_samples(training_temperatures) # P x N x M x C 
        # S_obs = S[..., :self.C_obs]
        # S_unobs = S[..., self.C_obs:] # P x N x M x C_unobs
        # D_unexplained = training_spectra -  (S_obs @ training_weights[..., None])[...,0] # P x N x M training_weights 
        # Ma = (1/self.sigma2 * S_unobs.transpose(-2,-1) @ S_unobs + torch.eye(self.C_unobs)) # P x N x C_unobs x C_unobs
        # Ma_chol = torch.linalg.cholesky(Ma)
        # Ma_inv = torch.cholesky_inverse(Ma_chol)
        # sample_quadratic_terms =  - 1/(2*self.sigma2)*(D_unexplained[..., None, :]@(torch.eye(self.M) - 1/self.sigma2*S_unobs@Ma_inv@S_unobs.transpose(-2,-1))@D_unexplained[...,:, None]).mean(dim=0)
        # log_det_term = torch.log(torch.linalg.det(Ma_inv)).mean(dim=0)
        # return log_det_term.sum() + sample_quadratic_terms.sum()
    
    
    def _kernel(self, x, y):
        if x.shape[-1] != y.shape[-1]:
            raise ValueError(f"input to kernel have differing dimensionalies: {x.shape[-1]} and {y.shape[-1]}")
        return self.sigma_s2[:, None, None]*torch.exp(-1/2*((x[None, :,None,:]-y[None, None,:,:])**2/self.beta[:,None, None,:]**2).sum(dim=-1))
    
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

    def get_K_obsV(self, temps):
        """
        Returns shape  N x C x L
        """
        K = self._kernel(temps, self.V)
        return K.transpose(0,1)
    
    def get_K_obsV_matrix(self, temps):
        K_obsV = self.get_K_obsV(temps)
        K_obsV_matrix = torch.zeros(temps.shape[0],self.C, self.C, self.L)
        K_obsV_matrix[:, range(self.C), range(self.C), :] = K_obsV[:, range(self.C),:]
        return K_obsV_matrix.reshape(temps.shape[0], self.C, self.L*self.C)
    

    def get_mu_u(self):
        """
        Returns shape C x L 
        """
        return self._mu_u.reshape(self.M, self.C, self.L)
        
    def get_Sigma_u(self):
        """
        Returns shape C x C x L x L
        """
        Sigma_u = self.get_matrix_Sigma_u()
        Sigma_u = Sigma_u.reshape(self.M, self.C, self.L, self.C, self.L)
        
        return Sigma_u.permute(0,1,3,2,4)

    def get_vector_mu_u(self):
        return self._mu_u

    def get_matrix_Sigma_u(self):
        chol = self.get_chol_Sigma_u()
        return chol @ chol.transpose(-2,-1) # + (1e-3*torch.eye(self.L*self.C))

    def get_chol_Sigma_u(self):
        return torch.tril(self._raw_chol) 
    
    def get_q_u_dist(self):
        mu_u = self.get_vector_mu_u()
        Sigma_u = self.get_matrix_Sigma_u() + 1e-5*torch.eye(self.C*self.L)
        return torch.distributions.MultivariateNormal(mu_u, Sigma_u)
    
    def get_u_prior(self):
        K_vv = self.get_K_VV()
        block_kernel_matrix = torch.zeros(self.C, self.C, self.L, self.L)
        block_kernel_matrix[range(self.C), range(self.C),:,:] = K_vv[range(self.C),:,:]
        block_kernel_matrix = block_kernel_matrix.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
        return torch.distributions.MultivariateNormal(torch.zeros(self.C * self.L), block_kernel_matrix)
    
    def get_kl_u(self):
        prior = self.get_u_prior()
        q_u = self.get_q_u_dist()
        return torch.distributions.kl_divergence(q_u, prior).sum()
    
    def get_mu_s(self, inputs):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        K_obsV = self.get_K_obsV(inputs) # N x C x L 
        mu_u = self.get_mu_u() # M x C x L 
        return (K_obsV[:, None, :, None, :]@K_VV_inv[None, None, :, :, :]@mu_u[None, :, :, :, None])[...,0,0]

    def get_Sigma_s_cov_C(self, inputs):
        # want to get to shape NxMxCxC
        K_obsobs_diag_elements = torch.diag_embed(torch.ones(inputs.shape[0], self.M, self.C)*self.sigma_s2[None, None, :]) # N x M x C x C 


        K_obsV = self.get_K_obsV_matrix(inputs) # N x CL x CL - yes this is messy but I think it works 
        A = self._get_conditional_cov_correction() # M x CL x CL

        Sigma_s = K_obsobs_diag_elements - (K_obsV[:,None,:,:] @ A[None,:,:,:]@ K_obsV.transpose(-2,-1)[:, None,:,:])
        return Sigma_s + 1e-3*torch.eye(self.C)
    

    def _get_conditional_cov_correction(self):
        Sigma_u = self.get_matrix_Sigma_u() # M x CL  x CL
        K_VV_inv = self.get_K_VV_inv_matrix() # CL x CL
        return (K_VV_inv
                 -  K_VV_inv @ Sigma_u @ K_VV_inv
                ) 
    
    def get_R_estimate_samples(self, test_spectra, test_temperatures, num_samples=100):
        
        dists = self.get_R_dist_samples(test_spectra, test_temperatures, num_samples)
        return dists.rsample()
    
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
    