import torch
import time

class OverlappingGP(torch.nn.Module):
    def __init__(self, C_obs, C_unobs, M,  V_temp,  sigma2 = 1, beta = None, sigma2_s = None):
        # torch.set_default_dtype(torch.float64)
        super(OverlappingGP, self).__init__()
        # Store dimensions
        # store variational parameters 
        self.V_temp = V_temp
        self.C = C_obs + C_unobs
        self.C_obs = C_obs 
        self.C_unobs = C_unobs
        self.M = M

        if beta is None:
            beta = [1]*self.C
        if sigma2_s is None:
            sigma2_s = [1]*self.C

        # store GP parameters
        self._log_beta = torch.nn.Parameter(torch.log(torch.tensor(beta)))
        # self._log_beta = torch.log(torch.tensor([beta]))
        self._log_sigma_s2 = torch.nn.Parameter( torch.log(torch.tensor(sigma2_s)))
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2)))
        # self._log_sigma2 = torch.log(torch.tensor(sigma2))

        self.L = self.V_temp.shape[0]
        # self._mu_u = torch.nn.Parameter(torch.randn(self.M, self.C*self.L ))
        self._mu_u = torch.nn.Parameter(torch.zeros(self.M, self.C*self.L ))

        K_VV = self.get_K_VV()
        chol = torch.linalg.cholesky(K_VV)
        new_tensor = torch.zeros(self.C, self.C, self.L, self.L)
        new_tensor[range(self.C), range(self.C), :, :] = chol[range(self.C), :, :]
        new_tensor = new_tensor.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
        self._raw_chol = torch.nn.Parameter(new_tensor[None,:,:].repeat(self.M,1,1))

        if len(V_temp.shape) != 1:
            raise ValueError("Inducing points for temperatures must be 1D")

    def forward(self):
        ...
    
    def elbo(self, training_spectra, training_weights, training_temperatures, 
             test_spectra, test_temperatures
             ):
        if training_spectra.shape[0] != training_weights.shape[0]:
            raise ValueError("Number of training weights needs to match number of training spectra")
        if training_weights.shape[1] != self.C:
            raise ValueError("Number of components of weights needs to match number of components of weights")
        training_fit = self.get_training_fit(
                                training_spectra=training_spectra, 
                                training_weights=training_weights,
                                training_temperatures=training_temperatures
                                )
        test_fit = self.get_test_fit(
            test_spectra=test_spectra,
            test_temperatures=test_temperatures,
        )
        KL_u = self.get_kl_u()
        const = self.M * (training_spectra.shape[0] 
                                        ) * (torch.log(torch.tensor([2*torch.pi])) + self._log_sigma2)
        return training_fit + test_fit - KL_u - const
        
    def get_test_fit(self, test_spectra, test_temperatures):
        S = self.get_S_samples(test_temperatures)
        Ma = (1/self.sigma2 * S.transpose(-2,-1) @ S + torch.eye(self.C))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        sample_quadratic_terms =  - 1/(2*self.sigma2)*(test_spectra[..., None, :]@(torch.eye(self.M) - 1/self.sigma2*S@Ma_inv@S.transpose(-2,-1))@test_spectra[...,:, None]).mean(dim=0)
        log_det_term = torch.log(torch.linalg.det(Ma_inv)).mean(dim=0)
        return log_det_term.sum() + sample_quadratic_terms.sum()

    def get_S_samples(self, temperatures, num_samples=4):
        mu_s = self.get_mu_s(temps = temperatures, )
        Sigma_s = self.get_Sigma_s_cov_C(temps=temperatures) 
        s_dist = torch.distributions.MultivariateNormal(mu_s, Sigma_s, validate_args=False)
        return s_dist.rsample((num_samples,))

    def get_training_fit(self, training_spectra, training_weights, training_temperatures):
        outer_observed_spectra = (training_spectra**2).sum() # 1
        mu_s = self.get_mu_s(training_temperatures) # N x M x C 
        cross_term =(training_spectra[:, :, None] * mu_s [:,:,:] *training_weights[:, None, :]).sum() # 1
        mu_s_outer = (mu_s[:, :, :, None] * mu_s[:, :, None, :]) # N x M x C x C 
        cov_s_marginal = self.get_Sigma_s_cov_C(training_temperatures) # N x M x C x C 
        gp_term = (training_weights[:, None, None, :] * training_weights[:, None, :, None] * (mu_s_outer + cov_s_marginal) ).sum()
        return -1/(2*self.sigma2) * (outer_observed_spectra - 2*cross_term + gp_term)

    
    
    def _temp_kernel(self, x, y):
        return self.sigma_s2[:, None,None]*torch.exp(-1/2*(x[None, :,None]-y[None, None,:])**2/self.beta[:,None, None])
    
    def get_K_VV(self):
        """
        Returns shape C x L x L 
        """
        return self._temp_kernel(self.V_temp, self.V_temp)
    
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
        K_Vtemp_Vtemp = self._temp_kernel(self.V_temp, self.V_temp)
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
        K = self._temp_kernel(temps, self.V_temp)
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
    
    def get_mu_s(self, temps):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        K_obsV = self.get_K_obsV(temps) # N x C x L 
        mu_u = self.get_mu_u() # M x C x L 
        return (K_obsV[:, None, :, None, :]@K_VV_inv[None, None, :, :, :]@mu_u[None, :, :, :, None])[...,0,0]

    def get_Sigma_s_cov_C(self, temps):
        # want to get to shape NxMxCxC
        K_obsobs_diag_elements = torch.diag_embed(torch.ones(temps.shape[0], self.M, self.C)*self.sigma_s2[None, None, :]) # N x M x C x C 


        K_obsV = self.get_K_obsV_matrix(temps) # N x CL x CL - yes this is messy but I think it works 
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
        return dists.sample()
    
    def get_R_dist_samples(self, test_spectra, test_temperatures, num_samples=100):
        S = self.get_S_samples(test_temperatures, num_samples=num_samples)
        Ma = (1/self.sigma2 * S.transpose(-2,-1) @ S + torch.eye(self.C))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        mu_r = 1/self.sigma2*(Ma_inv@S.transpose(-2,-1)@test_spectra[...,None])[...,0]
        dists = torch.distributions.MultivariateNormal(mu_r, Ma_inv, validate_args=False)
        return dists

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
    