import torch
import time

class OverlappingGP(torch.nn.Module):
    def __init__(self, C, M,  V_temp, beta = 1, sigma2 = 1, sigma2_s = 1):
        # torch.set_default_dtype(torch.float64)
        super(OverlappingGP, self).__init__()
        # Store dimensions
        # store variational parameters 
        self.V_temp = V_temp

        # store GP parameters
        self._log_beta = torch.nn.Parameter(torch.log(torch.tensor([beta])))
        # self._log_beta = torch.log(torch.tensor([beta]))
        self._log_sigma_s2 = torch.nn.Parameter( torch.log(torch.tensor([sigma2_s])))
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2)))
        # self._log_sigma2 = torch.log(torch.tensor(sigma2))

        self.C = C
        self.M = M
        self.L = self.V_temp.shape[0]
        self._mu_u = torch.nn.Parameter(torch.zeros(self.M, self.C*self.L ))

        K_VV = self.get_K_VV()
        K_VV = torch.block_diag(*[K_VV]*self.C)
        chol = torch.linalg.cholesky(K_VV)[None,...].repeat(self.M, 1,1)
        self._raw_chol = torch.nn.Parameter(chol)

        if len(V_temp.shape) != 1:
            raise ValueError("Inducing points for temperatures must be 1D")

    def forward(self):
        ...
    
    def elbo(self, training_spectra, training_weights, training_temperatures, wavelengths, 
             test_spectra, test_temperatures
             ):
        if training_spectra.shape[0] != training_weights.shape[0]:
            raise ValueError("Number of training weights needs to match number of training spectra")
        if training_weights.shape[1] != self.C:
            raise ValueError("Number of components of weights needs to match number of components of weights")
        training_fit = self.get_training_fit(
                                training_spectra=training_spectra, 
                                training_weights=training_weights,
                                wavelengths=wavelengths,
                                training_temperatures=training_temperatures
                                )
        test_fit = self.get_test_fit(
            test_spectra=test_spectra,
            test_temperatures=test_temperatures,
            wavelengths=wavelengths
        )
        KL_u = self.get_kl_u()
        const = wavelengths.shape[0] * (training_spectra.shape[0] 
                                        ) * (torch.log(torch.tensor([2*torch.pi])) + self._log_sigma2)
        return (training_fit 
                + test_fit 
                - KL_u 
                - const)
        
    def get_test_fit(self, test_spectra, test_temperatures, wavelengths):
        S = self.get_S_samples(test_temperatures, wavelengths)
        Ma = (1/self.sigma2 * S.transpose(-2,-1) @ S + torch.eye(self.C))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        sample_quadratic_terms =  - 1/(2*self.sigma2)*(test_spectra[..., None, :]@(torch.eye(wavelengths.shape[0]) - 1/self.sigma2*S@Ma_inv@S.transpose(-2,-1))@test_spectra[...,:, None]).mean(dim=0)
        log_det_term = torch.log(torch.linalg.det(Ma_inv)).mean(dim=0)
        return log_det_term.sum() + sample_quadratic_terms.sum()

    def get_S_samples(self, temperatures, wavelengths, num_samples=4):
        mu_s = self.get_mu_s(temps = temperatures, wavelengths=wavelengths)
        Sigma_s = self.get_Sigma_s_cov_C(temps=temperatures, wavelengths=wavelengths) 
        s_dist = torch.distributions.MultivariateNormal(mu_s, Sigma_s, validate_args=False)
        return s_dist.rsample((num_samples,))

    def get_training_fit(self, training_spectra, training_weights, training_temperatures, wavelengths):
        outer_observed_spectra = (training_spectra**2).sum() # 1
        mu_s = self.get_mu_s(training_temperatures, wavelengths) # N x M x C 
        cross_term =(training_spectra[:, :, None] * mu_s [:,:,:] *training_weights[:, None, :]).sum() # 1
        mu_s_outer = (mu_s[:, :, :, None] * mu_s[:, :, None, :]) # N x M x C x C 
        cov_s_marginal = self.get_Sigma_s_cov_C(training_temperatures, wavelengths) # N x M x C x C 
        gp_term = (training_weights[:, None, None, :] * training_weights[:, None, :, None] * (mu_s_outer + cov_s_marginal) ).sum()
        return -1/(2*self.sigma2) * (outer_observed_spectra - 2*cross_term + gp_term)

    
    
    def _temp_kernel(self, x, y):
        return self.sigma_s2*torch.exp(-1/2*(x[:,None]-y[None,:])**2/self.beta)
    
    def get_K_VV(self):
        """
        Returns shape L x L
        """
        return self._temp_kernel(self.V_temp, self.V_temp)
    
    def get_K_VV_inv(self):
        """
        Returns shape L x L
        """
        K_Vtemp_Vtemp = self._temp_kernel(self.V_temp, self.V_temp)
        chol_K_Vtemp_Vtemp = torch.linalg.cholesky(K_Vtemp_Vtemp)
        return torch.cholesky_inverse(chol_K_Vtemp_Vtemp)

    def get_K_obsV(self, temps):
        """
        Returns shape N x M x L
        """
        return self._temp_kernel(temps, self.V_temp)
    

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
        return chol @ chol.transpose(-2,-1) + (1e-4*torch.eye(self.L*self.C))

    def get_chol_Sigma_u(self):
        return torch.tril(self._raw_chol) 
    
    def get_q_u_dist(self):
        mu_u = self.get_vector_mu_u()
        Sigma_u = self.get_matrix_Sigma_u()
        return torch.distributions.MultivariateNormal(mu_u, Sigma_u)
    
    def get_u_prior(self):
        K_vv = self.get_K_VV()
        return torch.distributions.MultivariateNormal(torch.zeros(self.C * self.L), torch.block_diag(*[K_vv]*self.C))
    
    def get_kl_u(self):
        prior = self.get_u_prior()
        q_u = self.get_q_u_dist()
        return torch.distributions.kl_divergence(q_u, prior).sum()
    
    def get_mu_s(self, temps, wavelengths):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_inv() # L x L 
        K_obsV = self.get_K_obsV(temps) # N x M x L 
        mu_u = self.get_mu_u() # C x L 
        return (K_obsV[:, None, None, None, :]@K_VV_inv[None, None, None, :, :]@mu_u[None, :, :, :, None])[...,0,0]

    def get_Sigma_s_cov_C(self, temps, wavelengths):
        # want to get to shape NxMxCxC
        K_obsobs_diag_elements = torch.ones(temps.shape[0], wavelengths.shape[0])*self.sigma_s2 # N x M 


        K_obsV = self.get_K_obsV(temps) # N x L
        A = self._get_conditional_cov_correction() # M x C x C x L x L

        Sigma_s = (K_obsobs_diag_elements[:, :, None, None, None, None] - K_obsV[:, None, None, None,  None, :] @ 
                A[None, :, :, :, :, :] @ K_obsV[:, None, None, None, :, None]
        )[...,0,0]
        return Sigma_s + 5e-3*torch.eye(self.C)
    

    def _get_conditional_cov_correction(self):
        Sigma_u = self.get_Sigma_u() # C x C x L x L 
        K_VV_inv = self.get_K_VV_inv() # L x L 
        return (K_VV_inv[ None, None, :, :] 
                 -  K_VV_inv @ Sigma_u[ :, :, :, :] @ K_VV_inv
                ) 
    
    def get_R_estimate_samples(self, test_spectra, test_temperatures, wavelengths, num_samples=100):
        
        dists = self.get_R_dist_samples(test_spectra, test_temperatures, wavelengths, num_samples)
        return dists.sample()
    
    def get_R_dist_samples(self, test_spectra, test_temperatures, wavelengths, num_samples=100):
        S = self.get_S_samples(test_temperatures, wavelengths, num_samples=num_samples)
        Ma = (1/self.sigma2 * S.transpose(-2,-1) @ S + torch.eye(self.C))
        Ma_chol = torch.linalg.cholesky(Ma)
        Ma_inv = torch.cholesky_inverse(Ma_chol)
        mu_r = 1/self.sigma2*(Ma_inv@S.transpose(-2,-1)@test_spectra[...,None]).squeeze()
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
    