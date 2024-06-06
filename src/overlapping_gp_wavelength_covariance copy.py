import torch
import time

class OverlappingGPWavelengthCovar(torch.nn.Module):
    def __init__(self, C, V_temp, V_wavelengths):
        # torch.set_default_dtype(torch.float64)
        super(OverlappingGPWavelengthCovar, self).__init__()
        # Store dimensions
        # store variational parameters 
        self.V_temp = V_temp
        self.V_wavelengths = V_wavelengths

        # store GP parameters
        self._log_beta = torch.nn.Parameter(torch.zeros(1)-2)
        self._log_gamma = torch.nn.Parameter(torch.zeros(1)-2)
        self._log_sigma_s2 = torch.nn.Parameter(torch.zeros(1)-3)
        self._log_sigma2 = torch.nn.Parameter(torch.zeros(1)-3)


        self.C = C
        self.L = self.V_temp.shape[0] * self.V_wavelengths.shape[0]
        self._mu_u = torch.nn.Parameter(torch.zeros(self.L * C))

        K_VV = self.get_K_VV().to("cpu")
        K_VV = torch.block_diag(*[K_VV]*C)
        chol = torch.linalg.cholesky(K_VV)
        self._raw_chol = torch.nn.Parameter(chol.to(torch.get_default_device()))


        if len(V_temp.shape) != 1:
            raise ValueError("Inducing points for temperatures must be 1D")
        if len(V_wavelengths.shape) != 1:
            raise ValueError("Inducing points for wavelengths must be 1D")
            



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
                                        # + test_spectra.shape[0]
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

    def get_S_samples(self, temperatures, wavelengths, num_samples=10):
        mu_s = self.get_mu_s(temps = temperatures, wavelengths=wavelengths)
        #TODO: correct this to be not the marginal or rewrite maths it to be independent wavelengths
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

    
    def _wavelength_kernel(self, x, y):
        return torch.exp(-1/2*(x[:,None]-y[None,:])**2/self.gamma) 
    
    def _temp_kernel(self, x, y):
        return torch.exp(-1/2*(x[:,None]-y[None,:])**2/self.beta)
    
    def get_K_VV(self):
        """
        Returns shape L x L
        """
        K_Vtemp_Vtemp = self._temp_kernel(self.V_temp, self.V_temp)
        K_Vw_Vw = self._wavelength_kernel(self.V_wavelengths, self.V_wavelengths)
        return self.sigma_s2*torch.kron(K_Vtemp_Vtemp, K_Vw_Vw) 
    
    def get_K_VV_inv(self):
        """
        Returns shape L x L
        """
        K_Vtemp_Vtemp = self._temp_kernel(self.V_temp, self.V_temp)
        chol_K_Vtemp_Vtemp = torch.linalg.cholesky(K_Vtemp_Vtemp)
        K_Vtemp_Vtemp_inv = torch.cholesky_inverse(chol_K_Vtemp_Vtemp)
        K_Vw_Vw = self._wavelength_kernel(self.V_wavelengths, self.V_wavelengths)
        chol_K_Vw_Vw = torch.linalg.cholesky(K_Vw_Vw)
        K_Vw_Vw_inv = torch.cholesky_inverse(chol_K_Vw_Vw)
        return 1/self.sigma_s2*torch.kron(K_Vtemp_Vtemp_inv, K_Vw_Vw_inv)
    
    def get_K_obsV(self, temps, wavelengths):
        """
        Returns shape N x M x L
        """
        K_wavV = self._wavelength_kernel(wavelengths, self.V_wavelengths)
        K_tempV = self._temp_kernel(temps, self.V_temp)
        K_obsV = K_tempV[:, None, :, None] * K_wavV[None, :, None, :]
        return self.sigma_s2*K_obsV.reshape(temps.shape[0], wavelengths.shape[0], self.L)
    

    def get_mu_u(self):
        """
        Returns shape C x L 
        """
        return self._mu_u.reshape(self.C, self.L)
        
    def get_Sigma_u(self):
        """
        Returns shape C x C x L x L
        """
        Sigma_u = self.get_matrix_Sigma_u()
        Sigma_u = Sigma_u.reshape(self.C, self.L, self.C, self.L)
        
        return Sigma_u.permute(0,2,1,3)

    def get_vector_mu_u(self):
        return self._mu_u

    def get_matrix_Sigma_u(self):
        chol = self.get_chol_Sigma_u()
        return chol @ chol.T + (1e-4*torch.eye(self.L*self.C))

    def get_chol_Sigma_u(self):
        return torch.tril(self._raw_chol) 
    
    def get_q_u_dist(self):
        mu_u = self.get_vector_mu_u().to("cpu")
        Sigma_u = self.get_matrix_Sigma_u().to("cpu")
        return torch.distributions.MultivariateNormal(mu_u, Sigma_u)
    
    def get_u_prior(self):
        K_vv = self.get_K_VV()
        return torch.distributions.MultivariateNormal(torch.zeros(self.C * self.L).to("cpu"), torch.block_diag(*[K_vv]*self.C).to("cpu"))
    
    def get_kl_u(self):
        prior = self.get_u_prior()
        q_u = self.get_q_u_dist()
        return torch.distributions.kl_divergence(q_u, prior)
    
    def get_mu_s(self, temps, wavelengths):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_inv() # L x L 
        K_obsV = self.get_K_obsV(temps, wavelengths) # N x M x L 
        mu_u = self.get_mu_u() # C x L 
        return (K_obsV[:, :, None, None, :]@K_VV_inv[None, None, None, :, :]@mu_u[None, None, :, :, None])[...,0,0]

    def get_Sigma_s_cov_C(self, temps, wavelengths):
        # want to get to shape NxMxCxC
        K_obsobs_diag_elements = torch.ones(temps.shape[0], wavelengths.shape[0])*self.sigma_s2 # N x M 


        K_obsV = self.get_K_obsV(temps, wavelengths) # N x M x L
        A = self._get_conditional_cov_correction() # C x C x L x L

        Sigma_s = (K_obsobs_diag_elements[:, :, None, None, None, None] - K_obsV[:, :, None, None,  None, :] @ 
                A[None, None, :, :, :, :] @ K_obsV[:, :, None, None, :, None]
        )[...,0,0]
        return Sigma_s + 1e-4*torch.eye(self.C)
    

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
    