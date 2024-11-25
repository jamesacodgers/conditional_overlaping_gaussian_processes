import torch
import time

class WLIndependentGPs:
    """
    GPs provide a function estimate
    Note, cannot be initiated until either MF or overlapping module has been intiated
    """
    def __init__(self, C, M, V, beta = None, sigma2_s = None, trainable_V = True, beta_regularizer = None):
        # torch.set_default_dtype(torch.float64)
        # super(MultiSparseGPs, self).__init__()
        # Store dimensions
        # store variational parameters 
        assert self.C == C 
        assert self.M == M 
        if trainable_V:
            self.V = torch.nn.Parameter(V)
        else:
            self.V = V
        self.input_dims = V.shape[1]
        if beta is None:
            beta = [[1]*self.input_dims]*self.C
        if sigma2_s is None:
            sigma2_s = [1]*self.C

        self.beta_regularizer = beta_regularizer

        # store GP parameters
        self._log_beta = torch.nn.Parameter(torch.log(torch.tensor(beta))) 
        self._log_sigma2_s = torch.nn.Parameter( torch.log(torch.tensor(sigma2_s)))
        self.L = self.V.shape[0]
        self._mu_u = torch.nn.Parameter(torch.zeros(self.M, self.C*self.L ))
        self._mu_u = torch.nn.Parameter(torch.randn(self.M, self.C*self.L )/100)
        # self._log_beta = torch.log(torch.tensor(beta))
        # self._log_sigma2_s = torch.log(torch.tensor(sigma2_s))
        # self.L = self.V.shape[0]
        # self._mu_u = torch.zeros(self.M, self.C*self.L )
        # self._mu_u = torch.randn(self.M, self.C*self.L )/100

        assert self._log_beta.shape == torch.Size([self.C, self.input_dims])
        assert self.V.shape == torch.Size([self.L, self.input_dims])

        K_VV = self.get_K_VV()
        chol = torch.linalg.cholesky(K_VV)
        new_tensor = torch.zeros(self.C, self.C, self.L, self.L)
        new_tensor[range(self.C), range(self.C), :, :] = chol[range(self.C), :, :]
        new_tensor = new_tensor.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
        self._raw_chol = torch.nn.Parameter(new_tensor[None,:,:].repeat(self.M,1,1))
        # self._raw_chol = torch.nn.Parameter(new_tensor[None,:,:])
        # self._raw_chol = torch.nn.Parameter(new_tensor)


    def forward(self):
        ...
        
    def kernel(self, x, y):
        """
        x:   num_points_x x num_input_dims 
        y:   num_points_y x num_input_dims

        returns: num_components  x num_points_x x num_points_y
        """
        return self.sigma2_s[:,None, None]*torch.exp(-1/2*(torch.sum((x[...,None,:,None,:]-y[None,None,:,:])**2/self.beta[:,None, None, :], dim=-1)))
    
    def get_K_VV(self):
        """
        Returns shape C x L x L 
        """
        return self.kernel(self.V, self.V) + torch.eye(self.L)*1e-5
    
    def get_K_VV_matrix(self):
        """
        Returns block diagonal of shape CL x CL 
        """
        K_VV_matrix  = torch.zeros(self.C, self.C, self.L, self.L)
        K_VV = self.get_K_VV()
        K_VV_matrix[range(self.C), range(self.C), :, :] = K_VV[range(self.C),:,:]
        return K_VV_matrix.transpose(1,2).reshape(self.C*self.L, self.C*self.L)
    
    def get_K_VV_inv(self):
        """
        Returns shape C x L x L
        """
        K_VV = self.get_K_VV()
        chol_K_VV = torch.linalg.cholesky(K_VV)
        return torch.cholesky_inverse(chol_K_VV)
    
    def get_K_VV_inv_matrix(self):
        """
        Returns block diagonal of shape CL x CL 
        """
        K_VV_inv_matrix  = torch.zeros(self.C, self.C, self.L, self.L)
        K_VV = self.get_K_VV_inv()
        K_VV_inv_matrix[range(self.C), range(self.C), :, :] = K_VV[range(self.C),:,:]
        return K_VV_inv_matrix.transpose(1,2).reshape(self.C*self.L, self.C*self.L)

    def get_K_obsV(self, inputs):
        """
        Returns shape  N x C x L
        """
        K = self.kernel(inputs, self.V)
        return K.transpose(-3,-2)
    
    def get_K_obsV_matrix(self, temps):
        """
        Returns shape N x CL 
        """
        K_obsV = self.get_K_obsV(temps)
        K_obsV_matrix = torch.zeros(temps.shape[0],self.C, self.C, self.L)
        K_obsV_matrix[:, range(self.C), range(self.C), :] = K_obsV[:, range(self.C),:]
        return K_obsV_matrix.reshape(temps.shape[0], self.C, self.L*self.C)
    

    def get_mu_u(self):
        """
        Returns shape M x C x L 
        """
        return self._mu_u.reshape(self.M, self.C, self.L)
        
    def get_Sigma_u(self):
        """
        Returns shape C x C x L x L
        """
        Sigma_u = self.get_matrix_Sigma_u()
        Sigma_u = Sigma_u.reshape(self.M, self.C, self.L, self.C, self.L)
        # Sigma_u = Sigma_u.reshape(1, self.C, self.L, self.C, self.L)
        
        return Sigma_u.permute(0,1,3,2,4)

    def get_vector_mu_u(self):
        return self._mu_u

    def get_matrix_Sigma_u(self):
        chol = self.get_chol_Sigma_u()
        # return chol @ chol.transpose(-2,-1)[None, :,:] # + (1e-3*torch.eye(self.L*self.C))
        return chol @ chol.transpose(-2,-1)[:, :,:] + (1e-5*torch.eye(self.L*self.C))

    def get_chol_Sigma_u(self):
        return torch.tril(self._raw_chol) 
    
    def get_q_u_dist(self):
        mu_u = self.get_vector_mu_u()
        Sigma_u = self.get_matrix_Sigma_u() 
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
        # return 0.5*torch.distributions.kl_divergence(q_u, prior).sum()
        kl = torch.distributions.kl_divergence(q_u, prior).sum()
        if self.beta_regularizer is not None:
            kl += self.beta.sum()*self.beta_regularizer
        return kl
    
    def get_gp_expectation(self, inputs):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        K_obsV = self.get_K_obsV(inputs) # N x C x L 
        mu_u = self.get_mu_u() # M x C x L 
        return (K_obsV[:, None, :, None, :]@K_VV_inv[None, None, :, :, :]@mu_u[None, :, :, :, None])[...,0,0]

    def get_gp_covariance(self, inputs):
        # want to get to shape NxMxCxC
        K_obsobs_diag_elements = torch.diag_embed(torch.ones(inputs.shape[0], self.M, self.C)*self.sigma2_s[None, None, :]) # N x M x C x C 


        K_obsV = self.get_K_obsV_matrix(inputs) # N x CL x CL - yes this is messy but I think it works 
        A = self._get_conditional_cov_correction() # M x CL x CL

        Sigma_s = K_obsobs_diag_elements - (K_obsV[:,None,:,:] @ A[None,:,:,:]@ K_obsV.transpose(-2,-1)[:, None,:,:])
        return Sigma_s + 1e-5*torch.eye(self.C)
    

    def _get_conditional_cov_correction(self):
        Sigma_u = self.get_matrix_Sigma_u() # M x CL  x CL
        K_VV_inv = self.get_K_VV_inv_matrix() # CL x CL
        return (K_VV_inv
                 -  K_VV_inv @ Sigma_u @ K_VV_inv
                ) 
    @property 
    def sigma2_s(self): 
        return  torch.exp(self._log_sigma2_s)
    @property 
    def beta(self): 
        return  torch.exp(self._log_beta) + 1e-5

## Current thinking of how to include wavelength covariance is to assume a kernel of the form K_c x K_x x K_w and make a mean field approximation on each of these for sigma_c, sigma_u, sigma_w. This would make a mean field approximation over c, which we currently do not make. Take this class and put it into a new wrapper which can take advantage of the simplifications that this would give. 


# class WLDependantGPs:
#     """
#     GPs provide a function estimate
#     Note, cannot be initiated until either MF or overlapping module has been intiated
#     """
#     def __init__(self, C, M, input_dims, V_x, V_w, gamma=None, beta = None, sigma2_s = None, trainable_V = True):
#         # torch.set_default_dtype(torch.float64)
#         # super(MultiSparseGPs, self).__init__()
#         # Store dimensions
#         # store variational parameters 
#         assert self.C == C 
#         assert self.M == M 
        
#         self.V_w = V_w
#         self.V_x = V_x

#         if beta is None:
#             beta = [[1]*input_dims]*self.C
#         if gamma is None:
#             beta = [[1]]*self.C
#         if sigma2_s is None:
#             sigma2_s = [1]*self.C

#         # store GP parameters
#         self._log_beta = torch.nn.Parameter(torch.log(torch.tensor(beta)))
#         self._log_gamma = torch.nn.Parameter(torch.log(torch.tensor(gamma)))
#         self._log_sigma2_s = torch.nn.Parameter( torch.log(torch.tensor(sigma2_s)))
#         self.L_x = self.V_x.shape[0]
#         self.L_w = self.V_w.shape[0]
#         self.L = self.L_x*self.L_w
#         self._mu_u = torch.nn.Parameter(torch.zeros(self.L_w, self.C*self.L_x ))

#         assert self._log_beta.shape == torch.Size([self.C, input_dims])
#         assert self.V_x.shape == torch.Size([self.L, input_dims])
#         assert self.V_w.shape == torch.Size([self.L_w,1])

#         K_VV = self.get_K_VV()
#         chol = torch.linalg.cholesky(K_VV)
#         new_tensor = torch.zeros(self.C, self.C, self.L, self.L)
#         new_tensor[range(self.C), range(self.C), :, :] = chol[range(self.C), :, :]
#         new_tensor = new_tensor.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
#         self._raw_chol = torch.nn.Parameter(new_tensor)
#         K_VV_w = self.get_K_VV_w()
#         self._chol_w = torch.nn.Parameter(torch.linalg.cholesky(K_VV_w))
        

#     def forward(self):
#         ...
        
#     def kernel_x(self, x, y):
#         """
#         x:   num_points_x x num_input_dims 
#         y:   num_points_y x num_input_dims

#         returns: num_components  x num_points_x x num_points_y
#         """
#         return self.sigma2_s[:,None, None]*torch.exp(-1/2*(torch.sum((x[...,None,:,None,:]-y[None,None,:,:])**2/self.beta[:,None, None, :], dim=-1)))
    
#     def kernel_w(self, x, y):
#         """
#         x:   num_points_x x 1 
#         y:   num_points_y x 1

#         returns: num_components  x num_points_x x num_points_y
#         """
#         return torch.exp(-1/2*(torch.sum((x[None,:,None,:]-y[None,None,:,:])**2/self.gamma[:,None, None, :], dim=-1)))
    

    
#     def get_K_VV_inv_x(self):
#         K_VV_x = self.get_K_VV_x()
#         chol = torch.linalg.cholesky(K_VV_x)
#         return torch.cholesky_inverse(chol)
    
#     def get_K_VV_inv_w(self):
#         K_VV_x = self.get_K_VV_x()
#         chol = torch.linalg.cholesky(K_VV_x)
#         return torch.cholesky_inverse(chol)

    
#     def get_K_VV_x(self):
#         """
#         Returns shape C x L x L 
#         """
#         return self.kernel_x(self.V_x, self.V_x) + torch.eye(self.L)*1e-5
    
#     def get_K_VV_w(self):
#         """
#         Returns shape C x L x L 
#         """
#         return self.kernel_w(self.V_l, self.V_l) + torch.eye(self.L)*1e-5

#     def get_K_obsV_x(self, inputs):
#         """
#         Returns shape  N x C x L
#         """
#         K = self.kernel_x(inputs, self.V_x)
#         return K.transpose(0,1)
    
#     def get_K_obsV_w(self, inputs):
#         """
#         Returns shape  M x C x Lw
#         """
#         K = self.kernel_w(inputs, self.V_w)
#         return K.transpose(0,1)
    
#     # def get_K_obsV_matrix_x(self, x):
#     #     """
#     #     Returns shape N x CL 
#     #     """
#     #     K_obsV = self.get_K_obsV_x(x)
#     #     K_obsV_matrix = torch.zeros(x.shape[0],self.C, self.C, self.L)
#     #     K_obsV_matrix[:, range(self.C), range(self.C), :] = K_obsV[:, range(self.C),:]
#     #     return K_obsV_matrix.reshape(x.shape[0], self.C, self.L*self.C)
    
#     # def get_K_obsV_matrix_w(self, w):
#     #     """
#     #     Returns shape N x CL 
#     #     """
#     #     K_obsV = self.get_K_obsV_w(w)
#     #     K_obsV_matrix = torch.zeros(w.shape[0],self.C, self.C, self.L)
#     #     K_obsV_matrix[:, range(self.C), range(self.C), :] = K_obsV[:, range(self.C),:]
#     #     return K_obsV_matrix.reshape(w.shape[0], self.C, self.L*self.C)

#     def get_mu_u(self):
#         """
#         Returns shape C x L_x x L_w
#         """
#         return self._mu_u
        
#     def get_Sigma_u_w(self):
#         """
#         Returns shape Lw x Lw
#         """
#         Sigma_u = self.get_matrix_Sigma_w()
#         Sigma_u = Sigma_u.reshape(self.M, self.C, self.L, self.C, self.L)
        
#         return Sigma_u.permute(0,1,3,2,4)
    
#     def get_Sigma_u_x(self):
#         """
#         Returns shape C x C x Lx x Lx
#         """
#         Sigma_u = self.get_matrix_Sigma_u_w()
#         Sigma_u = Sigma_u.reshape(self.C, self.L, self.C, self.L)
        
#         return Sigma_u.permute(0,2,1,3 )

#     def get_vector_mu_u(self):
#         return self._mu_u

#     # def get_matrix_Sigma_u(self):
#     #     chol = self.get_chol_Sigma_u()
#     #     return chol @ chol.transpose(-2,-1) # + (1e-3*torch.eye(self.L*self.C))

#     def get_chol_Sigma_u_x(self):
#         return torch.tril(self._chol_x) 
    
#     def get_chol_Sigma_u_w(self):
#         return torch.tril(self._chol_w) 
    
#     # def get_q_u_dist(self):
#     #     mu_u = self.get_mu_u().reshape(self.N*self.M*self.C)
#     #     Sigma_u_w = self.get_Sigma_u_w()
#     #     chol_w = torch.linalg.cholesky(Sigma_u_w)
#     #     Sigma_u_w_inv = torch.cholesky_inverse(chol_w)
#     #     Sigma_u_x = self.get_Sigma_u_x()
#     #     chol_x = torch.linalg.cholesky(Sigma_u_x)
#     #     Sigma_u_x_inv = torch.cholesky_inverse(chol_x)
#     #     Sigma_u = Sigma_u_x[:,:,None,:,None]*Sigma_u_w[:,None,:,None,:]
#     #     Sigma_u = Sigma_u.transpose(-3,-2).reshape(self.C, self.L, self.L)

#     #     return torch.distributions.MultivariateNormal(mu_u, Sigma_u)
    
#     # def get_u_prior(self):
#     #     # This bit scars me because I can't think how to test it properly
#     #     K_vv_inv_x = self.get_K_VV_inv_x() # C x Lx x Lx 
#     #     K_vv_inv_w = self.get_K_VV_inv_w() # C x Lw x Lw 
#     #     K_vv_inv = K_vv_inv_x[:,:,None,:,None]*K_vv_inv_w[:,None,:,None,:]
#     #     K_vv_inv = K_vv_inv.transpose(-3,-2).reshape(self.C,self.L,self.L)
#     #     block_kernel_matrix = torch.zeros(self.C, self.C, self.L, self.L)
#     #     block_kernel_matrix[range(self.C), range(self.C),:,:] = K_vv_inv[range(self.C),:,:]
#     #     block_kernel_matrix = block_kernel_matrix.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
#     #     return torch.distributions.MultivariateNormal(torch.zeros(self.C * self.L), precision_matrix=block_kernel_matrix)
     
#     def get_kl_u(self):


#         K_vv_x = self.get_K_VV_x()
#         K_vv_w = self.get_K_VV_w()
#         K_vv_inv_x = self.get_K_VV_inv_x()
#         K_vv_inv_w = self.get_K_VV_inv_w()

#         mu_u = self.get_mu_u()

#         Sigma_u_x = self.get_Sigma_u_x()
#         Sigma_u_w = self.get_Sigma_u_w()

#         log_det_terms = self.L_x*torch.log(torch.det(K_vv_x)/torch.det(Sigma_u_x)) + self.L_w*torch.log(torch.det(K_vv_w)/torch.det(Sigma_u_w))

#         trace_cross = torch.trace(mu_u.transpose(-2,-1)@K_vv_inv_x@mu_u@K_vv_inv_w)

#         trace_prod = torch.trace((K_vv_inv_x@Sigma_u_x))*torch.trace(K_vv_inv_w@Sigma_u_x)

#         return 1/2*(log_det_terms + trace_cross + trace_prod - self.L_w*self.L_x)

        
    
#     def get_gp_expectation(self, conditions, wavelengths):
#         """
#         Returns N x M x C 
#         """
#         K_VV_x_inv = self.get_K_VV_inv_x() # C x L x L 
#         K_obsV_x = self.get_K_obsV_x(conditions) # N x C x L 
#         K_VV_w_inv = self.get_K_VV_inv_x() # C x L x L 
#         K_obsV_w = self.get_K_obsV_x(wavelengths) # N x C x L 

#         mu_u = self.get_mu_u() # C x Lx x Lw
#         mu = K_obsV_x@K_VV_x_inv@mu_u@K_VV_w_inv.T@K_obsV_w.T # Should be C x N x M 
#         return mu.permute(1,2,0)

#     def get_gp_covariance_x(self, conditions):

#         # want to get to shape NxCxC
#         K_obsobs_diag_elements = torch.diag_embed(torch.ones(conditions.shape[0], self.M, self.C)*self.sigma2_s[None, None, :]) # N x M x C x C 


#         K_obsV = self.get_K_obsV_matrix(conditions) # N x CL x CL - yes this is messy but I think it works 
#         A = self._get_conditional_cov_correction() # M x CL x CL

#         Sigma_s = K_obsobs_diag_elements - (K_obsV[:,None,:,:] @ A[None,:,:,:]@ K_obsV.transpose(-2,-1)[:, None,:,:])
#         return Sigma_s + 1e-3*torch.eye(self.C)
    
#     def get_gp_covariance_w(self, wavelengths):
#         """
#         returns shape M 
#         """
#         # want to get to shape M 
#         K_ww_diag_elements = torch.ones(self.M)

#         K_obsV = self.get_K_obsV_w(wavelengths) # N x CL x CL - yes this is messy but I think it works 
#         A = self._get_conditional_cov_correction() # M x CL x CL

#         Sigma_s = K_obsobs_diag_elements - (K_obsV[:,None,:,:] @ A[None,:,:,:]@ K_obsV.transpose(-2,-1)[:, None,:,:])
#         return Sigma_s + 1e-3*torch.eye(self.C)
    

#     def _get_conditional_cov_correction_x(self):
#         Sigma_u_x = self.get_matrix_Sigma_u_x() # M x CL  x CL
#         K_VV_inv = self.get_K_VV_inv_matrix() # CL x CL
#         return (K_VV_inv
#                  -  K_VV_inv @ Sigma_u_x @ K_VV_inv
#                 ) 
    
#     def _get_conditional_cov_correction_w(self):
#         Sigma_u_w = self.get_matrix_Sigma_u_w() # M x CL  x CL
#         K_VV_inv = self.get_K_VV_inv_matrix() # CL x CL
#         return (K_VV_inv
#                  -  K_VV_inv @ Sigma_u_w @ K_VV_inv
#                 ) 
#     @property 
#     def sigma2_s(self): 
#         return  torch.exp(self._log_sigma2_s)
#     @property 
#     def beta(self): 
#         return  torch.exp(self._log_beta)
#     @property 
#     def gamma(self): 
#         return  torch.exp(self._log_gamma)
