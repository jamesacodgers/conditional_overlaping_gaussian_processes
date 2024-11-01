import torch
import time

class MultiSparseGPs:
    """
    GPs provide a function estimate
    Note, cannot be initiated until either MF or overlapping module has been intiated
    """
    def __init__(self, C, M, input_dims, V, beta = None, sigma2_s = None, trainable_V = True):
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

        if beta is None:
            beta = [[1]*input_dims]*self.C
        if sigma2_s is None:
            sigma2_s = [1]*self.C

        # store GP parameters
        self._log_beta = torch.nn.Parameter(torch.log(torch.tensor(beta)))
        self._log_sigma2_s = torch.nn.Parameter( torch.log(torch.tensor(sigma2_s)))
        self.L = self.V.shape[0]
        self._mu_u = torch.nn.Parameter(torch.zeros(self.M, self.C*self.L ))

        assert self._log_beta.shape == torch.Size([self.C, input_dims])
        assert self.V.shape == torch.Size([self.L, input_dims])

        K_VV = self.get_K_VV()
        chol = torch.linalg.cholesky(K_VV)
        new_tensor = torch.zeros(self.C, self.C, self.L, self.L)
        new_tensor[range(self.C), range(self.C), :, :] = chol[range(self.C), :, :]
        new_tensor = new_tensor.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
        self._raw_chol = torch.nn.Parameter(new_tensor[None,:,:].repeat(self.M,1,1))


    def forward(self):
        ...
        
    def kernel(self, x, y):
        """
        x:   num_points_x x num_input_dims 
        y:   num_points_y x num_input_dims

        returns: num_components  x num_points_x x num_points_y
        """
        return self.sigma2_s[:,None, None]*torch.exp(-1/2*(torch.sum((x[None,:,None,:]-y[None,None,:,:])**2/self.beta[:,None, None, :], dim=-1)))
    
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
        chol_K_Vtemp_Vtemp = torch.linalg.cholesky(K_VV)
        return torch.cholesky_inverse(chol_K_Vtemp_Vtemp)
    
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
        return K.transpose(0,1)
    
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
        return Sigma_s + 1e-3*torch.eye(self.C)
    

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
        return  torch.exp(self._log_beta)

    