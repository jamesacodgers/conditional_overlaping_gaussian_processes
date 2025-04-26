import torch
import time
from abc import ABC, abstractmethod

class AbstractGP(ABC, torch.nn.Module):
    def __init__(self, C, M, beta, sigma2_s, trainable_V, Vx):
        torch.nn.Module.__init__(self)
        
        self.C=C
        self.M=M

        if trainable_V:
            self.Vx = torch.nn.Parameter(Vx)
        else:
            self.Vx = Vx
        self.input_dims = Vx.shape[1]
        if beta is None:
            beta = [[1]*self.input_dims]*self.C
        if sigma2_s is None:
            sigma2_s = [1]*self.C

        self._log_beta = torch.nn.Parameter(torch.log(torch.tensor(beta))) 
        self._log_sigma2_s = torch.nn.Parameter( torch.log(torch.tensor(sigma2_s)))
        self.Lx = self.Vx.shape[0]

        assert self._log_beta.shape == torch.Size([self.C, self.input_dims])
        assert self.Vx.shape == torch.Size([self.Lx, self.input_dims])


    @abstractmethod
    def get_s_expectation(self):
        ...

    @abstractmethod
    def get_s_outer_expectation(self):
        ...

    @abstractmethod
    def get_kl_u(self):
        ...

    def kernel_x(self, x, y):
        """
        x:   num_points_x x num_input_dims 
        y:   num_points_y x num_input_dims

        returns: num_components  x num_points_x x num_points_y
        """
        return self.sigma2_s[:,None, None]*torch.exp(-1/2*(torch.sum((x[...,None,:,None,:]-y[None,None,:,:])**2/self.beta[:,None, None, :], dim=-1)))

    def get_K_VV_x(self):
        """
        Returns shape C x L x L 
        """
        return self.kernel_x(self.Vx, self.Vx) + torch.eye(self.Lx)*1e-5
    
    def get_K_xVx(self, inputs):
        """
        Returns shape  N x C x L
        """
        K = self.kernel_x(inputs, self.Vx)
        return K.transpose(-3,-2)
    
    def get_K_VV_x_inv(self):
        """
        Returns shape C x L x L
        """
        K_VVx = self.get_K_VV_x()
        chol_K_VVx = torch.linalg.cholesky(K_VVx)
        return torch.cholesky_inverse(chol_K_VVx)
    
    @property 
    def sigma2_s(self): 
        return  torch.exp(self._log_sigma2_s)
    
    @property 
    def beta(self): 
        return  torch.exp(self._log_beta) + 1e-5

class WLIndependentGPs(AbstractGP):
    """
    GPs provide a function estimate
    Note, cannot be initiated until either MF or overlapping module has been intiated
    """
    def __init__(self, C, M, Vx, beta = None, sigma2_s = None, trainable_V = True):

        super(WLIndependentGPs, self).__init__(C=C, M=M, Vx=Vx, beta=beta, sigma2_s=sigma2_s, trainable_V=trainable_V)

        self._mu_u = torch.nn.Parameter(torch.randn(self.M, self.C*self.Lx )/100)

        K_VVx = self.get_K_VV_x()
        chol = torch.linalg.cholesky(K_VVx)
        new_tensor = torch.zeros(self.C, self.C, self.Lx, self.Lx)
        new_tensor[range(self.C), range(self.C), :, :] = chol[range(self.C), :, :]
        new_tensor = new_tensor.transpose(1,2).reshape(self.Lx*self.C, self.Lx*self.C)
        self._raw_chol = torch.nn.Parameter(new_tensor[None,:,:].repeat(self.M,1,1))



    def forward(self):
        ...
        

    
    def get_K_VV_matrix(self):
        """
        Returns block diagonal of shape CL x CL 
        """
        K_VV_matrix  = torch.zeros(self.C, self.C, self.Lx, self.Lx)
        K_VV = self.get_K_VV_x()
        K_VV_matrix[range(self.C), range(self.C), :, :] = K_VV[range(self.C),:,:]
        return K_VV_matrix.transpose(1,2).reshape(self.C*self.Lx, self.C*self.Lx)
    
    
    def get_K_VV_inv_matrix(self):
        """
        Returns block diagonal of shape CL x CL 
        """
        K_VV_inv_matrix  = torch.zeros(self.C, self.C, self.Lx, self.Lx)
        K_VV = self.get_K_VV_x_inv()
        K_VV_inv_matrix[range(self.C), range(self.C), :, :] = K_VV[range(self.C),:,:]
        return K_VV_inv_matrix.transpose(1,2).reshape(self.C*self.Lx, self.C*self.Lx)


    
    def get_K_obsV_matrix(self, inputs):
        """
        Returns shape N x CL 
        """
        K_obsV = self.get_K_xVx(inputs)
        K_obsV_matrix = torch.zeros(inputs.shape[0],self.C, self.C, self.Lx)
        K_obsV_matrix[:, range(self.C), range(self.C), :] = K_obsV[:, range(self.C),:]
        return K_obsV_matrix.reshape(inputs.shape[0], self.C, self.Lx*self.C)
    

    def get_mu_u(self):
        """
        Returns shape M x C x L 
        """
        return self._mu_u.reshape(self.M, self.C, self.Lx)
        
    def get_Sigma_u(self):
        """
        Returns shape C x C x L x L
        """
        Sigma_u = self.get_matrix_Sigma_u()
        Sigma_u = Sigma_u.reshape(self.M, self.C, self.Lx, self.C, self.Lx)
        # Sigma_u = Sigma_u.reshape(1, self.C, self.L, self.C, self.L)
        
        return Sigma_u.permute(0,1,3,2,4)

    def get_vector_mu_u(self):
        return self._mu_u

    def get_matrix_Sigma_u(self):
        chol = self.get_chol_Sigma_u()
        # return chol @ chol.transpose(-2,-1)[None, :,:] # + (1e-3*torch.eye(self.L*self.C))
        return chol @ chol.transpose(-2,-1)[:, :,:] + (1e-5*torch.eye(self.Lx*self.C))

    def get_chol_Sigma_u(self):
        return torch.tril(self._raw_chol) 
    
    def get_q_u_dist(self):
        mu_u = self.get_vector_mu_u()
        Sigma_u = self.get_matrix_Sigma_u() 
        return torch.distributions.MultivariateNormal(mu_u, Sigma_u)
    
    def get_u_prior(self):
        K_vv = self.get_K_VV_x()
        block_kernel_matrix = torch.zeros(self.C, self.C, self.Lx, self.Lx)
        block_kernel_matrix[range(self.C), range(self.C),:,:] = K_vv[range(self.C),:,:]
        block_kernel_matrix = block_kernel_matrix.transpose(1,2).reshape(self.Lx*self.C, self.Lx*self.C)
        return torch.distributions.MultivariateNormal(torch.zeros(self.C * self.Lx), block_kernel_matrix)
    
    def get_kl_u(self):
        prior = self.get_u_prior()
        q_u = self.get_q_u_dist()
        # return 0.5*torch.distributions.kl_divergence(q_u, prior).sum()
        kl = torch.distributions.kl_divergence(q_u, prior).sum()
        return kl
    
    def get_gp_expectation(self, inputs):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_x_inv() # C x L x L 
        K_obsV = self.get_K_xVx(inputs) # N x C x L 
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
    

    
    def get_s_expectation(self, kernel_expectation):
        """
        returns (P x) N x M x C 
        """
        K_VV_inv = self.get_K_VV_x_inv() # C x L x L 
        mu_u = self.get_mu_u() # M x C x L
        return (kernel_expectation[...,:,None,:,None,:]@K_VV_inv[...,None,None,:,:,:]@mu_u[...,None,:,:,:,None])[...,0,0] # N x M x C 
    
    def get_s_outer_expectation(self, kernel_outer_expectation):
        """
        retuns  N x C x C 
        """
        K_VV = self.get_K_VV_matrix()
        K_VV_inv = self.get_K_VV_inv_matrix()
        mu_u = self.get_vector_mu_u() # M x LC
        Sigma_u = self.get_matrix_Sigma_u() # M x LC x LC (currently broacsts from 1 instead of M )
        Ma = (
            (mu_u[:,:,None]*mu_u[:,None,:]
             - K_VV[None,:,:]  + Sigma_u[:,:,:]).sum(dim=0)
             )
        Ma = (K_VV_inv@Ma@ K_VV_inv.T)  
        Ma = Ma.reshape(self.C, self.Lx, self.C, self.Lx).permute(0,2,1,3) 
        K_xx_expectation = torch.diag(self.sigma2_s)
        A = K_xx_expectation*self.M + (Ma[None,:,:,:,:]*kernel_outer_expectation[:,:,:,:,:]).sum(dim=[-2,-1]) # + torch.eye(self.C)*1e-3
        return A # N x C x C 

## Current thinking of how to include wavelength covariance is to assume a kernel of the form K_c x K_x x K_w and make a mean field approximation on each of these for sigma_c, sigma_u, sigma_w. This would make a mean field approximation over c, which we currently do not make. Take this class and put it into a new wrapper which can take advantage of the simplifications that this would give. 




class WLDependentGPs(AbstractGP):
    """
    GPs provide a function estimate
    Note, cannot be initiated until either MF or overlapping module has been intiated
    """
    def __init__(self, C, M, Vx, Vl, beta = None, gamma=None,  sigma2_s = None, trainable_V = True):

        super(WLDependentGPs, self).__init__(C=C, M=M, Vx=Vx, beta=beta, sigma2_s=sigma2_s, trainable_V=trainable_V)
        # if trainable_V:
        #     self.Vl = torch.nn.Parameter(Vl)
        # else: 
            # self.Vl = Vl
        self.Vl= Vl
        self.Ll = self.Vl.shape[0]
        self.psudo_wavelengths = torch.linspace(0,1,M).reshape(-1,1) # This may be the wrong shape 
        self._mu_u = torch.nn.Parameter(torch.randn(self.C, self.Lx,  self.Ll )/1000)
        if gamma is None:
            gamma = [[0.1]*self.input_dims]*self.C
        self._log_gamma = torch.nn.Parameter(torch.log(torch.tensor(gamma)))
        K_VVx = self.get_K_VV_x()
        
        chol_x = torch.linalg.cholesky(K_VVx)
        chol_l = torch.linalg.cholesky(self.get_K_VV_l())
        
        # Register parameters properly
        self.register_parameter('_raw_chol_x', torch.nn.Parameter(chol_x))
        self.register_parameter('_raw_chol_l', torch.nn.Parameter(chol_l))

        



    def forward(self):
        ...

    def get_mu_u(self):
        return self._mu_u
    
    def get_Sigma_u(self):
        Sigma_ux = self.get_Sigma_u_x()
        Sigma_ul = self.get_Sigma_u_l()
        return Sigma_ux[:,:,:,None,None]*Sigma_ul[:,None,None,:,:] # C x Lx x Lx Ll x Ll
        
    def get_Sigma_u_x(self):
        chol = self.get_chol_Sigma_u_x()
        return chol @ chol.transpose(-1,-2)
    
    def get_Sigma_u_l(self):
        chol = self.get_chol_Sigma_u_l()
        return chol @ chol.transpose(-1,-2)

    def get_chol_Sigma_u_x(self):
        return torch.tril(self._raw_chol_x) 
    
    def get_chol_Sigma_u_l(self):
        return torch.tril(self._raw_chol_l) 
    
    def get_kl_u(self):
        K_VV_l = self.get_K_VV_l()
        K_VV_x = self.get_K_VV_x()

        K_VV_l_inv = self.get_K_VV_l_inv() # C x Ll x Ll
        K_VV_x_inv = self.get_K_VV_x_inv() # C x Lx x Lx

        Sigma_u_x = self.get_Sigma_u_x() # C x Lx x Lx
        Sigma_u_l = self.get_Sigma_u_l() # C x Ll x Ll

        mu_u = self.get_mu_u() # C x Lx x Ll
        log_det = self.Lx*torch.log(torch.det(K_VV_l)/torch.det(Sigma_u_l)) + self.Ll*torch.log(torch.det(K_VV_x)/torch.det(Sigma_u_x)) # C 

        mean_term = torch.sum(torch.diagonal(mu_u @ K_VV_l_inv @ mu_u.transpose(-2,-1)@K_VV_x_inv, dim1=-2, dim2=-1), dim=-1) # C
        trace_prod = torch.sum(torch.diagonal((K_VV_l_inv @ Sigma_u_l), dim1=-2, dim2=-1), dim=-1) * torch.sum(torch.diagonal(K_VV_x_inv @ Sigma_u_x, dim1=-2, dim2=-1),dim=-1)# C
        const = self.Lx* self.Ll
        return 1/2*(log_det + trace_prod + mean_term - const).sum()
    
    def get_gp_expectation(self, inputs):
        K_lVl = self.get_K_lVl() # M x C x Ll
        K_VV_l_inv = self.get_K_VV_l_inv() # C x Ll x Ll
        mu_u = self.get_mu_u() # C x Lx x Ll
        K_VV_x_inv = self.get_K_VV_x_inv() # C x Lx x Lx
        K_xVx = self.get_K_xVx(inputs) # N x C x Lx
        mat =  K_xVx[:,None,:,None,:]@K_VV_x_inv[None, None,:,:,:]@mu_u[None, None,:,:,:]@K_VV_l_inv[None,None,:,:,:]@K_lVl[None,:,:,:,None] # N x M x C x 1 x 1 
        return mat.squeeze((-1,-2)) # N x M x C

    def get_gp_covariance(self, inputs):
        K_xx = self.sigma2_s # C 

        K_VV_x_inv = self.get_K_VV_x_inv() # C x Lx x Lx
        K_xVx = self.get_K_xVx(inputs) # N x C x Lx
        x_conditonal_correction = (K_xVx[:,:,None,:]@K_VV_x_inv[None,:,:,:]@K_xVx[:,:,:,None]).squeeze(dim=[-1,-2]) # N x C 

        K_lVl = self.get_K_lVl() # M x C x Ll
        K_VV_l_inv = self.get_K_VV_l_inv() # C x Ll x Ll
        l_conditonal_correction = (K_lVl[:,:,None,:]@K_VV_l_inv[None,:,:,:]@K_lVl[:,:,:,None]).squeeze(dim=[-1,-2]) # M x C

        conditonal_correction = x_conditonal_correction[:,None,:]*l_conditonal_correction[None,:,:]
    
        x_inducing_point_correction = (K_xVx[:,:,None,:]@K_VV_x_inv[None,:,:,:]@self.get_Sigma_u_x()[None,:,:,:]@K_VV_x_inv[None,:,:,:]@K_xVx[:,:,:,None]).squeeze(dim=[-1,-2]) # N x C
        l_inducing_point_correction = (K_lVl[:,:,None,:]@K_VV_l_inv[None,:,:,:]@self.get_Sigma_u_l()[None,:,:,:]@K_VV_l_inv[None,:,:,:]@K_lVl[:,:,:,None]).squeeze(dim=[-1,-2]) # M x C
        inducing_point_correction = x_inducing_point_correction[:,None,:]*l_inducing_point_correction[None,:,:] # N x M x C

        return K_xx - conditonal_correction + inducing_point_correction # N x M x C 


    def get_s_expectation(self, kernel_expectation):
        """
        requires input kernel_expectation of shape N x C x Lx 
        returns object of shape N x M x C 
        """
        K_lVl = self.get_K_lVl() # M x C x Ll
        K_VV_l_inv = self.get_K_VV_l_inv() # C x Ll x Ll
        mu_u = self.get_mu_u() # C x Lx x Ll
        K_VV_x_inv = self.get_K_VV_x_inv() # C x Lx x Lx
        mat =  kernel_expectation[:,None,:,None,:]@K_VV_x_inv[None, None,:,:,:]@mu_u[None, None,:,:,:]@K_VV_l_inv[None,None,:,:,:]@K_lVl[None,:,:,:,None] # N x M x C x 1 x 1 
        return mat.squeeze((-1,-2)) # N x M x C

    def get_s_outer_expectation(self, kernel_outer_expectation):
        """
        Takes in kernel_outer_expectation of shape N x C x C x Lx x Lx
        retuns  N x C x C 
        """
        K_VV_x_inv = self.get_K_VV_x_inv() # C x Lx x Lx
        mu_u = self.get_mu_u() # C x Lx x Ll
        K_VV_l_inv = self.get_K_VV_l_inv() # C x Ll x Ll
        K_lVl = self.get_K_lVl().permute(1,2,0) # C x Ll x M 
        mean_before_x = K_VV_x_inv@mu_u@K_VV_l_inv@K_lVl # C x Lx x M
        mean_outer_before_x = mean_before_x[:,None,:,:]@mean_before_x.transpose(-2,-1)[None,:,:,:] # C x C x Lx x Lx
        mean_outer_expectation = (mean_outer_before_x[None,:,:,:,:]*kernel_outer_expectation[:,:,:,:,:]).sum(dim=[-2,-1])



        K_lVl =self.get_K_lVl() # M x C x Ll - Note this is flipped to be convenient for covariance calculation 
        wl_conditional_correction = (K_lVl[:,:,None,:]@K_VV_l_inv[None,:,:,:]@K_lVl[:,:,:,None]).sum(dim=0).squeeze([-1,-2]) # C 
        wl_conditional_correction_diag = torch.diag(wl_conditional_correction)
        
        K_VV_x_inv_diag = torch.diag_embed(K_VV_x_inv.permute(1,2,0)).permute(2,3,0,1) # C x Lx x Lx
        # cond_cov_before_x = K_VV_x_inv*wl_variance_decrease_conditioned_on_inducing[:,None,None]
        x_conditional_correction = (kernel_outer_expectation[:,:,:,:,:]*K_VV_x_inv_diag[None,:,:,:,:]).sum(dim=[-2,-1]) # N x C x C

        conditional_correction = x_conditional_correction[:,:,:]*wl_conditional_correction_diag[None,:,:] # N x C x C

        Sigma_u_l = self.get_Sigma_u_l() # C x Ll x Ll
        wl_inducing_covariance = (K_lVl[:,:,None,:]@K_VV_l_inv[None,:,:,:]@ Sigma_u_l[None,:,:,:]@ K_VV_l_inv[None,:,:,:]@K_lVl[:,:,:,None]).sum(dim=0).squeeze([-1,-2]) # C 
        wl_inducing_covariance_diag = torch.diag(wl_inducing_covariance)

        Sigma_u_x = self.get_Sigma_u_x() # C x Lx x Lx
        x_variance_term_before_x = (K_VV_x_inv@Sigma_u_x@K_VV_x_inv) # C x Lx x Lx
        x_variance_term_before_x_diag = torch.diag_embed(x_variance_term_before_x.permute(1,2,0)).permute(2,3,0,1) # C x Lx x Lx
        x_inducing_covariance_expectation = (kernel_outer_expectation[:,:,:,:,:]*x_variance_term_before_x_diag[None,:,:,:,:]).sum(dim=[-1,-2]) # N x C x C

        inducing_covariance_expectation = wl_inducing_covariance_diag[None,:,:]*x_inducing_covariance_expectation # N x C x C

        K_xx_expectation = torch.diag(self.sigma2_s)
        A =  mean_outer_expectation + K_xx_expectation[None,:,:]*self.M - conditional_correction + inducing_covariance_expectation # + torch.eye(self.C)*1e-3
        return A # N x C x C 
    
    def kernel_l(self, x, y):
        """
        x:   num_points_x x num_input_dims 
        y:   num_points_y x num_input_dims

        returns: num_components  x num_points_x x num_points_y
        """
        return torch.exp(-1/2*(torch.sum((x[...,None,:,None,:]-y[None,None,:,:])**2/self.gamma[:,None, None, :], dim=-1)))

    def get_K_VV_l(self):
        """
        Returns shape C x L x L 
        """
        return self.kernel_l(self.Vl, self.Vl) + torch.eye(self.Ll)*1e-5
    
    def get_K_xVx(self, inputs):
        """
        Returns shape  N x C x L
        """
        K = self.kernel_x(inputs, self.Vx)
        return K.transpose(-3,-2)
    
    def get_K_lVl(self): # I am not sure about this
        """
        Returns shape  N x C x L
        """
        K = self.kernel_l(self.psudo_wavelengths, self.Vl)
        return K.transpose(-3,-2)
    
    def get_K_VV_l_inv(self):
        """
        Returns shape C x L x L
        """
        K_VVl = self.get_K_VV_l()
        chol_K_VVl = torch.linalg.cholesky(K_VVl)
        return torch.cholesky_inverse(chol_K_VVl)
    
    def get_K_VV(self):
        K_VV_x = self.get_K_VV_x() # C x Lx x Lx
        K_VV_l = self.get_K_VV_l() # C x Ll x Ll
        return K_VV_x[:,:,:, None, None]*K_VV_l[:, None, None, :,:]
    
    def get_K_VV_inv(self):
        K_VV_x = self.get_K_VV_x_inv()
        K_VV_l = self.get_K_VV_l_inv()
        return K_VV_x[:,:,:, None, None]*K_VV_l[:, None, None, :,:]
    
    @property 
    def gamma(self): 
        return torch.exp(self._log_gamma) + 1e-5


