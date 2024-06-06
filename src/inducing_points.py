import torch
from abc import ABC,abstractmethod

class InducingPointVariationalDist(torch.nn.Module,ABC):
    def __init__(self):
        super(InducingPointVariationalDist, self).__init__()
        self._mu_u = torch.nn.Parameter(torch.zeros(self.M, self.C*self.L ))
    
    def get_mu_u(self):
        """
        Returns shape C x L 
        """
        return self._mu_u.reshape(self.M, self.C, self.L)
    
    def get_vector_mu_u(self):
        return self._mu_u
    
    def get_q_u_dist(self):
        mu_u = self.get_vector_mu_u()
        Sigma_u = self.get_matrix_Sigma_u() + 1e-5*torch.eye(self.C*self.L)
        return torch.distributions.MultivariateNormal(mu_u, Sigma_u)
    
    def get_mu_s(self, inputs):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        K_obsV = self.get_K_obsV(inputs) # N x C x L 
        mu_u = self.get_mu_u() # M x C x L 
        return (K_obsV[...,:, None, :, None, :]@K_VV_inv[None, None, :, :, :]@mu_u[None, :, :, :, None])[...,0,0]
    
    @abstractmethod
    def get_Sigma_s_cov_C(self, inputs):
        ...
    
    @abstractmethod
    def get_Sigma_u(self):
        ...

    @abstractmethod
    def get_matrix_Sigma_u(self):
        ...

    @abstractmethod    
    def get_chol_Sigma_u(self):
        ...
    
class FullRankCovariance(InducingPointVariationalDist):
    def __init__(self, C, kernel_matrix):
        super(FullRankCovariance, self).__init__()


    def get_Sigma_u(self):
        """
        Returns shape C x C x L x L
        """
        super(FullRankCovariance, self).__init__()
        Sigma_u = self.get_matrix_Sigma_u()
        Sigma_u = Sigma_u.reshape(self.M, self.C, self.L, self.C, self.L)
        return Sigma_u.permute(0,1,3,2,4)
    
    def get_matrix_Sigma_u(self):
        chol = self.get_chol_Sigma_u()
        return chol @ chol.transpose(-2,-1) # + (1e-3*torch.eye(self.L*self.C))
    
    def get_chol_Sigma_u(self):
        return torch.tril(self._raw_chol) 
    
    def get_mu_s(self, inputs):
        """
        Returns N x M x C 
        """
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        K_obsV = self.get_K_obsV(inputs) # N x C x L 
        mu_u = self.get_mu_u() # M x C x L 
        return (K_obsV[...,:, None, :, None, :]@K_VV_inv[None, None, :, :, :]@mu_u[None, :, :, :, None])[...,0,0]
    
    def get_Sigma_s_cov_C(self, inputs):
        # want to get to shape NxMxCxC
        K_obsobs_diag_elements = torch.diag_embed(torch.ones(inputs.shape[0], self.M, self.C)*self.sigma_s2[None, None, :]) # P x N x M x C x C 

        K_obsV = self.get_K_obsV_matrix(inputs) # N x CL x CL - yes this is messy but I think it works 
        A = self._get_conditional_cov_correction() # M x CL x CL

        Sigma_s = K_obsobs_diag_elements[..., None,:,:,:] - (K_obsV[...,:,None,:,:] @ A[...,None,:,:,:]@ K_obsV.transpose(-2,-1)[...,:, None,:,:])
        return Sigma_s + 1e-3*torch.eye(self.C)
    

    def _get_conditional_cov_correction(self):
        Sigma_u = self.get_matrix_Sigma_u() # M x CL  x CL
        K_VV_inv = self.get_K_VV_inv_matrix() # CL x CL
        return (K_VV_inv
                 -  K_VV_inv @ Sigma_u @ K_VV_inv
                ) 
    


class MeanField(InducingPointVariationalDist):
    def __init__(self, C, kernel_matrix):
        super(FullRankCovariance, self).__init__()

    

    
