import torch
from abc import ABC,abstractmethod

class InducingPointVariationalDist(torch.nn.Module,ABC):
    def __init__(self, C, M, L):
        super(InducingPointVariationalDist, self).__init__()
        self._mu_phi = torch.nn.Parameter(torch.zeros(M, C, L))
        self.C = C
        self.M = M
        self.L = L
    
    def get_mu_phi(self):
        """
        Returns shape C x L 
        """
        return self._mu_u.reshape(self.M, self.C, self.L)
    
    def get_vector_mu_phi(self):
        return self._mu_u
    
    def get_q_phi_dist(self):
        mu_u = self.get_vector_mu_u()
        Sigma_u = self.get_matrix_Sigma_phi() + 1e-5*torch.eye(self.C*self.L)
        return torch.distributions.MultivariateNormal(mu_u, Sigma_u)
    
    def get_mu_s(self, K_VV_inv, K_obsV):
        """
        Returns N x M x C 
        """
        mu_u = self.get_mu_u() # M x C x L 
        return (K_obsV[...,:, None, :, None, :]@K_VV_inv[None, None, :, :, :]@mu_u[None, :, :, :, None])[...,0,0]
    
    def get_KL_phi(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_Sigma_s(self, inputs):
        ...

    @abstractmethod
    def get_matrix_Sigma_phi(self):
        ...
    
    @abstractmethod
    def get_S_samples(self):
        ...
    
class FullRankCovariance(InducingPointVariationalDist):
    def __init__(self, C, M, L, kernel_matrix):
        super(FullRankCovariance, self).__init__(C, M, L)
        chol = torch.linalg.cholesky(kernel_matrix)
        new_tensor = torch.zeros(self.C, self.C, self.L, self.L)
        new_tensor[range(self.C), range(self.C), :, :] = chol[range(self.C), :, :]
        new_tensor = new_tensor.transpose(1,2).reshape(self.L*self.C, self.L*self.C)
        self._raw_chol = torch.nn.Parameter(new_tensor[None,:,:].repeat(self.M,1,1))



    def get_Sigma_phi(self):
        """
        Returns shape C x C x L x L
        """
        super(FullRankCovariance, self).__init__()
        Sigma_u = self.get_matrix_Sigma_u()
        Sigma_u = Sigma_u.reshape(self.M, self.C, self.L, self.C, self.L)
        return Sigma_u.permute(0,1,3,2,4)
    
    def get_matrix_Sigma_phi(self):
        chol = self.get_chol_Sigma_u()
        return chol @ chol.transpose(-2,-1) 
    
    def get_chol_Sigma_phi(self):
        return torch.tril(self._raw_chol) 
    
    def get_Sigma_s(self, K_obsobs, K_obsV, K_VV_inv):
        # want to get to shape (Px)NxMxCxC
        K_obsV = self.get_K_obsV_matrix(K_obsV=K_obsV)
        K_VV_inv_matrix  = torch.zeros(self.C, self.C, self.L, self.L)
        K_VV_inv_matrix[range(self.C), range(self.C), :, :] = K_VV_inv[range(self.C),:,:]
        K_VV_inv_matrix = K_VV_inv_matrix.transpose(1,2).reshape(self.C*self.L, self.C*self.L)
        K_obsobs = torch.diag_embed(K_obsobs)
        A = self._get_conditional_cov_correction(K_VV_inv=K_VV_inv_matrix) # M x CL x CL
        Sigma_s = K_obsobs - (K_obsV[...,:,None,:,:] @ A[...,None,:,:,:]@ K_obsV.transpose(-2,-1)[...,:, None,:,:])
        return Sigma_s #+ 2.5e-3*torch.eye(self.C)
    
    def get_K_obsV_matrix(self, K_obsV):
        K_obsV_matrix = torch.zeros(*K_obsV.shape[:-2],self.C, self.C, self.L)
        K_obsV_matrix[..., range(self.C), range(self.C), :] = K_obsV[..., range(self.C),:]
        return K_obsV_matrix.reshape(*K_obsV.shape[:-2], self.C, self.L*self.C)
    
    def _get_conditional_cov_correction(self, K_VV_inv):
        Sigma_u = self.get_matrix_Sigma_u() # M x CL  x CL
        return (K_VV_inv
                 -  K_VV_inv @ Sigma_u @ K_VV_inv
                ) 
    
    def get_S_samples(self, K_obsobs, K_obsV, K_VV_inv, num_samples):
        mu_s = self.get_mu_s(K_VV_inv=K_VV_inv, K_obsV=K_obsV)
        Sigma_s = self.get_Sigma_s(K_obsobs=K_obsobs, K_obsV=K_obsV, K_VV_inv=K_VV_inv)
        dist = torch.distributions.MultivariateNormal(mu_s, Sigma_s)
        return dist.sample((num_samples,))

class MeanField(InducingPointVariationalDist):
    def __init__(self, C, M, L, kernel_matrix):
        super(MeanField, self).__init__(C, M, L)
        self._log_sigma_u = torch.nn.Parameter(torch.zeros(self.M,self.C, self.L))
    
    @property 
    def Sigma_u(self): 
        return torch.exp(self._log_sigma_u)
    
    def get_S_samples(self, K_obsobs, K_obsV, K_VV_inv, num_samples):
        mu_s = self.get_mu_s(K_VV_inv=K_VV_inv, K_obsV=K_obsV)
        Sigma_s = self.get_Sigma_s(K_obsobs=K_obsobs, K_obsV=K_obsV, K_VV_inv=K_VV_inv)
        dist = torch.distributions.Normal(mu_s, Sigma_s)
        return dist.sample((num_samples,))
    
    def get_Sigma_s(self, K_obsobs, K_obsV, K_VV_inv):
        return torch.diag_embed(self.get_Sigma_s(K_obsobs=K_obsobs, K_obsV=K_obsV, K_VV_inv=K_VV_inv))
    
    def get_Sigma_s(self, K_obsobs, K_obsV, K_VV_inv):
        # want to get to shape (Px)NxMxCxC
        A = self._get_conditional_cov_correction(K_VV_inv=K_VV_inv)
        Sigma_s = K_obsobs - (K_obsV[...,None,:,None,:] @ A @ K_obsV[...,None,:,:, None])[...,0,0]
        return Sigma_s + 1e-3
    
    def _get_conditional_cov_correction(self, K_VV_inv):
        return (K_VV_inv
                 -  K_VV_inv @ torch.diag_embed(self.Sigma_u) @ K_VV_inv
                ) 
    
    def get_matrix_Sigma_phi(self):
        Sigma_u = self.Sigma_u.reshape(self.M, self.C*self.L)
        return torch.diag_embed(Sigma_u)

    

    
