import torch 
import matplotlib.pyplot as plt 
from scipy.stats.qmc import LatinHypercube


from src.mf_methods import MFVariableSignalsObservedInputs, MFVariableSignals

class StrucVariableSignalsObservedInputs(MFVariableSignalsObservedInputs):
    def __init__(self, P, V, M, C_obs, C_unobs=0, sigma2=1, sum_to_one = False, beta=None,sigma2_s=None):
        super(StrucVariableSignalsObservedInputs, self). __init__(V, M, C_obs, C_unobs=C_unobs, sigma2=1, sum_to_one = False, beta=None,sigma2_s=None)
        self.P = P
        self._samples = torch.distributions.Normal(0,1).icdf((torch.quasirandom.SobolEngine(self.L*self.C).draw(P)+1/P)*(1-2/P))
        self._mu_u = torch.nn.Parameter(torch.zeros_like(self._mu_u))

    def get_u_samples(self):
        # dist = self.get_q_u_dist()
        samples = self.get_u_vector_samples()
        return samples.reshape(self.P, self.M, self.C, self.L)
    
    def get_u_vector_samples(self):
        # dist = self.get_q_u_dist()
        chol = torch.tril(self._raw_chol)
        samples = self.get_vector_mu_u()[None,:,:] + (chol[None,:,::]@self._samples[:,None,:,None])[...,0]
        return samples


    def get_s_expectation(self, inputs):
        """
        Returns shape P x N x M x C
        """
        u_samples = self.get_u_samples() # P x M x C x L 
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        K_obsV = self.get_K_obsV(inputs) # N x C x L 
        
        # with torch.no_grad():
        #     plt.figure()
        #     for i in range(1):
        #         plt.scatter(self.V.flatten(),u_samples[i,0,0,:])
        #     plt.show()
        mu_s = (K_obsV[None, :, None, :, None, :]@K_VV_inv[None,None, None, :, :, :]@u_samples[:, None, :, :, :,None])[...,0,0]
        # with torch.no_grad():
        #     plt.figure()
        #     for i in range(1):
        #         plt.scatter(inputs[:,0], mu_s[0,:,i,0], c = "orange")
        #         plt.scatter(self.V[:,0], u_samples[0,i,0,:])
        #     plt.show()

        return mu_s
    
    

    def get_s_outer_expectation(self, mu_x, sigma_x):
        """
        retuns P x N x C x C 
        """
        u_samples = self.get_u_samples() # P x M x C
        K_VV = self.get_K_VV_matrix()
        K_VV_inv = self.get_K_VV_inv_matrix()
        mu_u = self.get_vector_mu_u() # M x LC
        Sigma_u = self.get_matrix_Sigma_u() # M x LC x LC (currently broacsts from 1 instead of M )
        Ma = (
            ( u_samples[:,:,None,:] - u_samples[:,:,:,None]
             - K_VV[None, None,:,:] ).sum(dim=1)
             )
        # Ma = (
        #     (mu_u[:,:,None]*mu_u[:,None,:]).sum(dim=0)
        #      - (K_VV[None,:,:]  - Sigma_u[:,:,:])*self.M
        #      )
        Ma = (K_VV_inv@Ma@ K_VV_inv.T)  # LC x LC 
        # Ma = (Ma + Ma.T)/2 # Trying to make Ma symetric 
        # Ma = Ma.reshape(self.C, self.L, self.C, self.L).transpose(1,2) # C x C x L x L
        Ma = Ma.reshape(self.C, self.L, self.C, self.L).permute(0,1,3,2,4) # C x C x L x L
        # Z = self._get_kernel_outer_expectation(mu_x,sigma_x) # N x C x C x L x L
        Z = self._get_kernel_outer_expectation(mu_x,sigma_x)#.transpose(1,2) # N x C x C x L x L
        K_xx_expectation = torch.diag(self.sigma2_s)
        A = K_xx_expectation*self.M + (Ma[None,:,:,:,:]*Z[:,:,:,:,:]).sum(dim=[-2,-1]) # + torch.eye(self.C)*1e-3
        return A # N x C x C 

    def _get_kernel_outer_expectation(self, mu_x, Sigma_x) -> torch.Tensor:
        """
        returns N x C x C x L x L
        """
        # samples = torch.distributions.Normal(mu_x,Sigma_x).rsample((16,))
        # K = self.kernel(samples, self.V).transpose(1,2)
        # K_outer = K[:,:,None,:,None,:]*K[:,:,:,None,:,None]
        # return K_outer.mean(dim=0)

        d = (
              self.beta[None,None, None,:,:]*Sigma_x[:,:,None,None,:] 
            + self.beta[None,None,:,None,:]*Sigma_x[:,:,None,None,:]
            + self.beta[None,None,:,None,:]*self.beta[None,None,None,:,:]
        ) # P x N x C x C x A 
        e = (
              self.beta[None,None,None,:,None,None,:]*Sigma_x[None,:,None,None,None,None,:]*self.V[None,None,None,None,:,None,:]
            + self.beta[None,None,:,None,None,None,:]*Sigma_x[None,:,None,None,None,None,:]*self.V[None,None,None,None,None,:,:]
            + self.beta[None,None,:,None,None,None,:]*self.beta[None,None,None,:,None,None,:]*mu_x[None,:,None,None,None,None,:]
        ) # P x N x C x C x L x L x A
        f = (
              self.beta[None,None,None,:,None,None,:]*Sigma_x[:,:,None,None,None,None,:]*(self.V[None,None,None,None,:,None,:]**2)
                + self.beta[None,None,:,None,None,None,:]*Sigma_x[:,:,None,None,None,None,:]*(self.V[None,None,None,None,None,:,:]**2)
                + self.beta[None,None,:,None,None,None,:]*self.beta[None,None,None,:,None,None,:]*(mu_x[:,:,None,None,None,None,:]**2)
        ) # N x C x C x L x L x A
        g_intermediate = 1/(self.beta[None,None,:,None,None,None,:]*self.beta[None,None,None,:,None,None,:]*Sigma_x[:,:,None,None,None,None,:])*(-e**2/d[:,:,:,:,None,None,:] + f)
        g = -1/2*g_intermediate.sum(dim=-1) # N x C x C x L x L
        Z = self.sigma2_s[None,None,:,None,None,None]*self.sigma2_s[None,None,None,:,None,None]*torch.sqrt((self.beta[None,None,:,None,None,None,:]*self.beta[None,None,None,:,None,None,:]/d[::,:,:,None,None,:]).prod(dim=-1))*torch.exp(g) 
        return Z # P x N x C x C x L x L 


    def elbo(self,D_train, R_train, D_test,x_train, x_test):
        assert R_train.shape[1] == self.C_obs
        s_train_expectation = self.get_s_expectation(x_train)

        s_train_outer_expectation = self.get_s_outer_expectation(x_train)
        if D_test is None:
            return self._elbo(s_train_expectation=s_train_expectation, s_train_outer_expectation=s_train_outer_expectation,s_test_expectation=None, s_test_outer_expectation=None, D_train=D_train, R_train=R_train, D_test=None )
        else:
            s_test_expectation = self.get_gp_expectation(x_test)
            s_test_outer_expectation = self.get_s_outer_expectation(x_test)
    
            return self._elbo(s_train_expectation=s_train_expectation, s_train_outer_expectation=s_train_outer_expectation,s_test_expectation=s_test_expectation, s_test_outer_expectation=s_test_outer_expectation, D_train=D_train, R_train=R_train, D_test=D_test )
    
    
class StrucVariableSignals(MFVariableSignals):
    def __init__(self,P, V, M, C_obs, mu_x_train, mu_x_test, C_unobs=0, sigma2=1, sum_to_one = False, beta=None,sigma2_s=None, ):
        super(StrucVariableSignals,self).__init__(V, M, C_obs, mu_x_train, mu_x_test, C_unobs=C_unobs, sigma2=1, sum_to_one = False, beta=None,sigma2_s=None, )
        self.P = P
        samples = torch.distributions.Normal(0,1).icdf((torch.quasirandom.SobolEngine(self.M*self.C).draw(P)+1/P)*(1-2/P))

        self._mu_u = torch.nn.Parameter(torch.zeros_like(self._mu_u))

        self.mu_x_train = torch.nn.Parameter(self.mu_x_train[None,...].repeat(self.P,1,1))
        self.mu_x_test = torch.nn.Parameter(self.mu_x_test[None,...].repeat(self.P,1,1))
        self._log_sigma_x_train = torch.nn.Parameter(self.Sigma_x_train[None,...].repeat(self.P,1,1))
        self._log_sigma_x_test = torch.nn.Parameter(self.Sigma_x_test[None,...].repeat(self.P,1,1))

        self.P = P
        self._samples = torch.distributions.Normal(0,1).icdf((torch.quasirandom.SobolEngine(self.L*self.C, seed=1, scramble=True).draw(P*M).reshape(self.P, self.M, self.L*self.C)+1/P)*(1-2/P))

    def get_u_samples(self):
        # dist = self.get_q_u_dist()
        # samples = dist.sample((self.P,))
        samples = self.get_u_vec_samples()
        return samples.reshape(self.P, self.M, self.C, self.L)
    
    def get_u_vec_samples(self):
        # dist = self.get_q_u_dist()
        chol = torch.tril(self._raw_chol)
        samples = self.get_vector_mu_u()[None,:,:] + (chol[None,:,::]@self._samples[:,:,:,None])[...,0]
        # chol = torch.tril(self._raw_chol)
        # samples = self.get_vector_mu_u()[None,:,:] + (chol[None,:,::]@self._samples[:,None,:,None])[...,0]
        return samples
    
    def get_kl_x(self):
        x_prior = torch.distributions.Normal(torch.zeros(self.A), torch.ones(self.A))
        x_train = torch.distributions.Normal(self.mu_x_train, self.Sigma_x_train)
        x_test = torch.distributions.Normal(self.mu_x_test, self.Sigma_x_test)
        kl_train = torch.distributions.kl_divergence(x_train, x_prior).mean(dim=0).sum()
        kl_test = torch.distributions.kl_divergence(x_test, x_prior).mean(dim=0).sum()
        return kl_train + kl_test
    
    def get_s_expectation(self, mu_x, sigma_x):
        """
        returns P x N x M x C 
        """
        # K_VV_inv = self.get_K_VV_inv() # C x L x L 
        # mu_u = self.get_mu_u() # M x C x L
        # kernel_expectation = self.get_kernel_expectation(mu_x,sigma_x)
        # return torch.einsum("nck,ckl,mcl->nmc", kernel_expectation, K_VV_inv, mu_u)
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        u_samples = self.get_u_samples() # P x M x C x L
        kernel_expectation = self._get_kernel_expectation(mu_x,sigma_x) # P x N x C x L 
        return (kernel_expectation[:,:,None,:,None,:]@K_VV_inv[None,None,None,:,:,:]@u_samples[:,None,:,:,:,None])[...,0,0] # P x N x M x C 
    
    def get_s_outer_expectation(self, mu_x, sigma_x):
        K_VV = self.get_K_VV_matrix()
        K_VV_inv = self.get_K_VV_inv_matrix()

        u_samples = self.get_u_vec_samples()


        Ma = (
            (u_samples[:,:,:,None]*u_samples[:,:,None,:]
             - K_VV[None,:,:]).sum(dim=1)
             )
        # Ma = (
        #     (mu_u[:,:,None]*mu_u[:,None,:]).sum(dim=0)
        #      - (K_VV[None,:,:]  - Sigma_u[:,:,:])*self.M
        #      )
        Ma = (K_VV_inv@Ma@ K_VV_inv.T)  # LC x LC 
        # Ma = (Ma + Ma.T)/2 # Trying to make Ma symetric 
        # Ma = Ma.reshape(self.C, self.L, self.C, self.L).transpose(1,2) # C x C x L x L
        Ma = Ma.reshape(self.P,self.C, self.L, self.C, self.L).permute(0,1,3,2,4) # C x C x L x L
        # Z = self._get_kernel_outer_expectation(mu_x,sigma_x) # N x C x C x L x L
        Z = self._get_kernel_outer_expectation(mu_x,sigma_x) # P, N x C x C x L x L
        K_xx_expectation = torch.diag(self.sigma2_s)
        A = K_xx_expectation*self.M + (Ma[:,None,:,:,:,:]*Z[:,:,:,:,:,:]).sum(dim=[-2,-1]) # + torch.eye(self.C)*1e-3
        return A # Px N x C x C 
    
    def _get_kernel_expectation(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor) -> torch.Tensor:
        """
        Returns object of shape P x N x C x L
        """
        d = (
              Sigma_x[:,:,None,:] 
            + self.beta[None,None,:,:]
        ) # P x N x C x A 
        e = (
              Sigma_x[:,:,None,None,:]*self.V[None,None,None,:,:]
            + self.beta[None,None,:,None,:]*mu_x[:,:,None, None,:]
        ) # P x N x C x L x A
        f = (
              Sigma_x[:,:,None,None,:]*(self.V[None,None,None,:,:]**2)
            + self.beta[None,None,:,None,:]*(mu_x[:,:,None,None,:]**2)
        ) # P x N x C x L x A
        g_intermediate = 1/(self.beta[None,None,:,None,:]*Sigma_x[:,:,None,None,:])*(-e**2/d[:,:,:,None,:] + f)
        g = -1/2*g_intermediate.sum(dim=-1) # P x N x C x L 
        Z = self.sigma2_s[None, None,:,None]*torch.sqrt((self.beta[None,None,:,None,:]/d[:,:,:,None,:]).prod(dim=-1))*torch.exp(g) 
        return Z # P x N x C x L 
    
    def _get_kernel_outer_expectation(self, mu_x, Sigma_x) -> torch.Tensor:
        """
        returns P x N x C x C x L x L
        """
        # samples = torch.distributions.Normal(mu_x,Sigma_x).rsample((16,))
        # K = self.kernel(samples, self.V).transpose(1,2)
        # K_outer = K[:,:,None,:,None,:]*K[:,:,:,None,:,None]
        # return K_outer.mean(dim=0)

        d = (
              self.beta[None,None,None,:,:]*Sigma_x[:, :,None,None,:] 
            + self.beta[None,None,:,None,:]*Sigma_x[:,:,None,None,:]
            + self.beta[None,None,:,None,:]*self.beta[None,None,None,:,:]
        ) # P x N x C x C x A 
        e = (
              self.beta[None,None,None,:,None,None,:]*Sigma_x[:,:,None,None,None,None,:]*self.V[None,None,None,None,:,None,:]
            + self.beta[None,None,:,None,None,None,:]*Sigma_x[:,:,None,None,None,None,:]*self.V[None,None,None,None,None,:,:]
            + self.beta[None,None,:,None,None,None,:]*self.beta[None,None,None,:,None,None,:]*mu_x[:,:,None,None,None,None,:]
        ) # P x N x C x C x L x L x A
        f = (
              self.beta[None,None,None,:,None,None,:]*Sigma_x[:,:,None,None,None,None,:]*(self.V[None,None,None,None,:,None,:]**2)
                + self.beta[None,None,:,None,None,None,:]*Sigma_x[:,:,None,None,None,None,:]*(self.V[None,None,None,None,None,:,:]**2)
                + self.beta[None,None,:,None,None,None,:]*self.beta[None,None,None,:,None,None,:]*(mu_x[:,:,None,None,None,None,:]**2)
        ) # P x N x C x C x L x L x A
        g_intermediate = 1/(self.beta[None,None,:,None,None,None,:]*self.beta[None,None,None,:,None,None,:]*Sigma_x[:,:,None,None,None,None,:])*(-e**2/d[:,:,:,:,None,None,:] + f)
        g = -1/2*g_intermediate.sum(dim=-1) # P x N x C x C x L x L
        Z = self.sigma2_s[None,None,:,None,None,None]*self.sigma2_s[None,None,None,:,None,None]*torch.sqrt((self.beta[None,None,:,None,None,None,:]*self.beta[None,None,None,:,None,None,:]/d[:,:,:,:,None,None,:]).prod(dim=-1))*torch.exp(g) 
        return Z # P x N x C x C x L x L 
    

