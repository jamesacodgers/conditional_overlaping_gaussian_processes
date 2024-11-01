import torch 
from src.utils import generate_regular_simplex
from src.inner_optimisers import MultivariateNormalSolver
from src.gps import MultiSparseGPs

from abc import ABC, abstractmethod

class MFModels(torch.nn.Module, ABC):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, sum_to_one = False):
        super(MFModels, self).__init__()
        self.M = M 
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.C = self.C_obs + self.C_unobs
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        self.inner_train_solver = MultivariateNormalSolver(self.C_unobs)
        if sum_to_one:
            self.linear_map = torch.vstack([torch.zeros(self.C-1), torch.eye(self.C-1)])
            self.linear_map[:self.C_obs, :self.C_obs-1] = generate_regular_simplex(self.C_obs-1)
            self.linear_map[:-self.C_unobs, ]
            self.omega = torch.zeros(self.C)
            self.omega[:self.C_obs] = torch.ones(self.C_obs)/self.C_obs
            self.inner_test_solver = MultivariateNormalSolver(self.C -1)
        else:
            self.linear_map = None
            self.inner_test_solver = MultivariateNormalSolver(self.C)

    @property 
    def sigma2(self): 
        return torch.exp(self._log_sigma2)+1e-6
    
    def get_A_test(self, s_train_outer_expectation):
        if self.linear_map is not None:
            return self.linear_map.T@(1/(self.sigma2)*(s_train_outer_expectation))@self.linear_map # (C-1) x (C-1) 
        else:
            return 1/(self.sigma2)*(s_train_outer_expectation) # C x C 


    def get_b_test(self, D_test, s_expectation: torch.Tensor):
        if self.linear_map is not None: 
            return 1/(self.sigma2)*((D_test - self.omega[None, :]@s_expectation.T)[:, :, None, None]*s_expectation[None,:,:,None]*self.linear_map[None,None,:,:]).sum(dim=[1,2]) # N x (C-1)
        else:
            return 1/(self.sigma2)*D_test@s_expectation # N x C 
                
    def get_const_test(self, D_test, s_expectation: torch.Tensor, s_outer_expectation: torch.Tensor):
        if self.linear_map is not None:
            return -(self.M * D_test.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_test**2).sum() - 2 * (D_test[:, :, None] * s_expectation[None, :, :] *self.omega[None,None,:]).sum()  + D_test.shape[0]*(self.omega[None,:]*s_outer_expectation*self.omega[:, None]).sum())
        else: 
            return -(self.M * D_test.shape[0] )/2*torch.log(self.sigma2)- 1/(2*self.sigma2)*(D_test**2).sum()
        
    def get_A_train(self, s_unobs_outer_expectation):
        return 1/(self.sigma2)*(s_unobs_outer_expectation)
    
    def get_b_train(self, D_train, R_train, s_unobs_expectation, s_outer_cross_expectation):
        return 1/(self.sigma2)*(D_train@s_unobs_expectation - R_train@s_outer_cross_expectation)
                
    def get_const_train(self, D_train, R_train, s_obs_expectation, s_obs_outer_expectation):
        return -(self.M * D_train.shape[0])/2*torch.log(self.sigma2) - 1/(2*self.sigma2)*((D_train**2).sum() - 2 * torch.einsum("nm, mc, cn", D_train, s_obs_expectation, R_train.T) + torch.sum(R_train[:,None,:]@s_obs_outer_expectation@R_train[:,:,None]))
                                                                                        #    torch.einsum("nc, ncd, dn",R_train,s_obs_outer_expectation,R_train.T))
    
    def _elbo(self, s_expectation, s_outer_expectation, D_train, R_train, D_test=None):
        # s_expectation = self.get_s_expectation()
        s_obs_expectation = s_expectation[:,:self.C_obs]
        s_unobs_expectation = s_expectation[:,self.C_obs:]
        # s_outer_expectation = self.get_s_outer_expectation()
        s_obs_outer_expectation = s_outer_expectation[:, :self.C_obs,:self.C_obs]
        s_cross_outer_expectation = s_outer_expectation[:, :self.C_obs,self.C_obs:]
        s_unobs_outer_expectation = s_outer_expectation[:, self.C_obs:,self.C_obs:]
        A_train = self.get_A_train(s_unobs_outer_expectation) # C_unobs x C_unobs
        b_train = self.get_b_train(D_train, R_train, s_unobs_expectation, s_cross_outer_expectation) # N x C_unobs
        const_train = self.get_const_train(D_train, R_train, s_obs_expectation, s_obs_outer_expectation)
        training_inner_elbo = self.inner_train_solver(A_train, b_train) + const_train
        kl = self.get_kl()
        if D_test is None: 
            return training_inner_elbo - kl
        else: 
            A_test = self.get_A_test(s_outer_expectation)
            b_test = self.get_b_test(D_test, s_expectation)
            const_test = self.get_const_test(D_test, s_expectation, s_outer_expectation)
            test_inner_elbo = self.inner_test_solver(A_test, b_test) + const_test
            return training_inner_elbo + test_inner_elbo - kl

    def elbo(self, R_train, D_train, D_test=None):
        s_expectation = self.get_s_expectation()
        s_outer_expectation = self.get_s_outer_expectation()
        return super(MFStaticSignals, self)._elbo(s_expectation, s_outer_expectation, R_train, D_train,D_test)
    
    def predict_weights(self, D_test):
        s_expectation = self.get_s_expectation()
        s_outer_expectation = self.get_s_outer_expectation()
        A_test = self.get_A_test(s_outer_expectation)
        b_test = self.get_b_test(D_test, s_expectation)
        if self.linear_map is not None:
            mu_rho, Sigma_rho = self.inner_test_solver.get_weight_params(A_test, b_test)
            mu_r = mu_rho @ self.linear_map.T + self.omega
            Sigma_r = self.linear_map@Sigma_rho@self.linear_map.T + 1e-8
        else:
            mu_r, Sigma_r = self.inner_test_solver.get_weight_params(A_test, b_test)
        return torch.distributions.MultivariateNormal(mu_r,Sigma_r, validate_args=False)
    
    @abstractmethod
    def get_kl(self):
        ...

    @abstractmethod
    def get_s_expectation(self):
        ...

    @abstractmethod
    def get_s_outer_expectation(self):
        ...

class MFStaticSignals(MFModels):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, sum_to_one = False):
        super(MFStaticSignals, self).__init__(M, C_obs, C_unobs, sigma2, sum_to_one)
        # self.M = M 
        # self.C_obs = C_obs
        # self.C_unobs = C_unobs
        # self.C = self.C_obs + self.C_unobs
        # self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        self.mu_S_T = torch.nn.Parameter(torch.randn(M, self.C))
        self._raw_chol_S_T = torch.nn.Parameter(torch.eye(self.C))
        # self.inner_train_solver = MultivariateNormalSolver(self.C_unobs)
        # if sum_to_one:
        #     self.linear_map = torch.vstack([torch.zeros(self.C-1), torch.eye(self.C-1)])
        #     self.linear_map[:self.C_obs, :self.C_obs-1] = generate_regular_simplex(self.C_obs-1)
        #     self.linear_map[:-self.C_unobs, ]
        #     self.omega = torch.zeros(self.C)
        #     self.omega[:self.C_obs] = torch.ones(self.C_obs)/self.C_obs
        #     self.inner_test_solver = MultivariateNormalSolver(self.C -1)
        # else:
        #     self.linear_map = None
        #     self.inner_test_solver = MultivariateNormalSolver(self.C)
        self._log_priors_T = torch.nn.Parameter(torch.zeros(self.C_unobs))


    def forward(self):
        ...


        
    def get_s_expectation(self):
        return self.mu_S_T
    
    def get_s_outer_expectation(self):
        return (self.mu_S_T.T@ self.mu_S_T + self.M*self.Sigma_S_T)[None,...]
        
    # def get_unobs_train_weights(self, D_train, R_train):
    #     b_train = 1/(self.sigma2)*(D_train - R_train@self.mu_S.T)@self.mu_T # N x C 
    #     A_train = torch.linalg.inv(1/(self.sigma2)*(self.mu_T.T@ self.mu_T + self.M*self.Sigma_T)) # C x C 
    #     return self.inner_train_solver.get_weight_params(A_train, b_train)
    
    
    def get_kl(self):
        variational_dist = torch.distributions.MultivariateNormal(self.mu_S_T, self.Sigma_S_T)
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), self.prior_Sigma)
        return torch.distributions.kl_divergence(variational_dist, prior).sum()

    @property 
    def mu_S(self): 
        return self.mu_S_T[:,:self.C_obs]
    @property
    def mu_T(self): 
        return self.mu_S_T[:,self.C_obs:]
    @property 
    def Sigma_S_T(self): 
        chol = torch.tril(self._raw_chol_S_T)
        return chol.T @ chol
    @property 
    def Sigma_S(self): 
        return self.Sigma_S_T[:self.C_obs, :self.C_obs]
    @property 
    def Sigma_T(self): 
        return  self.Sigma_S_T[self.C_obs:, self.C_obs:]
    @property
    def Sigma_cross(self):
        return self.Sigma_S_T[self.C_obs:, :self.C_obs]
    # @property 
    # def sigma2(self): 
    #     return torch.exp(self._log_sigma2)+1e-6
    @property 
    def prior_Sigma(self): 
        return  torch.diag(torch.concat((torch.ones(self.C_obs), torch.exp(self._log_priors_T)+1e-6)))
    
class MFVariableSignalsObservedInputs(MFModels, MultiSparseGPs):
    def __init__(self, V, input_dims, M, C_obs, C_unobs=0, sigma2=1, sum_to_one = False, beta=None,sigma2_s=None):
        MFModels.__init__(self,M, C_obs, C_unobs, sigma2, sum_to_one)
        MultiSparseGPs.__init__(self, self.C, M, input_dims, V, beta = None, sigma2_s = None, trainable_V = True)

    def get_kl(self):
        return self.get_kl_u()
    
    def get_gp_expectation(self, inputs):
        return MultiSparseGPs.get_gp_expectation(self, inputs)
    
    def get_s_outer_expectation(self,inputs):
        mu_s = MultiSparseGPs.get_gp_expectation(self, inputs)
        Sigma_s = MultiSparseGPs.get_gp_covariance(self, inputs)
        return torch.sum(mu_s[..., None]*mu_s[...,None,:] + Sigma_s, dim=1)