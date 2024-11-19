import torch 
from src.utils import generate_regular_simplex
from src.inner_optimisers import MultivariateNormalSolver
from src.gps import WLIndependentGPs

from abc import ABC, abstractmethod

class MFModels(torch.nn.Module, ABC):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, weight_distribution="continuos"):
        torch.nn.Module.__init__(self)
        weight_distribution_options = {"continuos","sum_to_one","categorical"}
        assert weight_distribution in weight_distribution_options
        self.M = M 
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.C = self.C_obs + self.C_unobs
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        if weight_distribution=="continuos":
            self.coordinator = NormalCoordinator(module=self,sum_to_one=False)
        elif weight_distribution=="sum_to_one":
            self.coordinator = NormalCoordinator(module=self,sum_to_one=True)
        elif weight_distribution=="categorical":
            assert self.C_unobs==0
            self.coordinator = ClassificationCoordinator(module=self)
        else:
            raise AssertionError("You should not be here")

    @property 
    def sigma2(self): 
        return torch.exp(self._log_sigma2)+1e-6
    
    def _elbo(self, s_train_expectation, s_train_outer_expectation, D_train, R_train,  s_test_expectation=None, s_test_outer_expectation=None,D_test=None):
        return self.coordinator._elbo(
                            s_train_expectation=s_train_expectation, 
                            s_train_outer_expectation=s_train_outer_expectation, 
                            D_train=D_train, 
                            R_train=R_train,  
                            s_test_expectation=s_test_expectation, 
                            s_test_outer_expectation=s_test_outer_expectation,
                            D_test=D_test
                        )
    
    def predict_weights(self, D_test):
        return self.coordinator.predict_weights(D_test)

    @abstractmethod
    def get_kl(self):
        ...

    @abstractmethod
    def get_s_expectation(self):
        ...

    @abstractmethod
    def get_s_outer_expectation(self):
        ...

class ClassificationCoordinator:
    def __init__(self, module:MFModels):
        self.module = module 

    def _elbo(self, s_train_expectation, s_train_outer_expectation, D_train, R_train,  s_test_expectation=None, s_test_outer_expectation=None,D_test=None):
        assert R_train.shape == torch.Size([D_train.shape[0]])

        s_outer_diag_expextation = s_train_outer_expectation[...,range(self.module.C), range(self.module.C)]
        if len(s_train_expectation.shape)==3:
            s_train_expectation_c = s_train_expectation.gather(dim=-1, index=R_train[:,None,None].expand(-1,self.module.M,-1)).squeeze(-1)
            s_train_squared_expectation_c = s_outer_diag_expextation.gather(dim=-1, index=R_train[:,None]).squeeze(-1)
        elif len(s_train_expectation.shape)==4:
            s_train_expectation_c = s_train_expectation.gather(dim=-1, index=R_train[None,:,None,None].expand(s_train_expectation.shape[0],-1,self.module.M,-1)).squeeze(-1)
            s_train_squared_expectation_c = s_outer_diag_expextation.gather(dim=-1, index=R_train[None,:,None].expand(s_train_expectation.shape[0],-1,-1)).squeeze(-1)
        else:
            raise AssertionError("I should not be here")
        if D_test == None:
            elbo = -1/(2*self.module.sigma2)*((D_train**2).sum() - (2*D_train[...,:,:]*s_train_expectation_c).sum(dim=[-1,-2]).mean() + s_train_squared_expectation_c.sum(dim=[-1]).mean())- self.module.M*D_train.shape[0]/2*torch.log(2*torch.pi*self.module.sigma2)
            return elbo
        else:
            ...

    def predict_weights(self, D_test, s_test_expectation, s_test_outer_expectation):
        s_test_squared_expectation = torch.diag(s_test_outer_expectation)
        -1/(2*self.module.sigma2)*((D_test**2).sum(dim=-2) - 2*D_test[:,:,None]*s_test_expectation + s_test_squared_expectation)

class NormalCoordinator:
    def __init__(self,module:MFModels,sum_to_one:bool):
        self.module=module
        self.inner_train_solver = MultivariateNormalSolver(self.module.C_unobs)
        if sum_to_one:
            self.linear_map = torch.vstack([torch.zeros(self.module.C-1), torch.eye(self.module.C-1)])
            self.linear_map[:self.module.C_obs, :self.module.C_obs-1] = generate_regular_simplex(self.module.C_obs-1)
            self.h = torch.zeros(self.module.C)
            self.h[:self.module.C_obs] = torch.ones(self.module.C_obs)/self.module.C_obs
            self.inner_test_solver = MultivariateNormalSolver(self.module.C -1)
        else:
            self.linear_map = None
            self.inner_test_solver = MultivariateNormalSolver(self.module.C)

    def get_A_test(self, s_outer_expectation):
        if self.linear_map is not None:
            return self.linear_map.T@(1/(self.module.sigma2)*(s_outer_expectation))@self.linear_map # [N] x (C-1) x (C-1) 
        else:
            return 1/(self.module.sigma2)*(s_outer_expectation) # [N] x C x C 
        
    def get_b_test(self, D_test, s_expectation: torch.Tensor):
        if self.linear_map is not None: 
            return 1/(self.module.sigma2)*((D_test - (self.h[..., :]*s_expectation).sum(dim=-1))[..., None, None]*s_expectation[...,None]*self.linear_map[...,None,None,:,:]).sum(dim=[-2,-3]) # N x (C-1)
        else:
            return 1/(self.module.sigma2)*(D_test[:,:,None]*s_expectation).sum(dim=-2) # N x C 

                
    def get_const_test(self, D_test, s_expectation: torch.Tensor, s_outer_expectation: torch.Tensor):
        if self.linear_map is not None:
            # If something breaks for structured VI, look here
            return -(D_test.shape[1] * D_test.shape[0])/2*torch.log(self.module.sigma2) - 1/(2*self.module.sigma2)*((D_test**2).sum() - 2 * (D_test[..., None] * s_expectation *self.h[...,None,None,:]).sum(dim=[-3,-2,-1]).mean()  + (self.h[...,None, None, :]*s_outer_expectation*self.h[...,None, :, None]).sum(dim=[-3,-2,-1]).mean())
        else: 
            const_test= -(D_test.shape[1] * D_test.shape[0] )/2*torch.log(self.module.sigma2)- 1/(2*self.module.sigma2)*(D_test**2).sum()
            if len(s_expectation.shape)==3:
                return const_test
            if len(s_expectation.shape)==4:
                return const_test[None,...]
            else:
                raise AssertionError("I should never be here")

    def get_A_train(self, s_unobs_outer_expectation):
        return 1/(self.module.sigma2)*(s_unobs_outer_expectation)
    
    def get_b_train(self, D_train, R_train, s_unobs_expectation, s_outer_cross_expectation):
        # return 1/(self.parent.sigma2)*((D_train[:,:,None]*s_unobs_expectation).sum(dim=1) - (R_train[:,:,None]*s_outer_cross_expectation).sum(dim=1))
        return 1/(self.module.sigma2)*((D_train[:,:,None]*s_unobs_expectation).sum(dim=-2) - (R_train[:,:,None]*s_outer_cross_expectation).sum(dim=-2))
                
    def get_const_train(self, D_train, R_train, s_obs_expectation, s_obs_outer_expectation):
        return -(D_train.shape[1] * D_train.shape[0])/2*torch.log(self.module.sigma2) - 1/(2*self.module.sigma2)*((D_train**2).sum(dim=[-1,-2]) - 2 * torch.sum(D_train[...,None,:] @ s_obs_expectation @ R_train[...,None], dim=[-1,-2,-3]) + torch.sum(R_train[...,None,:]@s_obs_outer_expectation@R_train[...,None],dim=[-1,-2,-3]))
               
               # torch.einsum("nm, mc, cn", D_train, s_obs_expectation, R_train.T)                                                                          #    torch.einsum("nc, ncd, dn",R_train,s_obs_outer_expectation,R_train.T))
    
    def _elbo(self, s_train_expectation, s_train_outer_expectation, D_train, R_train,  s_test_expectation=None, s_test_outer_expectation=None,D_test=None):
        assert R_train.shape[1] == self.module.C_obs
        s_train_obs_expectation = s_train_expectation[...,:self.module.C_obs]
        s_train_unobs_expectation = s_train_expectation[...,self.module.C_obs:]

        s_train_obs_outer_expectation = s_train_outer_expectation[..., :self.module.C_obs,:self.module.C_obs]
        s_train_cross_outer_expectation = s_train_outer_expectation[..., :self.module.C_obs,self.module.C_obs:]
        s_train_unobs_outer_expectation = s_train_outer_expectation[..., self.module.C_obs:,self.module.C_obs:]

        

        A_train = self.get_A_train(s_train_unobs_outer_expectation) # C_unobs x C_unobs
        b_train = self.get_b_train(D_train, R_train, s_train_unobs_expectation, s_train_cross_outer_expectation) # N x C_unobs
        const_train = self.get_const_train(D_train, R_train, s_train_obs_expectation, s_train_obs_outer_expectation)
        training_inner_elbo = self.inner_train_solver(A_train, b_train).mean() + const_train.mean()
        kl = self.module.get_kl()
        if D_test is None: 
            return training_inner_elbo.mean() - kl
        else: 
            assert s_test_expectation is not None
            assert s_test_outer_expectation is not None
            A_test = self.get_A_test(s_test_outer_expectation)
            b_test = self.get_b_test(D_test, s_test_expectation)
            const_test = self.get_const_test(D_test, s_test_expectation, s_test_outer_expectation)
            test_inner_elbo = self.inner_test_solver(A_test, b_test).mean() + const_test
            return (training_inner_elbo
                    + test_inner_elbo
                    - kl)
    
    def predict_weights(self, D_test):
        s_expectation = self.module.get_s_expectation(D_test.shape[0])
        s_outer_expectation = self.module.get_s_outer_expectation(D_test.shape[0])
        A_test = self.get_A_test(s_outer_expectation)
        b_test = self.get_b_test(D_test, s_expectation)
        if self.linear_map is not None:
            mu_rho, Sigma_rho = self.inner_test_solver.get_weight_params(A_test, b_test)
            mu_r = mu_rho @ self.linear_map.T + self.h
            Sigma_r = self.linear_map@Sigma_rho@self.linear_map.T + 1e-8
        else:
            mu_r, Sigma_r = self.inner_test_solver.get_weight_params(A_test, b_test)
        return torch.distributions.MultivariateNormal(mu_r,Sigma_r, validate_args=False)




class MFStaticSignals(MFModels):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, weight_distribution = "sum_to_one"):
        super(MFStaticSignals, self).__init__(M=M, C_obs=C_obs, C_unobs=C_unobs, sigma2=sigma2, weight_distribution=weight_distribution)
   
        self.mu_S = torch.nn.Parameter(torch.randn(M, self.C))
        self._raw_chol_S = torch.nn.Parameter(torch.eye(self.C))
 
        self._log_priors_T = torch.nn.Parameter(torch.zeros(self.C_unobs))


    def forward(self):
        ...

    def elbo(self, R_train, D_train, D_test=None):
        s_train_expectation = self.get_s_expectation(D_train.shape[0])
        s_train_outer_expectation = self.get_s_outer_expectation(D_train.shape[0])
        if D_test is not None:
            s_test_expectation = self.get_s_expectation(D_test.shape[0])
            s_test_outer_expectation = self.get_s_outer_expectation(D_test.shape[0])
            return self._elbo(s_train_expectation=s_train_expectation, s_train_outer_expectation=s_train_outer_expectation, s_test_expectation=s_test_expectation, s_test_outer_expectation=s_test_outer_expectation, R_train=R_train, D_train=D_train,D_test=D_test)
        else:
            return self._elbo(s_train_expectation=s_train_expectation, s_train_outer_expectation=s_train_outer_expectation,   R_train=R_train, D_train=D_train)
        
    def get_s_expectation(self,N):
        return self.mu_S[None,:,:].repeat(N,1,1)
    
    def get_s_outer_expectation(self,N):
        return (self.mu_S.T@ self.mu_S + self.M*self.Sigma_S)[None,...].repeat(N,1,1)
    

        
    # def get_unobs_train_weights(self, D_train, R_train):
    #     b_train = 1/(self.sigma2)*(D_train - R_train@self.mu_S.T)@self.mu_T # N x C 
    #     A_train = torch.linalg.inv(1/(self.sigma2)*(self.mu_T.T@ self.mu_T + self.M*self.Sigma_T)) # C x C 
    #     return self.inner_train_solver.get_weight_params(A_train, b_train)
    
    
    def get_kl(self):
        variational_dist = torch.distributions.MultivariateNormal(self.mu_S, self.Sigma_S)
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), self.prior_Sigma_S)
        return torch.distributions.kl_divergence(variational_dist, prior).sum()

    # @property 
    # def mu_S(self): 
    #     return self.mu_S_T[:,:self.C_obs]
    # @property
    # def mu_T(self): 
    #     return self.mu_S_T[:,self.C_obs:]
    @property 
    def Sigma_S(self): 
        chol = torch.tril(self._raw_chol_S)
        return chol.T @ chol
    # @property 
    # def Sigma_S(self): 
    #     return self.Sigma_S_T[:self.C_obs, :self.C_obs]
    # @property 
    # def Sigma_T(self): 
    #     return  self.Sigma_S_T[self.C_obs:, self.C_obs:]
    # @property
    # def Sigma_cross(self):
    #     return self.Sigma_S_T[self.C_obs:, :self.C_obs]
    @property 
    def prior_Sigma_S(self): 
        return  torch.diag(torch.concat((torch.ones(self.C_obs), torch.exp(self._log_priors_T)+1e-6)))
    
class MFVariableSignalsObservedInputs(MFModels, WLIndependentGPs):
    def __init__(self, V, M, C_obs, C_unobs=0, sigma2=1, sum_to_one = False, beta=None,sigma2_s=None):
        MFModels.__init__(self,M, C_obs, C_unobs, sigma2, sum_to_one)
        WLIndependentGPs.__init__(self, self.C, M, V, beta = beta, sigma2_s = None, trainable_V = True)

    def get_kl(self):
        return self.get_kl_u()

    
    def elbo(self,D_train, R_train, D_test,x_train, x_test):
        assert R_train.shape[1] == self.C_obs
        s_train_expectation = self.get_gp_expectation(x_train)

        s_train_outer_expectation = self.get_s_outer_expectation(x_train)
        if D_test is None:
            return self._elbo(s_train_expectation=s_train_expectation, s_train_outer_expectation=s_train_outer_expectation,s_test_expectation=None, s_test_outer_expectation=None, D_train=D_train, R_train=R_train, D_test=None )
        else:
            s_test_expectation = self.get_gp_expectation(x_test)
            s_test_outer_expectation = self.get_s_outer_expectation(x_test)
    
            return self._elbo(s_train_expectation=s_train_expectation, s_train_outer_expectation=s_train_outer_expectation,s_test_expectation=s_test_expectation, s_test_outer_expectation=s_test_outer_expectation, D_train=D_train, R_train=R_train, D_test=D_test )

    def get_s_expectation(self, inputs):
        return WLIndependentGPs.get_gp_expectation(self, inputs)
    
    def get_s_outer_expectation(self,inputs):
        mu_s = WLIndependentGPs.get_gp_expectation(self, inputs)
        Sigma_s = WLIndependentGPs.get_gp_covariance(self, inputs)
        return torch.sum(mu_s[..., None]*mu_s[...,None,:] + Sigma_s, dim=1)
    
    def predict_weights(self, D_test, x_test):
        s_expectation = self.get_s_expectation(x_test)
        s_outer_expectation = self.get_s_outer_expectation(x_test)
        A_test = self.coordinator.get_A_test(s_outer_expectation)
        b_test = self.coordinator.get_b_test(D_test, s_expectation)
        if self.coordinator.linear_map is not None:
            mu_rho, Sigma_rho = self.coordinator.inner_test_solver.get_weight_params(A_test, b_test)
            mu_r = mu_rho @ self.coordinator.linear_map.T + self.h
            Sigma_r = self.coordinator.linear_map@Sigma_rho@self.coordinator.linear_map.T + 1e-8
        else:
            mu_r, Sigma_r = self.coordinator.inner_test_solver.get_weight_params(A_test, b_test)
        return torch.distributions.MultivariateNormal(mu_r,Sigma_r, validate_args=False)
    
class MFVariableSignals(MFModels, WLIndependentGPs):
    def __init__(self, V, M, C_obs, mu_x_train, mu_x_test, C_unobs=0, sigma2=1, weight_distribution="continuos", beta=None,sigma2_s=None, ):
        MFModels.__init__(self,M=M, C_obs=C_obs, C_unobs=C_unobs, sigma2=sigma2, weight_distribution=weight_distribution)
        WLIndependentGPs.__init__(self, self.C, M, V, beta = None, sigma2_s = None, trainable_V = True)
        self.A = self.input_dims
        assert mu_x_train.shape[1] == self.A
        assert mu_x_test.shape[1] == self.A
        self.mu_x_train = torch.nn.Parameter(mu_x_train)
        self.mu_x_test = torch.nn.Parameter(mu_x_test)

        self._log_sigma_x_train = torch.nn.Parameter(torch.zeros_like(mu_x_train))
        self._log_sigma_x_test = torch.nn.Parameter(torch.zeros_like(mu_x_test))
        

    def get_kl(self):
        return self.get_kl_u() + self.get_kl_x()

    def get_kl_x(self):
        x_prior = torch.distributions.Normal(torch.zeros(self.A), torch.ones(self.A))
        x_train = torch.distributions.Normal(self.mu_x_train, self.Sigma_x_train)
        x_test = torch.distributions.Normal(self.mu_x_test, self.Sigma_x_test)
        kl_train = torch.distributions.kl_divergence(x_train, x_prior).sum()
        kl_test = torch.distributions.kl_divergence(x_test, x_prior).sum()
        return kl_train + kl_test
    
    def elbo(self,D_train, R_train, D_test=None):
        s_train_expectation = self.get_s_expectation(self.mu_x_train, self.Sigma_x_train)
        s_train_outer_expectation = self.get_s_outer_expectation(self.mu_x_train, self.Sigma_x_train)
        if D_test is None: 
            return self._elbo(s_train_expectation, s_train_outer_expectation, D_train, R_train )
        else:
            s_test_expectation = self.get_s_expectation(self.mu_x_test, self.Sigma_x_test)
            s_test_outer_expectation = self.get_s_outer_expectation(self.mu_x_test, self.Sigma_x_test)
            return self._elbo(
                    s_train_expectation, 
                    s_train_outer_expectation, 
                    D_train, 
                    R_train, 
                    D_test=D_test, 
                    s_test_expectation=s_test_expectation, 
                    s_test_outer_expectation=s_test_outer_expectation 
                )
    

    def get_s_expectation(self, mu_x, sigma_x):
        """
        returns P x N x M x C 
        """
        # K_VV_inv = self.get_K_VV_inv() # C x L x L 
        # mu_u = self.get_mu_u() # M x C x L
        # kernel_expectation = self.get_kernel_expectation(mu_x,sigma_x)
        # return torch.einsum("nck,ckl,mcl->nmc", kernel_expectation, K_VV_inv, mu_u)
        K_VV_inv = self.get_K_VV_inv() # C x L x L 
        mu_u = self.get_mu_u() # M x C x L
        kernel_expectation = self._get_kernel_expectation(mu_x,sigma_x) # N x C x L 
        return (kernel_expectation[:,None,:,None,:]@K_VV_inv[None,None,:,:,:]@mu_u[None,:,:,:,None])[...,0,0] # N x M x C 
    
    def get_s_outer_expectation(self, mu_x, sigma_x):
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
        # Ma = (
        #     (mu_u[:,:,None]*mu_u[:,None,:]).sum(dim=0)
        #      - (K_VV[None,:,:]  - Sigma_u[:,:,:])*self.M
        #      )
        Ma = (K_VV_inv@Ma@ K_VV_inv.T)  # LC x LC 
        # Ma = (Ma + Ma.T)/2 # Trying to make Ma symetric 
        # Ma = Ma.reshape(self.C, self.L, self.C, self.L).transpose(1,2) # C x C x L x L
        Ma = Ma.reshape(self.C, self.L, self.C, self.L).permute(0,2,1,3) # C x C x L x L
        # Z = self._get_kernel_outer_expectation(mu_x,sigma_x) # N x C x C x L x L
        Z = self._get_kernel_outer_expectation(mu_x,sigma_x)#.transpose(1,2) # N x C x C x L x L
        K_xx_expectation = torch.diag(self.sigma2_s)
        A = K_xx_expectation*self.M + (Ma[None,:,:,:,:]*Z[:,:,:,:,:]).sum(dim=[-2,-1]) # + torch.eye(self.C)*1e-3
        return A # N x C x C 
    
    def get_prediction_observed_inputs(self,inputs):
        return WLIndependentGPs.get_gp_expectation(self,inputs)

    def predict_weights(self, D_test):
        """
        Predicts the weights

        Needs to be redefined for this class 
        """
        s_expectation = self.get_s_expectation(self.mu_x_test, self.Sigma_x_test)
        s_outer_expectation = self.get_s_outer_expectation(self.mu_x_test, self.Sigma_x_test)
        A_test = self.coordinator.get_A_test(s_outer_expectation)
        b_test = self.coordinator.get_b_test(D_test, s_expectation)
        if self.coordinator.linear_map is not None:
            mu_rho, Sigma_rho = self.coordinator.inner_test_solver.get_weight_params(A_test, b_test)
            mu_r = mu_rho @ self.coordinator.linear_map.T + self.coordinator.h
            Sigma_r = self.coordinator.linear_map@Sigma_rho@self.coordinator.linear_map.T + 1e-8
        else:
            mu_r, Sigma_r = self.coordinator.inner_test_solver.get_weight_params(A_test, b_test)
        return torch.distributions.MultivariateNormal(mu_r,Sigma_r, validate_args=False)
    
    @property 
    def Sigma_x_train(self): 
        return torch.exp(self._log_sigma_x_train)
    
    @property 
    def Sigma_x_test(self): 
        return torch.exp(self._log_sigma_x_test)
    
    
    def _get_kernel_expectation(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor) -> torch.Tensor:
        """
        Returns object of shape N x C x L
        """
        # samples = torch.distributions.Normal(mu_x,Sigma_x).rsample((64,))
        # kernels_expectation = self.kernel(samples, self.V).mean(dim=0).transpose(0,1)

        # return kernels_expectation
        # scales = self.beta[None,:,:] + Sigma_x[:,None,:]  #   N x C x  L
        # det_beta = self.beta.prod(axis=-1) # C
        # det_scales = scales.prod(axis=-1) #  N x C 
        # differences = mu_x[:,None,:] - self.V[None,:,:]  # N x L x A 
        # L2_scaled_differences = (differences[:,None ,:,:]**2 / scales[:,:, None, :]).sum(
        #     axis=-1
        # )  # N x C x L 
        # return torch.sqrt(det_beta [None, :, None] / det_scales[:, :, None]) * torch.exp(
        #     -1 / 2 * L2_scaled_differences
        # )  #  N x C x L 
        d = (
              Sigma_x[:,None,:] 
            + self.beta[None,:,:]
        ) # N x C x A 
        e = (
              Sigma_x[:,None,None,:]*self.V[None,None,:,:]
            + self.beta[None,:,None,:]*mu_x[:,None, None,:]
        ) # N x C x L x A
        f = (
              Sigma_x[:,None,None,:]*(self.V[None,None,:,:]**2)
            + self.beta[None,:,None,:]*(mu_x[:,None,None,:]**2)
        ) # N x C x L x A
        g_intermediate = 1/(self.beta[None,:,None,:]*Sigma_x[:,None,None,:])*(-e**2/d[:,:,None,:] + f)
        g = -1/2*g_intermediate.sum(dim=-1) # N x C x L 
        Z = self.sigma2_s[None,:,None]*torch.sqrt((self.beta[None,:,None,:]/d[:,:,None,:]).prod(dim=-1))*torch.exp(g) 
        return Z # N x C x L 
    
    
    def _get_kernel_outer_expectation(self, mu_x, Sigma_x) -> torch.Tensor:
        """
        returns N x C x C x L x L
        """
        # samples = torch.distributions.Normal(mu_x,Sigma_x).rsample((16,))
        # K = self.kernel(samples, self.V).transpose(1,2)
        # K_outer = K[:,:,None,:,None,:]*K[:,:,:,None,:,None]
        # return K_outer.mean(dim=0)

        d = (
              self.beta[None, None,:,:]*Sigma_x[:,None,None,:] 
            + self.beta[None,:,None,:]*Sigma_x[:,None,None,:]
            + self.beta[None,:,None,:]*self.beta[None,None,:,:]
        ) # N x C x C x A 
        e = (
              self.beta[None,None,:,None,None,:]*Sigma_x[:,None,None,None,None,:]*self.V[None,None,None,:,None,:]
            + self.beta[None,:,None,None,None,:]*Sigma_x[:,None,None,None,None,:]*self.V[None,None,None,None,:,:]
            + self.beta[None,:,None,None,None,:]*self.beta[None,None,:,None,None,:]*mu_x[:,None,None,None,None,:]
        ) # N x C x C x L x L x A
        f = (
              self.beta[None,None,:,None,None,:]*Sigma_x[:,None,None,None,None,:]*(self.V[None,None,None,:,None,:]**2)
                + self.beta[None,:,None,None,None,:]*Sigma_x[:,None,None,None,None,:]*(self.V[None,None,None,None,:,:]**2)
                + self.beta[None,:,None,None,None,:]*self.beta[None,None,:,None,None,:]*(mu_x[:,None,None,None,None,:]**2)
        ) # N x C x C x L x L x A
        g_intermediate = 1/(self.beta[None,:,None,None,None,:]*self.beta[None,None,:,None,None,:]*Sigma_x[:,None,None,None,None,:])*(-e**2/d[:,:,:,None,None,:] + f)
        g = -1/2*g_intermediate.sum(dim=-1) # N x C x C x L x L
        Z = self.sigma2_s[None,:,None,None,None]*self.sigma2_s[None,None,:,None,None]*torch.sqrt((self.beta[None,:,None,None,None,:]*self.beta[None,None,:,None,None,:]/d[:,:,:,None,None,:]).prod(dim=-1))*torch.exp(g) 
        return Z # N x C x C x L x L 
    
