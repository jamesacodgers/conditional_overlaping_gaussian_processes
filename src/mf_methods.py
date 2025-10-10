import torch 
from src.utils import generate_regular_simplex
from src.inner_optimisers import MultivariateNormalSolver
from src.gps import WLIndependentGPs, WLDependentGPs

from abc import ABC, abstractmethod

class   MFModels(torch.nn.Module, ABC):
    def __init__(self,M, C_obs, C_unobs=0, sigma2=1, weight_distribution="continuous"):
        torch.nn.Module.__init__(self)
        weight_distribution_options = {"continuous","sum_to_one","categorical"}
        assert weight_distribution in weight_distribution_options
        self.M = M 
        self.C_obs = C_obs
        self.C_unobs = C_unobs
        self.C = self.C_obs + self.C_unobs
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2, dtype=torch.float)))
        if weight_distribution=="continuous":
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
        return torch.nn.functional.softplus(self._log_sigma2, beta=1.0, threshold=20.0)+1e-4
    
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
    

    @abstractmethod
    def get_kl(self):
        ...
    
    @abstractmethod
    def get_kl_from(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor):
        ...
    




class ClassificationCoordinator:
    def __init__(self, module:MFModels):
        self.module = module 

    def _elbo(self, s_train_expectation, s_train_outer_expectation, D_train, R_train,  s_test_expectation=None, s_test_outer_expectation=None,D_test=None):
        assert R_train.shape == torch.Size([D_train.shape[0]])

        s_train_outer_diag_expextation = s_train_outer_expectation[...,range(self.module.C), range(self.module.C)]
        if len(s_train_expectation.shape)==3:
            s_train_expectation_c = s_train_expectation.gather(dim=-1, index=R_train[:,None,None].expand(-1,self.module.M,-1)).squeeze(-1)
            s_train_squared_expectation_c = s_train_outer_diag_expextation.gather(dim=-1, index=R_train[:,None]).squeeze(-1)
        elif len(s_train_expectation.shape)==4:
            s_train_expectation_c = s_train_expectation.gather(dim=-1, index=R_train[None,:,None,None].expand(s_train_expectation.shape[0],-1,self.module.M,-1)).squeeze(-1)
            s_train_squared_expectation_c = s_train_outer_diag_expextation.gather(dim=-1, index=R_train[None,:,None].expand(s_train_expectation.shape[0],-1,-1)).squeeze(-1)
        else:
            raise AssertionError("I should not be here")
        training_fit_term = -1/(2*self.module.sigma2)*((D_train**2).sum() - (2*D_train[...,:,:]*s_train_expectation_c).sum(dim=[-1,-2]).mean() + s_train_squared_expectation_c.sum(dim=[-1]).mean())- self.module.M*D_train.shape[0]/2*torch.log(2*torch.pi*self.module.sigma2)
        kl = self.module.get_kl() 
        if D_test == None:
            return training_fit_term - kl
        else:
            s_test_outer_diag_expectation = s_test_outer_expectation[...,range(self.module.C), range(self.module.C)]
            expected_dists = -1/(2*self.module.sigma2)*((D_test[:,:,None]**2).sum(dim=-2) - 2*(D_test[:,:,None]*s_test_expectation[:,:,:]).sum(dim=-2)  + s_test_outer_diag_expectation[:,:]) # N x C
            expected_dists
            max_val, _ = torch.max(expected_dists, dim=-1)
            renormalised_dists = expected_dists - max_val[...,None]
            # test_fit_term = torch.log(torch.exp(expected_dists).mean(dim=-1)).sum(dim=-1) - self.module.M*D_test.shape[0]/2*torch.log(2*torch.pi*self.module.sigma2)
            test_fit_term = (max_val + torch.log(torch.exp(renormalised_dists).mean(dim=-1))).sum(dim=-1).mean()  - self.module.M*D_test.shape[0]/2*torch.log(2*torch.pi*self.module.sigma2)# -torch.log(self.module.C)).sum(dim=-2)
            return training_fit_term + test_fit_term - kl
            

            

             

    def predict_weights(self, D, s_expectation, s_outer_expectation):
        s_outer_diag_expectation = s_outer_expectation[...,range(self.module.C), range(self.module.C)]
        expected_dists = -1/(2*self.module.sigma2)*((D[:,:,None]**2).sum(dim=-2) - 2*(D[:,:,None]*s_expectation[:,:,:]).sum(dim=-2)  + s_outer_diag_expectation[:,:])
        return torch.softmax(expected_dists, dim=-1)

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
        return 1/(self.module.sigma2)*((D_train[:,:,None]*s_unobs_expectation).sum(dim=-2) - (R_train[...,:,:,None]*s_outer_cross_expectation).sum(dim=-2))
                
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
    
    def predict_weights(self, D, s_expectation, s_outer_expectation):
        # s_expectation = self.module.get_s_expectation(D_test.shape[0])
        # s_outer_expectation = self.module.get_s_outer_expectation(D_test.shape[0])
        A_test = self.get_A_test(s_outer_expectation)
        b_test = self.get_b_test(D, s_expectation)
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
    
    def predict_weights(self,D):
        s_expectation = self.get_s_expectation(D.shape[0])
        s_outer_expectation = self.get_s_outer_expectation(D.shape[0])
        return self.coordinator.predict_weights(D=D, s_expectation=s_expectation, s_outer_expectation=s_outer_expectation)
    
    
    def get_kl(self):
        variational_dist = torch.distributions.MultivariateNormal(self.mu_S, self.Sigma_S)
        prior = torch.distributions.MultivariateNormal(torch.zeros(self.C), self.prior_Sigma_S)
        return torch.distributions.kl_divergence(variational_dist, prior).sum()

    @property 
    def Sigma_S(self): 
        chol = torch.tril(self._raw_chol_S)
        return chol.T @ chol

    @property 
    def prior_Sigma_S(self): 
        return  torch.diag(torch.concat((torch.ones(self.C_obs), torch.exp(self._log_priors_T)+1e-6)))
    
    
class MF_WS_GPLVM(MFModels):
    def __init__(self, Vx, M, C_obs, mu_x_train, mu_x_test=None, C_unobs=0, sigma2=1, weight_distribution="continuous", beta=None,sigma2_s=None, Vl=None, gamma=None, trainable_V = True):
        MFModels.__init__(self,M=M, C_obs=C_obs, C_unobs=C_unobs, sigma2=sigma2, weight_distribution=weight_distribution)
        if Vl is None:
            self.gp = WLIndependentGPs(C = self.C, M = M, Vx=Vx, beta = beta, sigma2_s = sigma2_s, trainable_V = trainable_V)
        else:
            self.gp = WLDependentGPs(C = self.C, M = M, Vx=Vx, beta = beta, sigma2_s = sigma2_s, trainable_V = trainable_V, Vl=Vl, gamma=gamma)
        self.add_module('gp', self.gp)

        self.A = Vx.shape[-1]
        self.mu_x_train = torch.nn.Parameter(mu_x_train)

        # If there are test points, register them 
        if mu_x_test is None:
            self.mu_x_test = None
        else:
            assert mu_x_train.shape[1] == self.A
            self.mu_x_test = torch.nn.Parameter(mu_x_test)
            self._log_sigma_x_test = torch.nn.Parameter(torch.zeros_like(mu_x_test))

        self._log_sigma_x_train = torch.nn.Parameter(torch.zeros_like(mu_x_train))


        

    def get_kl(self):
        return self.gp.get_kl_u() + self.get_kl_x()

    def get_kl_x(self):
        x_prior = torch.distributions.Normal(torch.zeros(self.A), torch.ones(self.A))
        x_train = torch.distributions.Normal(self.mu_x_train, self.Sigma_x_train)
        kl_train = torch.distributions.kl_divergence(x_train, x_prior).sum()
        if self.mu_x_test is None:
            return kl_train
        else: 
            x_test = torch.distributions.Normal(self.mu_x_test, self.Sigma_x_test)
            kl_test = torch.distributions.kl_divergence(x_test, x_prior).sum()
            return kl_train + kl_test
        
    def get_kl_from(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor):
        return self.gp.get_kl_u() + self.get_kl_x_from(mu_x, Sigma_x)
        
    def get_kl_x_from(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor) -> torch.Tensor:
        """
        KL_x for diagonal Gaussian q(x) from passed parameters.
        Expects Sigma_x to be the *variance* (diag), not std.
        Supports shapes (N, Q) or (Q,).
        """
        # normalize to (N, Q)
        if mu_x.dim() == 1:
            mu_x = mu_x.unsqueeze(0)
            Sigma_x = Sigma_x.unsqueeze(0)

        # prior N(0, I)
        Q = mu_x.shape[-1]
        prior = torch.distributions.Normal(
            torch.zeros(Q, device=mu_x.device, dtype=mu_x.dtype),
            torch.ones(Q,  device=mu_x.device, dtype=mu_x.dtype),
        )
        # q(x) with std = sqrt(variance)
        post = torch.distributions.Normal(mu_x, Sigma_x.clamp_min(1e-12).sqrt())

        return torch.distributions.kl_divergence(post, prior).sum()

    
    def elbo(self,D_train, R_train, D_test=None):
        kernel_train_expectation = self._get_lv_kernel_expectation(self.mu_x_train, self.Sigma_x_train)
        kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(self.mu_x_train,self.Sigma_x_train)
        s_train_expectation = self.get_s_expectation(kernel_expectation=kernel_train_expectation)
        s_train_outer_expectation = self.get_s_outer_expectation(kernel_outer_expectation=kernel_train_outer_expectation)
        if D_test is None: 
            return self._elbo(s_train_expectation, s_train_outer_expectation, D_train, R_train )
        else:
            kernel_test_expectation = self._get_lv_kernel_expectation(self.mu_x_test, self.Sigma_x_test)
            kernel_test_outer_expectation = self._get_lv_kernel_outer_expectation(self.mu_x_test,self.Sigma_x_test)
            s_test_expectation = self.gp.get_s_expectation(kernel_test_expectation)
            s_test_outer_expectation = self.gp.get_s_outer_expectation(kernel_test_outer_expectation)
            return self._elbo(
                    s_train_expectation, 
                    s_train_outer_expectation, 
                    D_train, 
                    R_train, 
                    D_test=D_test, 
                    s_test_expectation=s_test_expectation, 
                    s_test_outer_expectation=s_test_outer_expectation 
                )
    
    
    def get_mean_prediction_observed_inputs(self,inputs):
        """
        inputs: shape N x A

        returns: shape N x M x C 
        """
        K_VV_x_inv = self.gp.get_K_VV_x_inv() # C x Lx x Lx 
        K_obsV_x = self.gp.kernel_x(inputs, self.gp.Vx).transpose(0,1) # N x c x L
        mu_u = self.gp.get_mu_u()
        mu_s =  (K_obsV_x @ K_VV_x_inv)[:,None,:,None,:] @ mu_u[None,:,:,:,None]
        return mu_s.squeeze(dim=[-1,-2])

    @property 
    def Sigma_x_train(self): 
        return torch.nn.functional.softplus(self._log_sigma_x_train, beta=1.0, threshold=20.0)
    
    @property 
    def Sigma_x_test(self): 
        return torch.nn.functional.softplus(self._log_sigma_x_test, threshold=20.0)
    
    
    def _get_lv_kernel_expectation(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor) -> torch.Tensor:
        """
        Returns object of shape N x C x L
        """
        d = (
              Sigma_x[:,None,:] 
            + self.gp.beta[None,:,:]
        ) # N x C x A 
        e = (
              Sigma_x[:,None,None,:]*self.gp.Vx[None,None,:,:]
            + self.gp.beta[None,:,None,:]*mu_x[:,None, None,:]
        ) # N x C x L x A
        f = (
              Sigma_x[:,None,None,:]*(self.gp.Vx[None,None,:,:]**2)
            + self.gp.beta[None,:,None,:]*(mu_x[:,None,None,:]**2)
        ) # N x C x L x A
        g_intermediate = 1/(self.gp.beta[None,:,None,:]*Sigma_x[:,None,None,:])*(-e**2/d[:,:,None,:] + f)
        g = -1/2*g_intermediate.sum(dim=-1) # N x C x L 
        Z = self.gp.sigma2_s[None,:,None]*torch.sqrt((self.gp.beta[None,:,None,:]/d[:,:,None,:]).prod(dim=-1))*torch.exp(g) 
        return Z # N x C x L 
    
    
    def _get_lv_kernel_outer_expectation(self, mu_x, Sigma_x) -> torch.Tensor:
        """
        returns N x C x C x L x L
        """
        d = (
              self.gp.beta[None, None,:,:]*Sigma_x[:,None,None,:] 
            + self.gp.beta[None,:,None,:]*Sigma_x[:,None,None,:]
            + self.gp.beta[None,:,None,:]*self.gp.beta[None,None,:,:]
        ) # N x C x C x A 
        e = (
              self.gp.beta[None,None,:,None,None,:]*Sigma_x[:,None,None,None,None,:]*self.gp.Vx[None,None,None,:,None,:]
            + self.gp.beta[None,:,None,None,None,:]*Sigma_x[:,None,None,None,None,:]*self.gp.Vx[None,None,None,None,:,:]
            + self.gp.beta[None,:,None,None,None,:]*self.gp.beta[None,None,:,None,None,:]*mu_x[:,None,None,None,None,:]
        ) # N x C x C x L x L x A
        f = (
              self.gp.beta[None,None,:,None,None,:]*Sigma_x[:,None,None,None,None,:]*(self.gp.Vx[None,None,None,:,None,:]**2)
                + self.gp.beta[None,:,None,None,None,:]*Sigma_x[:,None,None,None,None,:]*(self.gp.Vx[None,None,None,None,:,:]**2)
                + self.gp.beta[None,:,None,None,None,:]*self.gp.beta[None,None,:,None,None,:]*(mu_x[:,None,None,None,None,:]**2)
        ) # N x C x C x L x L x A
        g_intermediate = 1/(self.gp.beta[None,:,None,None,None,:]*self.gp.beta[None,None,:,None,None,:]*Sigma_x[:,None,None,None,None,:])*(-e**2/d[:,:,:,None,None,:] + f)
        g = -1/2*g_intermediate.sum(dim=-1) # N x C x C x L x L
        Z = self.gp.sigma2_s[None,:,None,None,None]*self.gp.sigma2_s[None,None,:,None,None]*torch.sqrt((self.gp.beta[None,:,None,None,None,:]*self.gp.beta[None,None,:,None,None,:]/d[:,:,:,None,None,:]).prod(dim=-1))*torch.exp(g) 
        return Z # N x C x C x L x L 
    
    def predict_weights(self, D):
        kernel_expectation = self._get_lv_kernel_expectation(self.mu_x_test, self.Sigma_x_test)
        kernel_outer_expectation = self._get_lv_kernel_outer_expectation(self.mu_x_test,self.Sigma_x_test)

        s_expectation = self.gp.get_s_expectation(kernel_expectation=kernel_expectation)
        s_outer_expectation = self.gp.get_s_outer_expectation(kernel_outer_expectation=kernel_outer_expectation)
        return self.coordinator.predict_weights(D, s_expectation=s_expectation, s_outer_expectation=s_outer_expectation)
