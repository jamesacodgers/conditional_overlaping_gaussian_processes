import torch 
import torch.nn as nn
from src.struc_models import Struc_WS_GPLVM, MFModels
from src.inner_optimisers import MultivariateNormalSolver
from src.utils import train_lbfgs, generate_regular_simplex
import numpy as np

class GPLVM_SSVI(Struc_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64,  trainable_V=True):
        super().__init__(P=P,  V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, sigma2=sigma2, weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s, trainable_V=trainable_V)
        
        # Inducing size and latent dim
        self.L = self.gp.Vx.shape[-2]     # #inducing, expects Vx: (L, Q) or (1, L, Q)
        self.Q = self.gp.Vx.shape[-1]

        # Natural params per output m=1..M (each u_m is L-dim)
        self._Lambda = nn.ParameterList([nn.Parameter(torch.eye(self.L)) for _ in range(M)])
        self._nu     = nn.ParameterList([nn.Parameter(torch.zeros(self.L))   for _ in range(M)])

        # Robbins-Monro schedule rho_t = s * t^{-kappa}
        self.ssvi_s = 1.0
        self.ssvi_kappa = 0.75
        self._iter = 0
        
        self.coordinator = NormalCoordinatorCustom(module=self,sum_to_one=False)
        
    @property 
    def mu_x(self): 
        return  self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return  self.Sigma_x_train
    
    @property
    def Lambda(self):
        """Natural parameter Lambda (precision matrices)"""
        return [param for param in self._Lambda]
    
    @property  
    def nu(self):
        """Natural parameter nu (precision-weighted means)"""
        return [param for param in self._nu]
        
    def get_global_mu(self, j: int):
        """
        Compute mu^(t) = (Lambda^(t))^(-1) nu^(t) for output dimension j
        Following Algorithm 1, line 5
        """
        try:
            return torch.solve(self._nu[j].unsqueeze(-1), self._Lambda[j])[0].squeeze(-1)
        except:
            # Numerical fallback
            return torch.inverse(self._Lambda[j] + 1e-6 * torch.eye(self._Lambda[j].shape[0])) @ self._nu[j]
    
    def get_global_Sigma(self, j: int):
        """
        Compute Sigma^(t) = (Lambda^(t))^(-1) for output dimension j
        Use Cholesky decomposition with jitter for numerical stability
        """
        # Start with small jitter and increase if needed
        jitter = 1e-6
        max_tries = 5
        
        for attempt in range(max_tries):
            try:
                # Add jitter to Lambda for numerical stability
                jittered_Lambda = self._Lambda[j] + jitter * torch.eye(self._Lambda[j].shape[0], device=self._Lambda[j].device)
                
                # Try Cholesky decomposition to check positive definiteness
                L = torch.linalg.cholesky(jittered_Lambda)
                
                # If successful, compute inverse using Cholesky solve
                # Sigma = Lambda^(-1) = (L L^T)^(-1)
                identity = torch.eye(self._Lambda[j].shape[0], device=self._Lambda[j].device)
                Sigma = torch.cholesky_solve(identity, L)
                
                return Sigma
                
            except RuntimeError:
                # Cholesky failed, increase jitter and try again
                jitter *= 10
                if attempt == max_tries - 1:
                    # Final fallback: return well-conditioned identity
                    return 1e-3 * torch.eye(self._Lambda[j].shape[0], device=self._Lambda[j].device)
        
        # Should never reach here, but safety fallback
        return 1e-3 * torch.eye(self._Lambda[j].shape[0], device=self._Lambda[j].device)
    
    
    def get_ssvi_mean_prediction(self, inputs):
        """
        SSVI-specific prediction that works directly with natural parameters
        
        Args:
            inputs: Test input locations [N_test, latent_dim]
            
        Returns:
            Predicted means [N_test, C] 
        """
        N_test = inputs.shape[0]
        predictions = torch.zeros(N_test, self.C)
        
        # Compute kernel between test inputs and inducing points
        # kernel_x returns [C, N_test, L] 
        K_test_Z_all = self.gp.kernel_x(inputs, self.gp.Vx)  # C x N_test x L
        
        for j in range(self.C):  # For each output dimension
            # Get SSVI global mean for this output dimension
            mu_j = self.get_global_mu(j)  # L-dimensional vector
            
            # Extract kernel for this output dimension and transpose
            K_test_Z_j = K_test_Z_all[j]  # N_test x L
            K_ZZ_inv = self.gp.get_K_VV_x_inv()[j]  # L x L
            
            # GP prediction: mu_f = K_test_Z @ K_ZZ_inv @ mu_j
            predictions[:, j] = K_test_Z_j @ K_ZZ_inv @ mu_j
        
        return predictions
    
    def ssvi_local_step(self, Y_batch: torch.Tensor, batch_indices: torch.Tensor) -> None:
        """
        SSVI Algorithm Steps 1-3:
        1. Use provided mini-batch B in {1,...,N}
        2. Compute global variational parameters mu^(t), Sigma^(t) and sample U^(t) ~ N(mu^(t), Sigma^(t))  
        3. For each n in B: optimize local q(x_i) and compute local natural parameters
        
        Args:
            Y_batch: Mini-batch of observations [batch_size x D]
            batch_indices: Indices of data points in the batch [batch_size]
        """
        batch_size = Y_batch.shape[0]
        N = self.mu_x_train.shape[0]  # Total dataset size
        
        # Compute step size: rho_t = s * t^(-kappa) (Algorithm 1, line 3)
        self.iteration += 1
        rho_t = self.step_size_params['s'] * (self.iteration ** (-self.step_size_params['kappa']))
        
        # SSVI Step 2: Compute global variational parameters and sample U^(t) (Algorithm 1, lines 5-6)
        # mu^(t) = (Lambda^(t))^(-1) nu^(t), Sigma^(t) = (Lambda^(t))^(-1)
        # Sample U^(t) ~ N(mu^(t), Sigma^(t)) - multiple samples for Monte Carlo approximation
        
        # Accumulate local contributions (proper implementation)
        Delta_Lambda = [torch.zeros_like(self._Lambda[j]) for j in range(self.C)]
        Delta_nu = [torch.zeros_like(self._nu[j]) for j in range(self.C)]
        
        # For each data point in mini-batch (Algorithm 1, lines 7-10)
        for i, idx in enumerate(batch_indices):
            y_i = Y_batch[i]  # D
            
            # Step 1: Optimize local variational distribution q(x_i) (Algorithm 1, line 7)
            # Use configurable parameters (can be overridden by subclasses)
            max_iter = getattr(self, 'local_opt_iterations', 1)
            lr_local = getattr(self, 'lr_local', 0.01)
            # Detach U_samples for local optimization to avoid gradient graph conflicts
            U_samples_detached = U_samples.detach()
            self.optimize_local_variational_distribution(idx, y_i, U_samples_detached, max_iter=max_iter, lr_local=lr_local)
            
            # Step 2: Compute local natural parameters using existing infrastructure
            for j in range(self.C):
                # Natural parameters don't depend on U, so compute once per data point
                h_i, H_i = self.compute_local_natural_parameters(idx, y_i[j], j)
                
                # Step 3: Apply curvature correction V(U, Lambda) * Delta_eta_i  
                # In SSVI-A, curvature correction is identity, so no averaging needed
                h_i_corrected, H_i_corrected = self.apply_curvature_correction(h_i, H_i, None, j)
                
                Delta_nu[j] += h_i_corrected
                Delta_Lambda[j] += H_i_corrected
        
        # Store local contributions for later natural parameter update
        scaling = N / batch_size
        self._pending_Delta_Lambda = Delta_Lambda
        self._pending_Delta_nu = Delta_nu
        self._pending_rho_t = rho_t
        self._pending_scaling = scaling


    def update_natural_parameters(self):
        """
        Update natural parameters using accumulated local contributions
        Called after GP hyperparameter update
        """
        if not hasattr(self, '_pending_Delta_Lambda'):
            return  # No pending updates
            
        # Global update (Algorithm 1, line 12)
        for j in range(self.C):
            # Prior natural parameters  
            nu_prior = torch.zeros_like(self._nu[j])  # nu^prior = 0
            K_ZZ_inv = self.gp.get_K_VV_x_inv()[j]  # K_ZZ^(-1)
            Lambda_prior = K_ZZ_inv  # Lambda^prior = K_ZZ^(-1)
            
            # Natural parameter updates with prior terms
            self._Lambda[j].data = ((1 - self._pending_rho_t) * self._Lambda[j].data + 
                                  self._pending_rho_t * (Lambda_prior + self._pending_scaling * self._pending_Delta_Lambda[j]))
            self._nu[j].data = ((1 - self._pending_rho_t) * self._nu[j].data + 
                              self._pending_rho_t * (nu_prior + self._pending_scaling * self._pending_Delta_nu[j]))
        
        # Clean up pending updates
        delattr(self, '_pending_Delta_Lambda')
        delattr(self, '_pending_Delta_nu')
        delattr(self, '_pending_rho_t') 
        delattr(self, '_pending_scaling')
        
    
    def optimize_local_variational_distribution(self, data_idx: int, y_i: torch.Tensor, U_samples: torch.Tensor, max_iter: int = None, lr_local: float = 0.01):
        """
        Optimize local variational distribution q(x_i) for data point i
        Following Algorithm 1, line 7: "Compute local variational distribution q(x_i | U^(t))"
        
        Optimizes local ELBO L_i(U) w.r.t. mu_i, Sigma_i using gradient ascent
        """
        # Create local parameters for this data point (detached from main params)
        mu_x_i = self.mu_x_train[data_idx].clone().detach().requires_grad_(True)
        log_sigma_x_i = self._log_sigma_x_train[data_idx].clone().detach().requires_grad_(True)
        
        # Use default max_iter if not specified (for backwards compatibility)
        if max_iter is None:
            max_iter = 1
            
        # Local optimization loop with configurable learning rate
        local_optimizer = torch.optim.Adam([mu_x_i, log_sigma_x_i], lr=lr_local)
        
        for _ in range(max_iter):
            local_optimizer.zero_grad()
            
            # Compute local ELBO L_i(U) using existing infrastructure
            # CORRECT HERE, WE WANT TO USE gplvm_elbo(self, y) FOR A SINGLE POINT AND AS I UNDERSTAND TO MAKE IT LOCAL ELBO WE NEED TO SUBSTRACT KL_U
            local_elbo = self.gplvm_elbo(y_i, mu_x_i, torch.exp(log_sigma_x_i))
            
            # Maximize ELBO (minimize negative ELBO)
            (-local_elbo).backward()
            local_optimizer.step()
        
        # Update the main parameters with optimized values
        with torch.no_grad():
            self.mu_x_train[data_idx] = mu_x_i.detach()
            self._log_sigma_x_train[data_idx] = log_sigma_x_i.detach()
    
    def elbo(self,D_train, R_train, D_test=None, mu_x_train=None, Sigma_x_train=None):
        # needed to override to make the inputs configurable for the local elbo to pass single x_i if needed
        # so changed only using mu_x_train, Sigma_x_train instead of self.mu_x_train, self.Sigma_x_train
        kernel_train_expectation = self._get_lv_kernel_expectation(mu_x_train, Sigma_x_train)
        kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x_train,Sigma_x_train)
        s_train_expectation = self.get_s_expectation(kernel_expectation=kernel_train_expectation)
        s_train_outer_expectation = self.get_s_outer_expectation(kernel_outer_expectation=kernel_train_outer_expectation)
        if D_test is None: 
            return self._elbo(s_train_expectation, s_train_outer_expectation, D_train, R_train)
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
            
    def _elbo(self, s_train_expectation, s_train_outer_expectation, D_train, R_train, mu_x_train, Sigma_x_train,  s_test_expectation=None, s_test_outer_expectation=None,D_test=None):
    # Same reason to override as in elbo     
        return self.coordinator._elbo(
                            s_train_expectation=s_train_expectation, 
                            s_train_outer_expectation=s_train_outer_expectation, 
                            D_train=D_train, 
                            R_train=R_train,  
                            mu_x_train=mu_x_train, 
                            Sigma_x_train=Sigma_x_train,
                            s_test_expectation=s_test_expectation, 
                            s_test_outer_expectation=s_test_outer_expectation,
                            D_test=D_test
                        )
    
    def gplvm_elbo(self, y, mu_x_train, Sigma_x_train):
        R_train = torch.ones((y.shape[0],1))
        return self.elbo(y, R_train, mu_x_train=mu_x_train, Sigma_x_train=Sigma_x_train)
    
    def compute_local_natural_parameters(self, data_idx: int, y_ij: float, output_dim: int):
        """
        Compute local natural parameters h_i, H_i following tex document Eq. (349-352)
        
        From tex: h_i = beta * y_ij * psi_1i * K_ZZ^(-1)
                 H_i = beta * vec(K_ZZ^(-1) * psi_2i * K_ZZ^(-1))
        
        Uses existing kernel expectation infrastructure
        """
        # Get current local variational parameters  
        mu_x_i = self.mu_x_train[data_idx:data_idx+1]  # 1 x Q
        Sigma_x_i = self.Sigma_x_train[data_idx:data_idx+1]  # 1 x Q (diagonal format)
        
        # Compute psi-statistics using existing infrastructure
        kernel_expectation = self._get_lv_kernel_expectation(mu_x_i, Sigma_x_i)  # 1 x C x L
        kernel_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x_i, Sigma_x_i)  # 1 x C x C x L x L
        
        # Extract for this output dimension
        psi_1 = kernel_expectation[0, output_dim, :]  # L
        psi_2 = kernel_outer_expectation[0, output_dim, output_dim, :, :]  # L x L  
        
        # Get kernel matrices
        K_ZZ_inv = self.gp.get_K_VV_x_inv()[output_dim]  # L x L
        beta = 1.0 / self.sigma2  # precision
        
        # Compute local natural parameters
        h_i = beta * y_ij * psi_1 @ K_ZZ_inv  # L
        H_i = beta * (K_ZZ_inv @ psi_2 @ K_ZZ_inv)  # L x L 
        
        return h_i, H_i
    
    def apply_curvature_correction(self, h_i: torch.Tensor, H_i: torch.Tensor, U_j: torch.Tensor = None, output_dim: int = None):
        """
        Mock function for curvature correction in SSVI-A
        In SSVI-A, curvature correction is not used, so we return original values unchanged
        """
        # SSVI-A: No curvature correction applied
        return h_i, H_i
    
    
class NormalCoordinatorCustom:
    def __init__(self, module:MFModels, sum_to_one:bool):
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
    
    def _elbo(self, s_train_expectation, s_train_outer_expectation, D_train, R_train, mu_x_train, Sigma_x_train, s_test_expectation=None, s_test_outer_expectation=None,D_test=None):
        # Again same reason to override as in elbo
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
        # HERE IS THE CORRECTION TO USE mu_x_train, Sigma_x_train FROM THE ARGUMETNS
        kl = self.module.get_kl_from(mu_x_train, Sigma_x_train)
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
    
    