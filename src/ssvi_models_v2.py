import torch 
import torch.nn as nn
from src.struc_models import Struc_WS_GPLVM
from src.mf_methods import MFModels
from src.inner_optimisers import MultivariateNormalSolver
from src.utils import train_lbfgs, generate_regular_simplex
import numpy as np

class GPLVM_SSVI(Struc_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64,  trainable_V=True):
        super().__init__(P=P,  V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, sigma2=sigma2, weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s, trainable_V=trainable_V)
        
        # Inducing size and latent dim
        self.L = self.gp.Vx.shape[-2]     # #inducing, expects Vx: (L, Q) or (1, L, Q)
        self.Q = self.gp.Vx.shape[-1]

        # Natural params per measurement m=1..M (each u_m is L-dim)
        # Initialize Lambda to prior precision K_ZZ^{-1}, nu to zeros
        K_ZZ_inv_0 = self.gp.get_K_VV_x_inv()[0].detach()
        self._Lambda = nn.ParameterList([nn.Parameter(K_ZZ_inv_0.clone()) for _ in range(M)])
        self._nu     = nn.ParameterList([nn.Parameter(torch.zeros(self.L, device=K_ZZ_inv_0.device, dtype=K_ZZ_inv_0.dtype)) for _ in range(M)])

        # Robbins-Monro schedule rho_t = s * t^{-kappa}
        self.ssvi_s = 1.0
        self.ssvi_kappa = 0.75
        self.iteration = 0
        self.step_size_params = {"s": self.ssvi_s, "kappa": self.ssvi_kappa}

        # SSVI uses natural parameters to define q(u); we sync them into gp's (mu, Sigma)
        # We can optionally draw explicit U samples per-iteration (P samples) for ELBOs
        
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
    
    def get_kl_x_from(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor):
        x_prior = torch.distributions.Normal(torch.zeros(self.A), torch.ones(self.A))
        x_train = torch.distributions.Normal(mu_x, Sigma_x)
        kl_train = torch.distributions.kl_divergence(x_train, x_prior).mean(dim=0).sum()
        if self.mu_x_test is None:
            return kl_train
        else:
            x_test = torch.distributions.Normal(self.mu_x_test, self.Sigma_x_test)
            kl_test = torch.distributions.kl_divergence(x_test, x_prior).mean(dim=0).sum()
            return kl_train + kl_test
        
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
    
    
    def get_ssvi_mean_prediction(self, inputs):
        """
        Mean prediction for GP-LVM using synchronized q(u).
        Returns [N_test, M] for C_obs=1 by squeezing the last dimension.
        """
        # Ensure gp variational params reflect current natural params
        self.synchronize_gp_from_natural_params()
        mu_s = self.get_mean_prediction_observed_inputs(inputs)  # [N_test, M, C]
        if mu_s.shape[-1] == 1:
            return mu_s[..., 0]
        return mu_s

    # -- U sampling and expectations conditioned on U --
    def get_u_samples_ssvi(self, num_samples: int) -> torch.Tensor:
        """
        Synchronize gp q(u) from (Lambda, nu), then sample U ~ q(u).
        Returns U_samples of shape [P=num_samples, M, C, L].
        """
        self.synchronize_gp_from_natural_params()
        samples = self.get_u_vec_samples()
        return samples.reshape(self.P, self.M, self.C, self.Lx)


    def _s_expectation_given_U(self, kernel_expectation: torch.Tensor, U_samples: torch.Tensor) -> torch.Tensor:
        """
        kernel_expectation: [P, N, C, L]
        U_samples: [P, M, C, L]
        Returns: [P, N, M, C]
        """
        K_VV_inv = self.gp.get_K_VV_x_inv()  # C x L x L
        return (
            kernel_expectation[:, :, None, :, None, :] @
            K_VV_inv[None, None, None, :, :, :] @
            U_samples[:, None, :, :, :, None]
        )[..., 0, 0]

    def _s_outer_expectation_given_U(self, kernel_outer_expectation: torch.Tensor, U_samples: torch.Tensor) -> torch.Tensor:
        """
        kernel_outer_expectation: [P, N, C, C, L, L]
        U_samples: [P, M, C, L]
        Returns: [P, N, C, C]
        """
        K_VV = self.gp.get_K_VV_matrix()               # (C*L) x (C*L)
        K_VV_inv_matrix = self.gp.get_K_VV_inv_matrix() # (C*L) x (C*L)

        # Reshape U from [P, M, C, L] to [P, M, C*L]
        Pn = U_samples.shape[0]
        u_vec = U_samples.reshape(Pn, self.M, self.C * self.Lx)

        Ma = (u_vec[:, :, :, None] * u_vec[:, :, None, :] - K_VV[None, :, :]).sum(dim=1)  # [P, CL, CL]
        Ma = (K_VV_inv_matrix @ Ma @ K_VV_inv_matrix.transpose(-2, -1))
        Ma = Ma.reshape(Pn, self.C, self.Lx, self.C, self.Lx).permute(0, 1, 3, 2, 4)      # [P, C, C, L, L]

        Z = kernel_outer_expectation  # [P, N, C, C, L, L]
        K_xx_expectation = torch.diag(self.gp.sigma2_s)
        A = K_xx_expectation * self.M + (Ma[:, None, :, :, :, :] * Z).sum(dim=[-2, -1])
        return A

    def synchronize_gp_from_natural_params(self):
        """
        Update gp variational parameters (mu_u, Sigma_u) to match current
        natural parameters (Lambda, nu). Assumes C == 1 (GP-LVM case).
        """
        if self.C != 1:
            raise NotImplementedError("Synchronization currently implemented for C==1 (GP-LVM) only.")
        device = self._Lambda[0].device
        eye_L = torch.eye(self.L, device=device)
        # Update each measurement block m
        for m in range(self.M):
            # Sigma = Lambda^{-1}
            jitter = 1e-6
            max_tries = 5
            for _ in range(max_tries):
                try:
                    Lchol = torch.linalg.cholesky(self._Lambda[m] + jitter * eye_L)
                    Sigma_m = torch.cholesky_solve(eye_L, Lchol)
                    break
                except RuntimeError:
                    jitter *= 10
            else:
                Sigma_m = 1e-3 * eye_L
            mu_m = Sigma_m @ self._nu[m]
            # Write into gp parameters (C==1 -> CL == L)
            with torch.no_grad():
                self.gp._mu_u.data[m, :self.L] = mu_m
                chol_Sigma = torch.linalg.cholesky(Sigma_m + 1e-9 * eye_L)
                self.gp._raw_chol.data[m, :self.L, :self.L] = chol_Sigma
    
    def ssvi_local_step(self, Y_batch: torch.Tensor, batch_indices: torch.Tensor, U_samples: torch.Tensor) -> None:
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
        
        # Require externally provided U_samples each iteration (no silent sampling)
        assert U_samples is not None, "U_samples must be provided for ssvi_local_step"
        assert U_samples.dim() == 4, "U_samples must be [P, M, C, L]"
        P_u, M_u, C_u, L_u = U_samples.shape
        assert M_u == self.M and C_u == self.C and L_u == self.L, "U_samples dims must match model (M, C, L)"
        self._pending_U_samples = U_samples
        
        # Accumulate local contributions per measurement m (since C_obs=1)
        Delta_Lambda = [torch.zeros_like(self._Lambda[m]) for m in range(self.M)]
        Delta_nu = [torch.zeros_like(self._nu[m]) for m in range(self.M)]
        
        # For each data point in mini-batch (Algorithm 1, lines 7-10)
        for i, idx in enumerate(batch_indices):
            y_i = Y_batch[i]  # D
            
            # Step 1: Optimize local variational distribution q(x_i) (Algorithm 1, line 7)
            # Use configurable parameters (can be overridden by subclasses)
            max_iter = getattr(self, 'local_opt_iterations', 1)
            lr_local = getattr(self, 'lr_local', 0.01)
            # Optimize local q(x_i | U) using current gp params (no explicit U sampling here)
            self.optimize_local_variational_distribution(idx, y_i, U_samples.detach(), max_iter=max_iter, lr_local=lr_local)
            
            # Step 2: Compute local natural parameters for each measurement m
            for m in range(self.M):
                h_i, H_i = self.compute_local_natural_parameters(idx, y_i[m], output_dim=0)
                h_i_corrected, H_i_corrected = self.apply_curvature_correction(h_i, H_i, None, output_dim=0)
                Delta_nu[m] += h_i_corrected
                Delta_Lambda[m] += H_i_corrected
        
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
            
        # Global update (Algorithm 1, line 12) over measurements m (C_obs=1)
        K_ZZ_inv_0 = self.gp.get_K_VV_x_inv()[0]
        for m in range(self.M):
            nu_prior = torch.zeros_like(self._nu[m])  # nu^prior = 0
            Lambda_prior = K_ZZ_inv_0                 # Lambda^prior = K_ZZ^(-1)

            self._Lambda[m].data = (
                (1 - self._pending_rho_t) * self._Lambda[m].data
                + self._pending_rho_t * (Lambda_prior + self._pending_scaling * self._pending_Delta_Lambda[m])
            )
            self._nu[m].data = (
                (1 - self._pending_rho_t) * self._nu[m].data
                + self._pending_rho_t * (nu_prior + self._pending_scaling * self._pending_Delta_nu[m])
            )
        
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
        # Create local parameters for this data point (shared across P replicates)
        mu_x_i = self.mu_x_train[:, data_idx, :].mean(dim=0).clone().detach().requires_grad_(True)
        log_sigma_x_i = self._log_sigma_x_train[:, data_idx, :].mean(dim=0).clone().detach().requires_grad_(True)
        
        # Use default max_iter if not specified (for backwards compatibility)
        if max_iter is None:
            max_iter = 1
            
        # Local optimization loop with configurable learning rate
        local_optimizer = torch.optim.Adam([mu_x_i, log_sigma_x_i], lr=lr_local)
        
        assert U_samples.dim() == 4, "U_samples must be [P, M, C, L]"
        U_samples = U_samples.detach()
        P_u = U_samples.shape[0]
        for _ in range(max_iter):
            local_optimizer.zero_grad()
            
            # Single-point local ELBO (exclude KL_u): ELBO_i + KL_u
            y_i_batch = y_i.unsqueeze(0)  # [1, M]
            # Repeat local q(x_i) across P (structured model uses P samples)
            mu_x_local = mu_x_i.unsqueeze(0).unsqueeze(0).repeat(P_u, 1, 1)           # [P_u, 1, Q]
            Sigma_x_local = torch.exp(log_sigma_x_i).unsqueeze(0).unsqueeze(0).repeat(P_u, 1, 1)  # [P_u, 1, Q]
            local_elbo_full = self.gplvm_elbo(y_i_batch, mu_x_local, Sigma_x_local, U_samples=U_samples)
            local_elbo = local_elbo_full + self.gp.get_kl_u()
            
            # Maximize ELBO (minimize negative ELBO)
            (-local_elbo).backward()
            local_optimizer.step()
        
        # Update the main parameters with optimized values
        with torch.no_grad():
            self.mu_x_train[:, data_idx, :] = mu_x_i.detach()
            self._log_sigma_x_train[:, data_idx, :] = log_sigma_x_i.detach()
    
    # def elbo(self,D_train, R_train, D_test=None, mu_x_train=None, Sigma_x_train=None, U_samples: torch.Tensor = None):
        # needed to override to make the inputs configurable for the local elbo to pass single x_i if needed
        # so changed only using mu_x_train, Sigma_x_train instead of self.mu_x_train, self.Sigma_x_train
    def elbo(self,D_train, R_train, D_test=None, mu_x_train=None, Sigma_x_train=None, U_samples: torch.Tensor = None):
        kernel_train_expectation = self._get_lv_kernel_expectation(mu_x_train, Sigma_x_train)
        kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x_train,Sigma_x_train)
        assert U_samples is not None, "U_samples must be provided for ELBO computation"
        # Replaced _s_expectation and _s_outer_expectation with _s_expectation_given_U and _s_outer_expectation_given_U
        s_train_expectation = self._s_expectation_given_U(kernel_expectation=kernel_train_expectation,
                                                         U_samples=U_samples)
        s_train_outer_expectation = self._s_outer_expectation_given_U(kernel_outer_expectation=kernel_train_outer_expectation,
                                                                     U_samples=U_samples)
        if D_test is None: 
            return self._elbo(
                s_train_expectation,
                s_train_outer_expectation,
                D_train,
                R_train,
                mu_x_train,
                Sigma_x_train,
            )
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
                mu_x_train,
                Sigma_x_train,
                D_test=D_test,
                s_test_expectation=s_test_expectation,
                s_test_outer_expectation=s_test_outer_expectation,
            )
            
    def _elbo(self, s_train_expectation, s_train_outer_expectation, D_train, R_train, mu_x_train, Sigma_x_train,  s_test_expectation=None, s_test_outer_expectation=None,D_test=None):
    # Same reason to override as in elbo - to have mu_x_train, Sigma_x_train as arguments    
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
    
    def gplvm_elbo(self, y, mu_x_train, Sigma_x_train, U_samples: torch.Tensor = None):
        R_train = torch.ones((y.shape[0],1), device=y.device, dtype=y.dtype)
        return self.elbo(y, R_train, mu_x_train=mu_x_train, Sigma_x_train=Sigma_x_train, U_samples=U_samples)
    
    def compute_local_natural_parameters(self, data_idx: int, y_ij: float, output_dim: int):
        """
        Compute local natural parameters h_i, H_i following tex document Eq. (349-352)
        
        From tex: h_i = beta * y_ij * psi_1i * K_ZZ^(-1)
                 H_i = beta * vec(K_ZZ^(-1) * psi_2i * K_ZZ^(-1))
        
        Uses existing kernel expectation infrastructure
        """
        # Get current local variational parameters across P samples: [P, 1, Q]
        mu_x_i = self.mu_x_train[:, data_idx:data_idx+1, :]
        Sigma_x_i = self.Sigma_x_train[:, data_idx:data_idx+1, :]
        
        # Compute psi-statistics using existing infrastructure
        kernel_expectation = self._get_lv_kernel_expectation(mu_x_i, Sigma_x_i)  # P x 1 x C x L
        kernel_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x_i, Sigma_x_i)  # P x 1 x C x C x L x L
        
        # Extract for this output dimension, average across P
        psi_1 = kernel_expectation[:, 0, output_dim, :].mean(dim=0)  # L
        psi_2 = kernel_outer_expectation[:, 0, output_dim, output_dim, :, :].mean(dim=0)  # L x L  
        
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
        #kl = self.module.get_kl_from(mu_x_train, Sigma_x_train)
        # Match Struc_WS_GPLVM: KL_u + KL_x (averaged over P if provided)
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
    
    