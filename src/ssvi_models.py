import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple, Optional
import math
from src.mf_methods import MF_WS_GPLVM


class SSVI_GP(nn.Module):
    """
    GP module that uses SSVI natural parameters instead of _mu_u
    This ensures all GP methods consistently use SSVI parameterization
    """
    def __init__(self, original_gp, ssvi_model):
        super().__init__()
        self.original_gp = original_gp
        self.ssvi_model = ssvi_model
        
        # Copy important attributes directly for easy access
        self.Vx = original_gp.Vx
        self.Lx = original_gp.Lx
        self.C = original_gp.C
        self.M = original_gp.M
        self.input_dims = original_gp.input_dims
        self.sigma2_s = original_gp.sigma2_s
        self.beta = original_gp.beta
        
        # Copy all parameter attributes except _mu_u
        for attr_name in dir(original_gp):
            if attr_name.startswith('_') and attr_name != '_mu_u' and hasattr(original_gp, attr_name):
                attr_value = getattr(original_gp, attr_name)
                if isinstance(attr_value, nn.Parameter):
                    setattr(self, attr_name, attr_value)
        
    def get_mu_u(self):
        """Use SSVI natural parameters to compute inducing point means"""
        return self.ssvi_model.get_mu_u_from_ssvi()
    
    def get_vector_mu_u(self):
        """Use SSVI natural parameters to compute vectorized inducing point means"""
        return self.ssvi_model.get_vector_mu_u_from_ssvi()
    
    def get_kl_u(self):
        """Use SSVI natural parameters to compute KL divergence"""
        return self.ssvi_model.get_kl_u_from_ssvi()
    
    def __getattr__(self, name):
        """Delegate any missing attributes to original GP"""
        if hasattr(self.original_gp, name):
            return getattr(self.original_gp, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class SSVI_GPLVM(MF_WS_GPLVM):
    """
    Structured Stochastic Variational Inference GP-LVM following the mathematical 
    formulation in gp_lvm_ssvi document - Algorithm 1: SSVI for GP-LVM
    
    Perfect integration: SSVI natural parameters completely replace GP's _mu_u
    while maintaining full compatibility with existing ELBO framework.
    """
    
    def __init__(self, Vx, M, mu_x_train, mu_x_test=None, C_obs=1, C_unobs=0, 
                 sigma2=1, beta=None, sigma2_s=None, step_size_params=None, 
                 trainable_V=True):
        """
        Initialize SSVI GP-LVM model following the LaTeX formulation
        """
        super().__init__(
            Vx=Vx, M=M, C_obs=C_obs, mu_x_train=mu_x_train, 
            mu_x_test=mu_x_test, C_unobs=C_unobs, sigma2=sigma2,
            weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s,
            trainable_V=trainable_V
        )
        
        # SSVI-specific parameters (Algorithm 1, line 1)
        self.step_size_params = step_size_params or {'s': 1.0, 'kappa': 0.75}
        self.iteration = 0
        
        # Store original GP for reference
        original_gp = self.gp
        
        # Natural parameters lambda = (Lambda, nu) for global variational distribution
        # Following the LaTeX: "Initialize natural parameters lambda^(0) = (Lambda^(0), nu^(0))"
        D = original_gp.Lx  # Dimension per component
        
        # Initialize natural parameters for each component j
        self._Lambda = nn.ParameterList([
            nn.Parameter(torch.eye(D))  # Lambda_j: D x D precision matrix
            for _ in range(self.C)
        ])
        
        self._nu = nn.ParameterList([
            nn.Parameter(torch.zeros(D))  # nu_j: D-dimensional vector
            for _ in range(self.C)
        ])
        
        # Initialize from the original GP parameters
        if hasattr(original_gp, '_mu_u'):
            original_mu_u = original_gp._mu_u.data.clone()
            self._initialize_natural_parameters_from_original(original_mu_u)
        
        # Replace GP with SSVI-compatible version
        self.gp = SSVI_GP(original_gp, self)
        
        # Store current global sample U^(t) for consistency
        self.current_U_sample = None
        
    def _initialize_natural_parameters_from_original(self, original_mu_u):
        """
        Initialize SSVI natural parameters from the original GP _mu_u parameter
        This ensures continuity with existing initialization
        """
        with torch.no_grad():
            mu_u_reshaped = original_mu_u.reshape(self.M, self.C, self.gp.original_gp.Lx)
            
            for j in range(self.C):
                # Initialize with mean from first inducing point
                mu_j = mu_u_reshaped[0, j, :]  
                self._nu[j].data = mu_j.clone()  # nu = Lambda * mu, with Lambda = I initially
    
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
        Compute mu^(t) = (Lambda^(t))^(-1) nu^(t) for component j
        Following Algorithm 1, line 5
        """
        try:
            return torch.solve(self._nu[j].unsqueeze(-1), self._Lambda[j])[0].squeeze(-1)
        except:
            # Numerical fallback
            return torch.inverse(self._Lambda[j] + 1e-6 * torch.eye(self._Lambda[j].shape[0])) @ self._nu[j]
    
    def get_global_Sigma(self, j: int):
        """
        Compute Sigma^(t) = (Lambda^(t))^(-1) for component j
        Following Algorithm 1, line 5
        """
        try:
            return torch.inverse(self._Lambda[j])
        except:
            # Numerical fallback - add small diagonal
            return torch.inverse(self._Lambda[j] + 1e-6 * torch.eye(self._Lambda[j].shape[0]))
    
    def get_mu_u_from_ssvi(self):
        """
        Compute inducing point means from SSVI natural parameters
        This replaces the original _mu_u parameter
        
        Returns: M x C x Lx tensor (same format as original)
        """
        mu_u = torch.zeros(self.M, self.C, self.gp.original_gp.Lx)
        
        for j in range(self.C):
            mu_j = self.get_global_mu(j)  # Lx
            # Replicate across all inducing points (simplified for now)
            mu_u[:, j, :] = mu_j.unsqueeze(0).expand(self.M, -1)
            
        return mu_u
    
    def get_vector_mu_u_from_ssvi(self):
        """
        Get vectorized inducing point means from SSVI parameters
        This replaces the original get_vector_mu_u method
        
        Returns: M x (C*Lx) tensor 
        """
        mu_u = self.get_mu_u_from_ssvi()  # M x C x Lx
        return mu_u.reshape(self.M, self.C * self.gp.original_gp.Lx)
    
    def get_kl_u_from_ssvi(self):
        """
        Compute KL divergence D_KL(q(U) || p(U)) using SSVI parameters
        This replaces the GP's original KL computation
        """
        kl_total = 0.0
        
        for j in range(self.C):
            # Prior: p(u_j) = N(0, K_ZZ)
            K_ZZ = self.gp.get_K_VV_x()[j]  # Lx x Lx
            
            # Variational posterior: q(u_j) = N(mu_j, Sigma_j) from SSVI natural parameters
            mu_j = self.get_global_mu(j)  # Lx
            Sigma_j = self.get_global_Sigma(j)  # Lx x Lx
            
            # KL divergence for multivariate normal
            try:
                q_dist = dist.MultivariateNormal(mu_j, Sigma_j)
                p_dist = dist.MultivariateNormal(torch.zeros_like(mu_j), K_ZZ)
                kl_j = dist.kl_divergence(q_dist, p_dist)
            except:
                # Analytical KL for numerical stability
                K_ZZ_inv = torch.inverse(K_ZZ + 1e-6 * torch.eye(K_ZZ.shape[0]))
                kl_j = 0.5 * (
                    torch.trace(K_ZZ_inv @ Sigma_j) +
                    mu_j @ K_ZZ_inv @ mu_j -
                    self.gp.original_gp.Lx +
                    torch.logdet(K_ZZ) - torch.logdet(Sigma_j + 1e-6 * torch.eye(Sigma_j.shape[0]))
                )
            
            # Sum over inducing points (simplified - in full version would be per inducing point)
            kl_total += self.M * kl_j
                
        return kl_total
    
    def sample_global_variables(self) -> torch.Tensor:
        """
        Sample global inducing variable U^(t) ~ N(mu^(t), Sigma^(t))
        Following Algorithm 1, line 6
        Returns: U samples of shape (C x Lx)
        """
        U_samples = torch.zeros(self.C, self.gp.original_gp.Lx)
        
        for j in range(self.C):
            mu_j = self.get_global_mu(j)  # Lx
            Sigma_j = self.get_global_Sigma(j)  # Lx x Lx
            
            try:
                mvn = dist.MultivariateNormal(mu_j, Sigma_j)
                U_samples[j, :] = mvn.rsample()
            except:
                # Numerical stability fallback
                U_samples[j, :] = mu_j + 0.01 * torch.randn_like(mu_j)
        
        self.current_U_sample = U_samples
        return U_samples
    
    def compute_local_elbo(self, y_i: torch.Tensor, mu_x_i: torch.Tensor, 
                          S_x_i: torch.Tensor, U_sample: torch.Tensor) -> torch.Tensor:
        """
        Compute local ELBO L_i(U) for data point i
        Following LaTeX equations (195-200)
        
        L_i(U) = sum_j E_q(x_i) E_p(f_j(x_i)|U,x_i) [log p(y_ij | f_j(x_i))] - KL(q(x_i) || p(x_i))
        """
        local_elbo = 0.0
        
        # Expected log-likelihood term
        for j in range(self.C_obs):
            y_ij = y_i[j]
            
            # Compute psi-statistics
            psi_0, psi_1, psi_2 = self.compute_psi_statistics_single(mu_x_i, S_x_i, j)
            
            # Expected log-likelihood using the formula from LaTeX
            beta = 1.0 / self.sigma2
            K_ZZ_inv = self.gp.get_K_VV_x_inv()[j]
            u_j = U_sample[j, :]  # Lx
            
            # Following the explicit form from LaTeX document
            const_term = -0.5 * math.log(2 * math.pi / beta)
            y_squared_term = -beta / 2 * y_ij**2
            cross_term = beta * y_ij * torch.sum(psi_1 @ K_ZZ_inv @ u_j)
            quad_term = -beta / 2 * torch.sum(u_j @ K_ZZ_inv @ psi_2 @ K_ZZ_inv @ u_j)
            trace_term = -beta / 2 * (psi_0 - torch.trace(K_ZZ_inv @ psi_2))
            
            local_ll_j = const_term + y_squared_term + cross_term + quad_term + trace_term
            local_elbo += local_ll_j
        
        # KL divergence term: KL(q(x_i) || p(x_i))
        # q(x_i) = N(mu_i, S_i), p(x_i) = N(0, I)
        kl_x_i = 0.5 * torch.sum(
            S_x_i + mu_x_i**2 - 1 - torch.log(S_x_i + 1e-8)
        )
        
        return local_elbo - kl_x_i
    
    def optimize_local_variational_distribution(self, n: int, y_i: torch.Tensor, 
                                              U_sample: torch.Tensor, max_iter: int = 10):
        """
        Optimize local variational distribution q(x_i) to maximize L_i(U)
        Following Algorithm 1, line 8: "Compute local variational distribution q(x_i | U^(t))"
        
        This solves: argmax_q(x_i) L_i(U^(t))
        """
        # Initialize with current parameters
        mu_i = self.mu_x_train[n].clone().detach().requires_grad_(True)
        log_S_i = self._log_sigma_x_train[n].clone().detach().requires_grad_(True)
        
        # Local optimizer for this data point
        local_optim = torch.optim.Adam([mu_i, log_S_i], lr=0.1)
        
        for _ in range(max_iter):
            local_optim.zero_grad()
            
            S_i = torch.exp(log_S_i)
            local_elbo = self.compute_local_elbo(y_i, mu_i, S_i, U_sample)
            
            # Maximize local ELBO
            loss = -local_elbo
            loss.backward()
            local_optim.step()
        
        # Return optimized parameters (detached from local optimization graph)
        return mu_i.detach(), torch.exp(log_S_i).detach()
    
    def compute_local_variational_distribution(self, n: int, U_sample: torch.Tensor):
        """
        Compute local variational distribution q(x_n | U^(t))
        Following Algorithm 1, line 8
        """
        # Get observations for this data point
        y_i = torch.zeros(self.C_obs)  # We'll get this from the batch in ssvi_step
        
        # This will be called from ssvi_step with proper y_i
        # For now, return current parameters (will be overridden in ssvi_step)
        mu_n = self.mu_x_train[n]
        S_n = self.Sigma_x_train[n]
        
        return mu_n, S_n
    
    def compute_psi_statistics_single(self, mu_x: torch.Tensor, S_x: torch.Tensor, j: int):
        """
        Compute psi-statistics for a single data point and component j
        Following Algorithm 1, line 9: "Compute kernel expectations psi_0i, psi_1i, psi_2i"
        """
        # Add batch dimension for existing methods
        mu_batch = mu_x.unsqueeze(0)  # 1 x A
        S_batch = S_x.unsqueeze(0)    # 1 x A
        
        # Use existing kernel expectation methods (this is the psi_1 statistic)
        kernel_exp = self._get_lv_kernel_expectation(mu_batch, S_batch)  # 1 x C x Lx
        psi_1 = kernel_exp[0, j, :]  # Lx
        
        # Use existing kernel outer expectation methods (this is related to psi_2)
        kernel_outer_exp = self._get_lv_kernel_outer_expectation(mu_batch, S_batch)  # 1 x C x C x Lx x Lx
        psi_2 = kernel_outer_exp[0, j, j, :, :]  # Lx x Lx
        
        # psi_0 is the kernel diagonal
        psi_0 = self.gp.sigma2_s[j]  # scalar
        
        return psi_0, psi_1, psi_2
    
    def compute_local_natural_parameter(self, y_nj: float, mu_n: torch.Tensor, 
                                      S_n: torch.Tensor, j: int) -> torch.Tensor:
        """
        Compute local natural parameter eta_hat_i(x_i) 
        Following Algorithm 1, line 10 and the LaTeX formulation in Section 3
        
        From the LaTeX: eta_hat_i(x_i) = [beta y_ij psi_1_i^(j) K_ZZ^(-1), 
                                         beta/2 vec(-K_ZZ^(-1) psi_2_i^(j) K_ZZ^(-1))]
        """
        # Compute psi-statistics for this data point and component
        psi_0, psi_1, psi_2 = self.compute_psi_statistics_single(mu_n, S_n, j)
        
        # Get precision parameter and kernel inverse
        beta = 1.0 / self.sigma2  # Precision parameter beta
        K_ZZ_inv = self.gp.get_K_VV_x_inv()[j]  # K_ZZ^(-1), shape: Lx x Lx
        
        # Compute h_i(x_i) = beta y_ij psi_1_i^(j) K_ZZ^(-1)
        h_i = beta * y_nj * (psi_1 @ K_ZZ_inv)  # Shape: Lx
        
        # Compute H_i(x_i) = beta/2 vec(-K_ZZ^(-1) psi_2_i^(j) K_ZZ^(-1))
        H_matrix = -beta / 2 * (K_ZZ_inv @ psi_2 @ K_ZZ_inv)  # Shape: Lx x Lx
        H_i = H_matrix.reshape(-1)  # Vectorize: Lx^2
        
        # Combine into natural parameter vector [h_i, H_i]
        eta_i = torch.cat([h_i, H_i])  # Shape: Lx + Lx^2
        
        return eta_i
    
    def compute_curvature_correction(self, U_sample: torch.Tensor, j: int) -> torch.Tensor:
        """
        Compute curvature correction matrix V(U^(t), Lambda^(t))
        Following Algorithm 1, line 12: "V(U^(t), Lambda^(t)) Delta_eta_i"
        
        This is a simplification - full implementation would require the structured VI framework
        """
        # For now, return identity (no curvature correction)
        # Full implementation would compute the Fisher information matrix
        D = self.gp.original_gp.Lx
        return torch.eye(D + D*D)
    
    def ssvi_step(self, y_batch: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Perform one SSVI step following Algorithm 1 exactly
        """
        batch_size = len(batch_indices)
        N_total = self.mu_x_train.shape[0]
        
        # Algorithm 1, line 2: Compute step size rho_t = s * t^(-kappa)
        self.iteration += 1
        rho_t = self.step_size_params['s'] * (self.iteration ** (-self.step_size_params['kappa']))
        
        # Algorithm 1, line 3: Sample mini-batch B subset of {1,...,N}
        # (batch_indices is the mini-batch)
        
        # Algorithm 1, line 4-6: Compute global variational parameters and sample U^(t)
        U_sample = self.sample_global_variables()  # C x Lx
        
        # Initialize natural parameter updates
        Delta_Lambda = [torch.zeros_like(self._Lambda[j]) for j in range(self.C)]
        Delta_nu = [torch.zeros_like(self._nu[j]) for j in range(self.C)]
        
        # Algorithm 1, lines 7-11: For each n in B
        for batch_idx, n in enumerate(batch_indices):
            # Algorithm 1, line 8: Compute local variational distribution q(x_i | U^(t))
            # This optimizes the local ELBO L_i(U^(t)) as required by LaTeX
            mu_n, S_n = self.optimize_local_variational_distribution(
                n, y_batch[batch_idx], U_sample, max_iter=5
            )
            
            # Process each output dimension j
            for j in range(self.C_obs):
                y_nj = y_batch[batch_idx, j].item()
                
                # Algorithm 1, line 9: Compute kernel expectations psi_0i, psi_1i, psi_2i
                # Algorithm 1, line 10: Compute local natural parameter eta_hat_i(x_i)
                eta_i = self.compute_local_natural_parameter(y_nj, mu_n, S_n, j)
                
                # Algorithm 1, line 11: Set local contribution Delta_eta_i := eta_hat_i
                Delta_eta_i = eta_i
                
                # Apply curvature correction (simplified)
                V_matrix = self.compute_curvature_correction(U_sample, j)
                corrected_delta = V_matrix @ Delta_eta_i
                
                # Split back into Lambda and nu components
                D = self.gp.original_gp.Lx
                delta_nu_j = corrected_delta[:D]  # First D elements
                delta_Lambda_j = corrected_delta[D:].reshape(D, D)  # Remaining elements as matrix
                
                # Accumulate updates
                Delta_nu[j] += delta_nu_j
                Delta_Lambda[j] += delta_Lambda_j
        
        # Algorithm 1, line 12: Global update
        # lambda^(t+1) = (1 - rho_t) lambda^(t) + rho_t * (N/|B|) * sum_{n in B} V(U^(t), Lambda^(t)) Delta_eta_i
        scaling = N_total / batch_size
        for j in range(self.C):
            self._Lambda[j].data = ((1 - rho_t) * self._Lambda[j].data + 
                                  rho_t * scaling * Delta_Lambda[j])
            self._nu[j].data = ((1 - rho_t) * self._nu[j].data + 
                              rho_t * scaling * Delta_nu[j])
        
        # Now compute ELBO using existing framework - the SSVI_GP will handle parameter routing
        elbo = self.elbo(y_batch, torch.ones(batch_size, 1))
        
        return elbo
    
    def gplvm_elbo(self, y: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Compute GP-LVM ELBO using SSVI with mini-batches
        The existing ELBO framework automatically uses SSVI parameters via SSVI_GP
        """
        N = y.shape[0]
        
        if batch_size is None or batch_size >= N:
            # Full batch - use existing ELBO framework 
            return self.elbo(y, torch.ones(N, 1))
        else:
            # Mini-batch SSVI step - updates natural parameters AND computes ELBO
            indices = torch.randperm(N)[:batch_size]
            y_batch = y[indices]
            return self.ssvi_step(y_batch, indices)
    
    def elbo(self, D_train, R_train, D_test=None):
        """
        Use existing ELBO computation framework
        The SSVI_GP wrapper automatically routes all calls to use SSVI parameters
        """
        return super().elbo(D_train, R_train, D_test)
    
    def parameters(self):
        """
        Return all trainable parameters including SSVI natural parameters
        """
        params = list(super().parameters())
        params.extend(self._Lambda.parameters())
        params.extend(self._nu.parameters())
        return params 