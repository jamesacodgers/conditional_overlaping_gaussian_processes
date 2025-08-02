import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple, Optional
import math
from src.mf_methods import MF_WS_GPLVM


class SSVI_GPLVM(MF_WS_GPLVM):
    """
    Structured Stochastic Variational Inference GP-LVM following the mathematical 
    formulation in gp_lvm_ssvi document - Algorithm 1: SSVI for GP-LVM
    
    Key insight: Standard GP-LVM is a special case of mixture model where:
    - Each output dimension is its own "component" 
    - R_train = ones matrix (each output has weight 1) - this overrides weight_distribution
    - C_obs = D (output dimensions), C_unobs = 0
    - weight_distribution="continuous" sets up NormalCoordinator(sum_to_one=False)
    """
    
    def __init__(self, Y_shape, latent_dim, Vx, beta=None, sigma2_s=None, 
                 sigma2=1, step_size_params=None, trainable_V=True):
        """
        Initialize SSVI GP-LVM by setting up standard GP-LVM as mixture model special case
        """
        N, D = Y_shape
        
        # Initialize latent variables 
        mu_x_train = torch.randn(N, latent_dim) * 0.1
        
        # Standard GP-LVM setup: each output dimension is a "component"
        super().__init__(
            Vx=Vx,
            M=Vx.shape[0],  # Number of inducing points
            C_obs=D,        # Each output dimension is an observed component  
            mu_x_train=mu_x_train,
            mu_x_test=None,
            C_unobs=0,      # No unobserved components in standard GP-LVM
            sigma2=sigma2,
            weight_distribution="continuous",  # Sets up NormalCoordinator(sum_to_one=False)
            beta=beta,
            sigma2_s=sigma2_s,
            trainable_V=trainable_V
        )
        
        # SSVI-specific parameters (Algorithm 1, line 1)
        self.step_size_params = step_size_params or {'s': 1.0, 'kappa': 0.75}
        self.iteration = 0
        
        # Natural parameters lambda = (Lambda, nu) for global variational distribution  
        # Following the LaTeX: "Initialize natural parameters lambda^(0) = (Lambda^(0), nu^(0))"
        L = self.gp.Lx  # Number of inducing points
        
        # Initialize natural parameters for each output dimension j
        self._Lambda = nn.ParameterList([
            nn.Parameter(torch.eye(L))  # Lambda_j: L x L precision matrix
            for _ in range(D)
        ])
        
        self._nu = nn.ParameterList([
            nn.Parameter(torch.zeros(L))  # nu_j: L-dimensional vector  
            for _ in range(D)
        ])
        
        # Initialize from GP's initial _mu_u
        self._initialize_natural_parameters_from_gp()
        
        # Store current global sample U^(t) for consistency  
        self.current_U_sample = None
        
    def _initialize_natural_parameters_from_gp(self):
        """
        Initialize SSVI natural parameters from the GP's _mu_u parameter
        """
        with torch.no_grad():
            if hasattr(self.gp, '_mu_u'):
                # _mu_u shape: [M, C * L] where C = D (output dimensions)
                mu_u = self.gp._mu_u.data  # M x (D*L)
                mu_u_reshaped = mu_u.reshape(self.M, self.C, self.gp.Lx)
                
                for j in range(self.C):  # C = D (output dimensions)
                    # Take mean over inducing points for initialization
                    mu_j = mu_u_reshaped[:, j, :].mean(dim=0)  # L
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
        """
        try:
            return torch.inverse(self._Lambda[j])
        except:
            return torch.inverse(self._Lambda[j] + 1e-6 * torch.eye(self._Lambda[j].shape[0]))
    
    def override_gp_parameters(self):
        """
        Override GP's inducing point parameters with SSVI values
        """
        # Compute means for each output dimension and update GP's _mu_u
        mu_u_per_dim = []
        for j in range(self.C):  # C = D (output dimensions)
            mu_j = self.get_global_mu(j)  # L
            mu_u_per_dim.append(mu_j)
        
        # Stack and reshape to match GP's expected format [M, C*L]
        mu_u_stacked = torch.stack(mu_u_per_dim, dim=0)  # C x L
        self.gp._mu_u.data = mu_u_stacked.T.reshape(self.M, -1)  # M x (C*L)
    
    def ssvi_step(self, Y_batch: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Perform one SSVI step following Algorithm 1 from tex document
        
        Args:
            Y_batch: Mini-batch of observations [batch_size, D]
            batch_indices: Indices of data points in batch
            
        Returns:
            ELBO value for the batch
        """
        batch_size = Y_batch.shape[0]
        N_total = self.mu_x_train.shape[0]
        
        # Step size: rho_t = s * t^(-kappa) (Algorithm 1, line 3)
        self.iteration += 1
        rho_t = self.step_size_params['s'] * (self.iteration ** (-self.step_size_params['kappa']))
        
        # Sample global variables: U^(t) ~ q(U) (Algorithm 1, line 6)
        U_samples = []
        for j in range(self.C):  # C = D (output dimensions)
            mu_j = self.get_global_mu(j)
            Sigma_j = self.get_global_Sigma(j)
            U_j = torch.distributions.MultivariateNormal(mu_j, Sigma_j).rsample()
            U_samples.append(U_j)
        U_samples = torch.stack(U_samples, dim=0)  # C x L
        
        # Accumulate local contributions (proper implementation)
        Delta_Lambda = [torch.zeros_like(self._Lambda[j]) for j in range(self.C)]
        Delta_nu = [torch.zeros_like(self._nu[j]) for j in range(self.C)]
        
        # For each data point in mini-batch (Algorithm 1, lines 7-10)
        for i, idx in enumerate(batch_indices):
            y_i = Y_batch[i]  # D
            
            # Step 1: Optimize local variational distribution q(x_i) (Algorithm 1, line 7)
            self.optimize_local_variational_distribution(idx, y_i, U_samples)
            
            # Step 2: Compute local natural parameters using existing infrastructure
            for j in range(self.C):
                h_i, H_i = self.compute_local_natural_parameters(idx, y_i[j], U_samples[j], j)
                
                # Step 3: Apply curvature correction V(U, Lambda) * Delta_eta_i  
                h_i_corrected, H_i_corrected = self.apply_curvature_correction(h_i, H_i, U_samples[j], j)
                
                Delta_nu[j] += h_i_corrected
                Delta_Lambda[j] += H_i_corrected
        
        # Global update (Algorithm 1, line 12)
        scaling = N_total / batch_size
        for j in range(self.C):
            # Prior natural parameters  
            nu_prior = torch.zeros_like(self._nu[j])  # nu^prior = 0
            K_ZZ_inv = self.gp.get_K_VV_x_inv()[j]  # K_ZZ^(-1)
            Lambda_prior = K_ZZ_inv  # Lambda^prior = K_ZZ^(-1)
            
            # Natural parameter updates with prior terms
            self._Lambda[j].data = ((1 - rho_t) * self._Lambda[j].data + 
                                  rho_t * (Lambda_prior + scaling * Delta_Lambda[j]))
            self._nu[j].data = ((1 - rho_t) * self._nu[j].data + 
                              rho_t * (nu_prior + scaling * Delta_nu[j]))
        
        # Override GP parameters and compute ELBO using existing infrastructure
        self.override_gp_parameters()
        
        # For standard GP-LVM: R_train = all ones (each output has weight 1)
        # This overrides the weight_distribution by explicitly providing weights
        R_batch = torch.ones(batch_size, self.C)  # batch_size x C
        
        # Use existing ELBO computation infrastructure
        return self.elbo(Y_batch, R_batch)
    
    def gplvm_elbo(self, Y: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Compute GP-LVM ELBO using SSVI with mini-batches or existing infrastructure
        """
        N = Y.shape[0]
        
        if batch_size is None or batch_size >= N:
            # Full batch - use existing ELBO infrastructure
            # For standard GP-LVM: R_train = all ones (each output has weight 1)
            # This overrides the weight_distribution by explicitly providing weights
            R_train = torch.ones(N, self.C)  # N x C
            return self.elbo(Y, R_train)
        else:
            # Mini-batch SSVI step - updates natural parameters AND computes ELBO
            indices = torch.randperm(N)[:batch_size]
            y_batch = Y[indices]
            return self.ssvi_step(y_batch, indices)
    
    def parameters(self):
        """
        Return all trainable parameters including SSVI natural parameters
        """
        params = list(super().parameters())
        params.extend(self._Lambda.parameters())
        params.extend(self._nu.parameters())
        return params
    
    def optimize_local_variational_distribution(self, data_idx: int, y_i: torch.Tensor, U_samples: torch.Tensor, max_iter: int = 5):
        """
        Optimize local variational distribution q(x_i) for data point i
        Following Algorithm 1, line 7: "Compute local variational distribution q(x_i | U^(t))"
        
        Optimizes local ELBO L_i(U) w.r.t. mu_i, Sigma_i using gradient ascent
        REUSES existing ELBO infrastructure
        """
        # Create local parameters for this data point (detached from main params)
        mu_x_i = self.mu_x_train[data_idx].clone().detach().requires_grad_(True)
        log_sigma_x_i = self._log_sigma_x_train[data_idx].clone().detach().requires_grad_(True)
        
        # Local optimization loop
        local_optimizer = torch.optim.Adam([mu_x_i, log_sigma_x_i], lr=0.01)
        
        for _ in range(max_iter):
            local_optimizer.zero_grad()
            
            # Compute local ELBO L_i(U) using existing infrastructure
            local_elbo = self.compute_local_elbo(data_idx, y_i, mu_x_i, torch.exp(log_sigma_x_i), U_samples)
            
            # Maximize ELBO (minimize negative ELBO)
            (-local_elbo).backward()
            local_optimizer.step()
        
        # Update the main parameters with optimized values
        with torch.no_grad():
            self.mu_x_train[data_idx] = mu_x_i.detach()
            self._log_sigma_x_train[data_idx] = log_sigma_x_i.detach()
    
    def compute_local_elbo(self, data_idx: int, y_i: torch.Tensor, mu_x_i: torch.Tensor, sigma_x_i: torch.Tensor, U_samples: torch.Tensor):
        """
        Compute local ELBO L_i(U) for data point i given current U samples
        Following tex document Eq. (196-199)
        
        L_i(U) = Sum_j E_q(x_i) E_p(f_j(x_i)|U,x_i) [log p(y_ij | f_j(x_i))] - KL(q(x_i) || p(x_i))
        
        KEY INSIGHT: Full ELBO = Sum_i L_i(U) - KL_u, so:
        L_i(U) = (Full ELBO for single point + KL_u) / N
        We can leverage existing infrastructure
        """
        # Temporarily update the latent parameters for this data point
        original_mu_x_i = self.mu_x_train[data_idx].clone()
        original_sigma_x_i = self.Sigma_x_train[data_idx].clone()
        
        with torch.no_grad():
            self.mu_x_train[data_idx] = mu_x_i
            self.Sigma_x_train[data_idx] = sigma_x_i
        
        try:
            # Use existing ELBO infrastructure for single data point
            y_i_batch = y_i.unsqueeze(0)  # 1 x D
            R_i_batch = torch.ones(1, self.C)  # 1 x C (weight = 1 for each output)
            
            # Compute full ELBO for this single data point (includes all KL terms)
            single_point_elbo = self.elbo(y_i_batch, R_i_batch)
            
            # Add back the KL_u terms to get local ELBO: L_i(U) = ELBO_i + KL_u
            # From tex: Full ELBO = Sum_i L_i(U) - Sum_j KL(q(u_j) || p(u_j))
            kl_u_terms = self.gp.get_kl_u()  # KL terms for inducing variables
            local_elbo = single_point_elbo + kl_u_terms
            
            return local_elbo
            
        finally:
            # Restore original parameters
            with torch.no_grad():
                self.mu_x_train[data_idx] = original_mu_x_i
                self.Sigma_x_train[data_idx] = original_sigma_x_i
    
    def compute_local_natural_parameters(self, data_idx: int, y_ij: float, U_j: torch.Tensor, output_dim: int):
        """
        Compute local natural parameters h_i, H_i following tex document Eq. (349-352)
        
        From tex: h_i = beta * y_ij * psi_1i * K_ZZ^(-1)
                 H_i = (beta/2) * vec(-K_ZZ^(-1) * psi_2i * K_ZZ^(-1))
        
        Uses existing kernel expectation infrastructure
        """
        # Get current local variational parameters  
        mu_x_i = self.mu_x_train[data_idx:data_idx+1]  # 1 x Q
        Sigma_x_i = torch.diag(self.Sigma_x_train[data_idx]).unsqueeze(0)  # 1 x Q x Q
        
        # Compute psi-statistics using existing infrastructure
        kernel_expectation = self._get_lv_kernel_expectation(mu_x_i, Sigma_x_i)  # 1 x C x L
        kernel_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x_i, Sigma_x_i)  # 1 x C x C x L x L
        
        # Extract for this output dimension
        psi_1 = kernel_expectation[0, output_dim, :]  # L
        psi_2 = kernel_outer_expectation[0, output_dim, output_dim, :, :]  # L x L  
        
        # Get kernel matrices
        K_ZZ_inv = self.gp.get_K_VV_x_inv()[output_dim]  # L x L
        beta = 1.0 / self.sigma2  # precision
        
        # Compute local natural parameters (tex Eq. 349-352)
        h_i = beta * y_ij * psi_1 @ K_ZZ_inv  # L
        H_i = -(beta / 2) * (K_ZZ_inv @ psi_2 @ K_ZZ_inv)  # L x L 
        
        return h_i, H_i
    
    def apply_curvature_correction(self, h_i: torch.Tensor, H_i: torch.Tensor, U_j: torch.Tensor, output_dim: int):
        """
        Mock function for curvature correction in SSVI-A
        In SSVI-A, curvature correction is not used, so we return original values unchanged
        """
        # SSVI-A: No curvature correction applied
        return h_i, H_i