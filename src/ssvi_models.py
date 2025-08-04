import torch
import torch.nn as nn
from typing import Optional
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
                 sigma2=1, step_size_params=None, trainable_V=True, latent_init_scale=0.1):
        """
        Initialize SSVI GP-LVM by setting up standard GP-LVM as mixture model special case
        """
        N, D = Y_shape
        
        # Initialize latent variables with configurable scale
        mu_x_train = torch.randn(N, latent_dim) * latent_init_scale
        
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
        num_samples = getattr(self, 'num_u_samples', 1)
        U_samples = []
        for s in range(num_samples):  # Multiple U samples 
            U_sample_s = []
            for j in range(self.C):  # C = D (output dimensions)
                mu_j = self.get_global_mu(j)      # mu_j^(t) = (Lambda_j^(t))^(-1) nu_j^(t)
                Sigma_j = self.get_global_Sigma(j) # Sigma_j^(t) = (Lambda_j^(t))^(-1)
                U_j = torch.distributions.MultivariateNormal(mu_j, Sigma_j).rsample()
                U_sample_s.append(U_j)
            U_samples.append(torch.stack(U_sample_s, dim=0))  # C x L
        U_samples = torch.stack(U_samples, dim=0)  # num_samples x C x L
        
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
            # Average over multiple U samples for Monte Carlo approximation
            for j in range(self.C):
                h_i_total = torch.zeros_like(self._nu[j])
                H_i_total = torch.zeros_like(self._Lambda[j])
                
                # Average over multiple U samples
                for s in range(num_samples):
                    u_j_s = U_samples[s, j, :]  # L
                    h_i_s, H_i_s = self.compute_local_natural_parameters(idx, y_i[j], u_j_s, j)
                    
                    # Step 3: Apply curvature correction V(U, Lambda) * Delta_eta_i  
                    h_i_s_corrected, H_i_s_corrected = self.apply_curvature_correction(h_i_s, H_i_s, u_j_s, j)
                    
                    h_i_total += h_i_s_corrected
                    H_i_total += H_i_s_corrected
                
                # Average the contributions
                Delta_nu[j] += h_i_total / num_samples
                Delta_Lambda[j] += H_i_total / num_samples
        
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
    

    def get_ssvi_kl_u(self):
        """
        Compute KL divergence for SSVI using natural parameters
        KL(q(U) || p(U)) where q(U) is parameterized by natural parameters
        """
        # Get prior over inducing variables
        prior = self.gp.get_u_prior()
        
        # For SSVI, q(U) is parameterized by natural parameters (Lambda, nu)
        # Convert natural parameters to moment parameters
        total_kl = 0.0
        for j in range(self.gp.C):
            # Get moment parameters for output dimension j
            mu_j = self.get_global_mu(j)  # L
            Sigma_j = self.get_global_Sigma(j)  # L x L
            
            # Create q_u distribution for this output dimension
            q_u_j = torch.distributions.MultivariateNormal(mu_j, Sigma_j)
            
            # Prior for this output dimension (extract from block diagonal)
            start_idx = j * self.gp.Lx
            end_idx = (j + 1) * self.gp.Lx
            prior_mean_j = prior.mean[start_idx:end_idx]
            prior_cov_j = prior.covariance_matrix[start_idx:end_idx, start_idx:end_idx]
            prior_j = torch.distributions.MultivariateNormal(prior_mean_j, prior_cov_j)
            
            # KL divergence for this output dimension
            kl_j = torch.distributions.kl_divergence(q_u_j, prior_j)
            total_kl += kl_j
        
        return total_kl

    def compute_gplvm_elbo(self, Y: torch.Tensor) -> torch.Tensor:
        """
        GP-LVM ELBO following exact tex document formulation:
        L = sum_i L_i(U) - KL_u
        
        Simply sum local ELBOs and subtract KL_u
        """
        N = Y.shape[0]
        
        # Sum all local ELBOs using existing compute_local_elbo method
        total_local_elbo = 0.0
        for i in range(N):
            y_i = Y[i]
            mu_x_i = self.mu_x_train[i]
            sigma_x_i = self.Sigma_x_train[i] 
            
            # Compute local ELBO for data point i
            local_elbo_i = self.compute_local_elbo(i, y_i, mu_x_i, sigma_x_i, None)
            total_local_elbo += local_elbo_i
        
        # Global ELBO = sum of local ELBOs - KL_u (using SSVI-specific KL)
        kl_u = self.get_ssvi_kl_u()
        return total_local_elbo - kl_u
    
    def parameters(self):
        """
        Return all trainable parameters including SSVI natural parameters
        """
        params = list(super().parameters())
        params.extend(self._Lambda.parameters())
        params.extend(self._nu.parameters())
        return params
    
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
        Compute local ELBO L_i(U) for data point i following exact tex document formulation:
        L_i(U) = sum_j E_q(x_i) E_p(f_j(x_i)|U,x_i) [log p(y_ij | f_j(x_i))] - KL(q(x_i)||p(x_i))
        
        Efficient implementation without full tensor cloning
        """
        # Create single-point tensors directly (no expensive cloning!)
        mu_x_i_batch = mu_x_i.unsqueeze(0)  # 1 x latent_dim
        sigma_x_i_batch = sigma_x_i.unsqueeze(0)  # 1 x latent_dim (diagonal format)
        
        # Compute psi-statistics using existing infrastructure  
        kernel_expectation_i = self._get_lv_kernel_expectation(mu_x_i_batch, sigma_x_i_batch)  # 1 x C x L
        kernel_outer_expectation_i = self._get_lv_kernel_outer_expectation(mu_x_i_batch, sigma_x_i_batch)  # 1 x C x C x L x L
        
        # Local ELBO computation following tex document
        D = y_i.shape[0]
        local_elbo_i = 0.0
        
        for j in range(D):  # For each output dimension
            y_ij = y_i[j]
            
            # psi-statistics for dimension j
            psi_1_ij = kernel_expectation_i[0, j, :]  # L (corresponds to psi_1_i^(j))
            psi_2_ij = kernel_outer_expectation_i[0, j, j, :, :]  # L x L (corresponds to psi_2_i^(j))
            psi_0_ij = self.gp.sigma2_s[j]  # scalar (corresponds to psi_0_i^(j))
            
            K_ZZ_inv = self.gp.get_K_VV_x_inv()[j]  # L x L
            beta = 1.0 / self.sigma2  # precision
            
            # Compute log-likelihood: average over multiple U samples (Monte Carlo)
            if U_samples is None:
                # For compute_gplvm_elbo, use current mean from natural parameters
                u_j = self.get_global_mu(j)  # L
                log_likelihood_ij = (
                    -0.5 * torch.log(2 * torch.pi / beta)
                    - 0.5 * beta * (
                        y_ij**2 
                        - 2 * y_ij * (psi_1_ij @ K_ZZ_inv @ u_j)
                        + u_j @ K_ZZ_inv @ psi_2_ij @ K_ZZ_inv @ u_j
                        + psi_0_ij 
                        - torch.trace(K_ZZ_inv @ psi_2_ij)
                    )
                )
            else:
                # For local optimization: compute log-likelihood for each U sample, then average
                num_samples = U_samples.shape[0]
                log_likelihood_contributions = []
                
                for s in range(num_samples):
                    u_j_s = U_samples[s, j, :]  # L (individual U sample)
                    
                    log_likelihood_ij_s = (
                        -0.5 * torch.log(2 * torch.pi / beta)
                        - 0.5 * beta * (
                            y_ij**2 
                            - 2 * y_ij * (psi_1_ij @ K_ZZ_inv @ u_j_s)
                            + u_j_s @ K_ZZ_inv @ psi_2_ij @ K_ZZ_inv @ u_j_s
                            + psi_0_ij 
                            - torch.trace(K_ZZ_inv @ psi_2_ij)
                        )
                    )
                    log_likelihood_contributions.append(log_likelihood_ij_s)
                
                # Average the log-likelihood contributions (Monte Carlo approximation)
                log_likelihood_ij = torch.mean(torch.stack(log_likelihood_contributions))
            
            local_elbo_i += log_likelihood_ij
        
        # KL term for latent variable x_i: -KL(q(x_i)||p(x_i))
        x_prior = torch.distributions.Normal(torch.zeros_like(mu_x_i), torch.ones_like(sigma_x_i))
        x_posterior = torch.distributions.Normal(mu_x_i, sigma_x_i)
        kl_x_i = torch.distributions.kl_divergence(x_posterior, x_prior).sum()
        
        local_elbo_i -= kl_x_i
        
        return local_elbo_i
    
    def compute_local_natural_parameters(self, data_idx: int, y_ij: float, U_j: torch.Tensor, output_dim: int):
        """
        Compute local natural parameters h_i, H_i following tex document Eq. (349-352)
        
        From tex: h_i = beta * y_ij * psi_1i * K_ZZ^(-1)
                 H_i = (beta/2) * vec(-K_ZZ^(-1) * psi_2i * K_ZZ^(-1))
        
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
        H_i = -(beta / 2) * (K_ZZ_inv @ psi_2 @ K_ZZ_inv)  # L x L 
        
        return h_i, H_i
    
    def apply_curvature_correction(self, h_i: torch.Tensor, H_i: torch.Tensor, U_j: torch.Tensor, output_dim: int):
        """
        Mock function for curvature correction in SSVI-A
        In SSVI-A, curvature correction is not used, so we return original values unchanged
        """
        # SSVI-A: No curvature correction applied
        return h_i, H_i