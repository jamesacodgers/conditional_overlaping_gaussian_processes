from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
from tqdm import tqdm
from src.struc_models import Struc_WS_GPLVM
from src.mf_methods import MFModels
from src.inner_optimisers import MultivariateNormalSolver
from src.utils import train_lbfgs, generate_regular_simplex
from src.ssvi_metrics import compute_nlpd
import numpy as np

# Set default dtype to float64 for all tensor operations
torch.set_default_dtype(torch.float64)

class GPLVM_SSVI_AM_INF(Struc_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64, trainable_V=True, 
                 encoder_hidden_dims=[64, 32], local_lr=0.001, global_lr=0.001, device="auto", 
                 enable_missing_data=False):
        # Resolve device if auto
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Ensure input tensors are on the correct device
        Vx = Vx.to(device)
        mu_x = mu_x.to(device)
        
        super().__init__(P=P, V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, 
                        sigma2=sigma2, weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s, 
                        trainable_V=trainable_V)
        

        # Amortized inference encoder network
        # Input: y_n (observation) + u_samples (P samples from global U distribution)
        # Output: mu_n (Q) + log_sigma_n (Q) = 2*Q total
        self.encoder = self._build_encoder(encoder_hidden_dims, 2 * self.A)
        
        # Move encoder to device
        self.encoder = self.encoder.to(device)
        
        # Missing data capability
        self.enable_missing_data = enable_missing_data
        if self.enable_missing_data:
            print("Missing data reconstruction enabled")
        
        # Separate optimizers for different learning rates
        self._setup_optimizers(local_lr, global_lr)
        
        self.coordinator = NormalCoordinatorCustom(module=self, sum_to_one=False)

    def _build_encoder(self, hidden_dims, output_dim):
        """Build amortized inference encoder network"""
        layers = []
        # Calculate correct input dimension:
        # y_n: [M] + u_samples: [P * M * Lx] 
        input_dim = self.M + self.M * self.Lx

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Output layer: mu_n (Q) + log_sigma_n (Q)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)

    def _setup_optimizers(self, local_lr, global_lr):
        """Setup separate optimizers for different learning rates"""
        # Local parameters (amortized inference encoder)
        local_params = list(self.encoder.parameters())
        self.optimizer_local = torch.optim.AdamW(local_params, lr=local_lr, weight_decay=1e-4)
        
        # Global parameters (GP hyperparameters, inducing points, and U distribution)
        global_params = [
            self.gp._log_sigma2_s,     # Signal variances  
            self.gp._log_beta,         # Lengthscales
            self._log_sigma2,       # Noise variance
            self.gp.Vx,               # Inducing inputs
            self.gp._mu_u,             # U distribution mean
            self.gp._raw_chol          # U distribution covariance (Cholesky)
        ]
        
        # Add inducing inputs if trainable
        if hasattr(self.gp, 'Vx') and isinstance(self.gp.Vx, torch.nn.Parameter):
            global_params.append(self.gp.Vx)
            
        self.optimizer_global = torch.optim.Adam(global_params, lr=global_lr)

    def encode_local_params(self, y_n, u_samples):
        """
        Amortized inference: encode observation y_n and u_samples to local variational parameters
        
        Args:
            y_n: [batch_size, M] - observations
            u_samples: [P, M, C, Lx] - samples from global U distribution (C=1 for GP-LVM)
            
        Returns:
            mu_n: [batch_size, A] - local variational means
            sigma_n: [batch_size, A] - local variational standard deviations
        """
        batch_size = y_n.shape[0] 
        y_n = y_n.unsqueeze(0).expand(self.P, -1, -1)  # [P, batch_size, M]

        u_samples = u_samples.squeeze(2)  # Remove C=1 dimension for GP-LVM: [P, M, Lx]
        u_samples = u_samples.unsqueeze(1).expand(-1, batch_size, -1, -1)  # [P, batch_size, M, Lx]
        u_samples = u_samples.reshape(self.P, batch_size, -1)  # [P, batch_size, M*Lx]

        encoder_input = torch.cat([y_n, u_samples], dim=-1) # [P, batch_size, M + M*Lx]
        
        # Encode to local parameters
        encoder_output = self.encoder(encoder_input)
        
        # Split output into mu_n and log_sigma_n
        mu_n = encoder_output[:, :, :self.A]
        log_sigma_n = encoder_output[:, :, self.A:]
        
        sigma_n = torch.nn.functional.softplus(log_sigma_n)
        
        return mu_n, sigma_n

    def get_mu_sigma_x_train_samples(self, Y_train_batch, U_samples=None):
        """
        Get latent variable samples for training data using amortized inference
        
        Args:
            Y_train_batch: Training observations [batch_size, M]
            U_samples: U samples from q(u) distribution [P, M, C, Lx]
            
        Returns:
            (mu_x, sigma_x): Training latent variable samples (P, batch_size, D)
        """
        
        mu_x, sigma_x = self.encode_local_params(Y_train_batch, U_samples)
        return mu_x, sigma_x
    
    def get_mu_sigma_x_test_samples(self, Y_test_batch, U_samples=None):
        # For SSVI, test and train use the same amortized inference
        mu_x, sigma_x = self.encode_local_params(Y_test_batch, U_samples)
        return mu_x, sigma_x
    
    
    def get_kl_x_from(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor):
        x_prior = torch.distributions.Normal(torch.zeros(self.A), torch.ones(self.A))
        x_train = torch.distributions.Normal(mu_x, Sigma_x)
        kl_train = torch.distributions.kl_divergence(x_train, x_prior).mean(dim=[0,1]).sum()
        if self.mu_x_test is None:
            return kl_train
        else:
            assert False
    
    def get_ssvi_mean_prediction(self, inputs):
        """
        Mean prediction for GP-LVM using synchronized q(u).
        Returns [N_test, M] for C_obs=1 by squeezing the last dimension.
        """
        mu_s = self.get_mean_prediction_observed_inputs(inputs)  # [N_test, M, C]
        if mu_s.shape[-1] == 1:
            return mu_s[..., 0]
        return mu_s

    # def get_u_samples_ssvi(self, num_samples: int) -> torch.Tensor:
    #     """Sample U from current q(u) in gp (already kept in sync by setters)."""
    #     samples = self.get_u_vec_samples()
    #     return samples.reshape(self.P, self.M, self.C, self.Lx)

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

    def amortized_inference_step(self, Y_batch: torch.Tensor, batch_indices: torch.Tensor, u_samples: torch.Tensor):
        """
        Amortized inference step: encode observations to local variational parameters
        
        Args:
            Y_batch: Mini-batch of observations [batch_size x M]
            batch_indices: Indices of data points in the batch [batch_size]
            u_samples: [P, M, C, Lx] - samples from global U distribution
            
        Returns:
            mu_n_batch: [batch_size, Q] - local variational means with gradients
            sigma_n_batch: [batch_size, Q] - local variational std devs with gradients
        """
        # Encode local parameters using amortized inference with u_samples
        mu_n_batch, sigma_n_batch = self.encode_local_params(Y_batch, u_samples)
        
        # Update parent class tensors for bookkeeping (detached copies)
        with torch.no_grad():
            self.mu_x_train[:, batch_indices, :] = mu_n_batch.detach()
            self._log_sigma_x_train[:, batch_indices, :] = torch.log(sigma_n_batch.detach())
        
        return mu_n_batch, sigma_n_batch
    


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
    
    def gplvm_elbo(self, y, mu_x, Sigma_x, U_samples, batch_size, total_size, return_components=False):
        #R_train = torch.ones((y.shape[0],1), device=y.device, dtype=y.dtype)
        #kernel_train_expectation = self._get_lv_kernel_expectation(mu_x, Sigma_x)
        #kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x,Sigma_x)

        #s_train_expectation = self._s_expectation_given_U(kernel_expectation=kernel_train_expectation,
        #                                                 U_samples=U_samples)
        #s_train_outer_expectation = self._s_outer_expectation_given_U(kernel_outer_expectation=kernel_train_outer_expectation,
        #                                                             U_samples=U_samples)

        #const_train = self.coordinator.get_const_train(y, R_train, s_train_expectation, s_train_outer_expectation)

        # Use consistent U_samples and latent parameters for NLPD computation
        training_inner_elbo = -self.compute_nlpd(y, U_samples, mu_x, Sigma_x) - self.get_kl_x_from(mu_x, Sigma_x)
        

        
        elbo = training_inner_elbo  - self.gp.get_kl_u()/total_size
        
        if return_components:
            # Compute components for tracking (no gradients needed)
            with torch.no_grad():
                nlpd_value = self.compute_nlpd(y, U_samples, mu_x, Sigma_x)
                likelihood = -nlpd_value  # Convert NLPD back to log-likelihood
                kl_x = self.get_kl_x_from(mu_x, Sigma_x)
                kl_u = self.gp.get_kl_u()
                return elbo, likelihood , kl_x , kl_u / total_size

        return elbo

    def compute_nlpd(self,
            y_true: torch.Tensor,
            U_samples: torch.Tensor,
            mu_x: torch.Tensor,
            sigma_x: torch.Tensor) -> torch.Tensor:
        """
        Compute Negative Log Predictive Density for uncertainty quantification
        
        This version accepts precomputed U_samples and latent parameters to ensure
        consistency in the ELBO computational graph and avoid recomputation.

        Args:
            y_true: True observations [N, D]
            U_samples: Precomputed U samples [P, M, C, Lx]
            mu_x: Precomputed latent means [P, N, latent_dim]
            sigma_x: Precomputed latent std devs [P, N, latent_dim]

        Returns:
            NLPD value (differentiable tensor, lower is better)
        """
        # Use provided U_samples and latent parameters (no resampling/reencoding)

        # Compute averaged latent variables once before the loop (optimization)
        x_dist = torch.distributions.Normal(mu_x, sigma_x)
        x_samples = x_dist.rsample()  # [n_samples, batch_size, latent_dim]

        # Sample latent variables using pre-computed averages

        # Compute predictive distribution through GP using proper GP methods

        # Get GP predictive mean using GP method directly
        f_mean = self.gp.get_gp_expectation_given_U(x_samples, U_samples)  # [batch_size, M, C]

        # Get proper GP predictive variance: GP covariance + observation noise
        gp_cov = self.gp.get_gp_covariance_given_U(x_samples, U_samples)  # [batch_size, M, C, C]

        # For observed data (C=1 case), extract diagonal and add observation noise
        if gp_cov.shape[-1] == 1:
            f_var = gp_cov[..., 0, 0] + self.sigma2  # [batch_size, M]
            f_mean = f_mean[..., 0]  # [batch_size, M] to match f_var
        else:
            # For multi-output case, use diagonal elements + observation noise
            f_var = torch.diagonal(gp_cov, dim1=-2, dim2=-1) + self.sigma2  # [batch_size, M, C]

        log_prob = -0.5 * torch.log(2 * torch.pi * f_var) - 0.5 * (y_true - f_mean) ** 2 / f_var
        log_prob = log_prob.sum(dim=-1).mean()

        # Average log probabilities (log of mixture)
        # Use logsumexp for numerical stability

        # Negative log predictive density (average over batch)
        nlpd = -log_prob.mean()

        # model.train()
        return nlpd

    def get_gradient_info(self):
        """
        Get gradient and parameter information for logging
        Returns dict with gradient norms and parameter info
        """
        info = {}
        
        # q(u) parameters gradients
        if hasattr(self.gp, '_mu_u') and self.gp._mu_u.grad is not None:
            info['mu_u_grad_norm'] = self.gp._mu_u.grad.norm().item()
            info['mu_u_param_norm'] = self.gp._mu_u.norm().item()
        
        if hasattr(self.gp, '_raw_chol') and self.gp._raw_chol.grad is not None:
            info['raw_chol_grad_norm'] = self.gp._raw_chol.grad.norm().item()
            info['raw_chol_param_norm'] = self.gp._raw_chol.norm().item()
        
        # Encoder gradients
        encoder_grad_norms = []
        encoder_param_norms = []
        for name, param in self.encoder.named_parameters():
            if param.grad is not None:
                encoder_grad_norms.append(param.grad.norm().item())
                encoder_param_norms.append(param.norm().item())
        
        if encoder_grad_norms:
            info['encoder_grad_norm_max'] = max(encoder_grad_norms)
            info['encoder_grad_norm_mean'] = sum(encoder_grad_norms) / len(encoder_grad_norms)
            info['encoder_param_norm_max'] = max(encoder_param_norms)
            info['encoder_param_norm_mean'] = sum(encoder_param_norms) / len(encoder_param_norms)
        
        # GP hyperparameters gradients
        if hasattr(self.gp, 'beta') and hasattr(self.gp, '_log_beta') and self.gp._log_beta.grad is not None:
            info['beta_grad_norm'] = self.gp._log_beta.grad.norm().item()
            info['beta_values'] = self.gp.beta.detach().cpu().numpy().tolist()
        
        if hasattr(self.gp, 'sigma2') and hasattr(self.gp, '_log_sigma2') and self.gp._log_sigma2.grad is not None:
            info['sigma2_grad_norm'] = self.gp._log_sigma2.grad.norm().item()
            info['sigma2_value'] = self.gp.sigma2.item()
        
        return info
    
    def reconstruct_missing_data(self, y_observed, missing_mask=None, n_samples=50):
        """
        Reconstruct missing dimensions using amortized inference and GP decoding.
        
        Args:
            y_observed: Data with missing values as NaN [batch_size, output_dim]
            missing_mask: Boolean mask [batch_size, output_dim] where True=missing (optional)
            n_samples: Number of samples for Monte Carlo reconstruction
            
        Returns:
            y_reconstructed: Complete data with reconstructed missing values
            reconstruction_std: Standard deviation of reconstructions
        """
        if not self.enable_missing_data:
            raise ValueError("Missing data reconstruction not enabled. Set enable_missing_data=True in model constructor.")
        
        from src.missing_data_utils import reconstruct_missing_dimensions
        return reconstruct_missing_dimensions(self, y_observed, missing_mask, n_samples)
    
    def gplvm_elbo_with_missing_data(self, y, mu_x, Sigma_x, U_samples, batch_size, total_size, missing_mask, return_components=False):
        """
        Extended gplvm_elbo that handles missing data by modifying the coordinator's likelihood computation.
        Uses the original ELBO structure but handles missing data in the coordinator.
        
        Raises:
            ValueError: If missing data functionality is not enabled or missing_mask is not provided
        """
        
        n_observed = (~missing_mask).sum().item()
        
        # STUFF

        kernel_train_expectation = self._get_lv_kernel_expectation(mu_x, Sigma_x)
        kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x,Sigma_x)

        s_train_expectation = self._s_expectation_given_U(kernel_expectation=kernel_train_expectation,
                                                         U_samples=U_samples)
        s_train_expectation = s_train_expectation[..., 0]  # [P, N, M] since C=1 for GP-LVM        
        
        y_sqr_clean = (y[~missing_mask] ** 2).sum() # 1
        cross_clean = (y[None,:,:] * s_train_expectation * (~missing_mask)[None,:,:]).sum(dim=[-1,-2]).mean() 
        # s_sqr_clean = .sum(dim=[-1,-2,-3]) # [P]

        K_VV = self.gp.get_K_VV_matrix()               # L x L 
        K_VV_inv_matrix = self.gp.get_K_VV_inv_matrix() # L x L
        kernel_train_outer_expectation = kernel_train_outer_expectation[..., 0, 0, :, :]  # [P, N, L, L] since C=1 for GP-LVM

        # Reshape U from [P, M, C, L] to [P, M, C*L]
        Pn = U_samples.shape[0]
        u_vec = U_samples.reshape(Pn, self.M, self.C * self.Lx)

        Ma = (u_vec[:, :, :, None] * u_vec[:, :, None, :] - K_VV[None, :, :]) # [P, M, L, L]
        Ma = (K_VV_inv_matrix @ Ma @ K_VV_inv_matrix.transpose(-2, -1))

        signal_squared = self.gp.sigma2_s[None, None,:] + (Ma[:, None, :, :, :] * kernel_train_outer_expectation[:,:,None,:,:]).sum(dim=[-2, -1])
        signal_sqr_clean = (signal_squared[:,~missing_mask]).sum(dim=-1).mean()

        normalising_const = -n_observed/2*torch.log(2*torch.pi*self.sigma2)

        training_inner_elbo = normalising_const - 1/(2*self.sigma2)*(y_sqr_clean - 2* cross_clean + signal_sqr_clean) - self.get_kl_x_from(mu_x, Sigma_x)
        
        # Scale const_train for mini-batches to represent full dataset contribution
        if batch_size != total_size:
            training_inner_elbo = training_inner_elbo * (total_size / batch_size)
        
        elbo = (training_inner_elbo  - self.gp.get_kl_u())/total_size
        
        if return_components:
            # Compute components for tracking (no gradients needed)
            with torch.no_grad():
                likelihood = normalising_const - 1/(2*self.sigma2)*(y_sqr_clean - 2* cross_clean + signal_sqr_clean)
                if batch_size != total_size:
                    likelihood = likelihood * (total_size / batch_size)
                kl_x = self.get_kl_x_from(mu_x, Sigma_x)
                kl_u = self.gp.get_kl_u()
                return elbo, likelihood / total_size, kl_x / total_size, kl_u / total_size

        return elbo


class NormalCoordinatorCustom:
    def __init__(self, module:MFModels, sum_to_one:bool):
        self.module=module
        self.inner_train_solver = MultivariateNormalSolver(self.module.C_unobs)
        if sum_to_one:
            device = getattr(self.module, 'device', 'cpu')
            self.linear_map = torch.vstack([torch.zeros(self.module.C-1, device=device), torch.eye(self.module.C-1, device=device)])
            self.linear_map[:self.module.C_obs, :self.module.C_obs-1] = generate_regular_simplex(self.module.C_obs-1)
            self.h = torch.zeros(self.module.C, device=device)
            self.h[:self.module.C_obs] = torch.ones(self.module.C_obs, device=device)/self.module.C_obs
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
            return 1/(self.module.sigma2)*((D_test - (self.h[..., :]*s_expectation).sum(dim=-1))[..., None, None]*s_expectation[...,None]*self.linear_map[...,None,None,:,:]).sum(dim=[-2,-3]) # N x (C-1)
        else:
            return 1/(self.module.sigma2)*(D_test[:,:,None]*s_expectation).sum(dim=-2) # N x C 

                
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
        # Handle missing data if missing_mask is provided
        if hasattr(self, 'missing_mask') and self.missing_mask is not None:
            return self._get_const_train_missing_data(D_train, R_train, s_obs_expectation, s_obs_outer_expectation)
        
        # Standard likelihood computation for complete data
        return -(D_train.shape[1] * D_train.shape[0])/2*torch.log(2*torch.pi*self.module.sigma2) - 1/(2*self.module.sigma2)*((D_train**2).sum(dim=[-1,-2]) - 2 * torch.sum(D_train[...,None,:] @ s_obs_expectation @ R_train[...,None], dim=[-1,-2,-3]) + torch.sum(R_train[...,None,:]@s_obs_outer_expectation@R_train[...,None],dim=[-1,-2,-3]))
    
    def _get_const_train_missing_data(self, D_train, R_train, s_obs_expectation, s_obs_outer_expectation):
        """
        Compute likelihood term accounting for missing data following SVI paper formulation.
        Only observed dimensions contribute to the likelihood.
        
        Mathematical formulation:
        log p(y_obs | mu, sigma^2) = -0.5 * N * log(2*pi*sigma^2) - 0.5 * sum((y_obs - mu)^2)/sigma^2
        where N = number of observed dimensions
        """
        missing_mask = self.missing_mask
        
        # Create observed mask (inverse of missing mask)
        observed_mask = ~missing_mask  # [batch_size, output_dim]
        
        # Count observed elements for normalization
        n_observed = observed_mask.sum()
        
        if n_observed == 0:
            # All data missing - return zero likelihood
            return torch.zeros(1, device=D_train.device)
        
        # Replace NaN values with zeros for computation (they'll be masked out)
        D_train_clean = torch.where(missing_mask, torch.zeros_like(D_train), D_train)
        
        # Compute likelihood terms only for observed dimensions
        # Data term: (D_train**2).sum() but only for observed dimensions
        data_squared_term = (D_train_clean**2 * observed_mask.float()).sum()
        
        # Cross term: D_train @ s_obs_expectation @ R_train but only for observed dimensions
        estimated_signal = (s_obs_expectation @ R_train[...,None]).squeeze(-1)  # [P, batch_size, M]
        cross_term = (D_train_clean[None,:,:] * estimated_signal * observed_mask.float()).sum(dim=[-1,-2])
        # cross_term = torch.sum(
        #     D_train_clean[...,None,:] @ s_obs_expectation @ R_train[...,None] * observed_mask[...,None,:],
        #     dim=[-1,-2,-3]
        # )
        
        # Quadratic term: R_train @ s_obs_outer_expectation @ R_train but only for observed dimensions
        quad_term = torch.sum(
            R_train[...,None,:]@s_obs_outer_expectation@R_train[...,None] * observed_mask[...,None,:],
            dim=[-1,-2,-3]
        )
        
        # Complete Gaussian log-likelihood: -0.5 * N * log(2*pi*sigma^2) - 0.5 * sum((y-mu)^2)/sigma^2
        log_likelihood = (
            -n_observed/2 * torch.log(2 * torch.pi * self.module.sigma2) -
            1/(2*self.module.sigma2) * (data_squared_term - 2 * cross_term + quad_term)
        )
        
        return log_likelihood


    
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
        
        # Scale const_train for mini-batches to represent full dataset contribution
        batch_size = D_train.shape[0]
        total_size = mu_x_train.shape[0] if mu_x_train.dim() == 2 else mu_x_train.shape[1]
        if batch_size != total_size:
            const_train = const_train * (total_size / batch_size)
        
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
    
    