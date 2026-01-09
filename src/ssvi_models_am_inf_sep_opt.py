from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
from tqdm import tqdm
from src.struc_models import Struc_WS_GPLVM
from src.mf_methods import MFModels
from src.inner_optimisers import MultivariateNormalSolver
from src.utils import train_lbfgs, generate_regular_simplex
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
        

        # Amortized inference encoder networks (separate for mean and variance, like Original GP-LVM)
        # Input: y_n (observation) + u_samples (P samples from global U distribution)
        # Output: mu_n (A) from mu_encoder, log_sigma_n (A) from sigma_encoder
        self.mu_encoder = self._build_encoder(encoder_hidden_dims, self.A)
        self.sigma_encoder = self._build_encoder(encoder_hidden_dims, self.A)
        
        # Move encoders to device
        self.mu_encoder = self.mu_encoder.to(device)
        self.sigma_encoder = self.sigma_encoder.to(device)
        
        # Missing data capability
        self.enable_missing_data = enable_missing_data
        if self.enable_missing_data:
            print("Missing data reconstruction enabled")
        
        # Separate optimizers for different learning rates
        self._setup_optimizers(local_lr, global_lr)

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
        
        # Output layer: outputs output_dim dimensions (A for mu_encoder or A for sigma_encoder)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)

    def _setup_optimizers(self, local_lr, global_lr):
        """Setup separate optimizers for different learning rates"""
        # Local parameters (amortized inference encoders - both mu and sigma)
        local_params = list(self.mu_encoder.parameters()) + list(self.sigma_encoder.parameters())
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
        
        # Encode to local parameters using separate networks (like Original GP-LVM)
        mu_n = self.mu_encoder(encoder_input)  # [P, batch_size, A]
        log_Sigma_n = self.sigma_encoder(encoder_input)  # [P, batch_size, A]
        
        Sigma_n = torch.nn.functional.softplus(log_Sigma_n) + 1e-6  # Add small epsilon to ensure positive
        
        return mu_n, Sigma_n

    def get_mu_sigma_x_train_samples(self, Y_train_batch, epsilon):
        """
        Get latent variable samples for training data using amortized inference
        
        Args:
            Y_train_batch: Training observations [batch_size, M]
            U_samples: U samples from q(u) distribution [P, M, C, Lx]
            
        Returns:
            (mu_x, sigma_x): Training latent variable samples (P, batch_size, D)
        """
        
        mu_x, sigma_x = self.encode_local_params(Y_train_batch, epsilon)
        return mu_x, sigma_x
    
    def get_mu_sigma_x_test_samples(self, Y_test_batch, epsilon):
        # For SSVI, test and train use the same amortized inference
        mu_x, sigma_x = self.encode_local_params(Y_test_batch, epsilon)
        return mu_x, sigma_x
    
    
    def get_kl_x_from(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor):
        x_prior = torch.distributions.Normal(torch.zeros(self.A), torch.ones(self.A))
        x_train = torch.distributions.Normal(mu_x, torch.sqrt(Sigma_x))
        kl_train = torch.distributions.kl_divergence(x_train, x_prior).mean(dim=0).sum()
        if self.mu_x_test is None:
            return kl_train
        else:
            x_test = torch.distributions.Normal(self.mu_x_test, self.Sigma_x_test)
            kl_test = torch.distributions.kl_divergence(x_test, x_prior).mean(dim=0).sum()
            return kl_train + kl_test
    
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

    def _f_expectation_given_U(self, kernel_expectation: torch.Tensor, U_samples: torch.Tensor) -> torch.Tensor:
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
    
    def _f_outer_expectation_given_U(self, kernel_outer_expectation: torch.Tensor, U_samples: torch.Tensor) -> torch.Tensor:
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
        
        # Encoder gradients (both mu and sigma encoders)
        encoder_grad_norms = []
        encoder_param_norms = []
        for name, param in self.mu_encoder.named_parameters():
            if param.grad is not None:
                encoder_grad_norms.append(param.grad.norm().item())
                encoder_param_norms.append(param.norm().item())
        for name, param in self.sigma_encoder.named_parameters():
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
    
    def ssvi_elbo(self, y, mu_x, Sigma_x, U_samples, batch_size, total_size, missing_mask = None):
        """
        Extended gplvm_elbo that handles missing data by modifying the coordinator's likelihood computation.
        Uses the original ELBO structure but handles missing data in the coordinator.
        
        Raises:
            ValueError: If missing data functionality is not enabled or missing_mask is not provided
        """
        if missing_mask is None:
            missing_mask = torch.zeros_like(y, dtype=torch.bool)
        n_observed = (~missing_mask).sum().item()
        
        # STUFF

        kernel_train_expectation = self._get_lv_kernel_expectation(mu_x, Sigma_x)
        kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(mu_x,Sigma_x)

        f_train_expectation = self._f_expectation_given_U(kernel_expectation=kernel_train_expectation,
                                                          U_samples=U_samples)
        f_train_expectation = f_train_expectation[..., 0]  # [P, N, M] since C=1 for GP-LVM
        
        y_sqr_clean = (y[~missing_mask] ** 2).sum() # 1
        cross_clean = (y[None,:,:] * f_train_expectation * (~missing_mask)[None,:,:]).sum(dim=[-1,-2]).mean()
        # s_sqr_clean = .sum(dim=[-1,-2,-3]) # [P]

        K_VV = self.gp.get_K_VV_matrix()               # L x L 
        K_VV_inv_matrix = self.gp.get_K_VV_inv_matrix() # L x L
        kernel_train_outer_expectation = kernel_train_outer_expectation[..., 0, 0, :, :]  # [P, N, L, L] since C=1 for GP-LVM

        # Reshape U from [P, M, C, L] to [P, M, C*L]
        u_vec = U_samples.reshape(self.P, self.M, self.C * self.Lx)

        Ma = (u_vec[:, :, :, None] * u_vec[:, :, None, :] - K_VV[None, :, :]) # [P, M, L, L]
        Ma = (K_VV_inv_matrix @ Ma @ K_VV_inv_matrix.transpose(-2, -1))

        signal_squared = self.gp.sigma2_s[None, None,:] + (Ma[:, None, :, :, :] * kernel_train_outer_expectation[:,:,None,:,:]).sum(dim=[-2, -1])
        signal_sqr_clean = (signal_squared[:,~missing_mask]).sum(dim=-1).mean()

        normalising_const = -n_observed/2*torch.log(2*torch.pi*self.sigma2)

        training_inner_elbo = normalising_const - 1/(2*self.sigma2)*(y_sqr_clean - 2* cross_clean + signal_sqr_clean) - self.get_kl_x_from(mu_x, Sigma_x)
        
        # Scale const_train for mini-batches to represent full dataset contribution
        if batch_size != total_size:
            training_inner_elbo = training_inner_elbo  / batch_size
        


        return training_inner_elbo  - (self.gp.get_kl_u())/total_size