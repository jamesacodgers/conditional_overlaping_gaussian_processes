"""
LVMOGP_SSVI: Proper mathematical LVMOGP based on GPLVM_SSVI_AM_INF

This implements the same mathematical structure as LVMOGP.py but uses
GPLVM_SSVI_AM_INF's SSVI computational machinery:
- X_data: Fixed observed input features [N, Q_x]
- H_data: Learned function-level latent positions [P, num_fns, Q_h] (P for SSVI)
- X_tilde = [X_data, H[fn_id]]: Full GP input [P, N, Q_x + Q_h]
- Uses SSVI with amortized inference for efficient learning
"""

import torch
import torch.nn as nn
import numpy as np
from src.struc_models import Struc_WS_GPLVM


class LVMOGP_SSVI(Struc_WS_GPLVM):
    """
    Latent Variable Multi-Output GP with SSVI and Amortized Inference
    
    Mathematical Structure (following LVMOGP.py):
    - X_data: [N, Q_x] - Fixed observed input features (e.g., x locations)
    - X_data_fn: [N] - Function IDs (which surface/task)
    - H: [P, num_fns, Q_h] - Learned function latent positions (P samples for SSVI)
    - X_tilde = [X_data, H[fn_id]]: [P, N, Q_x + Q_h] - Full GP input
    
    Inherits from Struc_WS_GPLVM to reuse SSVI computational machinery.
    """
    
    def __init__(self, X_data, X_data_fn, num_fns, Q_h, M, Vx,
                 sigma2=1, beta=None, sigma2_s=None, P=8,
                 trainable_V=True, encoder_hidden_dims=[128, 64],
                 local_lr=0.001, global_lr=0.001, device="auto",
                 enable_missing_data=False):
        """
        Initialize LVMOGP with SSVI.
        
        Parameters
        ----------
        X_data : torch.Tensor [N, Q_x]
            Fixed observed input features
        X_data_fn : torch.Tensor [N]
            Function IDs for each observation
        num_fns : int
            Number of functions/surfaces
        Q_h : int
            Dimensionality of function latent space
        M : int
            Output dimension (1 for scalar outputs)
        Vx : torch.Tensor [L, Q_x + Q_h]
            Inducing points in full space
        sigma2 : float
            Observation noise variance
        beta : list
            Lengthscales for (Q_x + Q_h) dimensions
        sigma2_s : list
            Signal variances
        P : int
            Number of SSVI samples
        trainable_V : bool
            Whether inducing points are trainable
        encoder_hidden_dims : list
            Encoder architecture
        local_lr : float
            Learning rate for encoder/H
        global_lr : float
            Learning rate for GP hyperparameters
        device : str
            Device
        enable_missing_data : bool
            Enable missing data handling
        """
        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Store LVMOGP-specific data
        self.X_data = X_data.to(device)  # [N, Q_x] - FIXED observed features
        self.X_data_fn = X_data_fn.to(device).long()  # [N] - Function IDs
        
        # Dimensions
        self.N = X_data.shape[0]
        self.Q_x = X_data.shape[1]  # Observed feature dimensions
        self.num_fns = num_fns
        self.Q_h = Q_h  # Function latent dimensions
        self.Q_total = self.Q_x + self.Q_h  # Total latent dimensions
        
        # Initialize dummy mu_x for parent class
        # Parent expects [P, N, A] for SSVI
        dummy_mu_x = torch.randn(P, self.N, self.Q_total, dtype=torch.float64, device=device) * 0.1
        
        # Ensure Vx and beta are correct dimensions
        Vx = Vx.to(device)
        if beta is None:
            beta = [[1.0] * self.Q_total]  # One set of lengthscales for C=1
        
        # Initialize parent (Struc_WS_GPLVM)
        # This sets up GP machinery, inducing points, q(u), etc.
        super().__init__(
            P=P, 
            V=Vx, 
            M=M, 
            C_obs=1,  # Single output per observation
            mu_x_train=dummy_mu_x,
            mu_x_test=None,
            C_unobs=0,
            sigma2=sigma2,
            weight_distribution="continuous",
            beta=beta,
            sigma2_s=sigma2_s,
            trainable_V=trainable_V
        )
        
        # Initialize function latent positions H [P, num_fns, Q_h]
        # P samples for SSVI, one H per function
        # This is stored for bookkeeping, like mu_x_train in parent
        self.H_mean = torch.randn(self.P, num_fns, Q_h, dtype=torch.float64, device=device) * 0.5
        self._H_log_sigma = torch.randn(self.P, num_fns, Q_h, dtype=torch.float64, device=device) * 0.1
        
        # Make H trainable
        self.H_mean.requires_grad_(True)
        self._H_log_sigma.requires_grad_(True)
        
        # Build encoder for amortized inference: Y + u_samples -> H
        # Aggregates observations per function and infers its H
        self.encoder_H = self._build_encoder_for_H(encoder_hidden_dims, 2 * Q_h)
        self.encoder_H = self.encoder_H.to(device)
        
        # Missing data capability
        self.enable_missing_data = enable_missing_data
        if self.enable_missing_data:
            print("Missing data reconstruction enabled")
        
        # Setup optimizers
        self._setup_lvmogp_optimizers(local_lr, global_lr)
    
    @property
    def H_sigma(self):
        return torch.nn.functional.softplus(self._H_log_sigma)
    
    def _build_encoder_for_H(self, hidden_dims, output_dim):
        """
        Build encoder for amortized inference: Y + u_samples -> H
        
        For each function, we aggregate its observations and infer H.
        Input: observation [M] + u_samples [M * Lx]
        Output: mu_H [Q_h] + log_sigma_H [Q_h]
        """
        layers = []
        # Input: y_n [M] + u_samples flattened [M * Lx]
        input_dim = self.M + self.M * self.Lx
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Output: mu_H [Q_h] + log_sigma_H [Q_h]
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _setup_lvmogp_optimizers(self, local_lr, global_lr):
        """Setup optimizers for LVMOGP"""
        # Local parameters: encoder for H (H itself is updated via amortized inference)
        local_params = list(self.encoder_H.parameters())
        self.optimizer_local = torch.optim.AdamW(local_params, lr=local_lr, weight_decay=1e-4)
        
        # Global parameters: GP hyperparameters, inducing points, q(u)
        global_params = [
            self.gp._log_sigma2_s,
            self.gp._log_beta,
            self._log_sigma2,
            self.gp._mu_u,
            self.gp._raw_chol
        ]
        
        if hasattr(self.gp, 'Vx') and isinstance(self.gp.Vx, torch.nn.Parameter):
            global_params.append(self.gp.Vx)
        
        self.optimizer_global = torch.optim.Adam(global_params, lr=global_lr)
    
    def get_X_tilde(self, X_data, X_data_fn, H_samples):
        """
        Construct X_tilde = [X_data, H[fn_id]]
        
        Parameters:
        -----------
        X_data : [N, Q_x]
        X_data_fn : [N]
        H_samples : [P, num_fns, Q_h]
        
        Returns:
        --------
        X_tilde : [P, N, Q_total]
        """
        # Gather H for each observation based on function ID
        H_for_obs = H_samples[:, X_data_fn, :]  # [P, N, Q_h]
        
        # Expand X_data for all P samples
        X_data_expanded = X_data.unsqueeze(0).expand(self.P, -1, -1)  # [P, N, Q_x]
        
        # Concatenate [X_data, H]
        X_tilde = torch.cat([X_data_expanded, H_for_obs], dim=-1)  # [P, N, Q_total]
        
        return X_tilde
    
    def encode_H_for_functions(self, Y_obs, X_data_fn, u_samples):
        """
        Amortized inference: For each unique function in batch, aggregate observations and infer H.
        
        This is the LVMOGP equivalent of encode_local_params() in GPLVM_SSVI_AM_INF.
        
        Parameters:
        -----------
        Y_obs : [batch_size, M]
            Observations
        X_data_fn : [batch_size]
            Function IDs for observations
        u_samples : [P, M, 1, Lx]
            Samples from q(u)
            
        Returns:
        --------
        mu_H_batch : [P, num_fns, Q_h]
            Mean of H for all functions (updated from current values)
        sigma_H_batch : [P, num_fns, Q_h]
            Std dev of H for all functions (updated from current values)
        """
        # For each unique function in batch, aggregate observations
        unique_fns = torch.unique(X_data_fn)
        
        # Start with current H values
        mu_H_batch = self.H_mean.clone()
        sigma_H_batch = self.H_sigma.clone()
        
        # Prepare u_samples for encoder
        u_flat = u_samples.squeeze(2)  # [P, M, Lx]
        
        for fn_id in unique_fns:
            # Find observations for this function
            fn_mask = (X_data_fn == fn_id)
            Y_fn = Y_obs[fn_mask]  # [n_obs_fn, M]
            
            if Y_fn.shape[0] == 0:
                continue
            
            # Aggregate observations for this function (mean)
            Y_fn_mean = Y_fn.mean(dim=0)  # [M]
            
            # Expand for P samples
            Y_fn_expanded = Y_fn_mean.unsqueeze(0).expand(self.P, -1)  # [P, M]
            
            # Flatten u_samples for encoder
            u_flat_expanded = u_flat.reshape(self.P, -1)  # [P, M*Lx]
            
            # Concatenate for encoder
            encoder_input = torch.cat([Y_fn_expanded, u_flat_expanded], dim=-1)  # [P, M + M*Lx]
            
            # Encode
            encoder_output = self.encoder_H(encoder_input)  # [P, 2*Q_h]
            
            # Split
            mu_H_fn = encoder_output[:, :self.Q_h]
            log_sigma_H_fn = encoder_output[:, self.Q_h:]
            sigma_H_fn = torch.nn.functional.softplus(log_sigma_H_fn)
            
            # Update H for this function
            mu_H_batch[:, fn_id, :] = mu_H_fn
            sigma_H_batch[:, fn_id, :] = sigma_H_fn
        
        return mu_H_batch, sigma_H_batch
    
    def amortized_inference_step_H(self, Y_batch, batch_indices, u_samples):
        """
        Amortized inference step for H (LVMOGP version of amortized_inference_step).
        
        Similar to parent's amortized_inference_step() but for function latent positions.
        
        Parameters:
        -----------
        Y_batch : [batch_size, M]
        batch_indices : [batch_size]
        u_samples : [P, M, 1, Lx]
        
        Returns:
        --------
        mu_H_batch : [P, num_fns, Q_h]
        sigma_H_batch : [P, num_fns, Q_h]
        """
        # Get function IDs for batch
        X_data_fn_batch = self.X_data_fn[batch_indices]
        
        # Encode H for functions in batch
        mu_H_batch, sigma_H_batch = self.encode_H_for_functions(Y_batch, X_data_fn_batch, u_samples)
        
        # Update stored H values (detached for bookkeeping)
        with torch.no_grad():
            self.H_mean.copy_(mu_H_batch.detach())
            self._H_log_sigma.copy_(torch.log(sigma_H_batch.detach() + 1e-8))
        
        return mu_H_batch, sigma_H_batch
    
    def get_mu_sigma_x_train_samples(self, Y_train_batch, U_samples=None):
        """
        Override parent method to construct X_tilde for LVMOGP.
        
        This constructs X_tilde = [X_data, H[fn_id]] which serves as the
        latent variable input to the GP.
        
        Returns [P, batch_size, Q_total], [P, batch_size, Q_total]
        """
        # Sample U if not provided
        if U_samples is None:
            U_samples = self.get_u_vec_samples()
            U_samples = U_samples.reshape(self.P, self.M, 1, self.Lx)
        
        # Sample H from variational distribution: H = mu_H + sigma_H * eps
        eps = torch.randn_like(self.H_mean)
        H_samples = self.H_mean + self.H_sigma * eps  # [P, num_fns, Q_h]
        
        # This will be called with batch, but we need to handle it properly
        # For now, return dummy values that won't be used
        # The actual X_tilde construction happens in train_step
        batch_size = Y_train_batch.shape[0]
        dummy_mu = torch.zeros(self.P, batch_size, self.Q_total, dtype=torch.float64, device=self.device)
        dummy_sigma = torch.ones(self.P, batch_size, self.Q_total, dtype=torch.float64, device=self.device) * 1e-5
        
        return dummy_mu, dummy_sigma
    
    def get_kl_x_from(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor):
        """
        Override parent's KL divergence computation.
        
        For LVMOGP:
        - X_data is FIXED (no KL needed)
        - Only H has KL divergence: KL[q(H) || p(H)]
        
        X_tilde = [X_data, H], so we only compute KL for the H dimensions.
        
        Parameters:
        -----------
        mu_x : [P, N, Q_total]
            Not used directly, we use self.H_mean
        Sigma_x : [P, N, Q_total]
            Not used directly, we use self.H_sigma
        """
        # KL[q(H) || p(H)] where p(H) = N(0, I)
        # H_mean: [P, num_fns, Q_h]
        # H_sigma: [P, num_fns, Q_h]
        
        # Create distributions (following parent class pattern)
        H_prior = torch.distributions.Normal(
            torch.zeros(self.Q_h, device=self.device), 
            torch.ones(self.Q_h, device=self.device)
        )
        H_variational = torch.distributions.Normal(self.H_mean, self.H_sigma)
        
        # Compute KL divergence using PyTorch's built-in (like parent class)
        # kl_divergence returns [P, num_fns, Q_h]
        kl = torch.distributions.kl_divergence(H_variational, H_prior)
        
        # Sum over all dimensions (num_fns, Q_h) and average over P samples
        kl_total = kl.sum(dim=[1, 2]).mean(dim=0)
        
        return kl_total
    
    def train_step(self, Y_batch, batch_indices, U_samples=None):
        """
        Training step for LVMOGP with proper SSVI structure.
        
        Following the pattern from oil_flow_GPLVM_SSVI_AM_INF.py train_step().
        
        Parameters:
        -----------
        Y_batch : [batch_size, M]
            Batch of observations
        batch_indices : [batch_size]
            Indices of observations in batch
        U_samples : [P, M, 1, Lx]
            Samples from q(u)
        """
        # Convert to float64
        Y_batch = Y_batch.to(dtype=torch.float64)
        
        # Zero gradients for local optimizer (encoder)
        self.optimizer_local.zero_grad()
        
        # Sample U if not provided
        if U_samples is None:
            U_samples = self.get_u_vec_samples()
            U_samples = U_samples.reshape(self.P, self.M, 1, self.Lx)
        
        # Amortized inference step: encode observations -> infer H
        mu_H_batch, sigma_H_batch = self.amortized_inference_step_H(Y_batch, batch_indices, U_samples)
        
        # Sample H: H = mu_H + sigma_H * eps
        eps = torch.randn_like(mu_H_batch)
        H_samples = mu_H_batch + sigma_H_batch * eps  # [P, num_fns, Q_h]
        
        # Construct X_tilde for batch
        X_batch = self.X_data[batch_indices]  # [batch_size, Q_x]
        fn_batch = self.X_data_fn[batch_indices]  # [batch_size]
        X_tilde_batch = self.get_X_tilde(X_batch, fn_batch, H_samples)  # [P, batch_size, Q_total]
        
        # Construct Sigma_x for X_tilde
        # X_data dimensions have tiny variance (fixed), H dimensions have sigma_H^2
        Sigma_x_batch = torch.ones_like(X_tilde_batch) * 1e-5
        H_sigma_for_batch = sigma_H_batch[:, fn_batch, :]  # [P, batch_size, Q_h]
        Sigma_x_batch[:, :, self.Q_x:] = H_sigma_for_batch ** 2
        
        # Zero gradients for global optimizer
        self.optimizer_global.zero_grad()
        
        # Handle missing data
        missing_mask = torch.isnan(Y_batch)
        if missing_mask.any():
            Y_clean = torch.where(missing_mask, torch.full_like(Y_batch, -9.0), Y_batch)
        else:
            Y_clean = Y_batch
            missing_mask = None
        
        # Compute ELBO using parent's ssvi_elbo with X_tilde
        elbo = self.ssvi_elbo(
            Y_clean,
            X_tilde_batch,
            Sigma_x_batch,
            U_samples,
            batch_size=Y_batch.shape[0],
            total_size=self.N,
            missing_mask=missing_mask
        )
        
        loss = -elbo
        
        # Backward
        loss.backward()
        
        # Update parameters
        self.optimizer_local.step()
        self.optimizer_global.step()
        
        return loss.item()
    
    def predict(self, X_new, X_new_fn, return_variance=True):
        """
        Make predictions at new locations.
        
        Parameters:
        -----------
        X_new : torch.Tensor [N_new, Q_x]
            New observed features
        X_new_fn : torch.Tensor [N_new]
            Function IDs for new locations
        return_variance : bool
            Whether to return variance
            
        Returns:
        --------
        mu : [N_new, M]
        var : [N_new, M] (if return_variance=True)
        """
        self.eval()
        
        with torch.no_grad():
            # Use mean H (average over P samples)
            H_mean_avg = self.H_mean.mean(dim=0)  # [num_fns, Q_h]
            H_mean_expanded = H_mean_avg.unsqueeze(0).expand(self.P, -1, -1)  # [P, num_fns, Q_h]
            
            # Construct X_tilde for predictions
            X_tilde_new = self.get_X_tilde(X_new, X_new_fn, H_mean_expanded)  # [P, N_new, Q_total]
            
            # Average over P samples for mean prediction
            X_tilde_mean = X_tilde_new.mean(dim=0)  # [N_new, Q_total]
            
            mu_pred = self.get_ssvi_mean_prediction(X_tilde_mean)  # [N_new, M]
            
            if return_variance:
                # Compute predictive variance
                noise_var = self.sigma2.item()
                
                # Kernel variance
                K_xx = self.gp.kernel_x(X_tilde_mean, X_tilde_mean)  # [M, N_new, N_new]
                signal_var = torch.diagonal(K_xx[0], dim1=0, dim2=1)  # [N_new]
                
                var_pred = noise_var + signal_var.unsqueeze(-1).expand(-1, self.M)  # [N_new, M]
                
                return mu_pred, var_pred
            else:
                return mu_pred
    
    def get_model_info(self):
        """Get model information"""
        return {
            'N': self.N,
            'Q_x': self.Q_x,
            'Q_h': self.Q_h,
            'Q_total': self.Q_total,
            'num_fns': self.num_fns,
            'M': self.M,
            'L': self.Lx,
            'P': self.P,
            'sigma2': self.sigma2.item(),
            'beta': self.gp.beta.detach().cpu().numpy(),
            'sigma2_s': self.gp.sigma2_s.detach().cpu().numpy()
        }
