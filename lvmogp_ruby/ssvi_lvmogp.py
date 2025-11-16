"""
LVMOGP with SSVI and Neural Processes inspired Permutation-Invariant Encoder

Structure:
- X_data: Fixed observed input features [N, Q_x]
- X_data_fn: Function IDs [N]
- H: Learned function-level latent positions [num_fns, Q_h]
- X_tilde = [X_data, H[fn_id]]: Full GP input [N, Q_x + Q_h]
- Uses SSVI with amortized inference for efficient learning

Mathematical Equivalence to LVMOGP.py:
1. fill_Hs(): Constructs X_tilde = [X_data, H[fn_id]]
2. elbo(): ELBO and KL[q(H) || p(H)]
3. predict_f(): Posterior predictions with proper variance
4. KL divergence only on H (X_data is fixed)

Neural Processes-Inspired Encoder (Permutation Invariant):

Stage 1 - Individual Encoder h:
  Input: (X, Y) observation pairs + U samples
  Output: representation r_i for each observation
  h: (x_i, y_i) -> r_i
  
Stage 2 - Aggregation for permutation invariance:
  Operation: Mean pooling across observations by fn_id
  r = mean(r_1, r_2, ..., r_n) for all observations of function
  Key property: Order of observations doesn't affect result
  Location: Inside encode_local_params() method
  
Stage 3 - Set Encoder g:
  Input: aggregated representation r
  Output: H parameters (mu_H, sigma_H) for function
  g: r -> (mu_H, sigma_H)

This ensures that:
- Different orderings of observations yield identical H distributions
- The encoder is a universal approximator for set functions
- Follows the Neural Processes framework for modeling distributions over functions
"""

import torch
import torch.nn as nn
import numpy as np
from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF


class LVMOGP_SSVI(GPLVM_SSVI_AM_INF):
    """
    Latent Variable Multi-Output GP with SSVI and Amortized Inference
    
    Mathematical Structure (following LVMOGP.py):
    - X_data: [N, Q_x] - Fixed observed input features (e.g., x locations)
    - X_data_fn: [N] - Function IDs (which surface/task)
    - H: [num_fns, Q_h] - Learned function latent positions
    - X_tilde = [X_data, H[fn_id]]: [N, Q_x + Q_h] - Full GP input
    
    Inherits from GPLVM_SSVI_AM_INF to use SSVI computational machinery.
    """
    
    def __init__(self, X_data, X_data_fn, num_fns, Q_h, M, Vx,
                 H_data_mean=None, H_data_var=None,
                 sigma2=1, beta=None, sigma2_s=None, P=8,
                 trainable_V=True, encoder_hidden_dims=[128, 64],
                 local_lr=0.001, global_lr=0.001, device="auto",
                 enable_missing_data=False,
                 H_prior_mean=None, H_prior_var=None):
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
        H_data_mean : torch.Tensor [num_fns, Q_h]
            Initial mean of function latent positions (following LVMOGP.py)
            If None, initialized randomly
        H_data_var : torch.Tensor [num_fns, Q_h]
            Initial variance of function latent positions (following LVMOGP.py)
            If None, initialized to small values
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
        H_prior_mean : torch.Tensor [num_fns, Q_h]
            Prior mean for H (default: zeros)
        H_prior_var : torch.Tensor [num_fns, Q_h]
            Prior variance for H (default: ones)
        """
        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        # Parent expects [N, A] and will add P dimension internally
        dummy_mu_x = torch.randn(self.N, self.Q_total, dtype=torch.float64, device=device) * 0.1
        
        # Ensure Vx and beta are correct dimensions
        Vx = Vx.to(device)
        if beta is None:
            beta = [[1.0] * self.Q_total]  # One set of lengthscales for C=1
        
        # Initialize parent (GPLVM_SSVI_AM_INF)
        # This sets up GP machinery, encoder, inducing points, q(u), etc.
        super().__init__(
            Vx=Vx,
            M=M,
            mu_x=dummy_mu_x,
            sigma2=sigma2,
            beta=beta,
            sigma2_s=sigma2_s,
            P=P,
            trainable_V=trainable_V,
            encoder_hidden_dims=encoder_hidden_dims,
            local_lr=local_lr,
            global_lr=global_lr,
            device=device,
            enable_missing_data=enable_missing_data
        )
        
        # Store dimensions for LVMOGP
        self.num_fns = num_fns
        self.Q_h = Q_h
        self.Q_x = X_data.shape[1]
        
        # Initialize H_data_mean and H_data_var with P dimension (like parent's mu_x_train)
        # Shape: [P, num_fns, Q_h] to store P samples per function
        if H_data_mean is None:
            # Initialize randomly (small values)
            H_data_mean = torch.randn(P, num_fns, Q_h, dtype=torch.float64, device=device) * 0.1
        else:
            H_data_mean = H_data_mean.to(device)
            # If provided without P dimension, expand it
            if H_data_mean.ndim == 2:
                H_data_mean = H_data_mean.unsqueeze(0).expand(P, -1, -1).clone()
        
        if H_data_var is None:
            # Initialize to small positive values
            H_data_var = torch.ones(P, num_fns, Q_h, dtype=torch.float64, device=device) * 0.1
        else:
            H_data_var = H_data_var.to(device)
            # If provided without P dimension, expand it
            if H_data_var.ndim == 2:
                H_data_var = H_data_var.unsqueeze(0).expand(P, -1, -1).clone()
        
        # Store as Parameters (not optimized, updated via encoder like parent's mu_x_train)
        self.H_data_mean = nn.Parameter(H_data_mean, requires_grad=False)
        self.H_data_var = nn.Parameter(H_data_var, requires_grad=False)
        
        # Setup H prior (following LVMOGP.py)
        if H_prior_mean is None:
            H_prior_mean = torch.zeros(num_fns, Q_h, dtype=torch.float64, device=device)
        if H_prior_var is None:
            H_prior_var = torch.ones(num_fns, Q_h, dtype=torch.float64, device=device)
        
        self.H_prior_mean = H_prior_mean.to(device)
        self.H_prior_var = H_prior_var.to(device)
        
        # Override parent's encoder to only output H parameters (not full X_tilde)
        # Parent encoder outputs 2*A where A=Q_total, but we only need 2*Q_h
        self.encoder = self._build_encoder_for_H(encoder_hidden_dims)
        
        # Add individual_encoder parameters to the existing local optimizer
        self.optimizer_local.add_param_group({'params': list(self.individual_encoder.parameters())})
    
    @property
    def H_data_var_positive(self):
        """Ensure H_data_var is positive (following LVMOGP.py's transform=positive())"""
        return torch.abs(self.H_data_var) + 1e-6
    
    def _build_encoder_for_H(self, hidden_dims):
        """
        Build permutation-invariant encoder for H following Neural Processes architecture.
        
        Two-stage architecture:
        1. Individual encoder: (X, Y) -> representation r_i
        2. Aggregation + set encoder: mean(r_i) -> H parameters
        
        This ensures permutation invariance: order of observations doesn't matter.
        
        Input: X [Q_x] + y [M] + u_samples [M * Lx]
        Output: mu_H [Q_h] + log_sigma_H [Q_h] = 2*Q_h
        """
        # Stage 1: Individual observation encoder
        # Input: X [Q_x] + Y [M] + u_samples [M * Lx]
        individual_input_dim = self.Q_x + self.M + self.M * self.Lx
        
        # Hidden representation dimension
        repr_dim = hidden_dims[0] if hidden_dims else 128
        
        individual_layers = []
        prev_dim = individual_input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            individual_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        # Output representation r_i
        individual_layers.append(nn.Linear(prev_dim, repr_dim))
        
        self.individual_encoder = nn.Sequential(*individual_layers).to(self.device)
        
        # Stage 2: Set encoder (after aggregation)
        # Input: aggregated representation [repr_dim]
        # Output: mu_H [Q_h] + log_sigma_H [Q_h]
        set_layers = []
        prev_dim = repr_dim
        for hidden_dim in hidden_dims:
            set_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        set_layers.append(nn.Linear(prev_dim, 2 * self.Q_h))
        
        return nn.Sequential(*set_layers).to(self.device)
    
    def fill_Hs(self, X_data=None, X_data_fn=None, H_mean=None, H_var=None):
        """
        Construct X_tilde = [X_data, H[fn_id]] following LVMOGP.py
        
        This is the core operation that constructs the full latent input by
        appending function-specific H values to the fixed X_data.
        
        Parameters:
        -----------
        X_data : [N, Q_x] or [batch_size, Q_x]
            Fixed observed features
        X_data_fn : [N] or [batch_size]
            Function IDs
        H_mean : [P, num_fns, Q_h]
            Mean of H distribution (from encoder)
        H_var : [P, num_fns, Q_h]
            Variance of H distribution (from encoder)
            
        Returns:
        --------
        X_tilde_mean : [P, batch_size, Q_total]
        X_tilde_var : [P, batch_size, Q_total]
        """
        if X_data is None:
            X_data = self.X_data
        if X_data_fn is None:
            X_data_fn = self.X_data_fn
        if H_mean is None or H_var is None:
            raise ValueError("H_mean and H_var must be provided")
        
        batch_size = X_data.shape[0]

        X_data_fn_flat = X_data_fn.flatten()  # Ensure 1D shape [batch_size]
        
        # Gather H for each observation based on function ID
        # H_mean: [P, num_fns, Q_h], X_data_fn_flat: [batch_size]
        H_for_obs_mean = torch.stack([H_mean[p, X_data_fn_flat, :] for p in range(self.P)], dim=0)  # [P, batch_size, Q_h]
        H_for_obs_var = torch.stack([H_var[p, X_data_fn_flat, :] for p in range(self.P)], dim=0)  # [P, batch_size, Q_h]
        
        # Expand X_data for all P samples
        X_data_expanded = X_data.unsqueeze(0).expand(self.P, -1, -1)  # [P, batch_size, Q_x]
        
        # X_data is fixed, so variance is tiny
        X_var_expanded = torch.ones(X_data_expanded.shape, device=self.device, dtype=X_data_expanded.dtype) * 1e-5
        
        # Concatenate [X_data, H]
        X_tilde_mean = torch.cat([X_data_expanded, H_for_obs_mean], dim=-1)  # [P, batch_size, Q_total]
        X_tilde_var = torch.cat([X_var_expanded, H_for_obs_var], dim=-1)  # [P, batch_size, Q_total]
        
        return X_tilde_mean, X_tilde_var
    
    def encode_local_params(self, y_n, u_samples, x_n, fn_ids):
        """
        Permutation-invariant encoding following Neural Processes architecture.
        
        Three-stage process (following Figure 1b of Neural Processes paper):
        1. Individual encoder h: encode each (X, Y) observation to representation r_i
        2. Aggregator a: mean pooling by fn_id (PERMUTATION INVARIANT)
        3. Set encoder g: map aggregated representation to H parameters
        
        This ensures permutation invariance: order of observations doesn't affect result.
        
        Args:
            y_n: [n_obs, M] - observations for a single function
            u_samples: [P, M, C, Lx] - samples from global U distribution (C=1)
            x_n: [n_obs, Q_x] - fixed X_data for observations
            fn_ids: [n_obs] - function IDs (should all be same for single function)
            
        Returns:
            H_mean: [P, Q_h] - H mean for this function
            H_var: [P, Q_h] - H variance for this function
        """
        n_obs = y_n.shape[0]
        
        # Expand y_n and x_n for P samples
        y_n_expanded = y_n.unsqueeze(0).expand(self.P, -1, -1)  # [P, n_obs, M]
        x_n_expanded = x_n.unsqueeze(0).expand(self.P, -1, -1)  # [P, n_obs, Q_x]
        
        # Prepare u_samples for encoder
        u_samples_flat = u_samples.squeeze(2)  # Remove C=1 dimension: [P, M, Lx]
        u_samples_flat = u_samples_flat.unsqueeze(1).expand(-1, n_obs, -1, -1)  # [P, n_obs, M, Lx]
        u_samples_flat = u_samples_flat.reshape(self.P, n_obs, -1)  # [P, n_obs, M*Lx]
        
        # Stage 1: Individual encoder h - encode each (X, Y) observation to r_i
        # h: (x_i, y_i) -> r_i
        individual_input = torch.cat([x_n_expanded, y_n_expanded, u_samples_flat], dim=-1)  
        # [P, n_obs, Q_x + M + M*Lx]
        
        representations = self.individual_encoder(individual_input)  # [P, n_obs, repr_dim]
        
        # Stage 2: Aggregator a - mean pooling (PERMUTATION INVARIANT)
        # a: {r_1, ..., r_n} -> r
        # This is the key to permutation invariance from Neural Processes paper
        aggregated_repr = representations.mean(dim=1)  # [P, repr_dim]
        
        # Stage 3: Set encoder g - map aggregated representation to H parameters
        # g: r -> (mu_H, sigma_H)
        encoder_output = self.encoder(aggregated_repr)  # [P, 2*Q_h]
        
        # Split into mu_H and log_sigma_H
        mu_H = encoder_output[:, :self.Q_h]  # [P, Q_h]
        log_sigma_H = encoder_output[:, self.Q_h:]  # [P, Q_h]
        
        # Convert log_sigma to sigma and compute variance
        sigma_H = torch.nn.functional.softplus(log_sigma_H)  # [P, Q_h]
        var_H = sigma_H ** 2  # [P, Q_h]
        
        return mu_H, var_H
    
    def get_mu_sigma_x_train_samples(self, Y_train_batch, U_samples=None, X_train_batch=None, fn_ids=None):
        """
        Override parent's method - not actively used in LVMOGP.
        
        For LVMOGP, we use infer_H_for_batch() directly in train_step.
        This method is kept for compatibility but properly implements per-function encoding.
        
        Args:
            Y_train_batch: [batch_size, M]
            U_samples: [P, M, C, Lx]
            X_train_batch: [batch_size, Q_x]
            fn_ids: [batch_size] - function IDs
            
        Returns:
            H_mean_batch: [P, batch_size, Q_h]
            H_var_batch: [P, batch_size, Q_h]
        """
        if X_train_batch is None or fn_ids is None:
            raise ValueError("X_train_batch and fn_ids must be provided for LVMOGP")
        
        batch_size = Y_train_batch.shape[0]
        
        # Initialize output tensors
        H_mean_batch = torch.zeros(self.P, batch_size, self.Q_h, 
                                    dtype=torch.float64, device=self.device)
        H_var_batch = torch.zeros(self.P, batch_size, self.Q_h,
                                   dtype=torch.float64, device=self.device)
        
        # Aggregate by function ID and encode each function separately
        unique_fns = torch.unique(fn_ids)
        
        for fn_id in unique_fns:
            # Get all observations for this function in batch
            fn_mask = (fn_ids == fn_id)
            Y_fn = Y_train_batch[fn_mask]  # [n_obs_fn, M]
            X_fn = X_train_batch[fn_mask]  # [n_obs_fn, Q_x]
            fn_ids_fn = fn_ids[fn_mask]  # [n_obs_fn]
            
            # Ensure proper 2D shape even if dimensions are 1
            # This is needed because boolean indexing can squeeze singleton dimensions
            if Y_fn.ndim == 1:
                Y_fn = Y_fn.unsqueeze(-1)  # [n_obs_fn] -> [n_obs_fn, 1]
            if X_fn.ndim == 1:
                X_fn = X_fn.unsqueeze(-1)  # [n_obs_fn] -> [n_obs_fn, 1]
            
            if Y_fn.shape[0] == 0:
                continue
            
            # Encode this function (with aggregation inside encode_local_params)
            mu_H_fn, var_H_fn = self.encode_local_params(Y_fn, U_samples, X_fn, fn_ids_fn)
            # Returns [P, Q_h] for this function
            
            # Broadcast to all observations of this function in batch
            H_mean_batch[:, fn_mask, :] = mu_H_fn.unsqueeze(1)  # [P, 1, Q_h] -> [P, n_obs_fn, Q_h]
            H_var_batch[:, fn_mask, :] = var_H_fn.unsqueeze(1)  # [P, 1, Q_h] -> [P, n_obs_fn, Q_h]
        
        return H_mean_batch, H_var_batch
    
    def infer_H_for_batch(self, Y_batch, batch_indices, U_samples):
        """
        Orchestrate Neural Processes encoding for all functions in batch.
        
        For each function:
        1. Gather all observations for that function
        2. Call encode_local_params() which does the full NP pipeline:
           - Stage 1 (h): Individual encoding
           - Stage 2 (a): Aggregation by mean (permutation invariant)
           - Stage 3 (g): Set encoding to H parameters
        
        Parameters:
        -----------
        Y_batch : [batch_size, M]
            Batch of observations
        batch_indices : [batch_size]
            Indices of observations
        U_samples : [P, M, C, Lx]
            Samples from q(u)
            
        Returns:
        --------
        X_tilde_mean : [P, batch_size, Q_total]
        X_tilde_var : [P, batch_size, Q_total]
        """
        # Get function IDs and X_data for batch
        X_data_fn_batch = self.X_data_fn[batch_indices]
        X_batch = self.X_data[batch_indices]  # [batch_size, Q_x]
        
        # Initialize H mean and var for ALL functions from stored values
        H_mean_all = self.H_data_mean.clone()  # [P, num_fns, Q_h]
        H_var_all = self.H_data_var.clone()  # [P, num_fns, Q_h]
        
        # For each function in batch, apply Neural Processes encoding
        unique_fns = torch.unique(X_data_fn_batch)
        
        for fn_id in unique_fns:
            # Get all observations for this function in batch
            fn_mask = (X_data_fn_batch == fn_id)
            Y_fn = Y_batch[fn_mask]  # [n_obs_fn, M]
            X_fn = X_batch[fn_mask]  # [n_obs_fn, Q_x]
            fn_ids = X_data_fn_batch[fn_mask]  # [n_obs_fn]
            
            # Ensure proper 2D shape even if dimensions are 1
            # This is needed because boolean indexing can squeeze singleton dimensions
            if Y_fn.ndim == 1:
                Y_fn = Y_fn.unsqueeze(-1)  # [n_obs_fn] -> [n_obs_fn, 1]
            if X_fn.ndim == 1:
                X_fn = X_fn.unsqueeze(-1)  # [n_obs_fn] -> [n_obs_fn, 1]
            
            if Y_fn.shape[0] == 0:
                continue
            
            # Apply full Neural Processes encoding (3 stages inside encode_local_params)
            mu_H_fn, var_H_fn = self.encode_local_params(Y_fn, U_samples, X_fn, fn_ids)
            # Returns [P, Q_h] for both
            
            # Store for this function
            H_mean_all[:, fn_id, :] = mu_H_fn
            H_var_all[:, fn_id, :] = var_H_fn
            
            # Update stored H_data_mean and H_data_var
            with torch.no_grad():
                self.H_data_mean[:, fn_id, :] = mu_H_fn.detach()
                self.H_data_var[:, fn_id, :] = var_H_fn.detach()
        
        # Construct X_tilde for batch using fill_Hs()
        X_tilde_mean, X_tilde_var = self.fill_Hs(X_batch, X_data_fn_batch, H_mean_all, H_var_all)
        
        return X_tilde_mean, X_tilde_var
    
    def get_kl_x_from(self, mu_x: torch.Tensor, Sigma_x: torch.Tensor):
        """
        Override parent's KL divergence computation for LVMOGP.
        
        For LVMOGP:
        - X_data is fixed (no KL needed)
        - Only H has KL divergence: KL[q(H) || p(H)]
        
        Parameters:
        -----------
        mu_x : [P, batch_size, Q_total]
            X_tilde mean (includes both X_data and H)
        Sigma_x : [P, batch_size, Q_total]
            X_tilde variance (includes both X_data and H)
        """
        # Extract H portion (last Q_h dimensions) from X_tilde
        H_mean = mu_x[..., self.Q_x:]  # [P, batch_size, Q_h]
        H_var = Sigma_x[..., self.Q_x:]  # [P, batch_size, Q_h]
        
        # Prior: N(0, 1) - same shape as H (on correct device)
        prior = torch.distributions.Normal(
            torch.zeros(self.Q_h, device=self.device, dtype=torch.float64), 
            torch.ones(self.Q_h, device=self.device, dtype=torch.float64)
        )
        
        # Variational: N(H_mean, sqrt(H_var)) with shape [P, batch_size, Q_h]
        variational = torch.distributions.Normal(H_mean, torch.sqrt(H_var))
        
        # Compute KL divergence following parent class pattern:
        # KL for each sample, then average over P, then sum over batch_size * Q_h
        KL = torch.distributions.kl_divergence(variational, prior).mean(dim=0).sum()
        
        return KL
    
    def train_step(self, Y_batch, batch_indices, U_samples=None):
        """
        Training step for LVMOGP following LVMOGP.py mathematical structure.
        
        Uses GPLVM_SSVI_AM_INF's computational machinery but with LVMOGP structure:
        1. Infer H parameters from Y using encoder
        2. Construct X_tilde = [X_data, H[fn_id]] via fill_Hs()
        3. Compute ELBO using parent's ssvi_elbo with X_tilde
        
        Parameters:
        -----------
        Y_batch : [batch_size, M]
            Batch of observations
        batch_indices : [batch_size]
            Indices of observations in batch
        U_samples : [P, M, C, Lx]
            Samples from q(u)
            
        Returns:
        --------
        loss : float
            Negative ELBO (loss value)
        """
        # Convert to float64
        Y_batch = Y_batch.to(dtype=torch.float64)
        
        # Zero gradients
        self.optimizer_local.zero_grad()
        self.optimizer_global.zero_grad()
        
        # Sample U if not provided
        if U_samples is None:
            U_samples = self.get_u_samples()  # Returns [P, M, C, Lx] directly
        
        # Amortized inference: infer H and construct X_tilde
        X_tilde_mean, X_tilde_var = self.infer_H_for_batch(Y_batch, batch_indices, U_samples)
        
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
            X_tilde_mean,
            X_tilde_var,
            U_samples,
            batch_size=Y_batch.shape[0],
            total_size=self.N,
            missing_mask=missing_mask
        )
        
        loss = -elbo
        
        # Backward and update
        loss.backward()
        self.optimizer_local.step()
        self.optimizer_global.step()
        
        return loss.item()
    
    def predict_f(self, X_new, X_new_fn, full_cov=False, full_output_cov=False):
        """
        Posterior prediction following LVMOGP.py predict_f() structure.
        
        Note: This is the LVMOGP equivalent of LVMOGP.py's predict_f() method.
        Uses SSVI machinery but follows LVMOGP.py's mathematical structure.
        
        Parameters:
        -----------
        X_new : torch.Tensor [N_new, Q_x]
            New observed features
        X_new_fn : torch.Tensor [N_new]
            Function IDs for new locations
        full_cov : bool
            Whether to return full covariance
        full_output_cov : bool
            Not implemented (same as LVMOGP.py)
            
        Returns:
        --------
        mean : [N_new, M]
        var : [N_new, M] or [N_new, N_new, M] if full_cov
        """
        if full_output_cov:
            raise NotImplementedError
        
        self.eval()
        
        with torch.no_grad():
            # Use stored H_data_mean and H_data_var (learned during training)
            # H_data_mean and H_data_var already have shape [P, num_fns, Q_h]
            H_mean_all = self.H_data_mean
            H_var_all = self.H_data_var
            
            # Construct X_tilde for predictions
            X_tilde_mean, X_tilde_var = self.fill_Hs(X_new, X_new_fn, H_mean_all, H_var_all)
            # X_tilde_mean: [P, N_new, Q_total], X_tilde_var: [P, N_new, Q_total]
            
            # Sample U from q(u) for prediction
            U_samples = self.get_u_samples()  # [P, M, C, Lx]
            
            # Get GP expectation - use proper GP method
            # For LVMOGP: C=1 (single output component)
            f_mean = self.gp.get_gp_expectation_given_U(X_tilde_mean, U_samples)  # [P, N_new, M, 1]
            mu_pred = f_mean[..., 0].mean(dim=0)  # [N_new, M]

            #gps.py question regarding "A" variable in get_gp_covariance and get_gp_covariance_given_U

            if full_cov:
                # Full covariance prediction (not implemented for simplicity)
                raise NotImplementedError("Full covariance prediction not implemented")
            else:
                # Get proper GP predictive variance using GP covariance method
                gp_cov = self.gp.get_gp_covariance_given_U(X_tilde_mean, U_samples)  # [P, N_new, M, 1, 1]
                
                # Extract diagonal: [P, N_new, M, 1, 1] -> [P, N_new, M]
                gp_var = gp_cov[..., 0, 0]  # [P, N_new, M]
                # Average over P samples
                gp_var_avg = gp_var.mean(dim=0)  # [N_new, M]
                # Add observation noise
                var_pred = gp_var_avg + self.sigma2
                
                return mu_pred, var_pred
    
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
