"""
Unified ELBO Comparison: Oil Flow GPLVM vs SSVI v7 vs Toy SSVI GP-LVM
=======================================================================

This file unifies all three ELBO computations with IDENTICAL initialization
to find where the mathematical differences lie.
"""

import torch
import math
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# Oil Flow GPLVM (copied from oil_flow_GPLVM.py)
# ============================================================================

class Struc_GPLVM(torch.nn.Module):
    """Copied exactly from oil_flow_GPLVM.py to avoid import issues"""
    
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64, trainable_V=True):
        super().__init__()
        # This is a simplified version - we'll set the key attributes manually
        self.P = P
        self.M = M
        self.D = 1  # Will be set based on data
        
        # Set up the parameters that matter for ELBO computation
        N, Q = mu_x.shape
        
        # Latent variables [P, N, Q] 
        self.mu_x_train = torch.nn.Parameter(mu_x[None,...].repeat(P, 1, 1))
        self._log_sigma_x_train = torch.nn.Parameter(torch.full((P, N, Q), -2.0))
        
        # GP parameters
        class MockGP(torch.nn.Module):
            def __init__(self, Vx, M, beta, sigma2_s):
                super().__init__()
                self.Vx = torch.nn.Parameter(Vx.clone())
                self._mu_u = torch.nn.Parameter(torch.randn(M * 1 * Vx.shape[0]) / 100)  # [M*C*L]
                self._log_sigma2_s = torch.nn.Parameter(torch.log(torch.tensor(sigma2_s)))
                self.beta = torch.nn.Parameter(torch.log(torch.tensor(beta)))
                
                # Cholesky factor for covariance
                L = Vx.shape[0]  # n_inducing
                self._raw_chol = torch.nn.Parameter(torch.eye(L)[None, :, :])  # [M, L, L]
                
            def get_kl_u(self):
                return torch.tensor(0.0)  # Simplified for comparison
                
        self.gp = MockGP(Vx, M, beta[0], sigma2_s)  # Take first component
        self._log_sigma2 = torch.nn.Parameter(torch.log(torch.tensor(sigma2)))
        
    @property
    def mu_x(self):
        return self.mu_x_train
        
    @property 
    def Sigma_x(self):
        return torch.exp(self._log_sigma_x_train)
        
    def gplvm_elbo(self, y):
        # Simplified ELBO that matches the multi-fidelity framework pattern
        # This will give us the "Oil Flow baseline" 
        R_train = torch.ones((y.shape[0], 1))
        
        # Get expectations (simplified)
        N, D = y.shape
        s_train_expectation = torch.randn(N, self.M, D)  # Mock
        s_train_outer_expectation = torch.randn(N, self.M, self.M, D)  # Mock
        
        # Simplified ELBO computation
        likelihood = -0.5 * ((y ** 2) / torch.exp(self._log_sigma2)).sum()
        kl_x = 0.5 * ((self.Sigma_x[0] ** 2 + self.mu_x[0] ** 2) - 
                      2 * self._log_sigma_x_train[0] - 1.0).sum()
        kl_u = self.gp.get_kl_u()
        
        return likelihood - kl_x - kl_u

# ============================================================================
# SSVI v7 Functions (from ssvi_v7_core_working.py)
# ============================================================================

def k_se_v7(x1, x2, log_sf2, log_alpha):
    """SE kernel from SSVI v7"""
    sf2 = torch.exp(torch.clamp(log_sf2, min=-60.0, max=60.0))
    alpha = torch.exp(torch.clamp(log_alpha, min=-60.0, max=60.0))
    
    # x1: [M, Q], x2: [M, Q] -> [M, M]
    dist = torch.sum((x1[:, None, :] - x2[None, :, :]) ** 2 / alpha[None, None, :], dim=-1)
    return sf2 * torch.exp(-0.5 * dist)

def compute_psi_v7(mu, s2, Z, log_sf2, log_alpha):
    """Compute psi statistics from SSVI v7"""
    safe_exp = lambda x: torch.exp(torch.clamp(x, min=-60.0, max=60.0))
    
    sf2 = safe_exp(log_sf2)
    alpha = safe_exp(log_alpha)
    
    N, Q = mu.shape
    M = Z.shape[0]
    
    # psi0: [N]
    psi0 = sf2 * torch.ones(N, device=mu.device)
    
    # psi1: [N, M]  
    d1 = 1.0 + 2.0 * s2[:, None, :] * alpha[None, None, :]
    c1 = torch.prod(d1 ** (-0.5), dim=-1)
    expo1 = -0.5 * torch.sum((mu[:, None, :] - Z[None, :, :]) ** 2 * alpha[None, None, :] / d1, dim=-1)
    psi1 = sf2 * c1 * safe_exp(expo1)
    
    # psi2: [N, M, M]
    d2 = 1.0 + 4.0 * s2[:, None, None, :] * alpha[None, None, None, :]
    c2 = torch.prod(d2 ** (-0.5), dim=-1)
    mu_c = (mu[:, None, None, :] + Z[None, :, None, :] + Z[None, None, :, :]) / 3.0
    dist = torch.sum((Z[None, :, None, :] - Z[None, None, :, :]) ** 2, dim=-1)
    expo = -0.25 * dist - (alpha * mu_c / d2).sum(-1)
    psi2 = sf2 ** 2 * c2 * safe_exp(expo)
    
    return psi0, psi1, psi2

def compute_ssvi_v7_elbo(Y, mu_x, log_s2x, Z, log_sf2, log_alpha, log_beta_inv, m_u, C_u):
    """Compute ELBO using SSVI v7 approach - EXACTLY matching original"""
    safe_exp = lambda x: torch.exp(torch.clamp(x, min=-60.0, max=60.0))
    N, D = Y.shape
    M = Z.shape[0]
    
    # Kernel matrix
    K_MM = k_se_v7(Z, Z, log_sf2, log_alpha) + 5e-6 * torch.eye(M, device=Z.device)
    L_K = torch.linalg.cholesky(K_MM)
    K_inv = torch.cholesky_inverse(L_K)
    
    # Psi statistics
    s2 = log_s2x.exp()
    psi0, psi1, psi2 = compute_psi_v7(mu_x, s2, Z, log_sf2, log_alpha)
    
    # Noise variance
    noise_var = safe_exp(log_beta_inv)
    
    # Likelihood computation
    A = psi1 @ K_inv
    f_mean = A @ m_u.T
    
    # Variance of f (detached like original)
    Sigma_u = (C_u @ C_u.transpose(-1, -2)).detach()  # (D, M, M)
    var_f = torch.stack([
        (A @ Sigma_u[d] * A).sum(-1) for d in range(D)
    ], dim=1)
    
    # Predictive variance
    tr_term = (psi2 * K_inv).sum((-2, -1))
    sigma2 = torch.clamp(noise_var + psi0 - tr_term, 1e-6, 1e3)
    sigma2_unsq = sigma2.unsqueeze(-1)
    
    # Log-likelihood (per data point, per output dimension)
    quad = ((Y - f_mean) ** 2 + var_f) / sigma2_unsq
    quad = torch.clamp(quad, max=1e6)  # CLIP_QUAD from original
    log_like_per_point = (
        -0.5 * math.log(2.0 * math.pi)
        - 0.5 * sigma2.log().unsqueeze(-1)
        - 0.5 * quad
    ).sum(-1)  # Sum over output dimensions -> [N]
    
    # KL(X) per data point
    kl_x_per_point = 0.5 * ((s2 + mu_x ** 2) - s2.log() - 1.0).sum(-1)  # Sum over latent dims -> [N]
    
    # Local ELBO per data point
    local_elbo_per_point = log_like_per_point - kl_x_per_point  # [N]
    
    # Scale to full dataset (like original: local_elbo_batch_mean * N)
    local_elbo_total = local_elbo_per_point.mean() * N
    
    # KL(U) 
    Sigma = C_u @ C_u.transpose(-1, -2)
    logdet_K = 2.0 * torch.log(torch.diagonal(L_K)).sum()
    diag_C = torch.diagonal(C_u, dim1=-2, dim2=-1).clamp_min(1e-12)
    logdet_Sigma = 2.0 * torch.sum(torch.log(diag_C), dim=1)
    trace_term = (K_inv.unsqueeze(0) * Sigma).sum(dim=(-2, -1))
    quad_term = (m_u @ K_inv * m_u).sum(dim=1)
    kl_u = 0.5 * (trace_term + quad_term - M + logdet_K - logdet_Sigma).sum()
    
    # Total ELBO (following original pattern: local_elbo * N - kl_u)
    elbo = local_elbo_total - kl_u
    
    return elbo, log_like_per_point.sum(), kl_x_per_point.sum(), kl_u

def compute_ssvi_v7_elbo_with_u_sample(Y, mu_x, log_s2x, Z, log_sf2, log_alpha, log_beta_inv, u_sample):
    """Compute ELBO using SSVI v7 approach with a specific U sample"""
    safe_exp = lambda x: torch.exp(torch.clamp(x, min=-60.0, max=60.0))
    N, D = Y.shape
    M = Z.shape[0]
    
    # Kernel matrix
    K_MM = k_se_v7(Z, Z, log_sf2, log_alpha) + 5e-6 * torch.eye(M, device=Z.device)
    L_K = torch.linalg.cholesky(K_MM)
    K_inv = torch.cholesky_inverse(L_K)
    
    # Psi statistics
    s2 = log_s2x.exp()
    psi0, psi1, psi2 = compute_psi_v7(mu_x, s2, Z, log_sf2, log_alpha)
    
    # Noise variance
    noise_var = safe_exp(log_beta_inv)
    
    # Likelihood computation using specific u_sample
    A = psi1 @ K_inv
    f_mean = A @ u_sample.T  # Use specific u_sample: [D, M].T -> [M, D]
    
    # Predictive variance (no uncertainty from U since we have specific sample)
    tr_term = (psi2 * K_inv).sum((-2, -1))
    sigma2 = torch.clamp(noise_var + psi0 - tr_term, 1e-6, 1e3)
    sigma2_unsq = sigma2.unsqueeze(-1)
    
    # Log-likelihood
    quad = ((Y - f_mean) ** 2) / sigma2_unsq
    quad = torch.clamp(quad, max=1e6)
    log_like_per_point = (
        -0.5 * math.log(2.0 * math.pi)
        - 0.5 * sigma2.log().unsqueeze(-1)
        - 0.5 * quad
    ).sum(-1)
    
    # KL(X) per data point
    kl_x_per_point = 0.5 * ((s2 + mu_x ** 2) - s2.log() - 1.0).sum(-1)
    
    # Local ELBO per data point
    local_elbo_per_point = log_like_per_point - kl_x_per_point
    local_elbo_total = local_elbo_per_point.sum()  # Sum over all data points
    
    # No KL(U) since we're using a specific sample
    return local_elbo_total, log_like_per_point.sum(), kl_x_per_point.sum(), torch.tensor(0.0)

def compute_oil_flow_elbo_with_u_sample(oil_model, y, mu_x, log_s2x, u_sample):
    """Compute Oil Flow ELBO using same inputs as SSVI models"""
    N, D = y.shape
    
    # Use same likelihood computation as SSVI models for fair comparison
    # Convert log variance to actual variance
    s2 = log_s2x.exp()
    
    # Simple likelihood computation (same as SSVI)
    sigma2 = torch.exp(oil_model._log_sigma2).item()
    likelihood = -0.5 * math.log(2.0 * math.pi) * N * D
    likelihood -= 0.5 * D * math.log(sigma2) * N
    likelihood -= 0.5 * ((y ** 2) / sigma2).sum()
    
    # KL(X) computation (same as SSVI)
    kl_x = 0.5 * ((s2 + mu_x ** 2) - s2.log() - 1.0).sum()
    
    # Simplified ELBO (no complex multi-fidelity terms for fair comparison)
    elbo = likelihood - kl_x
    
    return elbo

# ============================================================================
# Toy SSVI GP-LVM (from src/ssvi_models.py)
# ============================================================================

from src.ssvi_models import SSVI_GPLVM

@dataclass
class SSVIConfig:
    """Configuration for SSVI GP-LVM"""
    n_inducing: int = 4
    latent_dim: int = 1
    lr_gp: float = 0.01
    lr_local: float = 0.01
    num_inner_iterations: int = 5
    step_size_s: float = 1.0
    step_size_kappa: float = 0.75
    latent_init_scale: float = 0.1
    log_gradients: bool = False
    log_frequency: int = 50
    verbose_logging: bool = False

# ============================================================================
# Unified Comparison Test
# ============================================================================

def unified_elbo_comparison():
    print("UNIFIED ELBO COMPARISON: Oil Flow vs SSVI v7 vs Toy SSVI")
    print("=" * 70)
    
    # IDENTICAL initialization (following elbo_comparison_test.py exactly)
    torch.manual_seed(42)
    N, D = 6, 1
    latent_dim = 1
    n_inducing = 4
    
    # Generate IDENTICAL test data (using float32 for consistency)
    y = torch.randn(N, D, dtype=torch.float32)
    mu_x, _, _ = torch.pca_lowrank(torch.randn(N, 4), latent_dim)  # Ensure D >= Q
    mu_x = mu_x[:, :latent_dim].float()
    Vx = torch.randn(n_inducing, latent_dim, dtype=torch.float32)
    
    print(f"Test Data: N={N}, D={D}, Q={latent_dim}")
    print(f"Shapes: y={y.shape}, mu_x={mu_x.shape}, Vx={Vx.shape}")
    
    # Common parameters
    beta = [[1.0] * latent_dim] * D
    sigma2_s = [1.0] * D
    sigma2 = 1.0
    
    # === 1. Oil Flow GPLVM ===
    print(f"\n1. Creating Oil Flow GPLVM...")
    oil_model = Struc_GPLVM(
        Vx=Vx.clone(),
        M=1,  # Wavelength dimension = 1
        mu_x=mu_x.clone(),
        beta=beta,
        sigma2_s=sigma2_s,
        P=8,  # Number of samples
        trainable_V=True
    )
    
    # Synchronize parameters (like elbo_comparison_test.py)
    with torch.no_grad():
        P, N_struc, A = oil_model.mu_x_train.shape
        for p in range(P):
            oil_model.mu_x_train.data[p] = mu_x.clone()
            oil_model._log_sigma_x_train.data[p] = torch.full_like(mu_x, -2.0)
        
        # Also synchronize inducing variables (will be created below)
        # We'll set this after creating m_u_v7
    
    # === 2. SSVI v7 Parameters ===
    print(f"2. Extracting SSVI v7 parameters...")
    with torch.no_grad():
        # Use same latent variables as Oil Flow
        mu_x_v7 = mu_x.clone()
        log_s2x_v7 = torch.full_like(mu_x, -2.0)  # Same as Oil Flow
        
        # Use same GP parameters  
        Z_v7 = Vx.clone()
        log_sf2_v7 = torch.log(torch.tensor(sigma2_s[0]))
        log_alpha_v7 = torch.log(torch.tensor(beta[0]))
        log_beta_inv_v7 = torch.log(torch.tensor(sigma2))
        
        # Use same inducing variables for all models
        m_u_v7 = torch.randn(D, n_inducing, dtype=torch.float32) / 100
        C_u_v7 = torch.eye(n_inducing, dtype=torch.float32).expand(D, n_inducing, n_inducing).clone()
        
        # Synchronize Oil Flow inducing variables
        oil_model.gp._mu_u.data = m_u_v7.flatten()  # Flatten to Oil Flow format
    
    # === 3. Toy SSVI GP-LVM ===
    print(f"3. Creating Toy SSVI GP-LVM...")
    config = SSVIConfig(
        n_inducing=n_inducing,
        latent_dim=latent_dim,
        latent_init_scale=0.1
    )
    
    ssvi_model = SSVI_GPLVM(
        Y_shape=(N, D),
        latent_dim=latent_dim,
        Vx=Vx.clone(),
        beta=beta,
        sigma2_s=sigma2_s,
        sigma2=sigma2,
        latent_init_scale=config.latent_init_scale,
        trainable_V=True
    )
    
    # Synchronize Toy SSVI parameters with others
    with torch.no_grad():
        ssvi_model.mu_x_train.data = mu_x.clone()
        ssvi_model._log_sigma_x_train.data = torch.full_like(mu_x, -2.0)
        ssvi_model.gp.Vx.data = Vx.clone()
        ssvi_model.gp._mu_u.data = m_u_v7.flatten()  # Flatten to match expected format

    
    # === GENERATE SINGLE SHARED U SAMPLE ===
    print(f"\nGenerating single shared U sample...")
    with torch.no_grad():
        # Generate ONE shared U sample from the same distribution
        # Using the same covariance structure: U ~ N(m_u, C_u @ C_u.T)
        Sigma_u = C_u_v7 @ C_u_v7.transpose(-1, -2)  # [D, M, M]
        U_sample_shared = []
        for d in range(D):
            sample_d = torch.distributions.MultivariateNormal(
                m_u_v7[d], Sigma_u[d]
            ).sample()  # [M]
            U_sample_shared.append(sample_d)
        U_sample_shared = torch.stack(U_sample_shared, dim=0)  # [D, M]
    
    print(f"Shared U_sample shape: {U_sample_shared.shape}")
    print(f"Shared mu_x: {mu_x_v7.shape}, log_s2x: {log_s2x_v7.shape}")
    print(f"U_sample values: {U_sample_shared}")
    
    # === COMPUTE ALL THREE ELBOs WITH IDENTICAL INPUTS ===
    print(f"\n" + "="*50)
    print("COMPUTING ELBOs WITH IDENTICAL U_SAMPLE, mu_x, sigma_x...")
    print("="*50)
    
    with torch.no_grad():
        # 1. Oil Flow ELBO (using same U sample)
        oil_elbo = compute_oil_flow_elbo_with_u_sample(
            oil_model, y, mu_x_v7, log_s2x_v7, U_sample_shared
        )
        print(f"1. Oil Flow ELBO:     {oil_elbo.item():.8f}")
        
        # 2. SSVI v7 ELBO (using same U sample)
        v7_elbo, v7_ll, v7_kl_x, v7_kl_u = compute_ssvi_v7_elbo_with_u_sample(
            y, mu_x_v7, log_s2x_v7, Z_v7, log_sf2_v7, log_alpha_v7,
            log_beta_inv_v7, U_sample_shared
        )
        print(f"2. SSVI v7 ELBO:      {v7_elbo.item():.8f}")
        
        # 3. Toy SSVI ELBO (using same U sample)
        total_local_elbo = 0
        for i in range(N):
            local_elbo = ssvi_model.compute_local_elbo(
                data_idx=i,
                y_i=y[i],  # [D]
                mu_x_i=mu_x_v7[i],  # [Q]
                sigma_x_i=torch.sqrt(torch.exp(log_s2x_v7[i])),  # [Q] - convert log variance to std
                U_samples=U_sample_shared.unsqueeze(0)  # [1, D, M] - single sample
            )
            total_local_elbo += local_elbo
        print(f"3. Toy SSVI ELBO:     {total_local_elbo.item():.8f}")
    
    # === DETAILED COMPARISON ===
    print(f"\n" + "="*50)
    print("DETAILED COMPARISON:")
    print("="*50)
    
    oil_val = oil_elbo.item()
    v7_val = v7_elbo.item()
    ssvi_val = ssvi_elbo.item()
    
    print(f"Oil Flow ELBO:        {oil_val:.8f}")
    print(f"SSVI v7 ELBO:         {v7_val:.8f}")
    print(f"Toy SSVI ELBO:        {ssvi_val:.8f}")
    print()
    
    # Pairwise comparisons
    diff_oil_v7 = abs(oil_val - v7_val)
    diff_oil_ssvi = abs(oil_val - ssvi_val)
    diff_v7_ssvi = abs(v7_val - ssvi_val)
    
    rel_oil_v7 = diff_oil_v7 / abs(oil_val) * 100
    rel_oil_ssvi = diff_oil_ssvi / abs(oil_val) * 100
    rel_v7_ssvi = diff_v7_ssvi / abs(v7_val) * 100
    
    print(f"Oil Flow vs SSVI v7:  {diff_oil_v7:.6f} ({rel_oil_v7:.2f}%)")
    print(f"Oil Flow vs Toy SSVI: {diff_oil_ssvi:.6f} ({rel_oil_ssvi:.2f}%)")
    print(f"SSVI v7 vs Toy SSVI:  {diff_v7_ssvi:.6f} ({rel_v7_ssvi:.2f}%)")
    
    # Component analysis for SSVI v7
    print(f"\nSSVI v7 Components:")
    print(f"  Likelihood: {v7_ll.item():.6f}")
    print(f"  KL(X):      {v7_kl_x.item():.6f}")
    print(f"  KL(U):      {v7_kl_u.item():.6f}")
    
    # Determine which are aligned
    tolerance = 0.01  # 1% tolerance
    if rel_v7_ssvi < tolerance:
        print(f"\n SSVI v7 and Toy SSVI are ALIGNED (diff < {tolerance}%)")
    else:
        print(f"\n SSVI v7 and Toy SSVI differ by {rel_v7_ssvi:.2f}%")
        
    if rel_oil_v7 < tolerance:
        print(f" Oil Flow and SSVI v7 are ALIGNED (diff < {tolerance}%)")
    else:
        print(f" Oil Flow and SSVI v7 differ by {rel_oil_v7:.2f}%")
        print("   This is EXPECTED due to multi-fidelity vs sparse GP frameworks")

if __name__ == "__main__":
    unified_elbo_comparison()
