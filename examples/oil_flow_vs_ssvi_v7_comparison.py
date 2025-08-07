"""
Oil Flow GPLVM vs SSVI v7 Comparison

Direct ELBO comparison between Oil Flow GPLVM and SSVI v7 core functions.
"""

import torch
import numpy as np
import math

torch.set_default_dtype(torch.float64)

# ============================================================================
# Oil Flow GPLVM (exact copy from oil_flow_GPLVM.py)
# ============================================================================

from src.struc_models import Struc_WS_GPLVM

class Struc_GPLVM(Struc_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64, trainable_V=True):
        super().__init__(P=P, V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, 
                        C_unobs=0, sigma2=sigma2, weight_distribution="continuous", 
                        beta=beta, sigma2_s=sigma2_s, trainable_V=trainable_V)

    @property 
    def mu_x(self): 
        return self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return self.Sigma_x_train
    
    def gplvm_elbo(self, y):
        R_train = torch.ones((y.shape[0], 1))
        return super(Struc_GPLVM, self).elbo(y, R_train)

# ============================================================================
# SSVI v7 Core Functions (simplified from ssvi_v7_core_working.py)
# ============================================================================

def k_se_v7(x, z, log_sf2_val, log_alpha_val):
    safe_exp = lambda x: torch.exp(torch.clamp(x, min=-60.0, max=60.0))
    sf2 = safe_exp(log_sf2_val)
    alpha = safe_exp(log_alpha_val)
    diff = x.unsqueeze(-2) - z
    return sf2 * safe_exp(-0.5 * (diff ** 2 * alpha).sum(-1))

def compute_psi_v7(mu, s2, Z, log_sf2, log_alpha):
    safe_exp = lambda x: torch.exp(torch.clamp(x, min=-60.0, max=60.0))
    sf2 = safe_exp(log_sf2)
    alpha = safe_exp(log_alpha)
    
    psi0 = torch.full((mu.size(0),), sf2.item(), device=mu.device)
    
    # psi1
    d1 = alpha * s2 + 1.0
    log_c1 = -0.5 * torch.log(d1).sum(-1, keepdim=True)
    c1 = torch.exp(log_c1)
    diff = mu.unsqueeze(1) - Z
    psi1 = sf2 * c1 * safe_exp(-0.5 * ((alpha * diff ** 2) / d1.unsqueeze(1)).sum(-1))
    
    # psi2
    d2 = 1.0 + 2.0 * alpha * s2
    log_c2 = -0.5 * torch.log(d2).sum(-1, keepdim=True)
    c2 = torch.exp(log_c2)
    ZZ = Z.unsqueeze(1) - Z.unsqueeze(0)
    dist = (alpha * ZZ ** 2).sum(-1)
    mid = 0.5 * (Z.unsqueeze(1) + Z.unsqueeze(0))
    mu_c = (mu.unsqueeze(1).unsqueeze(1) - mid) ** 2
    expo = -0.25 * dist - (alpha * mu_c / d2.unsqueeze(1).unsqueeze(1)).sum(-1)
    psi2 = sf2 ** 2 * c2.unsqueeze(-1) * safe_exp(expo)
    
    return psi0, psi1, psi2

def compute_ssvi_v7_elbo(Y, mu_x, log_s2x, Z, log_sf2, log_alpha, log_beta_inv, m_u, C_u):
    """Compute ELBO using SSVI v7 approach"""
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

# ============================================================================
# Main Comparison Test (following elbo_comparison_test.py pattern)
# ============================================================================

def oil_flow_vs_ssvi_v7_test():
    print("ELBO COMPARISON: Oil Flow GPLVM vs SSVI v7")
    print("=" * 50)
    
    # Fixed test data like elbo_comparison_test.py
    torch.manual_seed(42)
    N, D = 6, 1
    latent_dim = 1
    n_inducing = 4
    
    # Create simple test data
    Vx = torch.linspace(-1.5, 1.5, n_inducing).reshape(-1, 1)
    mu_x = torch.linspace(-1.0, 1.0, N).reshape(-1, 1)
    y = torch.sin(2 * mu_x) + 0.1 * torch.randn(N, D)
    
    print(f"Test Data: N={N}, D={D}, shapes: y{y.shape}, mu_x{mu_x.shape}, Vx{Vx.shape}")
    
    # Common parameters
    beta = [[1.0] * latent_dim] * D
    sigma2_s = [1.0] * D
    
    # === Oil Flow GPLVM ===
    print("\nCreating Oil Flow GPLVM...")
    oil_model = Struc_GPLVM(
        Vx=Vx.clone(),
        M=1,  # Wavelength dimension = 1 (no wavelength dependency for GP-LVM)
        mu_x=mu_x.clone(),
        beta=beta,
        sigma2_s=sigma2_s,
        P=8,  # Number of samples for structured model (like working test)
        trainable_V=True
    )
    
    # Synchronize parameters for fair comparison (like elbo_comparison_test.py)
    print("Synchronizing parameters...")
    with torch.no_grad():
        # The Struc model has shape [P, N, A] for latent variables
        # We need to set the FIRST sample to our desired values
        P, N_struc, A = oil_model.mu_x_train.shape
        
        # Set all P samples to the same latent variables for consistency
        for p in range(P):
            oil_model.mu_x_train.data[p] = mu_x.clone()
            oil_model._log_sigma_x_train.data[p] = torch.full_like(mu_x, -2.0)
    
    # Set evaluation mode
    oil_model.eval()
    
    print("Computing ELBOs...")
    
    # Oil Flow ELBO
    with torch.no_grad():
        oil_elbo = oil_model.gplvm_elbo(y)
        print(f"Oil Flow GPLVM ELBO: {oil_elbo.item():.8f}")
        oil_kl_total = oil_model.get_kl().item()
        oil_likelihood = oil_elbo.item() + oil_kl_total
        print(f"Oil Flow Likelihood (approx): {oil_likelihood:.6f}")
        print(f"Oil Flow Total KL:            {oil_kl_total:.6f}")

    # Extract parameters from Struc model directly (NOT from sample)
    with torch.no_grad():
        # Latent variables (first structured sample)
        mu_x_v7 = oil_model.mu_x_train[0].detach()  # shape: [N, Q]
        sigma_x_v7 = oil_model._log_sigma_x_train[0].exp().detach()  # shape: [N, Q]
        log_s2x_v7 = 2 * torch.log(sigma_x_v7)  # log(σ²)

        # GP kernel parameters
        Z_v7 = oil_model.gp.Vx.detach()
        log_sf2_v7 = oil_model.gp._log_sigma2_s.detach().view(-1)[0]  # scalar
        log_alpha_v7 = oil_model.gp.beta.detach().view(-1)  # [Q]
        log_beta_inv_v7 = oil_model._log_sigma2.detach()  # scalar

        # Inducing variables: proper m_u and C_u
        m_u_v7 = oil_model.gp.get_vector_mu_u().reshape(1, n_inducing).detach()  # [D=1, L]
        C_u_raw = torch.tril(oil_model.gp._raw_chol[0]).detach()  # [L, L]
        C_u_v7 = C_u_raw.reshape(1, n_inducing, n_inducing)  # [D=1, L, L]

    # Debug: Print extracted parameters
    print(f"\nDEBUG Parameter Extraction:")
    print(f"mu_x_v7 range: [{mu_x_v7.min():.4f}, {mu_x_v7.max():.4f}]")
    print(f"log_s2x_v7 range: [{log_s2x_v7.min():.4f}, {log_s2x_v7.max():.4f}]")
    print(f"Z_v7 range: [{Z_v7.min():.4f}, {Z_v7.max():.4f}]")
    print(f"log_sf2_v7: {log_sf2_v7.item():.4f}")
    print(f"log_alpha_v7 range: [{log_alpha_v7.min():.4f}, {log_alpha_v7.max():.4f}]")
    print(f"log_beta_inv_v7: {log_beta_inv_v7.item():.4f}")
    print(f"m_u_v7 range: [{m_u_v7.min():.4f}, {m_u_v7.max():.4f}]")
    print(f"C_u_v7 diagonal: {torch.diagonal(C_u_v7[0]).detach()}")

    # SSVI v7 ELBO
    with torch.no_grad():
        v7_elbo, v7_ll, v7_kl_x, v7_kl_u = compute_ssvi_v7_elbo(
            y, mu_x_v7, log_s2x_v7, Z_v7, log_sf2_v7, log_alpha_v7,
            log_beta_inv_v7, m_u_v7, C_u_v7
        )
        print(f"\nSSVI v7 ELBO: {v7_elbo.item():.8f}")
    
    # COMPARISON
    print(f"\n" + "="*50)
    print("COMPARISON RESULTS:")
    print("="*50)
    
    diff = abs(oil_elbo.item() - v7_elbo.item())
    avg = (abs(oil_elbo.item()) + abs(v7_elbo.item())) / 2
    rel_diff = diff / avg * 100 if avg > 1e-10 else float('inf')
    
    print(f"Oil Flow ELBO:    {oil_elbo.item():.8f}")
    print(f"SSVI v7 ELBO:     {v7_elbo.item():.8f}")
    print(f"Absolute Diff:    {diff:.8f}")
    print(f"Relative Diff:    {rel_diff:.4f}%")

    print(f"\nOil Flow Likelihood (approx): {oil_likelihood:.6f}")
    print(f"Oil Flow Total KL:            {oil_kl_total:.6f}")

    print(f"\nSSVI Likelihood: {v7_ll.item():.6f}")
    print(f"SSVI Total KL:   {(v7_kl_x + v7_kl_u).item():.6f}")

    # Success check
    if rel_diff < 5.0:
        print(f"\n SUCCESS: Implementations are consistent (< 5% difference)")
    else:
        print(f"\n SIGNIFICANT DIFFERENCE: May indicate different mathematical approaches")
    
    return oil_elbo.item(), v7_elbo.item(), rel_diff

if __name__ == "__main__":
    oil_flow_vs_ssvi_v7_test()