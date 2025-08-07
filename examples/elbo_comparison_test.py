#!/usr/bin/env python3
"""
ELBO Comparison Test

Direct comparison between SSVI and MF_WS_GPLVM ELBO implementations.
This test validates that our SSVI ELBO computation matches the proven working
MF approach when given identical parameters.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.ssvi_models import SSVI_GPLVM
from src.struc_models import Struc_WS_GPLVM

class Struc_GPLVM(Struc_WS_GPLVM):
    """Exact copy of Struc_GPLVM from oil_flow_GPLVM.py (the proven working implementation)"""
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
        return self.elbo(y, R_train)  # Call our overridden elbo method, not super()
    
    def elbo(self, D_train, R_train, D_test=None):
        """Override elbo to handle P dimension correctly"""
        kernel_train_expectation = self._get_lv_kernel_expectation(self.mu_x_train, self.Sigma_x_train)
        kernel_train_outer_expectation = self._get_lv_kernel_outer_expectation(self.mu_x_train, self.Sigma_x_train)
        s_train_expectation = self.get_s_expectation(kernel_expectation=kernel_train_expectation)
        s_train_outer_expectation = self.get_s_outer_expectation(kernel_outer_expectation=kernel_train_outer_expectation)
        
        # Average over P samples for compatibility with base ELBO
        s_train_expectation = s_train_expectation.mean(dim=0)  # P x N x M x C -> N x M x C
        s_train_outer_expectation = s_train_outer_expectation.mean(dim=0)  # P x N x C x C -> N x C x C
        

        
        if D_test is None: 
            return self._elbo(s_train_expectation, s_train_outer_expectation, D_train, R_train)
        else:
            kernel_test_expectation = self._get_lv_kernel_expectation(self.mu_x_test, self.Sigma_x_test)
            kernel_test_outer_expectation = self._get_lv_kernel_outer_expectation(self.mu_x_test, self.Sigma_x_test)
            s_test_expectation = self.get_s_expectation(kernel_expectation=kernel_test_expectation)
            s_test_outer_expectation = self.get_s_outer_expectation(kernel_outer_expectation=kernel_test_outer_expectation)
            
            # Average over P samples
            s_test_expectation = s_test_expectation.mean(dim=0)
            s_test_outer_expectation = s_test_outer_expectation.mean(dim=0)
            
            return self._elbo(
                s_train_expectation, 
                s_train_outer_expectation, 
                D_train, 
                R_train, 
                D_test=D_test, 
                s_test_expectation=s_test_expectation, 
                s_test_outer_expectation=s_test_outer_expectation 
            )

def elbo_comparison_test():
    """Compare SSVI vs Struc_GPLVM ELBO directly"""
    print("ELBO COMPARISON TEST: SSVI vs Struc_GPLVM")
    print("="*50)
    
    # Fixed deterministic test data
    torch.manual_seed(42)
    N, D = 6, 1  # Start with 1D output for simplicity
    latent_dim = 1
    n_inducing = 4
    
    # Create test data
    Vx = torch.linspace(-1.5, 1.5, n_inducing).reshape(-1, 1)
    mu_x = torch.linspace(-1.0, 1.0, N).reshape(-1, 1)
    y = torch.sin(2 * mu_x) + 0.1 * torch.randn(N, D)
    
    print(f"Test Data: N={N}, D={D}, shapes: y{y.shape}, mu_x{mu_x.shape}, Vx{Vx.shape}")
    
    # Common parameters
    beta = [[1.0] * latent_dim] * D
    sigma2_s = [1.0] * D
    
    # === SSVI Model ===
    print("\nCreating SSVI model...")
    ssvi_model = SSVI_GPLVM(
        Y_shape=(N, D),
        latent_dim=latent_dim,
        Vx=Vx.clone(),
        beta=beta,
        sigma2_s=sigma2_s,
        latent_init_scale=0.1
    )
    

    
    # === Struc_GPLVM Model (Proven Working) ===
    print("Creating Struc_GPLVM model...")
    struc_model = Struc_GPLVM(
        Vx=Vx.clone(),
        M=1,  # Wavelength dimension = 1 (no wavelength dependency for GP-LVM)
        mu_x=mu_x.clone(),
        beta=beta,
        sigma2_s=sigma2_s,
        P=8,  # Number of samples for structured model (restored)
        trainable_V=True
    )
    
    # Synchronize parameters for fair comparison
    print("Synchronizing parameters...")
    with torch.no_grad():
        # The Struc model has shape [P, N, A] for latent variables
        # We need to set the FIRST sample to our desired values
        P, N_struc, A = struc_model.mu_x_train.shape
        
        # Set all P samples to the same latent variables for consistency
        for p in range(P):
            struc_model.mu_x_train.data[p] = mu_x.clone()
            struc_model._log_sigma_x_train.data[p] = torch.full_like(mu_x, -2.0)
        
        # Set SSVI latent variables
        ssvi_model.mu_x_train.data = mu_x.clone()
        ssvi_model._log_sigma_x_train.data = torch.full_like(mu_x, -2.0)
        
        # Copy GP parameters from Struc to SSVI for identical comparison
        ssvi_model.gp._mu_u.data = struc_model.gp._mu_u.data.clone()
        ssvi_model.gp._raw_chol.data = struc_model.gp._raw_chol.data.clone()
        ssvi_model.gp.Vx.data = struc_model.gp.Vx.data.clone()
        ssvi_model.gp._log_beta.data = struc_model.gp._log_beta.data.clone()
        ssvi_model.gp._log_sigma2_s.data = struc_model.gp._log_sigma2_s.data.clone()
        
        # Update SSVI model's M to match Struc model for initialization compatibility
        ssvi_model.M = struc_model.M  # M=1 for wavelength dimension
        
        # Initialize SSVI natural parameters from synchronized GP
        ssvi_model._initialize_natural_parameters_from_gp()
    
    # Set to evaluation mode
    ssvi_model.eval()
    struc_model.eval()
    
    print("\nComputing ELBOs...")
    
    # SSVI ELBO
    with torch.no_grad():
        try:
            ssvi_elbo = ssvi_model.compute_gplvm_elbo(y)
            print(f"SSVI ELBO: {ssvi_elbo.item():.8f}")
        except Exception as e:
            print(f"SSVI ELBO failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Struc_GPLVM ELBO (the proven working implementation)
    with torch.no_grad():
        try:
            # Use the gplvm_elbo method - exactly what oil_flow_GPLVM.py uses
            struc_elbo = struc_model.gplvm_elbo(y)
            print(f"Struc_GPLVM ELBO: {struc_elbo.item():.8f}")
        except Exception as e:
            print(f"Struc_GPLVM ELBO failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # COMPARISON
    print(f"\n" + "="*50)
    print("ELBO COMPARISON RESULTS:")
    print("="*50)
    
    diff = abs(ssvi_elbo.item() - struc_elbo.item())
    avg = (abs(ssvi_elbo.item()) + abs(struc_elbo.item())) / 2
    rel_diff = diff / avg * 100 if avg > 1e-10 else float('inf')
    
    print(f"SSVI ELBO:        {ssvi_elbo.item():.8f}")
    print(f"Struc_GPLVM ELBO: {struc_elbo.item():.8f}")
    print(f"Absolute Diff:    {diff:.8f}")
    print(f"Relative Diff:    {rel_diff:.4f}%")
    
    # Tolerance check
    rtol, atol = 1e-3, 1e-4
    matches = torch.allclose(ssvi_elbo, struc_elbo, rtol=rtol, atol=atol)
    
    if matches:
        print(f"\nSUCCESS: ELBOs match within tolerance!")
        print(f"   (rtol={rtol}, atol={atol})")
        print(f"\nCONCLUSION: SSVI ELBO implementation is mathematically correct!")
    else:
        print(f"\nMISMATCH: ELBOs differ beyond tolerance")
        print(f"   (rtol={rtol}, atol={atol})")
        
        # Component-wise debugging
        print(f"\nDETAILED ANALYSIS:")
        try:
            with torch.no_grad():
                # SSVI components
                ssvi_kl_u = ssvi_model.gp.get_kl_u()  # Use GP's KL for consistency
                ssvi_kl_x = ssvi_model.get_kl_x()
                ssvi_total_kl = ssvi_kl_u + ssvi_kl_x
                
                # Struc components
                struc_kl_total = struc_model.get_kl()
                
                print(f"  SSVI KL(U):      {ssvi_kl_u.item():.6f}")
                print(f"  SSVI KL(X):      {ssvi_kl_x.item():.6f}")
                print(f"  SSVI Total KL:   {ssvi_total_kl.item():.6f}")
                print(f"  Struc Total KL:  {struc_kl_total.item():.6f}")
                print(f"  KL Difference:   {abs(ssvi_total_kl.item() - struc_kl_total.item()):.6f}")
                
                # Approximate likelihood terms
                ssvi_likelihood = ssvi_elbo.item() + ssvi_total_kl.item()
                struc_likelihood = struc_elbo.item() + struc_kl_total.item()
                print(f"  SSVI Likelihood: {ssvi_likelihood:.6f}")
                print(f"  Struc Likelihood: {struc_likelihood:.6f}")
                print(f"  Likelihood Diff: {abs(ssvi_likelihood - struc_likelihood):.6f}")
                
        except Exception as e:
            print(f"  Component analysis failed: {e}")
    
    return matches, ssvi_elbo.item(), struc_elbo.item()

if __name__ == "__main__":
    elbo_comparison_test()
