"""
Simple example to test and debug LVMOGP_SSVI

Setup:
- Q_x = 2 (2D observed features, e.g., spatial locations)
- Q_h = 3 (3D function latent space)
- 3 functions (surfaces)
- Generate synthetic data
- Train for 3 steps
- Make predictions
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from lvmogp_ruby.ssvi_lvmogp import LVMOGP_SSVI


def generate_synthetic_data(num_fns=3, points_per_fn=20, Q_x=2, seed=42):
    """
    Generate synthetic data for LVMOGP with truly different functional forms.
    
    Function 0: Y = sin(x1) + cos(x2)
    Function 1: Y = exp(-0.5 * x1^2) + 0.5 * x2^2
    Function 2: Y = tanh(x1) * sin(x2)
    
    Returns:
    --------
    X_data : [N, Q_x] - Observed features (e.g., spatial locations)
    X_data_fn : [N] - Function IDs
    Y_data : [N, 1] - Observations
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X_data_list = []
    X_data_fn_list = []
    Y_data_list = []
    
    for fn_id in range(num_fns):
        # Generate random spatial locations for this function
        X_fn = torch.randn(points_per_fn, Q_x, dtype=torch.float64) * 2.0
        
        # Generate observations with DIFFERENT functional forms per function
        if fn_id == 0:
            # Function 0: Standard sinusoidal
            Y_fn = torch.sin(X_fn[:, 0]) + torch.cos(X_fn[:, 1])
        elif fn_id == 1:
            # Function 1: Gaussian-like with quadratic terms
            Y_fn = torch.exp(-0.5 * X_fn[:, 0]**2) + 0.5 * X_fn[:, 1]**2
        elif fn_id == 2:
            # Function 2: Mixed trigonometric
            Y_fn = torch.tanh(X_fn[:, 0]) * torch.sin(X_fn[:, 1])
        else:
            raise ValueError(f"Only 3 function IDs (0, 1, 2) are supported, got {fn_id}")
        
        Y_fn = Y_fn.unsqueeze(1)
        Y_fn += torch.randn(points_per_fn, 1, dtype=torch.float64) * 0.1  # Add noise
        
        # Function IDs
        fn_ids = torch.full((points_per_fn,), fn_id, dtype=torch.long)
        
        X_data_list.append(X_fn)
        X_data_fn_list.append(fn_ids)
        Y_data_list.append(Y_fn)
    
    X_data = torch.cat(X_data_list, dim=0)
    X_data_fn = torch.cat(X_data_fn_list, dim=0)
    Y_data = torch.cat(Y_data_list, dim=0)
    
    return X_data, X_data_fn, Y_data


def main():
    print("=" * 60)
    print("LVMOGP_SSVI Simple Example")
    print("=" * 60)
    
    # Configuration
    Q_x = 2  # Observed feature dimensions
    Q_h = 3  # Function latent dimensions
    num_fns = 3  # Number of functions
    M = 1  # Output dimension (scalar)
    P = 4  # Number of SSVI samples
    num_inducing = 15  # Number of inducing points
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Q_x: {Q_x}, Q_h: {Q_h}, num_fns: {num_fns}")
    print()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X_data, X_data_fn, Y_data = generate_synthetic_data(
        num_fns=num_fns, 
        points_per_fn=20, 
        Q_x=Q_x
    )
    
    N = X_data.shape[0]
    print(f"Total observations: {N}")
    print(f"X_data shape: {X_data.shape}")
    print(f"X_data_fn shape: {X_data_fn.shape}")
    print(f"Y_data shape: {Y_data.shape}")
    print()
    
    # Initialize inducing points in full space [Q_x + Q_h]
    Q_total = Q_x + Q_h
    Vx = torch.randn(num_inducing, Q_total, dtype=torch.float64) * 0.5
    
    # Initialize H values (optional - can be None)
    H_data_mean_init = torch.randn(num_fns, Q_h, dtype=torch.float64) * 0.1
    H_data_var_init = torch.ones(num_fns, Q_h, dtype=torch.float64) * 0.1
    
    print("Initializing LVMOGP_SSVI model...")
    model = LVMOGP_SSVI(
        X_data=X_data,
        X_data_fn=X_data_fn,
        num_fns=num_fns,
        Q_h=Q_h,
        M=M,
        Vx=Vx,
        H_data_mean=H_data_mean_init,
        H_data_var=H_data_var_init,
        sigma2=0.1,
        beta=[[1.0] * Q_total],  # Lengthscales for all dimensions
        sigma2_s=[1.0],  # Signal variance
        P=P,
        trainable_V=True,
        encoder_hidden_dims=[32, 16],
        local_lr=0.001,
        global_lr=0.001,
        device=device,
        enable_missing_data=False
    )
    
    print(f"Model initialized!")
    print(f"Model info: {model.get_model_info()}")
    print()
    
    # Training
    print("Training for 3 steps...")
    batch_size = 30
    
    for step in range(3):
        # Random batch
        indices = torch.randperm(N)[:batch_size]
        Y_batch = Y_data[indices]
        
        # Training step
        loss = model.train_step(Y_batch, indices)
        
        print(f"Step {step + 1}/3: Loss = {loss:.4f}")
        
        # Show learned H values (averaged over P samples)
        print(f"  H_data_mean[fn=0]: {model.H_data_mean[:, 0, :].mean(dim=0).detach().cpu().numpy()}")
        print(f"  H_data_var[fn=0]: {model.H_data_var[:, 0, :].mean(dim=0).detach().cpu().numpy()}")
    
    print()
    
    # Prediction
    print("Making predictions...")
    
    # Create test points for function 0
    X_test = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.5, -0.5]
    ], dtype=torch.float64, device=device)
    
    X_test_fn = torch.tensor([0, 0, 1], dtype=torch.long, device=device)
    
    print(f"Test X shape: {X_test.shape}")
    print(f"Test function IDs: {X_test_fn}")
    
    # Predict
    mu_pred, var_pred = model.predict_f(X_test, X_test_fn)
    
    print(f"Predictions shape: {mu_pred.shape}")
    print(f"Variance shape: {var_pred.shape}")
    print()
    print("Predictions:")
    for i in range(X_test.shape[0]):
        print(f"  Point {i} (fn={X_test_fn[i].item()}): "
              f"mu={mu_pred[i, 0].item():.4f}, "
              f"var={var_pred[i, 0].item():.4f}")
    
    print()
    
    # Show final learned H values (averaged over P samples)
    print("Final learned H values (averaged over P samples):")
    for fn_id in range(num_fns):
        print(f"  Function {fn_id}:")
        print(f"    H_mean: {model.H_data_mean[:, fn_id, :].mean(dim=0).detach().cpu().numpy()}")
        print(f"    H_var:  {model.H_data_var[:, fn_id, :].mean(dim=0).detach().cpu().numpy()}")
    
    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

