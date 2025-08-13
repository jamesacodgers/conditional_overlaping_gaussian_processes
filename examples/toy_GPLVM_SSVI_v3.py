# %% 
import torch 
import matplotlib.pyplot as plt
from src.ssvi_models_v2 import GPLVM_SSVI
from dataclasses import dataclass
from typing import Dict, Any

def setup_device():
    """Setup device with proper error handling - CUDA ONLY"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but required for this script!")
    
    device = torch.device("cuda")
    torch.cuda.set_device(0)  # Use first GPU
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Force all tensor creation to use CUDA
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device(device)
    else:
        # For older PyTorch versions, use this approach
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    return device

# Setup device at module level
device = setup_device()

@dataclass
class SSVIConfig:
    """Configuration for SSVI GP-LVM - Easy hyperparameter tuning"""
    
    # === Data Generation ===
    n_train: int = 256
    data_noise: float = 0.1  # Noise level in toy data generation
    random_seed: int = 42
    
    # === Model Architecture ===
    n_inducing_points: int = 32
    inducing_range: tuple = (-4, 4)
    latent_dim: int = 1
    
    # === SSVI Algorithm Parameters ===
    step_size_s: float = 0.1         # SSVI step size parameter 's'
    step_size_kappa: float = 0.75    # SSVI step size decay 'kappa' in (0.5, 1]
    local_opt_iterations: int = 10    # Inner optimization iterations for q(x_i)
    num_u_samples: int = 10           # Number of U samples for Monte Carlo approximation
    
    # === Training Configuration ===
    training_iterations: int = 100
    batch_size: int = 64
    
    # === Learning Rates ===
    lr_gp: float = 0.001              # GP hyperparameters (beta, sigma^2, etc.)
    lr_local: float = 0.001           # Local variational optimization in SSVI
    
    # === GP Hyperparameters ===
    sigma2: float = 1.0              # Observation noise variance
    beta: Any = None                 # Lengthscales (None = auto)
    sigma2_s: Any = None             # Signal variances (None = auto)
    trainable_V: bool = True         # Whether inducing points are trainable
    
    # === Initialization ===
    latent_init_scale: float = 0.3   # Scale for random latent variable initialization
    
    # === Visualization ===
    n_test_points: int = 100
    test_range: tuple = (-4, 4)
    plot_results: bool = True
    
    # === Debugging & Monitoring ===
    log_gradients: bool = False      # Enable gradient logging
    log_frequency: int = 10          # Log every N steps
    verbose_logging: bool = False    # Print detailed gradient stats
    
    def get_step_size_params(self) -> Dict[str, float]:
        """Get SSVI step size parameters"""
        return {'s': self.step_size_s, 'kappa': self.step_size_kappa}
    
    def print_config(self):
        """Print configuration summary"""
        print("=== SSVI GP-LVM Configuration ===")
        print(f"Data: {self.n_train} points, noise={self.data_noise}")
        print(f"Model: {self.n_inducing_points} inducing points, latent_dim={self.latent_dim}")
        print(f"SSVI: s={self.step_size_s}, kappa={self.step_size_kappa}, local_iter={self.local_opt_iterations}")
        print(f"Training: {self.training_iterations} iter, batch={self.batch_size}")
        print(f"Learning rates: GP={self.lr_gp}, local={self.lr_local}")
        print("=" * 35)

def generate_toy_data(config: SSVIConfig, device=None):
    """Generate toy data according to configuration - DEVICE FIXED"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(config.random_seed)
    
    # Create tensors on the correct device
    x = torch.randn((config.n_train, 1), device=device)
    
    # Generate nonlinear relationships  
    y1 = x**3 + torch.randn_like(x) * config.data_noise  # Cubic
    y2 = x + torch.randn_like(x) * config.data_noise     # Linear
    
    y = torch.hstack([y1, y2])
    y = (y - y.mean(dim=0)) / y.std(dim=0)  # Standardize
    
    # Generate inducing points - ensure on same device
    Vx = torch.linspace(config.inducing_range[0], config.inducing_range[1], 
                       config.n_inducing_points, device=device).reshape(-1, 1)
    
    # Initialize latent variables using PCA - ensure on same device
    mu_x, _, _ = torch.pca_lowrank(y, config.latent_dim)
    mu_x = mu_x / mu_x.std(dim=0)
    
    return y, Vx, mu_x

# === Pre-defined Configurations for Easy Experimentation ===

def get_fast_config():
    """Fast configuration for debugging"""
    return SSVIConfig(
        n_train=64,
        n_inducing_points=8,
        training_iterations=200,
        batch_size=32,
        local_opt_iterations=50
    )

def get_high_quality_config():
    """High-quality configuration for best results"""
    return SSVIConfig(
        n_train=512,
        n_inducing_points=8,
        training_iterations=200,
        batch_size=16,
        local_opt_iterations=50,
        lr_gp=0.005,
        lr_local=0.005
    )

def get_ablation_config():
    """Configuration for SSVI algorithm ablation studies"""
    return SSVIConfig(
        n_train=256,
        n_inducing_points=8,
        training_iterations=100,
        batch_size=64,
        step_size_s=0.05,      # Smaller step size
        step_size_kappa=0.8,   # Slower decay
        local_opt_iterations=2
    )


class SSVI_GPLVM_Wrapper(GPLVM_SSVI):
    """Wrapper class for SSVI GP-LVM with configuration support"""
    def __init__(self, config: SSVIConfig, Vx, mu_x):
        N, latent_dim = mu_x.shape
        D = 2  # For the toy example (y has 2 dimensions)
        
        # Get device from input tensors - MUST be CUDA
        device = Vx.device
        if device.type != 'cuda':
            raise RuntimeError(f"Expected CUDA device, but got {device}")
        
        print(f"Initializing model on device: {device}")
        
        # AGGRESSIVE DEVICE FORCING: Override default tensor type to force CUDA
        original_default_type = torch.get_default_dtype()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
        # Also set default device if available
        original_device = None
        if hasattr(torch, 'get_default_device'):
            try:
                original_device = torch.get_default_device()
                torch.set_default_device(device)
            except:
                pass
        
        try:
            # Initialize underlying SSVI GP-LVM with CUDA enforcement
            super().__init__(
                Vx=Vx,
                M=D,
                mu_x=mu_x,
                sigma2=config.sigma2,
                beta=config.beta,
                sigma2_s=config.sigma2_s,
                P=64,
                trainable_V=config.trainable_V,
            )
            print("Model initialization completed successfully!")
            
        except RuntimeError as e:
            if "device" in str(e).lower():
                print(f"Device error during model initialization: {e}")
                print("This suggests the underlying model code is creating tensors on CPU.")
                print("You may need to modify src/struc_models.py line 152 to use the correct device.")
                raise RuntimeError(f"CUDA device mismatch in model initialization: {e}")
            else:
                raise e
        finally:
            # Restore original settings
            torch.set_default_tensor_type(torch.FloatTensor)  # Reset to CPU default
            if original_device is not None and hasattr(torch, 'set_default_device'):
                try:
                    torch.set_default_device(original_device)
                except:
                    pass
        
        # Verify all model components are on CUDA
        self.to(device)
        
        # Double-check device placement
        for name, param in self.named_parameters():
            if param.device.type != 'cuda':
                print(f"WARNING: Parameter {name} is on {param.device}, moving to CUDA")
                param.data = param.data.to(device)
        
        # Sync step-size schedule from config
        self.step_size_params = config.get_step_size_params()
        
        # Store config for local optimization
        self.config = config
        
        # Set local optimization parameters from config
        self.local_opt_iterations = config.local_opt_iterations
        self.lr_local = config.lr_local
        self.batch_size = config.batch_size
        
        print(f"Model successfully initialized on {next(self.parameters()).device}")
        
        # Latent variables already initialized from mu_x in super().__init__

    @property 
    def mu_x(self): 
        return self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return self.Sigma_x_train


def train_ssvi_gplvm(model, y, config: SSVIConfig):
    """Training function for SSVI GP-LVM with configuration - DEVICE FIXED"""
    ll = []
    
    # Ensure model and data are on the same device
    model_device = next(model.parameters()).device
    y = y.to(model_device)
    
    print(f"Model device: {model_device}")
    print(f"Data device: {y.device}")
    
    # IMPORTANT: Three-level optimization structure:
    # 1. Local optimization: x_i updated via Adam (inside SSVI)
    # 2. GP hyperparameters: updated via gradient descent  
    # 3. Natural parameters (Lambda, nu): updated via SSVI algorithm
    
    # Only optimize GP hyperparameters and latent variables with gradient descent
    # Access GP parameters correctly from WLIndependentGPs
    gp_params = [
        model.gp._log_sigma2_s,     # Signal variances  
        model.gp._log_beta          # Lengthscales
    ]
    
    # Add inducing inputs if trainable
    if hasattr(model.gp, 'Vx') and isinstance(model.gp.Vx, torch.nn.Parameter):
        gp_params.append(model.gp.Vx)
        
    
    # Use learning rates from configuration  
    optim_gp = torch.optim.Adam(gp_params, lr=config.lr_gp)
    
    print(f"Training for {config.training_iterations} iterations (batch_size={config.batch_size})")
    
    for i in range(config.training_iterations):
        # Sample batch indices for this iteration (consistent across steps)
        N = y.shape[0]
        batch_indices = torch.randperm(N, device=model_device)[:config.batch_size]  # Ensure indices on correct device
        y_batch = y[batch_indices]
        
        # Steps 1-3: Core SSVI algorithm (sample U, sample batch, compute local params)
        U_samples = model.get_u_samples_ssvi(model.P)
        model.ssvi_local_step(y_batch, batch_indices, U_samples=U_samples)
        
        # Step 4: Full ELBO computation for hyperparameter optimization (one step)
        # Use SAME batch and SAME U_samples as local step
        optim_gp.zero_grad()
        R_batch = torch.ones((y_batch.shape[0], 1), dtype=y_batch.dtype, device=model_device)  # Ensure on correct device
        mu_x_batch = model.mu_x_train[:, batch_indices, :]
        Sigma_x_batch = model.Sigma_x_train[:, batch_indices, :]
        # Reuse same U_samples for global step
        loss = -model.elbo(y_batch, R_batch, mu_x_train=mu_x_batch, Sigma_x_train=Sigma_x_batch, U_samples=U_samples)
        loss.backward()
        optim_gp.step()
        
        # Step 5: Global natural parameter update (SSVI formula)
        model.update_natural_parameters()
        
        # Progress reporting
        if i % max(1, config.training_iterations // 10) == 0:
            print(f"Iteration {i}/{config.training_iterations}, ELBO: {-loss.item():.2f}")
        ll.append(loss.detach().item())

    print(f"Training completed! Final ELBO: {-ll[-1]:.4f}")
    return ll


def main(config: SSVIConfig = None):
    """Main function for running SSVI GP-LVM toy example with configuration - CUDA ONLY"""
    
    # Force CUDA usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available!")
    
    # Use default config if none provided
    if config is None:
        config = SSVIConfig()
    
    # Force CUDA device
    current_device = torch.device("cuda")
    print(f"Using device: {current_device}")
    
    # Print configuration
    config.print_config()
    
    # CRITICAL: Set default device and tensor type to CUDA
    print("Setting up CUDA environment...")
    if hasattr(torch, 'set_default_device'):
        torch.set_default_device(current_device)
        print(f"Set default device to: {current_device}")
    
    # Generate data according to configuration
    y, Vx, mu_x = generate_toy_data(config, device=current_device)
    
    # Ensure all tensors are on CUDA
    y = y.to(current_device)
    Vx = Vx.to(current_device) 
    mu_x = mu_x.to(current_device)
    
    print(f"Data shapes - y: {y.shape}, Vx: {Vx.shape}, mu_x: {mu_x.shape}")
    print(f"Data devices - y: {y.device}, Vx: {Vx.device}, mu_x: {mu_x.device}")
    
    # Verify all data is on CUDA
    assert y.device.type == 'cuda', f"y is on {y.device}, expected CUDA"
    assert Vx.device.type == 'cuda', f"Vx is on {Vx.device}, expected CUDA"
    assert mu_x.device.type == 'cuda', f"mu_x is on {mu_x.device}, expected CUDA"
    
    # Initialize SSVI model with configuration
    print("Initializing SSVI model...")
    ssvi_model = SSVI_GPLVM_Wrapper(config=config, Vx=Vx, mu_x=mu_x)
    
    # Verify model is on CUDA
    model_device = next(ssvi_model.parameters()).device
    assert model_device.type == 'cuda', f"Model is on {model_device}, expected CUDA"
    print(f"Model confirmed on device: {model_device}")

    # Train SSVI model
    print("\nTraining SSVI GP-LVM...")
    ll_ssvi = train_ssvi_gplvm(ssvi_model, y, config)
        
    # Generate test points for prediction - ensure on correct device
    x_test = torch.linspace(config.test_range[0], config.test_range[1], 
                           config.n_test_points, device=current_device).reshape(-1, 1)

    # Get predictions using SSVI-specific method (preserves natural parameter structure)
    mu_f_ssvi = ssvi_model.get_ssvi_mean_prediction(x_test)

    # Plot results if requested
    if config.plot_results:
        with torch.no_grad():
            # Generate true latent for coloring - ensure on correct device
            torch.manual_seed(config.random_seed)
            x_true = torch.randn((config.n_train, 1), device=current_device)
            
            # Move to CPU for plotting to avoid device mismatch errors
            y_cpu = y.cpu()
            mu_f_ssvi_cpu = mu_f_ssvi.cpu()
            x_true_cpu = x_true.cpu()
            
            plt.figure(figsize=(12, 4))
            
            # Plot 1: Data and predictions
            plt.subplot(1, 3, 1)
            plt.scatter(y_cpu[:,0], y_cpu[:,1], c=x_true_cpu.flatten(), alpha=0.6, label='Data')
            plt.plot(mu_f_ssvi_cpu[:,0], mu_f_ssvi_cpu[:,1], 'b-', label='SSVI', linewidth=2)
            plt.xlabel('y1')
            plt.ylabel('y2')
            plt.title('SSVI GP-LVM Predictions')
            plt.legend()
            
            # Plot 2: Training curve
            plt.subplot(1, 3, 2)
            plt.plot(ll_ssvi, label='SSVI', color='blue')
            plt.xlabel('Iteration')
            plt.ylabel('Negative ELBO')
            plt.title('SSVI Training Curve')
            plt.legend()
            
            # Plot 3: Latent space recovery
            plt.subplot(1, 3, 3)
            # ssvi_model.mu_x is [P, N, Q]; average across P for plotting
            mu_x_mean = ssvi_model.mu_x.mean(dim=0).cpu()  # [N, Q] - move to CPU
            mu_x_cpu = mu_x.cpu()
            plt.scatter(x_true_cpu.flatten(), mu_x_mean[:,0].detach(), 
                       color='blue', label='SSVI', alpha=0.7, s=20)
            plt.scatter(x_true_cpu.flatten(), mu_x_cpu[:,0], marker="x", color="black", 
                       label='PCA init', s=30)
            plt.xlabel('True LV')
            plt.ylabel('Estimated LV')
            plt.title('SSVI Latent Variable Recovery')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    return ssvi_model, ll_ssvi, config


if __name__ == "__main__":
    # === Usage Examples ===
    
    # 1. Default configuration
    #main()
    
    # 2. Fast configuration for debugging
    #main(get_fast_config())
    
    # 3. Gradient monitoring configuration
    # debug_config = SSVIConfig(
    #     training_iterations=1,
    #     log_gradients=True,
    #     log_frequency=5,
    #     verbose_logging=True,  # Set to False for compact logging
    #     lr_gp=0.001,
    #     lr_local=0.001,
    #     batch_size=128
    # )
    # main(debug_config)
    
    # 3. High-quality configuration for best results  
    main(get_high_quality_config())
    
    # 4. Custom configuration
    # custom_config = SSVIConfig(
    #     n_inducing_points=32,
    #     step_size_s=0.2,
    #     local_opt_iterations=3,
    #     training_iterations=150
    # )
    # main(custom_config)