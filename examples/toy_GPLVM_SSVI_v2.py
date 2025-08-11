# %% 
import torch 
import matplotlib.pyplot as plt
from src.mf_methods import MF_WS_GPLVM
from src.ssvi_models_v2 import GPLVM_SSVI
from dataclasses import dataclass
from typing import Dict, Any

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

def generate_toy_data(config: SSVIConfig):
    """Generate toy data according to configuration"""
    torch.manual_seed(config.random_seed)
    
    x = torch.randn((config.n_train, 1))
    
    # Generate nonlinear relationships  
    y1 = x**3 + torch.randn_like(x) * config.data_noise  # Cubic
    y2 = x + torch.randn_like(x) * config.data_noise     # Linear
    
    y = torch.hstack([y1, y2])
    y = (y - y.mean(dim=0)) / y.std(dim=0)  # Standardize
    
    # Generate inducing points
    Vx = torch.linspace(config.inducing_range[0], config.inducing_range[1], 
                       config.n_inducing_points).reshape(-1, 1)
    
    # Initialize latent variables using PCA
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
        # Initialize underlying SSVI GP-LVM
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
        # Sync step-size schedule from config
        self.step_size_params = config.get_step_size_params()
        
        # Store config for local optimization
        self.config = config
        
        # Set local optimization parameters from config
        self.local_opt_iterations = config.local_opt_iterations
        self.lr_local = config.lr_local
        self.batch_size = config.batch_size
        
        # Latent variables already initialized from mu_x in super().__init__

    @property 
    def mu_x(self): 
        return self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return self.Sigma_x_train


def train_ssvi_gplvm(model, y, config: SSVIConfig):
    """Training function for SSVI GP-LVM with configuration"""
    ll = []
    
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
        batch_indices = torch.randperm(N)[:config.batch_size]
        y_batch = y[batch_indices]
        
        # Steps 1-3: Core SSVI algorithm (sample U, sample batch, compute local params)
        U_samples = model.sample_U(model.P)
        model.ssvi_local_step(y_batch, batch_indices, U_samples=U_samples)
        
        # Step 4: Full ELBO computation for hyperparameter optimization (one step)
        # Use SAME batch and SAME U_samples as local step
        optim_gp.zero_grad()
        R_batch = torch.ones((y_batch.shape[0], 1), dtype=y_batch.dtype, device=y_batch.device)
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
    """Main function for running SSVI GP-LVM toy example with configuration"""
    
    # Use default config if none provided
    if config is None:
        config = SSVIConfig()
    
    # Print configuration
    config.print_config()
    
    # Generate data according to configuration
    y, Vx, mu_x = generate_toy_data(config)
    
    # Initialize SSVI model with configuration
    ssvi_model = SSVI_GPLVM_Wrapper(config=config, Vx=Vx, mu_x=mu_x)

    # Train SSVI model
    print("\nTraining SSVI GP-LVM...")
    ll_ssvi = train_ssvi_gplvm(ssvi_model, y, config)
        
    # Generate test points for prediction
    x_test = torch.linspace(config.test_range[0], config.test_range[1], 
                           config.n_test_points).reshape(-1, 1)

    # Get predictions using SSVI-specific method (preserves natural parameter structure)
    mu_f_ssvi = ssvi_model.get_ssvi_mean_prediction(x_test)

    # Plot results if requested
    if config.plot_results:
        with torch.no_grad():
            # Generate true latent for coloring
            torch.manual_seed(config.random_seed)
            x_true = torch.randn((config.n_train, 1))
            
            plt.figure(figsize=(12, 4))
            
            # Plot 1: Data and predictions
            plt.subplot(1, 3, 1)
            plt.scatter(y[:,0], y[:,1], c=x_true.flatten(), alpha=0.6, label='Data')
            plt.plot(mu_f_ssvi[:,0], mu_f_ssvi[:,1], 'b-', label='SSVI', linewidth=2)
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
            mu_x_mean = ssvi_model.mu_x.mean(dim=0)  # [N, Q]
            plt.scatter(x_true.flatten(), mu_x_mean[:,0].detach(), 
                       color='blue', label='SSVI', alpha=0.7, s=20)
            plt.scatter(x_true.flatten(), mu_x[:,0], marker="x", color="black", 
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
    debug_config = SSVIConfig(
        training_iterations=5,
        log_gradients=True,
        log_frequency=5,
        verbose_logging=True,  # Set to False for compact logging
        lr_gp=0.001,
        lr_local=0.001,
        batch_size=128
    )
    main(debug_config)
    
    # 3. High-quality configuration for best results  
    # main(get_high_quality_config())
    
    # 4. Custom configuration
    # custom_config = SSVIConfig(
    #     n_inducing_points=32,
    #     step_size_s=0.2,
    #     local_opt_iterations=3,
    #     training_iterations=150
    # )
    # main(custom_config) 