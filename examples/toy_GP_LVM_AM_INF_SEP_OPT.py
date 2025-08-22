# %% 
import torch 
import matplotlib.pyplot as plt
from src.ssvi_models_am_inf_sep_opt import GPLVM_SSVI_AM_INF
from dataclasses import dataclass
from typing import Dict, Any

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

@dataclass
class AM_INF_SSVIConfig:
    """Configuration for Amortized Inference SSVI GP-LVM with Separate Optimizers"""
    
    # === Data Generation ===
    n_train: int = 256
    data_noise: float = 0.1  # Noise level in toy data generation
    random_seed: int = 42
    
    # === Model Architecture ===
    n_inducing_points: int = 32
    inducing_range: tuple = (-4, 4)
    latent_dim: int = 1
    
    # === Amortized Inference Parameters ===
    encoder_hidden_dims: list = None  # Will be set to [64, 32] if None
    num_inner_steps: int = 5  # Number of inner optimization steps for local update
    
    # === Training Configuration ===
    training_iterations: int = 100
    batch_size: int = 64
    
    # === Learning Rates ===
    lr_local: float = 0.001              # Local parameters (encoder network)
    lr_global: float = 0.001             # Global parameters (GP hyperparameters)
    
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
    
    def __post_init__(self):
        """Set default encoder architecture if not specified"""
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [64, 32]
    
    def print_config(self):
        """Print configuration summary"""
        print("=== Amortized Inference SSVI GP-LVM Configuration ===")
        print(f"Data: {self.n_train} points, noise={self.data_noise}")
        print(f"Model: {self.n_inducing_points} inducing points, latent_dim={self.latent_dim}")
        print(f"Encoder: {self.encoder_hidden_dims} hidden dims, inner_steps={self.num_inner_steps}")
        print(f"Training: {self.training_iterations} iter, batch={self.batch_size}")
        print(f"Learning rates: Local={self.lr_local}, Global={self.lr_global}")
        print("=" * 50)

def generate_toy_data(config: AM_INF_SSVIConfig):
    """Generate toy data according to configuration"""
    torch.manual_seed(config.random_seed)
    
    # Generate structured latent variables
    x = torch.linspace(-2, 2, config.n_train).reshape(-1, 1)
    
    # Generate clear nonlinear relationships
    y1 = x**3 + torch.randn_like(x) * config.data_noise
    y2 = x + torch.randn_like(x) * config.data_noise
    
    y = torch.hstack([y1, y2])
    
    # Generate inducing points
    Vx = torch.linspace(config.inducing_range[0], config.inducing_range[1], 
                       config.n_inducing_points).reshape(-1, 1)
    
    # Initialize latent variables directly from true x values
    mu_x = x.clone()
    
    return y, Vx, mu_x

# === Pre-defined Configurations for Easy Experimentation ===

def get_fast_config():
    """Fast configuration for quick testing"""
    return AM_INF_SSVIConfig(
        n_train=128,
        n_inducing_points=16,
        training_iterations=100,
        batch_size=32,
        encoder_hidden_dims=[64, 32],
        num_inner_steps=3,
        lr_local=0.002,
        lr_global=0.003,
        data_noise=0.15
    )

def get_working_config():
    """Configuration that should produce reasonable results"""
    return AM_INF_SSVIConfig(
        n_train=128,
        n_inducing_points=32,
        training_iterations=200,
        batch_size=128,
        encoder_hidden_dims=[512, 256, 128, 64],
        num_inner_steps=100,
        lr_local=0.0001,
        lr_global=0.0005,
        data_noise=0.02,
        plot_results=True
    )

def get_high_quality_config():
    """High-quality configuration for best results"""
    return AM_INF_SSVIConfig(
        n_train=128,
        n_inducing_points=16,
        training_iterations=500,
        batch_size=64,
        encoder_hidden_dims=[128, 64, 32],
        lr_local=0.001,
        lr_global=0.005
    )

def get_ablation_config():
    """Configuration for algorithm ablation studies"""
    return AM_INF_SSVIConfig(
        n_train=128,
        n_inducing_points=8,
        training_iterations=100,
        batch_size=128,
        encoder_hidden_dims=[32],
        lr_local=0.01,
        lr_global=0.01
    )

def get_demo_config():
    """Ultra-fast demo configuration"""
    return AM_INF_SSVIConfig(
        n_train=64,
        n_inducing_points=8,
        training_iterations=50,
        batch_size=16,
        encoder_hidden_dims=[32, 16],
        num_inner_steps=2,
        lr_local=0.005,
        lr_global=0.005,
        data_noise=0.1,
        plot_results=True
    )

class AM_INF_SSVI_GPLVM_Wrapper(GPLVM_SSVI_AM_INF):
    """Wrapper class for Amortized Inference SSVI GP-LVM with configuration support"""
    def __init__(self, config: AM_INF_SSVIConfig, Vx, mu_x):
        N, latent_dim = mu_x.shape
        D = 2  # For the toy example (y has 2 dimensions)
        
        # Initialize underlying Amortized Inference SSVI GP-LVM
        super().__init__(
            Vx=Vx,
            M=D,
            mu_x=mu_x,
            sigma2=config.sigma2,
            beta=config.beta,
            sigma2_s=config.sigma2_s,
            P=64,
            trainable_V=config.trainable_V,
            encoder_hidden_dims=config.encoder_hidden_dims,
            num_inner_steps=config.num_inner_steps
        )
        
        # Store config
        self.config = config
        
        # Update learning rates from config
        for param_group in self.optimizer_local.param_groups:
            param_group['lr'] = config.lr_local
        for param_group in self.optimizer_global.param_groups:
            param_group['lr'] = config.lr_global

    @property 
    def mu_x(self): 
        return self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return self.Sigma_x_train

def train_am_inf_ssvi_gplvm(model, y, config: AM_INF_SSVIConfig):
    """Training function for Amortized Inference SSVI GP-LVM with configuration"""
    local_losses = []
    global_losses = []
    
    print(f"Training for {config.training_iterations} iterations (batch_size={config.batch_size})")
    
    for i in range(config.training_iterations):
        # Sample batch indices for this iteration
        N = y.shape[0]
        batch_indices = torch.randperm(N)[:config.batch_size]
        y_batch = y[batch_indices]
        
        # Sample U from current q(u) distribution
        U_samples = model.get_u_samples_ssvi(model.P)
        
        # Single training step with amortized inference and separate optimizers
        local_loss, global_loss = model.train_step(y_batch, batch_indices, U_samples)
        
        local_losses.append(local_loss)
        global_losses.append(global_loss)
        
        # Progress reporting
        if i % max(1, config.training_iterations // 10) == 0:
            print(f"Iteration {i}/{config.training_iterations}")
            print(f"  Local Loss: {local_loss:.4f}, Global Loss: {global_loss:.4f}")
            
            # Show inner step progress if available
            if hasattr(model, 'last_inner_losses') and model.last_inner_losses:
                inner_losses = model.last_inner_losses
                print(f"  Inner Steps: {len(inner_losses)} steps, final inner loss: {inner_losses[-1]:.4f}")

    print(f"Training completed!")
    print(f"Final Local Loss: {local_losses[-1]:.4f}")
    print(f"Final Global Loss: {global_losses[-1]:.4f}")
    
    return local_losses, global_losses

def main(config: AM_INF_SSVIConfig = None):
    """Main function for running Amortized Inference SSVI GP-LVM toy example"""
    
    import time
    start_time = time.time()
    
    # Use default config if none provided
    if config == None:
        config = AM_INF_SSVIConfig()
    
    # Print configuration
    config.print_config()
    
    # Generate data according to configuration
    y, Vx, mu_x = generate_toy_data(config)
    
    # Initialize Amortized Inference SSVI model with configuration
    am_inf_model = AM_INF_SSVI_GPLVM_Wrapper(config=config, Vx=Vx, mu_x=mu_x)

    # Train Amortized Inference SSVI model
    print("\nTraining Amortized Inference SSVI GP-LVM...")
    local_losses, global_losses = train_am_inf_ssvi_gplvm(am_inf_model, y, config)
        
    # Generate test points for prediction
    x_test = torch.linspace(config.test_range[0], config.test_range[1], 
                           config.n_test_points).reshape(-1, 1)

    # Test inner steps functionality
    print("\nTesting inner steps functionality...")
    test_batch_indices = torch.tensor([0, 1, 2, 3, 4])
    test_y_batch = y[test_batch_indices]
    test_U_samples = am_inf_model.get_u_samples_ssvi(am_inf_model.P)
    
    # Run a single training step to see inner steps in action
    local_loss, global_loss = am_inf_model.train_step(test_y_batch, test_batch_indices, test_U_samples)
    inner_losses = am_inf_model.get_last_inner_losses()
    print(f"Test step completed:")
    print(f"  Local Loss: {local_loss:.4f}, Global Loss: {global_loss:.4f}")
    print(f"  Inner Steps: {len(inner_losses)} steps")
    print(f"  Inner Losses: {[f'{l:.4f}' for l in inner_losses]}")

    # Get predictions using SSVI-specific method
    mu_f_am_inf = am_inf_model.get_ssvi_mean_prediction(x_test)

    # Plot results if requested
    if config.plot_results:
        with torch.no_grad():
            # Get true latent variables for coloring and comparison
            _, _, x_true = generate_toy_data(config)
            
            plt.figure(figsize=(15, 5))
            
            # Plot 1: Data and predictions
            plt.subplot(1, 4, 1)
            plt.scatter(y[:,0], y[:,1], c=x_true.flatten(), alpha=0.6, label='Data')
            plt.plot(mu_f_am_inf[:,0], mu_f_am_inf[:,1], 'r-', label='AM-INF SSVI', linewidth=2)
            plt.xlabel('y1')
            plt.ylabel('y2')
            plt.title('Amortized Inference SSVI GP-LVM Predictions')
            plt.legend()
            
            # Plot 2: Local loss curve
            plt.subplot(1, 4, 2)
            plt.plot(local_losses, label='Local Loss', color='red')
            plt.xlabel('Iteration')
            plt.ylabel('Local Loss')
            plt.title('Local Parameters Training Curve')
            plt.legend()
            
            # Plot 3: Global loss curve
            plt.subplot(1, 4, 3)
            plt.plot(global_losses, label='Global Loss', color='blue')
            plt.xlabel('Iteration')
            plt.ylabel('Global Loss')
            plt.title('Global Parameters Training Curve')
            plt.legend()
            
            # Plot 4: Latent space recovery
            plt.subplot(1, 4, 4)
            mu_x_mean = am_inf_model.mu_x.mean(dim=0)  # [N, Q]
            plt.scatter(x_true.flatten(), mu_x_mean[:,0].detach(), 
                       color='red', label='AM-INF SSVI', alpha=0.7, s=20)
            for p in range(am_inf_model.P):
                plt.scatter(x_true.flatten(), am_inf_model.mu_x[p,:,0].detach(), marker="x", color="black", 
                           s=30)
            plt.xlabel('True LV')
            plt.ylabel('Estimated LV')
            plt.title('AM-INF SSVI Latent Variable Recovery')
            plt.legend()
            
            plt.tight_layout()
            plt.show()

    # Print final model parameters
    print("\nFinal Model Parameters:")
    for name, param in am_inf_model.gp.named_parameters():
        print(f"  {name}: {param.data}")
    
    # Print timing information
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Training speed: {config.training_iterations/total_time:.2f} iterations/second")
    
    return am_inf_model, local_losses, global_losses, config

if __name__ == "__main__":
    # === Usage Examples ===
    
    # 1. Default configuration
    #main()
    
    # 2. Fast configuration for debugging
    #main(get_fast_config())
    
    # 3. Fast but reasonable configuration (default for testing)
    #main(get_fast_config())
    
    # 4. Working configuration for reasonable results
    main(get_working_config())
    
    # 5. High-quality configuration for best results  
    #main(get_high_quality_config())
    
    # 6. Ablation study configuration
    #main(get_ablation_config())
    
    # 7. Demo configuration (fastest but still reasonable)
    #main(get_demo_config())
    
    # 6. Custom configuration
    # custom_config = AM_INF_SSVIConfig(
    #     n_inducing_points=16,
    #     encoder_hidden_dims=[64, 32],
    #     training_iterations=200,
    #     lr_local=0.002,
    #     lr_global=0.003
    # )
    # main(custom_config)
