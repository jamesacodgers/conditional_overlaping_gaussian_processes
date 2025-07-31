# %% 
import torch 
import matplotlib.pyplot as plt
from src.mf_methods import MF_WS_GPLVM

from src.ssvi_models import SSVI_GPLVM


n_train = 256
x = torch.randn((n_train,1))

y1 = x**3 + torch.randn_like(x)/10
y2 = x + torch.randn_like(x)/10

y = torch.hstack([y1,y2])
y = (y - y.mean(dim=0))/y.std(dim=0)

R_train = torch.ones(n_train,1)

Vx = torch.linspace(-4,4,6).reshape(-1,1)
mu_x,_,_ =  torch.pca_lowrank(y,1)
mu_x = mu_x/mu_x.std(dim=0)




class SSVI_GPLVM_Wrapper(SSVI_GPLVM):
    """Wrapper class to match interface of other models"""
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, step_size_params=None, trainable_V=True):
        super().__init__(
            Vx=Vx, M=M, mu_x_train=mu_x, mu_x_test=None, C_obs=1, C_unobs=0, 
            sigma2=sigma2, beta=beta, sigma2_s=sigma2_s, 
            step_size_params=step_size_params, trainable_V=trainable_V
        )

    @property 
    def mu_x(self): 
        return self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return self.Sigma_x_train
    
    def gplvm_elbo(self, y, batch_size=32):
        """SSVI ELBO with mini-batch support"""
        return super().gplvm_elbo(y, batch_size=batch_size)





def train_ssvi_gplvm(model, y, plot=False):
    """Training function for SSVI GP-LVM"""
    ll = []
    
    # IMPORTANT: SSVI natural parameters (Lambda, nu) are NOT optimized with gradient descent
    # They are updated internally via the SSVI algorithm in ssvi_step()
    
    # Only optimize GP hyperparameters and latent variables with gradient descent
    original_gp = model.gp.original_gp
    gp_params = [
        original_gp._raw_chol,      # Covariance parameters
        original_gp.Vx,             # Inducing inputs  
        original_gp._log_sigma2_s,  # Signal variances
        original_gp._log_beta       # Lengthscales
    ]
    latent_params = [
        model.mu_x_train, 
        model._log_sigma_x_train
    ]
    
    # Only these parameters get gradient-based optimization
    optim_gp = torch.optim.Adam(gp_params, lr=0.01) 
    optim_latent = torch.optim.Adam(latent_params, lr=0.01)
    
    for i in range(2000):
        # Zero gradients for gradient-optimized parameters
        optim_gp.zero_grad()
        optim_latent.zero_grad()
        
        # SSVI step: updates natural parameters (Lambda, nu) internally + computes ELBO
        loss = -model.gplvm_elbo(y, batch_size=64)
        loss.backward()
        
        # Step gradient-based optimizers (natural parameters updated in gplvm_elbo via SSVI)
        optim_gp.step()
        optim_latent.step()
        
        if i % 100 == 0:
            print(i)
        ll.append(loss.detach().item())

    if plot:
        with torch.no_grad():
            plt.plot(ll)
            plt.show()
    return ll


# Initialize SSVI model
ssvi_model = SSVI_GPLVM_Wrapper(
    M=2, Vx=Vx, mu_x=mu_x,
    step_size_params={'s': 0.1, 'kappa': 0.75}
)

# Train SSVI model
print("Training SSVI GP-LVM...")
ll_ssvi = train_ssvi_gplvm(ssvi_model, y, plot=False)
    
x_test = torch.linspace(-4,4,100).reshape(-1,1)

# %%

mu_f_ssvi = ssvi_model.get_mean_prediction_observed_inputs(x_test)

with torch.no_grad():
    plt.figure(figsize=(10, 4))
    
    # Plot 1: Data and predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y[:,0], y[:,1], c=x.flatten(), alpha=0.6, label='Data')
    plt.plot(mu_f_ssvi[:,0], mu_f_ssvi[:,1], 'b-', label='SSVI', linewidth=2)
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.title('SSVI GP-LVM Predictions')
    plt.legend()
    
    # Plot 2: Training curve
    plt.subplot(1, 2, 2)
    plt.plot(ll_ssvi, label='SSVI', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Negative ELBO')
    plt.title('SSVI Training Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Separate plot for latent space
    plt.figure(figsize=(6, 4))
    plt.scatter(x.flatten(), ssvi_model.mu_x[:,0].detach(), 
               color='blue', label='SSVI', alpha=0.7, s=20)
    plt.scatter(x.flatten(), mu_x[:,0], marker="x", color="black", 
               label='PCA init', s=30)
    plt.xlabel('True LV')
    plt.ylabel('Estimated LV')
    plt.title('SSVI Latent Variable Recovery')
    plt.legend()
    plt.show()

# %%

print(f"Final SSVI ELBO: {-ll_ssvi[-1]:.4f}") 