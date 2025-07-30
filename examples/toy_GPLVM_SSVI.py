# %% 
import torch 
import matplotlib.pyplot as plt
from src.mf_methods import MF_WS_GPLVM
from src.struc_models import Struc_WS_GPLVM
from src.ssvi_models import SSVI_GPLVM
from src.utils import train_lbfgs

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

class Struc_GPLVM(Struc_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64):
        super().__init__(P=P, V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, sigma2=sigma2, weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s)

    @property 
    def mu_x(self): 
        return  self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return  self.Sigma_x_train
    
    def gplvm_elbo(self, y):
        R_train = torch.ones((y.shape[0],1))
        return super().elbo(y, R_train)


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


def train_gplvm(model,y,plot=False):
    ll = []
    optim_gp= torch.optim.Adam([model.gp._mu_u, model.gp._raw_chol, model.gp.Vx, model.gp._log_sigma2_s], lr=0.01)
    optim = torch.optim.Adam(model.parameters())
    for j in range(4000): 
        optim_gp.zero_grad()
        loss = -model.gplvm_elbo(y=y)
        loss.backward()
        optim_gp.step()
        if j%100==0:
            print(j)
        ll.append(loss.detach().item())

    for i in range(2000):
        optim.zero_grad()
        ll.append(loss.detach().item())
        loss = -model.gplvm_elbo(y=y)
        loss.backward()
        optim.step()
        if i%100==0:
            print(i)
        ll.append(loss.detach().item())

    if plot:
        with torch.no_grad():
            plt.plot(ll)
            plt.show()
    return ll


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


# Initialize models
struc_model = Struc_GPLVM(M=2,Vx=Vx, mu_x=mu_x)
ssvi_model = SSVI_GPLVM_Wrapper(
    M=2, Vx=Vx, mu_x=mu_x,
    step_size_params={'s': 0.1, 'kappa': 0.75}
)

# Train models
print("Training Structured GP-LVM...")
ll_struc = train_gplvm(struc_model,y,plot=False)

print("Training SSVI GP-LVM...")
ll_ssvi = train_ssvi_gplvm(ssvi_model, y, plot=False)
    
x_test = torch.linspace(-4,4,100).reshape(-1,1)

# %%

mu_f_struc = struc_model.get_mean_prediction_observed_inputs(x_test)
mu_f_ssvi = ssvi_model.get_mean_prediction_observed_inputs(x_test)

with torch.no_grad():
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Data and predictions
    plt.subplot(1, 3, 1)
    plt.scatter(y[:,0], y[:,1], c=x.flatten(), alpha=0.6, label='Data')
    plt.plot(mu_f_struc[:,0], mu_f_struc[:,1], 'r-', label='Structured', linewidth=2)
    plt.plot(mu_f_ssvi[:,0], mu_f_ssvi[:,1], 'b--', label='SSVI', linewidth=2)
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.title('GP-LVM Predictions')
    plt.legend()
    
    # Plot 2: Training curves
    plt.subplot(1, 3, 2)
    plt.plot(ll_struc, label='Structured', color='red')
    plt.plot(ll_ssvi, label='SSVI', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Negative ELBO')
    plt.title('Training Curves')
    plt.legend()
    
    # Plot 3: Latent space comparison
    plt.subplot(1, 3, 3)
    if hasattr(struc_model, 'P'):  # Structured model has multiple samples
        for i in range(min(3, struc_model.P)):
            plt.scatter(x.flatten(), struc_model.mu_x[i,:,0].detach(), 
                       alpha=0.3, s=10, color='red')
        plt.scatter(x.flatten(), struc_model.mu_x.mean(0)[:,0].detach(), 
                   color='red', label='Structured (mean)', s=20)
    
    plt.scatter(x.flatten(), ssvi_model.mu_x[:,0].detach(), 
               color='blue', label='SSVI', alpha=0.7, s=20)
    plt.scatter(x.flatten(), mu_x[:,0], marker="x", color="black", 
               label='PCA init', s=30)
    plt.xlabel('True LV')
    plt.ylabel('Estimated LV')
    plt.title('Latent Variable Recovery')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%

print(f"Final Structured ELBO: {-ll_struc[-1]:.4f}")
print(f"Final SSVI ELBO: {-ll_ssvi[-1]:.4f}") 