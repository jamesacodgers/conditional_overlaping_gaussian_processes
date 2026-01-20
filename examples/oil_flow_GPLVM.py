# %% 
import torch 
import matplotlib.pyplot as plt
from src.mf_methods import MF_WS_GPLVM
from src.struc_models import Struc_WS_GPLVM
from src.utils import train_lbfgs
import numpy as np

import urllib.request
import tarfile

# Import scalers for data standardization
from src.pytorch_scalers import MeanCenter, StandardScalerPytorch, MinMaxScalerPytorch, NoTransform

# === Data Standardization Configuration ===
# Options: "standard" (mean=0, std=1), "minmax" ([0,1]), "mean" (mean=0), or "none" (no transform)
standardization = "mean"  # Default: mean centering

url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
urllib.request.urlretrieve(url, '3PhData.tar.gz')
with tarfile.open('3PhData.tar.gz', 'r') as f:
    f.extract('DataTrn.txt')
    f.extract('DataTrnLbls.txt')

y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)

# === Data Standardization ===
# Fit scaler on data (before model initialization)
print(f"\n=== Data Standardization ===")
print(f"Standardization method: {standardization}")

if standardization == "standard":
    scaler = StandardScalerPytorch(device=y.device)
elif standardization == "minmax":
    scaler = MinMaxScalerPytorch(device=y.device)
elif standardization == "mean":
    scaler = MeanCenter(device=y.device)
elif standardization == "none":
    scaler = NoTransform(device=y.device)
else:
    raise ValueError(
        f"Unknown standardization method: {standardization}. Must be 'standard', 'minmax', 'mean', or 'none'"
    )

# Fit scaler on data
print(f"Fitting scaler on data ({y.shape[0]} samples)...")
scaler.fit(y)

# Transform data using the fitted scaler
print(f"Transforming data...")
print(f"  Before transform - Shape: {y.shape}")
print(f"  Before transform - Mean: {y.mean().item():.6f}, Std: {y.std().item():.6f}")

y = scaler.transform(y)

# Debug: Show statistics after transform
print(f"  After transform - Mean: {y.mean().item():.6f}, Std: {y.std().item():.6f}")
if hasattr(scaler, 'min'):
    print(f"  After transform - Min: {y.min().item():.6f}, Max: {y.max().item():.6f}")

print(f"Standardization completed.")
print("=" * 50)
A = 6
Vx = torch.randn(12,A)
mu_x,_,_ =  torch.pca_lowrank(y,A)
mu_x = mu_x/mu_x.std(dim=0)

class MF_GPLVM(MF_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, trainable_V=True):
        super().__init__(Vx=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, sigma2=sigma2, weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s, Vl=None, gamma=None, trainable_V=trainable_V)

    @property 
    def mu_x(self): 
        return  self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return  self.Sigma_x_train
    
    def gplvm_elbo(self, y):
        R_train = torch.ones((self.mu_x.shape[0],1))
        return super().elbo(y, R_train)
    
def train_GPLVM_lbgfs(model, y, plot=False):
    R_train = torch.ones(y.shape[0],1)
    loss_list = train_lbfgs(model,y,R_train,plot=plot)
    return loss_list


    
class Struc_GPLVM(Struc_WS_GPLVM):
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64,  trainable_V=True):
        super().__init__(P=P,  V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, sigma2=sigma2, weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s, trainable_V=trainable_V)

    @property 
    def mu_x(self): 
        return  self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return  self.Sigma_x_train
    
    def gplvm_elbo(self, y):
        R_train = torch.ones((y.shape[0],1))
        return super(Struc_GPLVM,self).elbo(y, R_train)
    
# mf_model = MF_GPLVM(M=2,Vx=Vx, mu_x=mu_x)
struc_model = Struc_GPLVM(M=12,Vx=Vx, mu_x=mu_x, P=8, trainable_V=True)
# struc_model._log_sigma2 = torch.nn.Parameter(torch.tensor([-2.]))
# print(mf_model.gplvm_elbo(y=y))


def train_gplvm(model,y,plot=False):
    ll = []
    # optim_gp= torch.optim.Adam([model.gp._mu_u, model.gp._raw_chol, model.gp.Vx, model.gp._log_sigma2_s], lr=0.01)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    # for j in range(4000): 
    #     optim_gp.zero_grad()
    #     loss = -model.gplvm_elbo(y=y)
    #     loss.backward()
    #     optim_gp.step()
    #     if j%100==0:
    #         print(j)
    #     ll.append(loss.detach().item())

    for i in range(1000):
        optim.zero_grad()
        loss = -model.gplvm_elbo(y=y)
        loss.backward()
        optim.step()
        if i%100==0:
            print(i)
            print("Loss: ", loss.detach().item())
        ll.append(loss.detach().item())

    if plot:
        with torch.no_grad():
            plt.plot(ll)
            plt.show()
    return ll
# ll = train_gplvm(mf_model,y,plot=True)

# %%
ll = train_gplvm(struc_model,y,plot=True)
   
# %%
c = ["r" ,"g", "b"]
colors = [c[int(i)-1] for i in labels]
with torch.no_grad():
    # plt.hlines(struc_model.gp.Vx[:,0], -3,3, color="blue", alpha=1 )
    for i in range(struc_model.P):
        plt.scatter(struc_model.mu_x[i,:,0], struc_model.mu_x[i,:,1], alpha=0.1, color=colors )
    # plt.scatter(x, mu_x[:,0], marker="x", color="black" )

    plt.xlabel("True LV ")
    plt.ylabel("Estimated LV ")
    plt.show()
# %%
c = ["r" ,"g", "b"]
colors = [c[int(i)-1] for i in labels]
with torch.no_grad():
    # plt.hlines(struc_model.gp.Vx[:,0], -3,3, color="blue", alpha=1 )
    for i in range(struc_model.P):
        sample_1 = torch.distributions.Normal(struc_model.mu_x[i,:,4], struc_model.Sigma_x[i,:,4].sqrt()).sample()
        sample_2 = torch.distributions.Normal(struc_model.mu_x[i,:,0], struc_model.Sigma_x[i,:,0].sqrt()).sample()
        # plt.scatter(struc_model.mu_x[i,:,4], struc_model.mu_x[i,:,1], alpha=0.1, color=colors )
        plt.scatter(sample_1, sample_2, alpha=0.1, color=colors )
    # plt.scatter(x, mu_x[:,0], marker="x", color="black" )

    plt.xlabel("True LV ")
    plt.ylabel("Estimated LV ")
    plt.show()
# %%
with torch.no_grad():
    plt.bar(range(Vx.shape[1]), 1/struc_model.gp.beta.flatten(), color="blue", alpha=1 )