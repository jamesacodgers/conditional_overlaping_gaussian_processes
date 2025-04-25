# %% 
import torch 
import matplotlib.pyplot as plt


n_train = 256
x = torch.randn((n_train,1))

y1 = x**2 + torch.randn_like(x)/10

y2 = x + torch.randn_like(x)/10

y = torch.hstack([y1,y2])
y = (y - y.mean(dim=0))/y.std(dim=0)

with torch.no_grad():
    plt.scatter(y[:,0], y[:,1], c=x)
    plt.show()

# %%

from src.mf_methods import WLIndependentGPs

R_train = torch.ones(n_train,1)


Vx = torch.linspace(-4,4,8).reshape(-1,1)
mu_x,_,_ =  torch.pca_lowrank(y,1)

model = WLIndependentGPs(C=1,M=2,Vx=Vx, mu_x_train=mu_x)

