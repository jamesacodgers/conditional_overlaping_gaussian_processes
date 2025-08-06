# %% 
import torch 
import matplotlib.pyplot as plt
from src.mf_methods import MF_WS_GPLVM
from src.struc_models import Struc_WS_GPLVM
from src.utils import train_lbfgs

n_train = 128
x = torch.randn((n_train,1))

y1 = x**3 + torch.randn_like(x)/10

y2 = x + torch.randn_like(x)/10

y = torch.hstack([y1,y2])
y = (y - y.mean(dim=0))/y.std(dim=0)

R_train = torch.ones(n_train,1)

Vx = torch.linspace(-4,4,8).reshape(-1,1)
mu_x,_,_ =  torch.pca_lowrank(y,1)
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
    def __init__(self, Vx, M, mu_x, sigma2=1, beta=None, sigma2_s=None, P=64, trainable_V=True):
        super().__init__(P=P, V=Vx, M=M, C_obs=1, mu_x_train=mu_x, mu_x_test=None, C_unobs=0, sigma2=sigma2, weight_distribution="continuous", beta=beta, sigma2_s=sigma2_s, trainable_V=trainable_V)

    @property 
    def mu_x(self): 
        return  self.mu_x_train
    
    @property 
    def Sigma_x(self): 
        return  self.Sigma_x_train
    
    def gplvm_elbo(self, y):
        R_train = torch.ones((y.shape[0],1))
        return super(Struc_GPLVM,self).elbo(y, R_train)
    
mf_model = MF_GPLVM(M=2,Vx=Vx, mu_x=mu_x)
struc_model = Struc_GPLVM(M=2,Vx=Vx, mu_x=mu_x, P=8, trainable_V=False)
# struc_model._log_sigma2 = torch.nn.Parameter(torch.tensor([-2.]))
# print(mf_model.gplvm_elbo(y=y))


def train_gplvm(model,y,plot=False):
    ll = []
    # optim_gp= torch.optim.Adam([model.gp._mu_u, model.gp._raw_chol, model.gp.Vx, model.gp._log_sigma2_s], lr=0.01)
    optim = torch.optim.Adam(model.parameters(), lr=0.005)
    # for j in range(4000): 
    #     optim_gp.zero_grad()
    #     loss = -model.gplvm_elbo(y=y)
    #     loss.backward()
    #     optim_gp.step()
    #     if j%100==0:
    #         print(j)
    #     ll.append(loss.detach().item())

    for i in range(2000):
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
    
x_test = torch.linspace(-4,4,100).reshape(-1,1)
# mu_f = mf_model.get_mean_prediction_observed_inputs(x_test)

# %%

mu_f = struc_model.get_mean_prediction_observed_inputs(x_test)
with torch.no_grad():
    plt.scatter(y[:,0], y[:,1], c = x.flatten() )
    plt.plot(mu_f[:,0], mu_f[:,1])
    plt.show()

# %%

with torch.no_grad():
    plt.hlines(struc_model.gp.Vx[:,0], -3,3, color="blue", alpha=1 )
    for i in range(struc_model.P):
        plt.scatter(x,struc_model.mu_x[i,:,0], color="orange", alpha=0.1 )
    # plt.scatter(x, mu_x[:,0], marker="x", color="black" )
    plt.xlabel("True LV ")
    plt.ylabel("Estimated LV ")
    plt.show()
# %%
