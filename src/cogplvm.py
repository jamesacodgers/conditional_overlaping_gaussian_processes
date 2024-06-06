import torch
from src.cogp import COGP



class COGPLMV(COGP):
    def __init__(self, 
                 C_obs, 
                 N_train,
                 N_test, 
                 M, 
                 num_latent_dims, 
                 V, 
                 C_unobs = 0, 
                 trainable_V = True, 
                 independent_LVs=True):
        super(COGPLMV, self).__init__(C_obs, C_unobs, M, V, trainable_V=trainable_V)
        self.mu_x_train = torch.nn.Parameter(torch.zeros(N_train,num_latent_dims, self.C))
        self._log_Sigma_x_train = torch.nn.Parameter(torch.ones(N_train,num_latent_dims, self.C))
        self.mu_x_test = torch.nn.Parameter(torch.zeros(N_test,num_latent_dims, self.C))
        self._log_Sigma_x_test = torch.nn.Parameter(torch.ones(N_test,num_latent_dims, self.C))

    @property 
    def Sigma_x_train(self): 
        return  torch.exp(self._log_Sigma_x_train)
    @property 
    def Sigma_x_test(self): 
        return  torch.exp(self._log_Sigma_x_test)
        
    def get_training_fit(self, training_spectra, training_weights, training_temperatures):
        return super().get_training_fit(training_spectra, training_weights, training_temperatures)

if __name__=="__main__":
    model = COGPLMV(1,1,100,100,20,2,torch.randn(4,2), False)
    for name, param in model.named_parameters():
        print(name)
