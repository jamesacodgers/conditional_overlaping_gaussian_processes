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
                 independent_LVs=True,
                 ):
        super(COGPLMV, self).__init__(C_obs = C_obs, 
                                      C_unobs=C_unobs, 
                                      M=M, 
                                      V=V, 
                                      trainable_V=trainable_V,
                                    problem_type="categorical"
                                    )
        if independent_LVs:
            self.mu_x_train = torch.nn.Parameter(torch.zeros(N_train,self.C,num_latent_dims))
            self._log_sigma_x_train = torch.nn.Parameter(torch.ones(N_train,self.C,num_latent_dims))
            self.mu_x_test = torch.nn.Parameter(torch.zeros(N_test,self.C,num_latent_dims))
            self._log_sigma_x_test = torch.nn.Parameter(torch.ones(N_test,self.C,num_latent_dims))
        else: 
            self.mu_x_train = torch.nn.Parameter(torch.zeros(N_train,1,num_latent_dims))
            self._log_sigma_x_train = torch.nn.Parameter(torch.ones(N_train,1,num_latent_dims))
            self.mu_x_test = torch.nn.Parameter(torch.zeros(N_test,1,num_latent_dims))
            self._log_sigma_x_test = torch.nn.Parameter(torch.ones(N_test,1,num_latent_dims))

    @property 
    def sigma_x_train(self): 
        return  torch.exp(self._log_sigma_x_train)
    @property 
    def sigma_x_test(self): 
        return  torch.exp(self._log_sigma_x_test)
    
    def get_train_distributions(self):
        return torch.distributions.Normal(self.mu_x_train, self.sigma_x_train)

    def get_test_distributions(self):
        return torch.distributions.Normal(self.mu_x_test, self.sigma_x_test)
    
    def get_train_samples(self, n_samples = 4):
        dist = self.get_train_distributions()
        return dist.rsample((n_samples, ))
    
    def get_test_samples(self, n_samples = 4):
        dist = self.get_test_distributions()
        return dist.rsample((n_samples, ))

    def elbo(self, training_targets, test_targets, training_weights,  n_samples = 16):
        x_train = self.get_train_samples(n_samples=n_samples)
        x_test = self.get_test_samples(n_samples=n_samples)
        training_fit = super().get_training_fit(inputs=x_train, targets=training_targets, training_weights=training_weights, n_samples = 1)
        test_fit = super().get_test_fit(test_inputs=x_test, test_targets=test_targets, n_samples = 1)
        KL_u = self.get_kl_u()
        const = self.get_const_term(training_targets.shape[0] + test_targets.shape[0])
        return (training_fit.mean() + test_fit.mean() - KL_u - const).squeeze()
    


if __name__=="__main__":
    model = COGPLMV(C_obs=1,C_unobs = 1,N_train=100,N_test=100,M=20,num_latent_dims=2,V= torch.randn(4,2), trainable_V=False, independent_LVs=True)
    for name, param in model.named_parameters():
        print(name)
