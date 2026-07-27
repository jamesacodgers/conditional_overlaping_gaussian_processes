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
                 sigma2 = 1,
                 trainable_V = True, 
                 independent_LVs=True,
                 variational_family = "mean_filed",
                 problem_type = "categorical"
                 ):
        super(COGPLMV, self).__init__(C_obs = C_obs, 
                                      C_unobs=C_unobs, 
                                      M=M, 
                                      V=V, 
                                      sigma2=sigma2,
                                      trainable_V=trainable_V,
                                    problem_type=problem_type,
                                    variational_family=variational_family
                                    )
        if independent_LVs:
            self.mu_x_train = torch.nn.Parameter(torch.zeros(N_train,self.C,num_latent_dims))
            self._log_sigma_x_train = torch.nn.Parameter(torch.zeros(N_train,self.C,num_latent_dims))
            self.mu_x_test = torch.nn.Parameter(torch.zeros(N_test,self.C,num_latent_dims))
            self._log_sigma_x_test = torch.nn.Parameter(torch.zeros(N_test,self.C,num_latent_dims))
        else: 
            self.mu_x_train = torch.nn.Parameter(torch.zeros(N_train,1,num_latent_dims))
            self._log_sigma_x_train = torch.nn.Parameter(torch.zeros(N_train,1,num_latent_dims))
            self.mu_x_test = torch.nn.Parameter(torch.zeros(N_test,1,num_latent_dims))
            self._log_sigma_x_test = torch.nn.Parameter(torch.zeros(N_test,1,num_latent_dims))

    @property 
    def sigma_x_train(self): 
        return  torch.exp(self._log_sigma_x_train)
    @property 
    def sigma_x_test(self): 
        return  torch.exp(self._log_sigma_x_test)
    
    def get_x_train_dist(self):
        return torch.distributions.Normal(self.mu_x_train, self.sigma_x_train)

    def get_x_test_dist(self):
        return torch.distributions.Normal(self.mu_x_test, self.sigma_x_test)
    
    def get_train_samples(self, n_samples = 4):
        dist = self.get_x_train_dist()
        return dist.rsample((n_samples, ))
    
    def get_test_samples(self, n_samples = 4):
        dist = self.get_x_test_dist()
        return dist.rsample((n_samples, ))

    def elbo(self, training_targets, test_targets, training_weights,  n_samples = 16):
        x_train = self.get_train_samples(n_samples=n_samples)#.transpose(2,1)
        x_test = self.get_test_samples(n_samples=n_samples)#.transpose(2,1)
        training_fit = super().get_training_fit(inputs=x_train, targets=training_targets, training_weights=training_weights, n_samples = 1)
        test_fit = super().get_test_fit(inputs=x_test, targets=test_targets, n_samples = 1)
        KL_u = self.get_kl_u()
        KL_x = self.get_kl_x()
        const = self.get_const_term(training_targets.shape[0] + test_targets.shape[0])
        return (training_fit + test_fit - KL_u -KL_x - const).squeeze()
    
    def get_x_prior(self):
        return torch.distributions.Normal(torch.zeros(1), torch.ones(1))
    
    def get_kl_x(self):
        p_x = self.get_x_prior()
        q_x_train = self.get_x_train_dist()
        q_x_test = self.get_x_test_dist()
        KL_train = torch.distributions.kl_divergence(q_x_train, p_x).sum()
        KL_test = torch.distributions.kl_divergence(q_x_test, p_x).sum()
        return  KL_train + KL_test
    
    def get_R_estimate_samples(self, targets, num_samples=4):
        #Note, this relies on the same order of targets and inputs
        x_test = self.get_test_samples(n_samples=num_samples)#.transpose(2,1)
        prob_samples = self.get_R_dist_samples(targets=targets, inputs=x_test, num_samples=1)
        return prob_samples.probs.mean(dim=[0,1])


if __name__=="__main__":
    model = COGPLMV(C_obs=1,C_unobs = 1,N_train=100,N_test=100,M=20,num_latent_dims=2,V= torch.randn(4,2), trainable_V=False, independent_LVs=True)
    for name, param in model.named_parameters():
        print(name)
