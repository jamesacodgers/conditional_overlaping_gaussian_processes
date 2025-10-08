"""
Simplified VAE for Tabular Data (MLP only)

Clean implementation without image/color/pixel concepts.
Works with flat vectors (batch, data_dim).
"""

import torch
from torch import nn
from likelihoods import GaussianLikelihood, BernoulliLikelihood
from posteriors import DiagonalGaussian, FullCovarianceGaussian
from networks import MLP
from typing import Optional, List


class VAE_MLP(nn.Module):
    """
    Variational Autoencoder for tabular/vector data using MLP architecture.
    
    Args:
        data_dim: dimensionality of input data vectors
        latent_dim: dimension of latent variable
        enc_hidden_dims: hidden layer dimensions for encoder MLP
        dec_hidden_dims: hidden layer dimensions for decoder MLP
        posterior_form: 'diagonal_gaussian' or 'full_covariance_gaussian'
        likelihood: 'gaussian' or 'bernoulli'
        noise: 'heteroscedastic' or 'homoscedastic'
        noise_std: initial noise std for homoscedastic case
        train_noise: whether to train homoscedastic noise
        prior_std: std of isotropic Gaussian prior
        likelihood_activation: activation for likelihood mean
        nonlinearity: activation between MLP layers
    """
    
    def __init__(
        self,
        data_dim: int = 12,
        latent_dim: int = 16,
        enc_hidden_dims: Optional[List[int]] = [50, 50],
        dec_hidden_dims: Optional[List[int]] = [50, 50],
        posterior_form: str = "diagonal_gaussian",
        likelihood: str = "gaussian",
        noise: str = "heteroscedastic",
        noise_std: float = 0.1,
        train_noise: bool = False,
        prior_std: torch.Tensor = torch.tensor(1.0),
        likelihood_activation: nn.Module = nn.Identity(),
        nonlinearity: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.posterior_form = posterior_form
        self.prior_std = prior_std
        self.nonlinearity = nonlinearity
        
        # Posterior
        if posterior_form == "diagonal_gaussian":
            self.posterior = DiagonalGaussian()
        elif posterior_form == "full_covariance_gaussian":
            self.posterior = FullCovarianceGaussian(latent_dim)
        else:
            raise NotImplementedError("Selected posterior not yet implemented")
        
        # Likelihood
        if likelihood == "gaussian":
            self.likelihood = GaussianLikelihood(
                noise=noise,
                std=noise_std,
                train_noise=train_noise,
                activation=likelihood_activation,
            )
        elif likelihood == "bernoulli":
            self.likelihood = BernoulliLikelihood()
        else:
            raise NotImplementedError("Likelihood chosen not yet implemented")
        
        # Encoder: data_dim → hidden → latent_dim * posterior.multiplier
        enc_dims = [data_dim] + enc_hidden_dims + [latent_dim * self.posterior.multiplier]
        self.encoder = MLP(dims=enc_dims, nonlinearity=nonlinearity)
        
        # Decoder: latent_dim → hidden → data_dim * likelihood.multiplier
        dec_dims = [latent_dim] + dec_hidden_dims + [data_dim * self.likelihood.multiplier]
        self.decoder = MLP(dims=dec_dims, nonlinearity=nonlinearity)
        
        # Prior
        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(latent_dim),
            covariance_matrix=prior_std**2 * torch.eye(latent_dim),
        )
    
    def encode(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """
        Encode data to latent space.
        
        Args:
            x: (batch, data_dim) input vectors
            
        Returns:
            Posterior distribution over latent variables
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.data_dim
        
        h = self.encoder(x)
        q = self.posterior(h)
        return q
    
    def decode(self, z: torch.Tensor) -> torch.distributions.Distribution:
        """
        Decode latent variables to data space.
        
        Args:
            z: (batch, num_samples, latent_dim) latent variables
            
        Returns:
            Likelihood distribution over observations
        """
        assert len(z.shape) == 3
        batch_size = z.shape[0]
        num_samples = z.shape[1]
        assert z.shape[2] == self.latent_dim
        
        # Flatten batch and samples
        z_flat = z.view(batch_size * num_samples, self.latent_dim)
        
        # Decode
        h = self.decoder(z_flat)
        
        # Reshape back: (batch*num_samples, data_dim*multiplier) → (batch, num_samples, data_dim*multiplier)
        h = h.view(batch_size, num_samples, self.data_dim * self.likelihood.multiplier)
        
        # Apply likelihood
        l = self.likelihood(h)
        return l
    
    def forward(self, x: torch.Tensor, num_samples: int = 1):
        """
        Forward pass through VAE.
        
        Args:
            x: (batch, data_dim) input vectors
            num_samples: number of latent samples
            
        Returns:
            q: Posterior distribution
            z: Latent samples (batch, num_samples, latent_dim)
            l: Likelihood distribution
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.data_dim
        batch_size = x.shape[0]
        
        # Encode
        q = self.encode(x)
        
        # Sample latent variables (reparameterization trick)
        z = q.rsample(sample_shape=torch.Size([num_samples])).permute(1, 0, 2)
        
        assert len(z.shape) == 3
        assert z.shape[0] == batch_size
        assert z.shape[1] == num_samples
        assert z.shape[2] == self.latent_dim
        
        # Decode
        l = self.decode(z)
        
        return q, z, l
    
    def elbo(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        beta_nll: bool = False,
        beta: float = 0.5,
    ):
        """
        Compute ELBO (Evidence Lower Bound).
        
        Args:
            x: (batch, data_dim) input vectors
            num_samples: number of samples for ELBO estimation
            beta_nll: whether to use beta-weighted NLL
            beta: beta weighting factor
            
        Returns:
            elbo: (batch,) ELBO values
            metrics: Dictionary with metrics (elbo, kl, ll, RMSE)
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.data_dim
        
        # Forward pass
        q, _, l = self(x, num_samples=num_samples)
        
        # KL divergence
        kl = torch.distributions.kl.kl_divergence(q, self.prior)
        
        # Log likelihood
        # x: (batch, data_dim) → repeated_x: (batch, num_samples, data_dim)
        repeated_x = x.unsqueeze(1).repeat(1, num_samples, 1)
        exp_ll = l.log_prob(repeated_x)
        
        # RMSE
        if isinstance(self.likelihood, GaussianLikelihood):
            rmse = ((l.loc - repeated_x) ** 2).mean().sqrt()
        elif isinstance(self.likelihood, BernoulliLikelihood):
            rmse = ((l.probs - repeated_x) ** 2).mean().sqrt()
        
        # ELBO: average over samples, sum over features
        if beta_nll:
            if isinstance(self.likelihood, GaussianLikelihood):
                var = l.scale ** 2
            elif isinstance(self.likelihood, BernoulliLikelihood):
                var = l.probs * (1 - l.probs)
            elbo = (exp_ll * (var.detach() ** beta)).mean(1).sum(dim=1) - kl
        else:
            elbo = exp_ll.mean(1).sum(dim=1) - kl
        
        exp_ll = exp_ll.mean(1).sum(dim=1)
        
        # Metrics
        elbo_metric_name = 'elbo' if not beta_nll else 'beta_nll_elbo'
        metrics = {
            elbo_metric_name: elbo.mean(),
            "ll": exp_ll.mean(),
            "kl": kl.mean(),
            "RMSE": rmse,
        }
        
        if isinstance(self.likelihood, GaussianLikelihood):
            if self.likelihood.noise == "homoscedastic":
                metrics["noise"] = self.likelihood.std
        
        return elbo, metrics
    
    def generate(self, z: torch.Tensor) -> torch.distributions.Distribution:
        """
        Generate data from latent variables.
        
        Args:
            z: Latent variables (any shape ending in latent_dim)
            
        Returns:
            Likelihood distribution over generated data
        """
        # Ensure z has shape (batch, num_samples, latent_dim)
        while len(z.shape) < 3:
            z = z.unsqueeze(0)
        
        return self.decode(z)

