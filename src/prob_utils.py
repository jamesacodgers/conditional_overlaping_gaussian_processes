import torch
from scipy.stats import norm
import numpy as np
from scipy.optimize import bisect

def get_mf_ci(preds, dim = 0, alpha = 0.05 ):
    with torch.no_grad():
        mu = preds.mean[:,dim].cpu().numpy()
        sigma2 = preds.mean[:,dim,dim].cpu().numpy()
        dist = norm(mu, np.sqrt(sigma2))
        return dist.ppf(alpha/2), dist.ppf(1-alpha/2)

def get_struc_ci(preds, dim=0, alpha=0.05):
    with torch.no_grad():
        mus = preds.mean[:,:,dim].cpu().numpy()
        sigma2s = preds.covariance_matrix[:,:,dim,dim].cpu().numpy()
        def f_lb(a,dists):
            return dists.cdf(a).mean() - alpha/2
        def f_ub(a,dists):
            return dists.cdf(a).mean() - (1-alpha/2)
        lb = np.zeros(mus.shape[1])
        ub = np.zeros(mus.shape[1])
        for i,(mu,sigma2) in enumerate(zip(mus.T, sigma2s.T)):
            dists = norm(mu, np.sqrt(sigma2))
            init_max = dists.ppf(1-alpha/2).max()
            init_min = dists.ppf(alpha/2).min()
            lb[i] = bisect(f=f_lb, a=-10, b=10, args=dists)
            ub[i] = bisect(f=f_ub, a=init_min, b=init_max, args=dists)
        return lb, ub



def _get_percentile(dists_1d, alpha):
        def f(a):
            return dists_1d.cdf(a).mean(axis = 0) - alpha
        return bisect(f,dists_1d.mean().mean(axis=0))