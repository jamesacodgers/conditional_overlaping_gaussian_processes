#!/usr/bin/env python3
"""
Training time and memory comparison between MF GP-LVM and SSVI GP-LVM on Oil Flow.

Usage (from ssvi_scripts_experiments/):
    python runtime_memory_comparison.py
"""

import sys
import os
import time
import gc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'generalised_gplvm_code'))

import torch
import numpy as np
from tqdm import trange

from src.original_gplvm_config import OriginalGPLVMConfig
from src.original_gplvm_config_utils import (
    create_argument_parser as mf_create_parser,
    load_config_with_overrides as mf_load_config,
)
from src.ssvi_config import OilFlowSSVIConfig
from src.ssvi_config_utils import (
    create_argument_parser as ssvi_create_parser,
    load_config_with_overrides as ssvi_load_config,
)
from src.data_utils import download_oilflow_data_if_needed
from utils.data import load_real_data
from sklearn.model_selection import train_test_split

from oil_flow_GPLVM import OilFlowModel
from oil_flow_GPLVM_SSVI_AM_INF import load_oil_flow_data
from ssvi_scripts_experiments.mnist_ood_GPLVM_SSVI_AM_INF import OilFlow_SSVI_GPLVM

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import MultivariateNormalPrior
from gpytorch.constraints import Positive
from torch.distributions import transforms
from models.latent_variable import NNEncoder

# ---------------------------------------------------------------------------

N_ITERS = 100
MF_CONFIG_PATH = os.path.join(os.path.dirname(__file__),
    "oil_flow_related/original_gp_lvm_configs/base_lr_0001_s7.yaml")
SSVI_CONFIG_PATH = os.path.join(os.path.dirname(__file__),
    "oil_flow_related/ssvi_configs/base_lr_0001_s7.yaml")


def _gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def _reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


# ===== MF GP-LVM (nn_gauss) =================================================

def run_mf_gplvm(n_iters: int):
    """Set up and train the MF GP-LVM (nn_gauss) for n_iters steps."""
    args = mf_create_parser().parse_args(['--config', MF_CONFIG_PATH])
    config, _ = mf_load_config(config_path=args.config, config_class=OriginalGPLVMConfig, args=args)

    config.save_results = False
    config.enable_testing = False
    config.save_plots = False

    torch.set_default_device(config.device)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    download_oilflow_data_if_needed()
    _, _, _, _, Y, labels = load_real_data(config.dataset)
    Y_cpu = Y.cpu() if Y.is_cuda else Y
    labels_cpu = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
    Y_np = Y_cpu.numpy()
    indices = np.arange(len(Y))
    _, _, Y_train_np, _, _, _ = train_test_split(
        indices, Y_np, labels_cpu,
        test_size=config.test_size, random_state=config.random_seed)
    Y_train = torch.tensor(Y_train_np, dtype=torch.float64, device=config.device)

    N = len(Y_train)
    data_dim = Y_train.shape[1]
    latent_dim = config.latent_dim
    n_inducing = config.n_inducing
    batch_size = config.batch_size

    mu_x_pca, _, _ = torch.pca_lowrank(Y_train, latent_dim)
    X_init = torch.nn.Parameter(mu_x_pca)
    X_prior_mean = mu_x_pca

    nn_layers = config.nn_layers
    prior_x = MultivariateNormalPrior(
        X_prior_mean,
        torch.eye(X_prior_mean.shape[1], dtype=torch.float64, device=config.device))
    X = NNEncoder(N, latent_dim, prior_x, data_dim, layers=nn_layers)

    model = OilFlowModel(N, data_dim, latent_dim, n_inducing, X,
                         nn_layers=nn_layers, device=config.device, config=config)
    likelihood = GaussianLikelihood(
        noise_constraint=Positive(transform=transforms.ExpTransform()))
    model.covar_module.base_kernel.register_constraint(
        "raw_lengthscale", Positive(transform=transforms.ExpTransform()))
    model.covar_module.register_constraint(
        "raw_outputscale", Positive(transform=transforms.ExpTransform()))

    with torch.no_grad():
        noise_scale = torch.tensor(config.init_output_noise_scale,
                                   dtype=torch.float64, device=config.device)
        likelihood.initialize(noise=noise_scale)

    elbo = VariationalELBO(likelihood, model, num_data=N)
    if config.device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
        if hasattr(model, 'X'):
            model.X = model.X.to(config.device)
    if hasattr(model.X, 'jitter'):
        model.X = model.X.to(config.device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=config.learning_rate)

    n_params = _count_params(model) + _count_params(likelihood)
    print(f"  MF GP-LVM (nn_gauss) — {n_params:,} trainable params")
    print(f"  N_train={N}, D={data_dim}, Q={latent_dim}, M={n_inducing}, batch={batch_size}")

    model.train()
    likelihood.train()
    _reset_gpu_stats()

    # Timed run
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in trange(n_iters, desc="MF GP-LVM", leave=True):
        idx = model._get_batch_idx(batch_size)
        idx_t = torch.tensor(idx, dtype=torch.int64, device=config.device)
        optimizer.zero_grad()
        s = model.sample_latent_variable(Y_train)
        out = model(s[idx_t])
        loss = -elbo(out, Y_train[idx_t].T).sum()
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = _gpu_mem_mb()

    # Free all model tensors before returning
    del model, likelihood, elbo, optimizer, Y_train, X, prior_x, mu_x_pca
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return elapsed, peak_mem, n_params


# ===== SSVI GP-LVM ==========================================================

def run_ssvi_gplvm(n_iters: int):
    """Set up and train the SSVI GP-LVM for n_iters steps."""
    args = ssvi_create_parser().parse_args(['--config', SSVI_CONFIG_PATH])
    config, _ = ssvi_load_config(config_path=args.config, config_class=OilFlowSSVIConfig, args=args)

    config.save_results = False
    config.enable_metrics = False
    config.enable_gradient_logging = False
    config.save_training_plots = False
    config.training_iterations = n_iters

    torch.set_default_device(config.device)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    y_train, labels_train, Vx, mu_x, split_data = load_oil_flow_data(config)

    model = OilFlow_SSVI_GPLVM(
        config=config, Vx=Vx, mu_x=mu_x, M=y_train.shape[1])

    N = y_train.shape[0]
    data_dim = y_train.shape[1]
    batch_size = config.batch_size

    n_params = _count_params(model)
    print(f"  SSVI GP-LVM — {n_params:,} trainable params")
    print(f"  N_train={N}, D={data_dim}, Q={config.latent_dim}, "
          f"M={config.n_inducing_points}, batch={batch_size}, P={config.P}")

    _reset_gpu_stats()

    # Timed run
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in trange(n_iters, desc="SSVI GP-LVM", leave=True):
        bi = torch.randperm(N, device=y_train.device)[:batch_size]
        model.train_step(y_train[bi], bi, prior_weight=1.0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = _gpu_mem_mb()

    del model, y_train, Vx, mu_x, split_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return elapsed, peak_mem, n_params


# ===== Main ==================================================================

def main():
    print("=" * 60)
    print(f"  Training-time & memory comparison  ({N_ITERS} iterations)")
    print("=" * 60)

    # --- SSVI GP-LVM ---
    print("\n>>> SSVI GP-LVM")
    ssvi_time, ssvi_mem, ssvi_params = run_ssvi_gplvm(N_ITERS)

    # Clean up before next run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float64)

    # --- MF GP-LVM ---
    print("\n>>> MF GP-LVM (nn_gauss)")
    mf_time, mf_mem, mf_params = run_mf_gplvm(N_ITERS)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"{'':30s} {'MF GP-LVM':>14s} {'SSVI GP-LVM':>14s}")
    print("-" * 60)
    print(f"{'Trainable parameters':30s} {mf_params:>14,} {ssvi_params:>14,}")
    print(f"{'Wall-clock time (s)':30s} {mf_time:>14.2f} {ssvi_time:>14.2f}")
    print(f"{'Time per iteration (ms)':30s} {mf_time/N_ITERS*1000:>14.2f} {ssvi_time/N_ITERS*1000:>14.2f}")
    if torch.cuda.is_available():
        print(f"{'Peak GPU memory (MB)':30s} {mf_mem:>14.1f} {ssvi_mem:>14.1f}")
    else:
        print(f"{'(GPU not available — memory not tracked)':30s}")
    print(f"{'Speed-up (MF / SSVI)':30s} {mf_time/ssvi_time:>14.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
