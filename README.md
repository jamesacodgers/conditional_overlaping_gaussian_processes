# Amortized Structured Stochastic Variational Inference for Gaussian Process Latent Variable Models

**Authors:** Maksym Tretiakov, Sarah Lucie Filippi, Vincent Fortuin, Ruth Misener, Ruby Sedgwick, and James Odgers.

This directory contains experimental code for the paper *Amortized Structured Stochastic Variational Inference for Gaussian Process Latent Variable Models*. We introduce the **Amortized Structured GPLVM (A-S-GPLVM)**, which replaces the mean-field independence assumption between latent variables and sparse GP inducing variables with a structured variational posterior, $q(X\mid U)q(U)$. The conditional latent posterior is parameterized by an amortized encoder, and the objective supports mini-batch optimization.

Experiments cover **Oilflow, qPCR, MNIST, and Fashion-MNIST**, comparing A-S-GPLVM against mean-field GPLVM baselines. The paper also reports an importance-weighted baseline (IWVI-GPLVM); the commands below document the A-S-GPLVM and mean-field implementations in this directory.

## Setup

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ssvi_scripts_experiments
```

Training scripts use paths relative to `ssvi_scripts_experiments/`. The paper configurations use CUDA; replace `--device cuda` with `--device cpu` to run on CPU.

Oilflow files are included under `data/`. qPCR is downloaded on first use, while MNIST and Fashion-MNIST are downloaded through `torchvision`.

## Models

| Model | Variational approximation | Entry point |
|---|---|---|
| **A-S-GPLVM** | Structured posterior with encoder conditioned on inducing variables $U$ | `*_GPLVM_SSVI_AM_INF.py` |
| **A-MF-GPLVM** | Mean-field posterior with amortized encoder | `*_GPLVM.py` (`nn_gauss`) |
| **MF-GPLVM** | Mean-field posterior with separately optimized local parameters | `*_GPLVM.py` (`gauss`) |

Mean-field configuration files can run both `gauss` and `nn_gauss` models. Configurations specify the random seed, number of inducing points, encoder architecture, optimizer settings, and number of training iterations. In structured configurations, `P` denotes the number of Monte Carlo samples of the inducing variables.

## Reproducing experiments

### Oilflow and qPCR

The main configuration folders contain seeds `0`–`4`, use 32 inducing points and batch size 32, and specify 100,000 training iterations.

**A-S-GPLVM:**

```bash
./run_ssvi_configs_folder.sh oil_flow_related/final_v1_ssvi --device cuda
./run_qpcr_ssvi_configs_folder.sh qpcr_related/final_v1_ssvi --device cuda
```

**MF-GPLVM and A-MF-GPLVM:**

```bash
./run_original_gplvm_configs_folder.sh oil_flow_related/final_v1_orig_gplvm --device cuda
./run_qpcr_original_gplvm_configs_folder.sh qpcr_related/final_v1_orig_gplvm --device cuda
```

To test a single configuration without completing the full run:

```bash
python oil_flow_GPLVM_SSVI_AM_INF.py \
  --config oil_flow_related/final_v1_ssvi/n_ind_32_arch_192_len_1_sig_1_b_32_s_0_lr_0001_n_steps_100000_P_64_miss_f_ratio_00_test_01.yaml \
  --device cuda --training-iterations 200
```

Omit `--training-iterations` to use the iteration count specified in the YAML.

### MNIST and Fashion-MNIST

Image experiments use 64 inducing points, batch size 32, and the configuration folders below. The non-amortized MF-GPLVM is not evaluated on these datasets.

**A-S-GPLVM:**

```bash
python mnist_ood_GPLVM_SSVI_AM_INF.py \
  --config ood_mnist_related/reverse_mnist_ood_experiment_final_v3_ssvi/ood_mnist_id_fmnist_ood_n_ind_64_arch_768_len_10_b_32_s_0_lr_00001_n_steps_200000_P_4_test_01.yaml \
  --device cuda
```

**A-MF-GPLVM:**

```bash
python mnist_ood_GPLVM.py \
  --config ood_mnist_related/reverse_mnist_ood_experiment_final_v3_orig_gplvm/ood_mnist_id_fmnist_ood_n_ind_64_arch_768_len_10_b_32_s_0_lr_00001_n_steps_200000_test_01.yaml \
  --device cuda
```

For Fashion-MNIST, use the corresponding configurations in `fashion_mnist_ood_experiment_final_v2_ssvi/` and `fashion_mnist_ood_experiment_final_v2_orig_gplvm/`. The checked-in image configuration folders contain only a subset of the seeds used for the paper; additional runs can be configured by changing `random_seed` in the YAML. For out-of-distribution evaluations, see `OOD_DETECTION_README.md`.

## Results and plots

Each run writes outputs next to its configuration, under `ssvi_run_results/` or `orig_gplvm_run_results/`. Outputs include `experiment_summary.json`, `config.json`, `training_log.txt`, model checkpoints, and diagnostic plots. Summaries contain train/test RMSE, NLPD, CRPS, empirical predictive-interval coverage, and final negative ELBO.

Aggregate Oilflow and qPCR runs into mean ± standard deviation tables (replace `26_07` with the relevant run date tag):

```bash
python oil_flow_related/aggregate_final_v1_metrics_table.py --dataset oilflow --date-tag 26_07
python oil_flow_related/aggregate_final_v1_metrics_table.py --dataset qpcr --date-tag 26_07
```

For image experiments:

```bash
python ood_mnist_related/aggregate_mnist_fmnist_metrics_table.py
```

Training-curve comparisons are produced with `automatic_results_comparison.py` (Oilflow/qPCR) and `mnist_automatic_results_comparison.py` (image experiments). The Oilflow ablation figures are generated with `ablation_study_plots.py`.
