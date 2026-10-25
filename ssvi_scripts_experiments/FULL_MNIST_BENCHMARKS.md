# Full-size MNIST / F-MNIST: A-MF-GPLVM vs A-S-GPLVM

These runs repeat the paper's MNIST and F-MNIST OOD experiments (arXiv 2610.03647) on the
full datasets, using the official 60k/10k torchvision split, seeds 0-4. The paper trained on
`id_samples_per_class: 1000` with `test_size: 0.1`, i.e. 9k train and 1k test images.

## What changed

- `use_official_split` (both arms, default off): load torchvision's own train and test splits
  instead of a seeded random split of the pooled 70k images. The helper is
  `src/data_utils.py: official_train_test_split`.
- `encode_batch_only` (MFVI arm only, default off): the `nn_gauss` encoder sees only the
  minibatch instead of all N images each step. `x_kl` is sum/n over the rows it encodes, so this
  is an unbiased estimate of the same objective; only the gradient noise differs.
- `full_mnist_generate_config_grid.py` writes `ood_mnist_related/full_*_v1_*/`. Each config is the
  paper's seed-0 config plus:
  - `id_samples_per_class: null`
  - `use_official_split: true`
  - the seed
  - `metrics_frequency` and `checkpoint_frequency` raised from 4000 to 20000, because each
    evaluation now passes over 70k images. This affects logging only.
  - in the `*_batchenc_*` folders, also `encode_batch_only: true`

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt pyyaml

# exact paper code for both arms (MFVI ~21 h/run), one run per GPU
GPUS="0 1 2 3" bash ssvi_scripts_experiments/run_full_mnist_benchmarks.sh

# or MFVI with encode_batch_only (~1.2 h/run)
GPUS="0 1 2 3" bash ssvi_scripts_experiments/run_full_mnist_benchmarks.sh \
  ood_mnist_related/full_mnist_v1_ssvi ood_mnist_related/full_mnist_v1_batchenc_orig_gplvm \
  ood_mnist_related/full_fmnist_v1_ssvi ood_mnist_related/full_fmnist_v1_batchenc_orig_gplvm

# mean ± std over seeds -> full_mnist_summary.{csv,md}
cd ssvi_scripts_experiments && ../.venv/bin/python full_mnist_summarise.py <folders...>
```

The launcher:
- runs one worker per GPU in `GPUS`, each taking the next unclaimed config from a shared queue
  (SSVI first, since those runs are longest, then by seed)
- also works with one launcher per node over a shared checkout
- skips configs that already have a `.done` marker; after a crash, delete the stale `.claimed`
  files so those runs restart from step 0
- logs each run to `<folder>/logs/`

## Timings on one RTX PRO 6000 Blackwell, float64, full MNIST

| arm | s/step | 200k steps | one full eval (70k) | per run |
|---|---|---|---|---|
| SSVI | 0.176 | 9.8 h | ~5 min | ~11 h |
| MFVI, exact code | 0.37 | 20.5 h | ~3 min | ~21 h |
| MFVI, `encode_batch_only` | 0.010 | 35 min | ~3 min | ~1.2 h |

- In the exact-code MFVI step, 344 ms of the 0.37 s is the encoder pass over all 60k images;
  over the 32-image batch it takes 2.5 ms.
- Two runs sharing one GPU each slow to 0.371 s/step (SSVI), so there's no gain: give each run
  its own GPU.

## Environment check against the paper

The unchanged paper MNIST seed-0 configs were run in this environment to step 4000 and compared
with the stored July runs:

| arm | metric | paper | here |
|---|---|---|---|
| SSVI | train RMSE / test RMSE | 0.2629 / 0.2625 | 0.2627 / 0.2621 |
| SSVI | train NLPD / test NLPD | 601.1 / 601.5 | 601.6 / 600.0 |
| SSVI | full-ELBO loss | 663.1 | 645.1 |
| MFVI | train RMSE | 0.2599 | 0.2599 |
| MFVI | train NLPD | 795.27 | 795.30 |
| MFVI | full-ELBO loss | 809.07 | 809.04 |

## Caveats

- The MFVI arm has no beta annealing; its loss is plain `-elbo`. Only the SSVI arm anneals
  beta from 10 to 1. These runs follow the code, not the paper's text.
- The OOD set is unchanged: 500 per class from the other dataset's test split.
- `ood_n_samples` stays as in each paper config: MFVI uses 16 on MNIST and 4 on F-MNIST.
