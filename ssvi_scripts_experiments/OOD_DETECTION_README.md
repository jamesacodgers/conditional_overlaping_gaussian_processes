# OOD Detection with GP-LVM SSVI

This document explains the Out-of-Distribution (OOD) detection functionality for the GP-LVM SSVI model.

## Overview

OOD detection aims to identify whether a new sample comes from the same distribution as the training data (in-distribution, ID) or from a different distribution (out-of-distribution, OOD).

**Key Idea**: A model trained on one dataset should assign higher uncertainty/reconstruction error to samples from a different dataset.

---

## Configuration Parameters

### OOD-Specific Parameters (in `OODSSVIConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id_dataset` | str | "FashionMNIST" | In-distribution dataset. Options: "MNIST", "FashionMNIST" |
| `ood_samples` | int | 1000 | Number of OOD samples for evaluation |
| `ood_score_types` | list | ["nlpd", "rmse", "variance"] | Score types to compute for OOD detection |

**Note**: OOD score computation uses the model's `P` parameter (inducing samples) for Monte Carlo sampling.

### Dataset Filtering Parameters

**MNIST filtering** (used when `id_dataset="MNIST"` for ID, or for OOD when ID is FashionMNIST):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mnist_digits` | list | None | Digits to include, e.g., `[0, 1, 2]`. `None` = all digits (0-9) |
| `samples_per_digit` | int | None | Samples per digit. `None` = use all available |

**FashionMNIST filtering** (used when `id_dataset="FashionMNIST"` for ID, or for OOD when ID is MNIST):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fashion_mnist_classes` | list | None | Classes to include, e.g., `[1, 2]`. `None` = all classes |
| `fashion_samples_per_class` | int | None | Samples per class. `None` = use all available |

**FashionMNIST Class Labels:**
| Label | Category |
|-------|----------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Dataset Pairs

The script automatically pairs datasets:
- If `id_dataset = "FashionMNIST"` → OOD dataset is "MNIST"
  - ID uses: `fashion_mnist_classes`, `fashion_samples_per_class`
  - OOD uses: `mnist_digits`, `samples_per_digit`
- If `id_dataset = "MNIST"` → OOD dataset is "FashionMNIST"
  - ID uses: `mnist_digits`, `samples_per_digit`
  - OOD uses: `fashion_mnist_classes`, `fashion_samples_per_class`

---

## Data Split

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IN-DISTRIBUTION DATASET                               │
│                    (e.g., FashionMNIST)                                  │
│                                                                          │
│    Full Dataset (70,000 samples)                                         │
│            │                                                             │
│            ▼                                                             │
│    ┌───────────────────────────────────────┐                            │
│    │ Filter by fashion_mnist_classes=[1,2] │                            │
│    │ + fashion_samples_per_class=100       │                            │
│    └───────────────────────────────────────┘                            │
│            │                                                             │
│            ▼                                                             │
│    Filtered Dataset (200 samples: 100 Trouser + 100 Pullover)           │
│            │                                                             │
│            ▼                                                             │
│    ┌───────────────────────────────────────┐                            │
│    │      Train/Test Split (test_size=0.2) │                            │
│    └───────────────────────────────────────┘                            │
│            │                                                             │
│       ┌────┴────┐                                                        │
│       ▼         ▼                                                        │
│   Training   ID Test                                                     │
│   (160)      (40)                                                        │
│       │         │                                                        │
│       │         └──── Same classes as training (1,2)                    │
│       │                                                                  │
│       └──── Used to train GP-LVM                                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    OUT-OF-DISTRIBUTION DATASET                           │
│                    (e.g., MNIST - automatically paired)                  │
│                                                                          │
│    Full Dataset (10,000 test samples)                                    │
│            │                                                             │
│            ▼                                                             │
│    ┌───────────────────────────────────────┐                            │
│    │ Filter by mnist_digits=[1,2]          │  (optional)                │
│    │ + samples_per_digit=100               │                            │
│    └───────────────────────────────────────┘                            │
│            │                                                             │
│            ▼                                                             │
│    ┌───────────────────────────────────────┐                            │
│    │ Sample ood_samples=200                │                            │
│    └───────────────────────────────────────┘                            │
│            │                                                             │
│            ▼                                                             │
│       OOD Test                                                           │
│       (200)                                                              │
│         │                                                                │
│         └──── Used for OOD evaluation                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Points

1. **Filtering happens BEFORE split**: Both training and ID test sets contain the same classes
2. **ID Test = Same distribution as Training**: This is correct for OOD detection baseline
3. **OOD uses its own filtering**: `mnist_digits` for MNIST, `fashion_mnist_classes` for FashionMNIST
4. **Automatic pairing**: If ID is FashionMNIST → OOD is MNIST (and vice versa)

### Split Details

1. **Training Data**: Samples from ID dataset used to train the GP-LVM
2. **ID Test Data**: Held-out samples from ID dataset (same classes as training)
3. **OOD Data**: Samples from OOD dataset (controlled by `ood_samples` + optional filtering)

### Example (Fast Config with `id_dataset="FashionMNIST"`)

```
FashionMNIST (ID):
  - Filter: classes [1, 2] (Trouser, Pullover), 100 samples each = 200 total
  - Training: 160 samples (80% of 200)
  - ID Test: 40 samples (20% of 200)
  - Classes in both: [1, 2] only

MNIST (OOD):
  - Filter: digits [1, 2], 100 samples each (if configured)
  - OOD Test: 200 samples (from ood_samples parameter)
```

### Using All Classes

For more comprehensive OOD detection, use all classes:

```python
OODSSVIConfig(
    # Use all 10 FashionMNIST classes
    fashion_mnist_classes=None,
    fashion_samples_per_class=500,  # 5000 total samples
    
    # Use all 10 MNIST digits
    mnist_digits=None,
    samples_per_digit=500,
    
    id_dataset="FashionMNIST",
    ood_samples=1000,
)
```

---

## OOD Score Types

Three methods to compute OOD scores for each sample:

### 1. NLPD (Negative Log Predictive Density)

**Mathematical Definition**:
```
NLPD(y) = -log p(y | model)
        = -log N(y | f_mean, f_var + sigma^2)
        = 0.5 * log(2*pi*var) + 0.5 * (y - mean)^2 / var
```

Where:
- `f_mean`: GP predictive mean
- `f_var`: GP predictive variance  
- `sigma^2`: Observation noise

**Intuition**: 
- NLPD measures how "surprising" a sample is to the model
- Higher NLPD = lower probability = more likely OOD
- Captures both reconstruction error AND uncertainty

### 2. RMSE (Root Mean Square Error)

**Mathematical Definition**:
```
RMSE(y) = sqrt( sum_d (y_d - f_mean_d)^2 )
```

**Intuition**:
- Measures reconstruction error
- Higher RMSE = worse reconstruction = more likely OOD
- Simpler than NLPD but ignores uncertainty

### 3. Variance (Latent Uncertainty)

**Mathematical Definition**:
```
Variance(y) = sum_q sigma_x_q^2
```

Where `sigma_x` is the latent variance from the encoder.

**Intuition**:
- Measures encoder uncertainty about latent position
- Higher variance = more uncertain = more likely OOD
- Purely uncertainty-based, ignores reconstruction

---

## How Scores Become Metrics

This section explains how NLPD/RMSE/Variance scores are used to compute AUROC, AUPRC, and FPR@80.

### Step-by-Step Process

**Step 1: Compute scores for ALL samples**
```
ID Test samples:  [y1, y2, ..., y40]   --> [score_1, score_2, ..., score_40]
OOD samples:      [z1, z2, ..., z200]  --> [score_1, score_2, ..., score_200]
```

**Step 2: Assign binary labels**
```
ID samples  --> label = 0 (negative class)
OOD samples --> label = 1 (positive class)

Combined:
scores = [0.5, 0.3, 0.8, ..., 2.1, 3.5, 1.9, ...]
labels = [0,   0,   0,   ..., 1,   1,   1,   ...]
         |---- ID ----|      |---- OOD ----|
```

**Step 3: Sweep through ALL possible thresholds (NO fixed threshold!)**
```
For each threshold t in sorted(all_scores):
    - Predict "OOD" if score > t
    - Compute TP, FP, TN, FN
    - Calculate TPR = TP/(TP+FN), FPR = FP/(FP+TN)
```

**Step 4: Compute metrics from the sweep**
```
AUROC = Area under (FPR, TPR) curve
AUPRC = Area under (Recall, Precision) curve  
FPR@80 = FPR when TPR = 0.80
```

### Visual Example

```
Suppose we have:
  ID scores:  [0.2, 0.3, 0.4, 0.5]  (4 samples, label=0)
  OOD scores: [0.6, 0.8, 1.2, 1.5]  (4 samples, label=1)

Threshold sweep:
  t=0.1: All predicted OOD  --> TPR=4/4=1.0, FPR=4/4=1.0
  t=0.35: Score>0.35 is OOD --> TPR=4/4=1.0, FPR=3/4=0.75
  t=0.55: Score>0.55 is OOD --> TPR=4/4=1.0, FPR=0/4=0.0  <-- Perfect!
  t=0.7:  Score>0.7 is OOD  --> TPR=3/4=0.75, FPR=0/4=0.0
  ...

ROC Curve points: (0,0) -> (0,1) -> (1,1)
AUROC = 1.0 (perfect separation in this example)
```

### Key Insight: No Fixed Threshold

- **We do NOT pick a single threshold** to classify samples
- Instead, we evaluate separation quality **across all possible thresholds**
- This gives a threshold-independent measure of how distinguishable ID and OOD are
- Higher scores should consistently appear for OOD samples

### Implementation (using scikit-learn)

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Combine scores
all_scores = np.concatenate([id_scores, ood_scores])

# Create labels: 0 for ID, 1 for OOD
labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])

# Compute metrics (scikit-learn handles threshold sweep internally)
auroc = roc_auc_score(labels, all_scores)
auprc = average_precision_score(labels, all_scores)
```

### Why This Works

| Score Type | ID Samples (should be low) | OOD Samples (should be high) |
|------------|---------------------------|------------------------------|
| NLPD | Model "knows" ID data -> low surprise | Model doesn't know OOD -> high surprise |
| RMSE | Good reconstruction -> low error | Poor reconstruction -> high error |
| Variance | Confident latent position -> low var | Uncertain position -> high variance |

If the model learned ID distribution well, OOD samples will have higher scores, making them separable.

---

## Evaluation Metrics

### 1. AUROC (Area Under ROC Curve)

**What it measures**: How well the scores separate ID from OOD samples.

**Mathematical Definition**:
```
AUROC = P(score_OOD > score_ID)
```

The probability that a randomly chosen OOD sample has a higher score than a randomly chosen ID sample.

**Interpretation**:
- AUROC = 0.5: Random guessing (no discrimination)
- AUROC = 1.0: Perfect separation
- AUROC > 0.9: Excellent OOD detection

**Visual**: Area under the curve plotting True Positive Rate vs False Positive Rate.

```
TPR |    ____
    |   /
    |  /
    | /
    |/_________ FPR
```

### 2. AUPRC (Area Under Precision-Recall Curve)

**What it measures**: Precision-recall tradeoff, especially useful for imbalanced datasets.

**Mathematical Definition**:
```
Precision = TP / (TP + FP)  -- Of detected OOD, how many are actually OOD?
Recall = TP / (TP + FN)     -- Of actual OOD, how many did we detect?
```

**Interpretation**:
- More informative than AUROC when classes are imbalanced
- AUPRC = 1.0: Perfect detection
- Baseline depends on class balance (not 0.5 like AUROC)

### 3. FPR@80 (False Positive Rate at 80% True Positive Rate)

**What it measures**: When we detect 80% of OOD samples correctly, what fraction of ID samples do we incorrectly flag?

**Mathematical Definition**:
```
FPR@80 = FP / (FP + TN)  when TPR = 0.80
```

**Interpretation**:
- Lower is better
- FPR@80 = 0.05: Only 5% of ID samples wrongly flagged when detecting 80% of OOD
- FPR@80 = 0.50: Half of ID samples wrongly flagged (poor detection)

**Why 80%?**: Common benchmark threshold - we want to catch most OOD samples (80%) while minimizing false alarms.

---

## Evaluation Process

### Important: OOD Data is Never Used During Training

- **Training**: Uses ONLY ID training data (e.g., FashionMNIST)
- **Validation during training**: Uses ID test data for RMSE/NLPD metrics
- **OOD Evaluation**: Happens ONLY at the end, after training completes
- **OOD data**: Never seen by the model until final evaluation

### Step-by-Step

```
1. TRAIN MODEL (ID data only)
   - Train GP-LVM SSVI on in-distribution (ID) training data
   - Model learns latent structure of ID data
   - OOD data is NOT used here

2. COMPUTE SCORES (after training)
   For each sample in ID Test and OOD Test:
   
   a) Encode: y → (mu_x, sigma_x)
      Pass through encoder to get latent distribution
   
   b) Sample: x ~ N(mu_x, sigma_x)
      Sample latent point
   
   c) Predict: x → (f_mean, f_var)
      Get GP predictive distribution
   
   d) Score: Compute NLPD, RMSE, or Variance
      Higher score = more likely OOD

3. COMPUTE METRICS
   - Combine ID scores and OOD scores
   - Create labels: ID=0, OOD=1
   - Compute AUROC, AUPRC, FPR@80

4. VISUALIZE
   - Plot score histograms (ID vs OOD distributions)
   - Good separation = distinct histograms
```

### Visual Example

```
Score Distribution:

        ID (FashionMNIST)          OOD (MNIST)
              |||                      |||
             |||||                    |||||
            |||||||                  |||||||
           |||||||||                |||||||||
    ______|||||||||||||________|||||||||||||______
    Low Score                              High Score
    
    ^ Good separation = High AUROC
```

---

## Usage Examples

```bash
# Run with FashionMNIST as ID (default)
python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py

# Run with MNIST as ID
python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py --id-dataset MNIST

# Custom OOD samples
python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py --ood-samples 2000

# Full training with more iterations
python ssvi_scripts_experiments/mnist_ood_GPLVM_SSVI_AM_INF.py \
    --training-iterations 1000 \
    --ood-samples 5000
```

---

## Plotting and Visualization

### OOD-Aware Plot Helper

The script uses `src/mnist_ood_plot_helper.py` which is a duplicate of `mnist_plot_helper.py` with **only legend changes** for OOD detection.

### Legend Format

All plots show dataset name and ID/OOD status in legends:

| Dataset | Label Example |
|---------|---------------|
| MNIST (ID) | `MNIST Digit 1 (ID)` |
| MNIST (OOD) | `MNIST Digit 2 (OOD)` |
| FashionMNIST (ID) | `FashionMNIST Trouser (ID)` |
| FashionMNIST (OOD) | `FashionMNIST Pullover (OOD)` |

### Plots with Legends

1. **Latent Space 2D Scatter** - Shows class separation with legend
2. **1D Histogram** - Distribution per class with legend  
3. **3D Scatter Plot** - Interactive plotly with legend
4. **OOD Score Histograms** - ID vs OOD score distributions

---

## Output Files

After running, results are saved to `ood_results/ssvi_results/`:

```
ood_ssvi_FashionMNIST_fast_seed_7_TIMESTAMP/
    plots/
        ood_hist_nlpd_7.png           # NLPD score histogram (ID vs OOD)
        ood_hist_rmse_7.png           # RMSE score histogram (ID vs OOD)
        ood_hist_variance_7.png       # Variance score histogram (ID vs OOD)
        training_results.png          # Training curve + latent space + lengthscales
        1d_histogram_most_important_dim.png  # 1D histogram with class legend
        3d_scatter_top3_dims.html     # Interactive 3D scatter with class legend
        metrics_history.png           # RMSE/NLPD over training
    models/
        ood_ssvi_7.pkl                # Trained model
    logs/
        training_log.txt              # Training progress
    experiment_summary.json           # All metrics including OOD results
```

---

## Interpreting Results

### Good OOD Detection
```
AUROC:  > 0.90
AUPRC:  > 0.90  
FPR@80: < 0.20
```

### Example Output
```
OOD DETECTION SUMMARY
============================================================
Dataset Pair: FashionMNIST (ID) vs MNIST (OOD)
------------------------------------------------------------

NLPD Scores:
  AUROC:  0.9526   <- Excellent separation
  AUPRC:  0.9971   <- Near-perfect precision-recall
  FPR@80: 0.0500   <- Only 5% false positives at 80% detection

RMSE Scores:
  AUROC:  0.9527
  AUPRC:  0.9971
  FPR@80: 0.0500
============================================================
```

This indicates the GP-LVM SSVI model can effectively distinguish FashionMNIST from MNIST samples.

---

## References

- Original OOD detection metrics from: "A Baseline for Detecting Misclassified and Out-of-Distribution Examples" (Hendrycks & Gimpel, 2017)
- FPR@80 threshold commonly used in OOD detection benchmarks

