"""
Create masks for Case 2: Row-wise CV with entry-wise masking on test set.

Case 2 setup:
- Split data into 5 folds by ROW (80% train, 20% test)
- Train rows: all entries observed (mask = True)
- Test rows: 80% entries observed, 20% masked for evaluation

Output: masks_case_2.npy with shape (5, 2730, 10)
"""

import numpy as np
from pathlib import Path
import pandas as pd

# Load data to get dimensions
data_path = Path(__file__).parent / "assay_to_yield_training_sequences.csv"
df = pd.read_csv(data_path)

SCORE_COLUMNS = [
    "Sort1_mean_score",
    "Sort2_mean_score",
    "Sort3_mean_score",
    "Sort4_mean_score",
    "Sort5_mean_score",
    "Sort6_mean_score",
    "Sort7_mean_score",
    "Sort8_mean_score",
    "Sort9_mean_score",
    "Sort10_mean_score",
]

data = df[SCORE_COLUMNS].values
n_rows, n_cols = data.shape
print(f"Data shape: {n_rows} rows x {n_cols} columns")

# Parameters
n_folds = 5
test_missing_ratio = 0.2  # 20% of test entries will be masked
seed = 42

np.random.seed(seed)

# Create row indices for each fold
indices = np.arange(n_rows)
np.random.shuffle(indices)

fold_size = n_rows // n_folds
fold_indices = []
for i in range(n_folds):
    start = i * fold_size
    if i == n_folds - 1:
        # Last fold gets remaining rows
        end = n_rows
    else:
        end = start + fold_size
    fold_indices.append(indices[start:end])

print(f"Fold sizes: {[len(f) for f in fold_indices]}")

# Create masks for each fold
# mask[fold, row, col] = True means entry is OBSERVED
# mask[fold, row, col] = False means entry is MISSING (held out for evaluation)
masks = np.ones((n_folds, n_rows, n_cols), dtype=bool)

for fold in range(n_folds):
    test_indices = fold_indices[fold]

    # For test rows, randomly mask some entries
    for row_idx in test_indices:
        # Randomly select entries to mask
        n_to_mask = int(n_cols * test_missing_ratio)
        cols_to_mask = np.random.choice(n_cols, size=n_to_mask, replace=False)
        masks[fold, row_idx, cols_to_mask] = False

# Also save train/test indices for convenience
train_test_splits = []
for fold in range(n_folds):
    test_idx = fold_indices[fold]
    train_idx = np.concatenate([fold_indices[i] for i in range(n_folds) if i != fold])
    train_test_splits.append({
        'train_indices': train_idx,
        'test_indices': test_idx
    })

# Summary statistics
print(f"\nMask shape: {masks.shape}")
for fold in range(n_folds):
    test_idx = fold_indices[fold]
    train_idx = np.concatenate([fold_indices[i] for i in range(n_folds) if i != fold])

    train_observed = masks[fold, train_idx, :].sum()
    train_total = len(train_idx) * n_cols
    test_observed = masks[fold, test_idx, :].sum()
    test_total = len(test_idx) * n_cols

    print(f"Fold {fold + 1}:")
    print(f"  Train: {len(train_idx)} rows, {train_observed}/{train_total} entries observed ({100*train_observed/train_total:.1f}%)")
    print(f"  Test:  {len(test_idx)} rows, {test_observed}/{test_total} entries observed ({100*test_observed/test_total:.1f}%)")

# Save masks
output_dir = Path(__file__).parent / "masks"
output_dir.mkdir(exist_ok=True)

np.save(output_dir / f"masks_case_2_seed{seed}.npy", masks)
print(f"\nSaved masks to: {output_dir / f'masks_case_2_seed{seed}.npy'}")

# Save train/test splits
np.save(output_dir / f"train_test_splits_case_2_seed{seed}.npy", train_test_splits)
print(f"Saved train/test splits to: {output_dir / f'train_test_splits_case_2_seed{seed}.npy'}")