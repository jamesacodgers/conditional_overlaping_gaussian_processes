import numpy as np
import os


def create_fold_indices(data_shape, n_folds, seed):
    """Create masks for cross validation, with entries being dropped out not rows."""

    np.random.seed(seed)

    # get the indices of every point in the data
    nrows, ncols = data_shape
    all_indices = np.arange(nrows * ncols)  # .reshape(nrows, ncols)
    np.random.shuffle(all_indices)

    fold_indices = []

    for fold in range(5):
        fold_size = (nrows * ncols) // 5
        fold_idx = all_indices[fold * fold_size : (fold + 1) * fold_size]
        fold_indices.append(fold_idx)
        print(f"Fold {fold+1} indices: {fold_idx}")

    # get the masks

    masks = []
    for fold_idx in fold_indices:
        mask = np.ones(data_shape, dtype=bool)
        mask[np.unravel_index(fold_idx, data_shape)] = False
        masks.append(mask)

    return fold_indices, masks


def main():

    data_shape = (195, 10)
    n_folds = 5
    seed = 42

    fold_indices, masks = create_fold_indices(data_shape, n_folds, seed)

    # check the masks
    binary_masks = []
    for i, mask in enumerate(masks):
        binary_mask = mask.astype(int)
        binary_masks.append(binary_mask)
        print(f"Mask for Fold {i+1}:\n{binary_mask}\n")

    # save the masks
    binary_masks = np.array(binary_masks)
    # make directory if it doesn't exist
    os.makedirs("masks", exist_ok=True)
    np.save("masks/cv_binary_masks.npy", binary_masks)

    assert binary_masks.sum(axis=0).min() == n_folds - 1
    assert binary_masks.sum(axis=0).max() == n_folds - 1

    # for each mask get the sums of the columns and of the rows
    for i, binary_mask in enumerate(binary_masks):
        row_sums = binary_mask.sum(axis=1)
        col_sums = binary_mask.sum(axis=0)
        print(f"Fold {i+1} - Row sums: {row_sums}")
        print(f"Fold {i+1} - Column sums: {col_sums}")


if __name__ == "__main__":
    main()
