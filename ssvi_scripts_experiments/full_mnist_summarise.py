"""
Mean +- std over seeds of every final metric in experiment_summary.json, per config
folder (dataset x arm), for the full-size MNIST / FashionMNIST runs.

Usage (from ssvi_scripts_experiments/):
    python full_mnist_summarise.py                       # the four full_*_v1_* folders
    python full_mnist_summarise.py <folder> [<folder>...]
Writes full_mnist_summary.csv and full_mnist_summary.md next to this script.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DEFAULT_FOLDERS = [
    HERE / "ood_mnist_related" / name
    for name in ["full_mnist_v1_ssvi", "full_mnist_v1_orig_gplvm", "full_fmnist_v1_ssvi", "full_fmnist_v1_orig_gplvm"]
]
# Metrics shown first in the markdown table; any other numeric key follows in the CSV
HEADLINE = [
    "train_rmse", "test_rmse", "train_nlpd", "test_nlpd", "final_full_loss",
    "test_crps", "crps", "coverage_90", "coverage_95", "ood_nlpd_auroc", "knn_knn_1",
]


def load_runs(folder):
    """One row per run: folder, arm, seed and every numeric metric of its summary."""
    rows = []
    for path in sorted((folder / "ood_run_results").glob("*/experiment_summary.json")):
        summary = json.loads(path.read_text())
        ((arm, metrics),) = [(k, v) for k, v in summary.items() if isinstance(v, dict) and "train_rmse" in v]
        seed = int(path.parent.name.split("_seed_")[1].split("_")[0])
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)}
        rows.append({"folder": folder.name, "arm": arm, "seed": seed, **numeric})
    return rows


def main():
    folders = [Path(p) for p in sys.argv[1:]] or DEFAULT_FOLDERS
    runs = pd.DataFrame([row for folder in folders for row in load_runs(folder)])
    grouped = runs.drop(columns="seed").groupby(["folder", "arm"])
    table = grouped.agg(["mean", "std"])
    table.insert(0, ("n_seeds", ""), grouped.size())
    table.to_csv(HERE / "full_mnist_summary.csv")

    shown = [m for m in HEADLINE if m in runs.columns]
    cells = {m: table[(m, "mean")].map("{:.4g}".format) + " ± " + table[(m, "std")].map("{:.2g}".format) for m in shown}
    lines = ["| folder | arm | n_seeds | " + " | ".join(shown) + " |", "|---" * (len(shown) + 3) + "|"]
    for (folder, arm), n in table[("n_seeds", "")].items():
        lines.append(f"| {folder} | {arm} | {n} | " + " | ".join(cells[m][(folder, arm)] for m in shown) + " |")
    markdown = "\n".join(lines)
    (HERE / "full_mnist_summary.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
