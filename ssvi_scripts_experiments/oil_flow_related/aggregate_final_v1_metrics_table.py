#!/usr/bin/env python3
"""Aggregate final_v1 experiment_summary.json runs into a mean±std CSV table.

Examples:
  # Oilflow (default)
  python ssvi_scripts_experiments/oil_flow_related/aggregate_final_v1_metrics_table.py \
    --date-tag 26_07

  # qPCR
  python ssvi_scripts_experiments/oil_flow_related/aggregate_final_v1_metrics_table.py \
    --dataset qpcr --date-tag 26_07
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "ssvi_scripts_experiments"

DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    "oilflow": {
        "dataset_name": "Oilflow",
        "ssvi_dir": SCRIPTS_ROOT / "oil_flow_related/final_v1_ssvi/ssvi_run_results",
        "orig_dir": SCRIPTS_ROOT / "oil_flow_related/final_v1_orig_gplvm/orig_gplvm_run_results",
        "output_stem": "oilflow_metrics_table",
        "output_dir": SCRIPTS_ROOT / "oil_flow_related",
    },
    "qpcr": {
        "dataset_name": "qPCR",
        "ssvi_dir": SCRIPTS_ROOT / "qpcr_related/final_v1_ssvi/ssvi_run_results",
        "orig_dir": SCRIPTS_ROOT / "qpcr_related/final_v1_orig_gplvm/orig_gplvm_run_results",
        "output_stem": "qpcr_metrics_table",
        "output_dir": SCRIPTS_ROOT / "qpcr_related",
    },
}

# Paper-style model labels.
MODEL_ORDER = ("MF", "A-MF", "A-S (Ours)")
ORIG_KEY_TO_LABEL = {"gauss": "MF", "nn_gauss": "A-MF"}
SSVI_LABEL = "A-S (Ours)"

# Metrics to report (source key -> column name).
METRIC_SPECS: Sequence[Tuple[str, str]] = (
    ("train_rmse", "Train RMSE"),
    ("test_rmse", "Test RMSE"),
    ("train_nlpd", "Train NLPD"),
    ("test_nlpd", "Test NLPD"),
    ("train_crps", "Train CRPS"),
    ("test_crps", "Test CRPS"),
    ("train_crps_posterior_predictive_sample", "Train CRPS PP sample"),
    ("test_crps_posterior_predictive_sample", "Test CRPS PP sample"),
    ("train_crps_posterior_predictive_mixture", "Train CRPS PP mixture"),
    ("test_crps_posterior_predictive_mixture", "Test CRPS PP mixture"),
    ("train_coverage_90", "Train Coverage 90"),
    ("test_coverage_90", "Test Coverage 90"),
    ("train_coverage_95", "Train Coverage 95"),
    ("test_coverage_95", "Test Coverage 95"),
    ("train_interval_width_90", "Train MPIW 90"),
    ("test_interval_width_90", "Test MPIW 90"),
    ("train_interval_width_95", "Train MPIW 95"),
    ("test_interval_width_95", "Test MPIW 95"),
    ("final_full_loss", "Final neg. ELBO"),
)

SEED_RE = re.compile(r"_seed_(\d+)_")
# SSVI dirs use DD_MM_HH_MM; original dirs use MM_DD_HH_MM.
DATE_TAIL_RE = re.compile(r"_(\d{2})_(\d{2})_(\d{2})_(\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate final_v1 experiment summaries into a mean±std CSV "
            "(oilflow or qpcr)."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS.keys()),
        default="oilflow",
        help="Dataset preset for default paths/labels. Default: oilflow",
    )
    parser.add_argument(
        "--date-tag",
        default="26_07",
        help=(
            "Date tag to match in run directory names. Accepts DD_MM (e.g. 26_07) "
            "and also matches the swapped MM_DD form used by original GPLVM runs "
            "(e.g. 07_26). Default: 26_07"
        ),
    )
    parser.add_argument(
        "--ssvi-dir",
        type=Path,
        default=None,
        help="Directory with SSVI run folders (overrides --dataset preset).",
    )
    parser.add_argument(
        "--orig-dir",
        type=Path,
        default=None,
        help="Directory with original GPLVM run folders (overrides --dataset preset).",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset label written in the table (overrides --dataset preset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: "
            "<preset-output-dir>/<stem>_<date-tag>.csv"
        ),
    )
    parser.add_argument(
        "--keep-all-duplicates",
        action="store_true",
        help=(
            "If set, keep all matching runs per (model, seed). "
            "Default keeps only the latest run per (model, seed)."
        ),
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    preset = DATASET_PRESETS[args.dataset]
    if args.ssvi_dir is None:
        args.ssvi_dir = preset["ssvi_dir"]
    if args.orig_dir is None:
        args.orig_dir = preset["orig_dir"]
    if args.dataset_name is None:
        args.dataset_name = preset["dataset_name"]
    args.output_stem = preset["output_stem"]
    args.output_dir = preset["output_dir"]
    return args


def extract_ssvi_metrics(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Prefer known SSVI keys, else first top-level '*_ssvi' dict."""
    for key in ("oilflow_ssvi", "qpcr_ssvi"):
        metrics = summary.get(key)
        if isinstance(metrics, dict):
            return metrics
    for key, value in summary.items():
        if isinstance(value, dict) and str(key).endswith("_ssvi"):
            return value
    return None


def date_patterns(date_tag: str) -> Tuple[str, ...]:
    parts = date_tag.strip("_").split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected date tag like '26_07', got: {date_tag!r}")
    a, b = parts
    patterns = (f"_{a}_{b}_",)
    if a != b:
        patterns = (f"_{a}_{b}_", f"_{b}_{a}_")
    return patterns


def matches_date(run_dir_name: str, patterns: Iterable[str]) -> bool:
    return any(p in run_dir_name for p in patterns)


def extract_seed(run_dir_name: str) -> Optional[int]:
    match = SEED_RE.search(run_dir_name)
    return int(match.group(1)) if match else None


def extract_timestamp_sort_key(run_dir_name: str) -> Tuple[int, int, int, int]:
    match = DATE_TAIL_RE.search(run_dir_name)
    if not match:
        return (0, 0, 0, 0)
    return tuple(int(x) for x in match.groups())  # type: ignore[return-value]


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def collect_ssvi_runs(
    ssvi_dir: Path,
    patterns: Sequence[str],
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not ssvi_dir.is_dir():
        return runs

    for run_dir in sorted(ssvi_dir.iterdir()):
        if not run_dir.is_dir() or not matches_date(run_dir.name, patterns):
            continue
        summary_path = run_dir / "experiment_summary.json"
        if not summary_path.exists():
            print(f"Skipping incomplete SSVI run (no summary): {run_dir.name}")
            continue
        seed = extract_seed(run_dir.name)
        if seed is None:
            print(f"Skipping SSVI run (no seed): {run_dir.name}")
            continue
        summary = load_summary(summary_path)
        metrics = extract_ssvi_metrics(summary)
        if metrics is None:
            print(f"Skipping SSVI run (missing *_ssvi metrics): {run_dir.name}")
            continue
        runs.append(
            {
                "model": SSVI_LABEL,
                "seed": seed,
                "run_dir": run_dir.name,
                "summary_path": str(summary_path),
                "timestamp": extract_timestamp_sort_key(run_dir.name),
                "metrics": metrics,
            }
        )
    return runs


def collect_orig_runs(
    orig_dir: Path,
    patterns: Sequence[str],
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not orig_dir.is_dir():
        return runs

    for run_dir in sorted(orig_dir.iterdir()):
        if not run_dir.is_dir() or not matches_date(run_dir.name, patterns):
            continue
        summary_path = run_dir / "experiment_summary.json"
        if not summary_path.exists():
            print(f"Skipping incomplete original run (no summary): {run_dir.name}")
            continue
        seed = extract_seed(run_dir.name)
        if seed is None:
            print(f"Skipping original run (no seed): {run_dir.name}")
            continue
        summary = load_summary(summary_path)
        for key, label in ORIG_KEY_TO_LABEL.items():
            metrics = summary.get(key)
            if not isinstance(metrics, dict):
                print(f"Skipping {label} in {run_dir.name} (missing '{key}')")
                continue
            runs.append(
                {
                    "model": label,
                    "seed": seed,
                    "run_dir": run_dir.name,
                    "summary_path": str(summary_path),
                    "timestamp": extract_timestamp_sort_key(run_dir.name),
                    "metrics": metrics,
                }
            )
    return runs


def dedupe_latest_per_seed(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for run in runs:
        key = (run["model"], run["seed"])
        current = best.get(key)
        if current is None or run["timestamp"] >= current["timestamp"]:
            if current is not None:
                print(
                    f"Replacing older duplicate {current['run_dir']} "
                    f"with {run['run_dir']} for {key}"
                )
            best[key] = run
    return sorted(best.values(), key=lambda r: (MODEL_ORDER.index(r["model"]), r["seed"]))


def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if not math.isfinite(mean):
        return "NaN"
    if not math.isfinite(std):
        return f"{mean:.{digits}f} ± NaN"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate(
    runs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    """model -> metric_key -> (mean, std, n)."""
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        model = run["model"]
        metrics = run["metrics"]
        for metric_key, _ in METRIC_SPECS:
            value = metrics.get(metric_key)
            if is_finite_number(value):
                values[model][metric_key].append(float(value))

    out: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for model, metric_map in values.items():
        out[model] = {}
        for metric_key, vals in metric_map.items():
            mean, std = mean_std(vals)
            out[model][metric_key] = (mean, std, len(vals))
    return out


def write_formatted_csv(
    path: Path,
    dataset_name: str,
    aggregated: Dict[str, Dict[str, Tuple[float, float, int]]],
    run_counts: Dict[str, int],
) -> None:
    fieldnames = ["Dataset", "GPLVM type", "n_runs"] + [name for _, name in METRIC_SPECS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODEL_ORDER:
            if model not in aggregated:
                continue
            row: Dict[str, Any] = {
                "Dataset": dataset_name,
                "GPLVM type": model,
                "n_runs": run_counts.get(model, 0),
            }
            for metric_key, col_name in METRIC_SPECS:
                stats = aggregated[model].get(metric_key)
                if stats is None:
                    row[col_name] = "NaN"
                else:
                    mean, std, _ = stats
                    row[col_name] = format_mean_std(mean, std)
            writer.writerow(row)


def write_numeric_csv(
    path: Path,
    dataset_name: str,
    aggregated: Dict[str, Dict[str, Tuple[float, float, int]]],
    run_counts: Dict[str, int],
) -> None:
    fieldnames = ["Dataset", "GPLVM type", "n_runs"]
    for _, col_name in METRIC_SPECS:
        fieldnames.extend([f"{col_name} mean", f"{col_name} std"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in MODEL_ORDER:
            if model not in aggregated:
                continue
            row: Dict[str, Any] = {
                "Dataset": dataset_name,
                "GPLVM type": model,
                "n_runs": run_counts.get(model, 0),
            }
            for metric_key, col_name in METRIC_SPECS:
                stats = aggregated[model].get(metric_key)
                if stats is None:
                    row[f"{col_name} mean"] = ""
                    row[f"{col_name} std"] = ""
                else:
                    mean, std, _ = stats
                    row[f"{col_name} mean"] = mean
                    row[f"{col_name} std"] = std
            writer.writerow(row)


def write_run_manifest(path: Path, runs: List[Dict[str, Any]]) -> None:
    fieldnames = ["GPLVM type", "seed", "run_dir", "summary_path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in sorted(
            runs, key=lambda r: (MODEL_ORDER.index(r["model"]), r["seed"], r["run_dir"])
        ):
            writer.writerow(
                {
                    "GPLVM type": run["model"],
                    "seed": run["seed"],
                    "run_dir": run["run_dir"],
                    "summary_path": run["summary_path"],
                }
            )


def main() -> None:
    args = resolve_paths(parse_args())
    patterns = date_patterns(args.date_tag)

    ssvi_runs = collect_ssvi_runs(args.ssvi_dir, patterns)
    orig_runs = collect_orig_runs(args.orig_dir, patterns)
    runs = ssvi_runs + orig_runs

    if not args.keep_all_duplicates:
        runs = dedupe_latest_per_seed(runs)

    if not runs:
        raise SystemExit(
            f"No completed runs found for date tag {args.date_tag!r} "
            f"(patterns: {patterns}) in:\n  {args.ssvi_dir}\n  {args.orig_dir}"
        )

    aggregated = aggregate(runs)
    run_counts = defaultdict(int)
    for run in runs:
        run_counts[run["model"]] += 1

    if args.output is None:
        out_dir = args.output_dir
        stem = f"{args.output_stem}_{args.date_tag}"
        formatted_path = out_dir / f"{stem}.csv"
        numeric_path = out_dir / f"{stem}_numeric.csv"
        manifest_path = out_dir / f"{stem}_runs.csv"
    else:
        formatted_path = args.output
        numeric_path = formatted_path.with_name(
            f"{formatted_path.stem}_numeric{formatted_path.suffix}"
        )
        manifest_path = formatted_path.with_name(
            f"{formatted_path.stem}_runs{formatted_path.suffix}"
        )

    write_formatted_csv(formatted_path, args.dataset_name, aggregated, run_counts)
    write_numeric_csv(numeric_path, args.dataset_name, aggregated, run_counts)
    write_run_manifest(manifest_path, runs)

    print(f"Included runs: {len(runs)}")
    for model in MODEL_ORDER:
        print(f"  {model}: {run_counts.get(model, 0)}")
    print(f"Wrote formatted table: {formatted_path}")
    print(f"Wrote numeric table:   {numeric_path}")
    print(f"Wrote run manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
