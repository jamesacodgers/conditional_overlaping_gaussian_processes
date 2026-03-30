#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study Plotting Script for GP-LVM Experiments

Compares A-MF (nn_gauss from orig_gplvm) vs A-S (ssvi) across varying hyperparameters.
For each ablation pair produces 3 plots:
  1. Train/Test NLPD vs ablation parameter
  2. Train/Test RMSE vs ablation parameter
  3. ELBO vs ablation parameter

Usage:
    python ablation_study_plots.py [--output_dir "path/to/output"]
"""

import os
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RunMetrics:
    """Final metrics from a single experiment run."""
    train_rmse: Optional[float] = None
    test_rmse: Optional[float] = None
    train_nlpd: Optional[float] = None
    test_nlpd: Optional[float] = None
    final_full_loss: Optional[float] = None


@dataclass
class ParsedRun:
    """Parsed information from a run directory name."""
    n_ind: int
    arch: int
    seed: int
    timestamp: str
    run_path: str
    batch_size: int = 0
    p_samples: int = 0
    lr: str = ''


class AblationStudyPlotter:
    """Generates ablation study plots comparing A-MF vs A-S."""

    MODEL_COLORS = {
        'nn_gauss': 'red',
        'ssvi': 'green',
    }

    MODEL_DISPLAY_NAMES = {
        'nn_gauss': 'A-MF',
        'ssvi': 'A-S',
    }

    MODEL_MARKERS_TRAIN = {
        'nn_gauss': 'o',
        'ssvi': 'o',
    }

    MODEL_MARKERS_TEST = {
        'nn_gauss': 's',
        'ssvi': 's',
    }

    LR_COLORS = {
        '0001': '#d62728',
        '00003': '#1f77b4',
    }

    P_SAMPLE_COLORS = {
        4: '#d62728',
        64: '#2ca02c',
    }

    LR_DISPLAY = {
        '0001': '1e-3',
        '00003': '3e-4',
    }

    @classmethod
    def _format_lr(cls, lr_str: str) -> str:
        return cls.LR_DISPLAY.get(lr_str, f'lr={lr_str}')

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _parse_run_dir_name(run_name: str) -> Optional[ParsedRun]:
        """Extract hyperparameters, seed, and timestamp from a run directory name."""
        pattern = (
            r'(?:orig_gplvm_|ssvi_)?'
            r'n_ind_(\d+)_arch_(\d+)_len_[\d]+_sig_[\d]+'
            r'_b_(\d+)_s_\d+_lr_([\d]+)_n_steps_\d+_P_(\d+)'
            r'_miss_[tf]_ratio_[\d]+(?:_test_\d+)?'
            r'_seed_(\d+)_(\d+_\d+_\d+_\d+)'
        )
        match = re.match(pattern, run_name)
        if not match:
            return None
        return ParsedRun(
            n_ind=int(match.group(1)),
            arch=int(match.group(2)),
            seed=int(match.group(6)),
            timestamp=match.group(7),
            run_path='',
            batch_size=int(match.group(3)),
            p_samples=int(match.group(5)),
            lr=match.group(4),
        )

    @staticmethod
    def _load_experiment_summary(summary_path: str, model_key: str) -> Optional[RunMetrics]:
        """Load metrics from experiment_summary.json for a specific model key."""
        try:
            with open(summary_path, 'r') as f:
                content = f.read().replace('NaN', 'null')
                data = json.loads(content)

            model_data = None
            if model_key in data:
                model_data = data[model_key]
            else:
                for key in data:
                    if model_key in key.lower():
                        model_data = data[key]
                        break

            if model_data is None:
                return None

            def safe_float(v):
                if v is None:
                    return None
                try:
                    val = float(v)
                    return None if np.isnan(val) else val
                except (ValueError, TypeError):
                    return None

            return RunMetrics(
                train_rmse=safe_float(model_data.get('train_rmse')),
                test_rmse=safe_float(model_data.get('test_rmse')),
                train_nlpd=safe_float(model_data.get('train_nlpd')),
                test_nlpd=safe_float(model_data.get('test_nlpd')),
                final_full_loss=safe_float(
                    model_data.get('final_full_loss', model_data.get('final_loss'))
                ),
            )
        except Exception as e:
            print(f"  Error loading {summary_path}: {e}")
            return None

    def _discover_runs(self, base_dir: str) -> List[ParsedRun]:
        """Find all valid run directories under base_dir (or base_dir/*_run_results/)."""
        results_dir = base_dir
        for entry in os.listdir(base_dir):
            if 'run_results' in entry and os.path.isdir(os.path.join(base_dir, entry)):
                results_dir = os.path.join(base_dir, entry)
                break

        runs = []
        for name in os.listdir(results_dir):
            full_path = os.path.join(results_dir, name)
            if not os.path.isdir(full_path):
                continue
            parsed = self._parse_run_dir_name(name)
            if parsed is None:
                continue
            parsed.run_path = full_path
            runs.append(parsed)
        return runs

    @staticmethod
    def _deduplicate_runs(runs: List[ParsedRun], ablation_param: str) -> List[ParsedRun]:
        """Keep only the latest run per (param_value, seed) pair."""
        best: Dict[tuple, ParsedRun] = {}
        for run in runs:
            param_val = getattr(run, ablation_param)
            key = (param_val, run.seed)
            if key not in best or run.timestamp > best[key].timestamp:
                best[key] = run
        return list(best.values())

    def collect_metrics(
        self, orig_gplvm_dir: str, ssvi_dir: str, ablation_param: str
    ) -> Dict[str, Dict[int, List[RunMetrics]]]:
        """
        Collect metrics grouped by model type and ablation parameter value.

        Returns: {model_type: {param_value: [RunMetrics, ...]}}
        """
        results: Dict[str, Dict[int, List[RunMetrics]]] = {
            'nn_gauss': defaultdict(list),
            'ssvi': defaultdict(list),
        }

        # A-MF from orig_gplvm
        print("  Scanning orig_gplvm directory...")
        orig_runs = self._deduplicate_runs(
            self._discover_runs(orig_gplvm_dir), ablation_param
        )
        for run in orig_runs:
            summary_path = os.path.join(run.run_path, 'experiment_summary.json')
            if not os.path.exists(summary_path):
                continue
            metrics = self._load_experiment_summary(summary_path, 'nn_gauss')
            if metrics is not None:
                param_val = getattr(run, ablation_param)
                results['nn_gauss'][param_val].append(metrics)

        # A-S from ssvi
        print("  Scanning ssvi directory...")
        ssvi_runs = self._deduplicate_runs(
            self._discover_runs(ssvi_dir), ablation_param
        )
        for run in ssvi_runs:
            summary_path = os.path.join(run.run_path, 'experiment_summary.json')
            if not os.path.exists(summary_path):
                continue
            metrics = self._load_experiment_summary(summary_path, 'ssvi')
            if metrics is not None:
                param_val = getattr(run, ablation_param)
                results['ssvi'][param_val].append(metrics)

        return results

    @staticmethod
    def _mean_std(metrics_list: List[RunMetrics], attr: str) -> Tuple[Optional[float], Optional[float]]:
        """Compute mean and std for a metric attribute across runs."""
        values = [getattr(m, attr) for m in metrics_list if getattr(m, attr) is not None]
        if not values:
            return None, None
        return float(np.mean(values)), float(np.std(values))

    @staticmethod
    def _format_yaxis(ax):
        """Integers without decimals, non-integers with up to 2 decimal places."""
        def fmt(value, _):
            if abs(value - round(value)) < 1e-9:
                return f'{int(round(value))}'
            return f'{value:.2f}'
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _plot_train_test_metric(
        self,
        results: Dict[str, Dict[int, List[RunMetrics]]],
        all_param_values: List[int],
        metric_attr_train: str,
        metric_attr_test: str,
        ylabel: str,
        title: str,
        param_label: str,
        save_name: str,
    ):
        """Plot a train/test metric with error bars for both models."""
        fs = 26
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        val_to_idx = {v: i for i, v in enumerate(all_param_values)}

        for model_type in ['nn_gauss', 'ssvi']:
            color = self.MODEL_COLORS[model_type]
            display = self.MODEL_DISPLAY_NAMES[model_type]

            x_pos, tr_m, tr_s, te_m, te_s = [], [], [], [], []
            for pv in all_param_values:
                runs = results[model_type].get(pv, [])
                if not runs:
                    continue
                tm, ts = self._mean_std(runs, metric_attr_train)
                tem, tes = self._mean_std(runs, metric_attr_test)
                if tm is not None and tem is not None:
                    x_pos.append(val_to_idx[pv])
                    tr_m.append(tm); tr_s.append(ts)
                    te_m.append(tem); te_s.append(tes)

            if x_pos:
                ax.errorbar(
                    x_pos, tr_m, yerr=tr_s,
                    label=f'{display} (Train)', color=color,
                    marker='o', markersize=10, linewidth=2, capsize=5, linestyle='-',
                )
                ax.errorbar(
                    x_pos, te_m, yerr=te_s,
                    label=f'{display} (Test)', color=color,
                    marker='s', markersize=10, linewidth=2, capsize=5, linestyle='--',
                )

        ax.set_xlabel(param_label, fontsize=fs)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.legend(fontsize=fs - 8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs - 4)
        ax.set_xticks(range(len(all_param_values)))
        ax.set_xticklabels([str(v) for v in all_param_values])
        self._format_yaxis(ax)
        plt.tight_layout()

        for ext in ['png', 'pdf']:
            path = os.path.join(self.output_dir, f'{save_name}.{ext}')
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_name}.{{png,pdf}}")

    def _plot_single_metric(
        self,
        results: Dict[str, Dict[int, List[RunMetrics]]],
        all_param_values: List[int],
        metric_attr: str,
        ylabel: str,
        title: str,
        param_label: str,
        save_name: str,
    ):
        """Plot a single metric with error bars for both models."""
        fs = 26
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        val_to_idx = {v: i for i, v in enumerate(all_param_values)}

        for model_type in ['nn_gauss', 'ssvi']:
            color = self.MODEL_COLORS[model_type]
            display = self.MODEL_DISPLAY_NAMES[model_type]

            x_pos, means, stds = [], [], []
            for pv in all_param_values:
                runs = results[model_type].get(pv, [])
                if not runs:
                    continue
                m, s = self._mean_std(runs, metric_attr)
                if m is not None:
                    x_pos.append(val_to_idx[pv])
                    means.append(m)
                    stds.append(s)

            if x_pos:
                ax.errorbar(
                    x_pos, means, yerr=stds,
                    label=display, color=color,
                    marker='o', markersize=10, linewidth=2, capsize=5,
                )

        ax.set_xlabel(param_label, fontsize=fs)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.legend(fontsize=fs - 4)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs - 4)
        ax.set_xticks(range(len(all_param_values)))
        ax.set_xticklabels([str(v) for v in all_param_values])
        self._format_yaxis(ax)
        plt.tight_layout()

        for ext in ['png', 'pdf']:
            path = os.path.join(self.output_dir, f'{save_name}.{ext}')
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_name}.{{png,pdf}}")

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def plot_ablation(
        self,
        results: Dict[str, Dict[int, List[RunMetrics]]],
        ablation_param: str,
        param_label: str,
        experiment_name: str,
    ):
        """Generate 3 ablation plots for a given parameter."""
        all_param_values = sorted(
            set(results['nn_gauss'].keys()) | set(results['ssvi'].keys())
        )
        if not all_param_values:
            print(f"  No data found for {ablation_param}")
            return

        print(f"\n  Data summary:")
        for model_type in ['nn_gauss', 'ssvi']:
            for pv in all_param_values:
                n = len(results[model_type].get(pv, []))
                print(f"    {self.MODEL_DISPLAY_NAMES[model_type]} @ {ablation_param}={pv}: {n} runs")

        print(f"\n  Generating plots...")

        self._plot_train_test_metric(
            results, all_param_values,
            'train_nlpd', 'test_nlpd',
            ylabel='NLPD', title='Train / Test NLPD',
            param_label=param_label,
            save_name=f'{experiment_name}_nlpd_ablation',
        )

        self._plot_train_test_metric(
            results, all_param_values,
            'train_rmse', 'test_rmse',
            ylabel='RMSE', title='Train / Test RMSE',
            param_label=param_label,
            save_name=f'{experiment_name}_rmse_ablation',
        )

        self._plot_single_metric(
            results, all_param_values,
            'final_full_loss',
            ylabel='ELBO', title='ELBO',
            param_label=param_label,
            save_name=f'{experiment_name}_elbo_ablation',
        )

    def save_json_summary(
        self,
        results: Dict[str, Dict[int, List[RunMetrics]]],
        ablation_param: str,
        param_label: str,
        experiment_name: str,
    ):
        """Save a JSON summary of all ablation results."""
        all_param_values = sorted(
            set(results['nn_gauss'].keys()) | set(results['ssvi'].keys())
        )
        summary = {
            'experiment': experiment_name,
            'ablation_parameter': ablation_param,
            'parameter_label': param_label,
            'parameter_values': all_param_values,
            'results': {},
        }

        for model_type in ['nn_gauss', 'ssvi']:
            display = self.MODEL_DISPLAY_NAMES[model_type]
            model_summary = {}
            for pv in all_param_values:
                runs = results[model_type].get(pv, [])
                point = {'n_runs': len(runs)}
                for metric in ['train_rmse', 'test_rmse', 'train_nlpd', 'test_nlpd', 'final_full_loss']:
                    m, s = self._mean_std(runs, metric)
                    point[metric] = (
                        {'mean': m, 'std': s, 'formatted': f"{m:.4f} ± {s:.4f}"}
                        if m is not None else None
                    )
                model_summary[str(pv)] = point
            summary['results'][display] = model_summary

        json_path = os.path.join(self.output_dir, f'{experiment_name}_ablation_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"    Saved summary: {json_path}")

    def run_ablation_study(
        self,
        orig_gplvm_dir: str,
        ssvi_dir: str,
        ablation_param: str,
        param_label: str,
        experiment_name: str,
    ):
        """Run a full ablation study for one parameter pair."""
        print(f"\n{'='*60}")
        print(f"Ablation Study: {experiment_name}")
        print(f"  Parameter : {param_label}")
        print(f"  orig_gplvm: {orig_gplvm_dir}")
        print(f"  ssvi      : {ssvi_dir}")
        print(f"{'='*60}")

        results = self.collect_metrics(orig_gplvm_dir, ssvi_dir, ablation_param)
        self.plot_ablation(results, ablation_param, param_label, experiment_name)
        self.save_json_summary(results, ablation_param, param_label, experiment_name)

    # ------------------------------------------------------------------
    # Variance ablation: A-MF vs A-S by LR + A-S P-samples effect
    # ------------------------------------------------------------------

    def _deduplicate_runs_full(self, runs: List[ParsedRun]) -> List[ParsedRun]:
        """Keep latest run per (batch_size, lr, p_samples, seed)."""
        best: Dict[tuple, ParsedRun] = {}
        for run in runs:
            key = (run.batch_size, run.lr, run.p_samples, run.seed)
            if key not in best or run.timestamp > best[key].timestamp:
                best[key] = run
        return list(best.values())

    def collect_metrics_variance(
        self, orig_gplvm_dir: str, ssvi_dir: str,
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[int, List[RunMetrics]]],
        Dict[str, Dict[int, Dict[int, List[RunMetrics]]]],
    ]:
        """
        Collect metrics for the variance ablation.

        Returns a tuple of:
          comparison: {(model_type, lr): {batch_size: [RunMetrics]}}
              Only runs with the highest P value (best-case for A-S).
          ssvi_p_detail: {lr: {p_samples: {batch_size: [RunMetrics]}}}
              A-S only, separated by P for P-samples effect analysis.
        """
        comparison: Dict[Tuple[str, str], Dict[int, List[RunMetrics]]] = defaultdict(
            lambda: defaultdict(list)
        )
        ssvi_p_detail: Dict[str, Dict[int, Dict[int, List[RunMetrics]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        print("  Scanning orig_gplvm directory...")
        orig_runs = self._deduplicate_runs_full(self._discover_runs(orig_gplvm_dir))
        max_p_orig = max((r.p_samples for r in orig_runs), default=0)
        print(f"    Using P={max_p_orig} for comparison")
        for run in orig_runs:
            if run.p_samples != max_p_orig:
                continue
            summary_path = os.path.join(run.run_path, 'experiment_summary.json')
            if not os.path.exists(summary_path):
                continue
            metrics = self._load_experiment_summary(summary_path, 'nn_gauss')
            if metrics is not None:
                comparison[('nn_gauss', run.lr)][run.batch_size].append(metrics)

        print("  Scanning ssvi directory...")
        ssvi_runs = self._deduplicate_runs_full(self._discover_runs(ssvi_dir))
        max_p_ssvi = max((r.p_samples for r in ssvi_runs), default=0)
        print(f"    Using P={max_p_ssvi} for comparison")
        for run in ssvi_runs:
            summary_path = os.path.join(run.run_path, 'experiment_summary.json')
            if not os.path.exists(summary_path):
                continue
            metrics = self._load_experiment_summary(summary_path, 'ssvi')
            if metrics is not None:
                ssvi_p_detail[run.lr][run.p_samples][run.batch_size].append(metrics)
                if run.p_samples == max_p_ssvi:
                    comparison[('ssvi', run.lr)][run.batch_size].append(metrics)

        return dict(comparison), dict(ssvi_p_detail)

    def _plot_comparison_train_test(
        self,
        comparison: Dict[Tuple[str, str], Dict[int, List[RunMetrics]]],
        batch_size: int,
        all_lrs: List[str],
        metric_attr_train: str,
        metric_attr_test: str,
        ylabel: str,
        title: str,
        save_name: str,
    ):
        """A-MF vs A-S, colored by model, x = LR, for a single batch size."""
        fs = 26
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        lr_to_idx = {v: i for i, v in enumerate(all_lrs)}

        for model_type in ['nn_gauss', 'ssvi']:
            display = self.MODEL_DISPLAY_NAMES[model_type]
            color = self.MODEL_COLORS[model_type]

            x_pos, tr_m, tr_s, te_m, te_s = [], [], [], [], []
            for lr_str in all_lrs:
                runs = comparison.get((model_type, lr_str), {}).get(batch_size, [])
                if not runs:
                    continue
                tm, ts = self._mean_std(runs, metric_attr_train)
                tem, tes = self._mean_std(runs, metric_attr_test)
                if tm is not None and tem is not None:
                    x_pos.append(lr_to_idx[lr_str])
                    tr_m.append(tm); tr_s.append(ts)
                    te_m.append(tem); te_s.append(tes)

            if x_pos:
                ax.errorbar(
                    x_pos, tr_m, yerr=tr_s,
                    label=f'{display} (Train)', color=color,
                    marker='o', markersize=10, linewidth=2, capsize=5, linestyle='-',
                )
                ax.errorbar(
                    x_pos, te_m, yerr=te_s,
                    label=f'{display} (Test)', color=color,
                    marker='s', markersize=10, linewidth=2, capsize=5, linestyle='--',
                )

        ax.set_xlabel('Learning Rate', fontsize=fs)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.legend(fontsize=fs - 8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs - 4)
        ax.set_xticks(range(len(all_lrs)))
        ax.set_xticklabels([self._format_lr(lr) for lr in all_lrs])
        self._format_yaxis(ax)
        plt.tight_layout()

        for ext in ['png', 'pdf']:
            path = os.path.join(self.output_dir, f'{save_name}.{ext}')
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_name}.{{png,pdf}}")

    def _plot_comparison_single(
        self,
        comparison: Dict[Tuple[str, str], Dict[int, List[RunMetrics]]],
        batch_size: int,
        all_lrs: List[str],
        metric_attr: str,
        ylabel: str,
        title: str,
        save_name: str,
    ):
        """A-MF vs A-S single metric, colored by model, x = LR, for a single batch size."""
        fs = 26
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        lr_to_idx = {v: i for i, v in enumerate(all_lrs)}

        for model_type in ['nn_gauss', 'ssvi']:
            display = self.MODEL_DISPLAY_NAMES[model_type]
            color = self.MODEL_COLORS[model_type]

            x_pos, means, stds = [], [], []
            for lr_str in all_lrs:
                runs = comparison.get((model_type, lr_str), {}).get(batch_size, [])
                if not runs:
                    continue
                m, s = self._mean_std(runs, metric_attr)
                if m is not None:
                    x_pos.append(lr_to_idx[lr_str])
                    means.append(m)
                    stds.append(s)

            if x_pos:
                ax.errorbar(
                    x_pos, means, yerr=stds,
                    label=display, color=color,
                    marker='o', markersize=10, linewidth=2, capsize=5,
                )

        ax.set_xlabel('Learning Rate', fontsize=fs)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.legend(fontsize=fs - 4)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs - 4)
        ax.set_xticks(range(len(all_lrs)))
        ax.set_xticklabels([self._format_lr(lr) for lr in all_lrs])
        self._format_yaxis(ax)
        plt.tight_layout()

        for ext in ['png', 'pdf']:
            path = os.path.join(self.output_dir, f'{save_name}.{ext}')
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_name}.{{png,pdf}}")

    def _plot_p_effect_train_test(
        self,
        ssvi_p_detail: Dict[str, Dict[int, Dict[int, List[RunMetrics]]]],
        batch_size: int,
        all_lrs: List[str],
        all_p_values: List[int],
        metric_attr_train: str,
        metric_attr_test: str,
        ylabel: str,
        title: str,
        save_name: str,
    ):
        """A-S only: colored by LR, x = P samples, for a single batch size."""
        fs = 26
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        p_to_idx = {v: i for i, v in enumerate(all_p_values)}

        for lr_str in all_lrs:
            color = self.LR_COLORS.get(lr_str, 'gray')
            lr_disp = self._format_lr(lr_str)

            x_pos, tr_m, tr_s, te_m, te_s = [], [], [], [], []
            for p_val in all_p_values:
                runs = ssvi_p_detail.get(lr_str, {}).get(p_val, {}).get(batch_size, [])
                if not runs:
                    continue
                tm, ts = self._mean_std(runs, metric_attr_train)
                tem, tes = self._mean_std(runs, metric_attr_test)
                if tm is not None and tem is not None:
                    x_pos.append(p_to_idx[p_val])
                    tr_m.append(tm); tr_s.append(ts)
                    te_m.append(tem); te_s.append(tes)

            if x_pos:
                ax.errorbar(
                    x_pos, tr_m, yerr=tr_s,
                    label=f'lr={lr_disp} (Train)', color=color,
                    marker='o', markersize=10, linewidth=2, capsize=5, linestyle='-',
                )
                ax.errorbar(
                    x_pos, te_m, yerr=te_s,
                    label=f'lr={lr_disp} (Test)', color=color,
                    marker='s', markersize=10, linewidth=2, capsize=5, linestyle='--',
                )

        ax.set_xlabel('P Samples', fontsize=fs)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.legend(fontsize=fs - 8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs - 4)
        ax.set_xticks(range(len(all_p_values)))
        ax.set_xticklabels([str(v) for v in all_p_values])
        self._format_yaxis(ax)
        plt.tight_layout()

        for ext in ['png', 'pdf']:
            path = os.path.join(self.output_dir, f'{save_name}.{ext}')
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_name}.{{png,pdf}}")

    def _plot_p_effect_single(
        self,
        ssvi_p_detail: Dict[str, Dict[int, Dict[int, List[RunMetrics]]]],
        batch_size: int,
        all_lrs: List[str],
        all_p_values: List[int],
        metric_attr: str,
        ylabel: str,
        title: str,
        save_name: str,
    ):
        """A-S only: single metric, colored by LR, x = P samples, for a single batch size."""
        fs = 26
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        p_to_idx = {v: i for i, v in enumerate(all_p_values)}

        for lr_str in all_lrs:
            color = self.LR_COLORS.get(lr_str, 'gray')
            lr_disp = self._format_lr(lr_str)

            x_pos, means, stds = [], [], []
            for p_val in all_p_values:
                runs = ssvi_p_detail.get(lr_str, {}).get(p_val, {}).get(batch_size, [])
                if not runs:
                    continue
                m, s = self._mean_std(runs, metric_attr)
                if m is not None:
                    x_pos.append(p_to_idx[p_val])
                    means.append(m)
                    stds.append(s)

            if x_pos:
                ax.errorbar(
                    x_pos, means, yerr=stds,
                    label=f'lr={lr_disp}', color=color,
                    marker='o', markersize=10, linewidth=2, capsize=5,
                )

        ax.set_xlabel('P Samples', fontsize=fs)
        ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.legend(fontsize=fs - 4)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fs - 4)
        ax.set_xticks(range(len(all_p_values)))
        ax.set_xticklabels([str(v) for v in all_p_values])
        self._format_yaxis(ax)
        plt.tight_layout()

        for ext in ['png', 'pdf']:
            path = os.path.join(self.output_dir, f'{save_name}.{ext}')
            plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_name}.{{png,pdf}}")

    def run_variance_ablation_study(
        self,
        orig_gplvm_dir: str,
        ssvi_dir: str,
        experiment_name: str,
    ):
        """Run variance ablation: comparison plots + per-LR P-effect plots."""
        variance_output_dir = os.path.join(self.output_dir, 'ablation_variance')
        os.makedirs(variance_output_dir, exist_ok=True)
        original_output_dir = self.output_dir
        self.output_dir = variance_output_dir

        print(f"\n{'='*60}")
        print(f"Variance Ablation Study: {experiment_name}")
        print(f"  X-axis    : Learning Rate")
        print(f"  Separate  : per Batch Size")
        print(f"  Comparison: A-MF vs A-S colored by model")
        print(f"  P-effect  : A-S only, colored by P (per batch size)")
        print(f"  orig_gplvm: {orig_gplvm_dir}")
        print(f"  ssvi      : {ssvi_dir}")
        print(f"  output    : {variance_output_dir}")
        print(f"{'='*60}")

        comparison, ssvi_p_detail = self.collect_metrics_variance(
            orig_gplvm_dir, ssvi_dir,
        )

        all_lrs = sorted({lr for (_, lr) in comparison})
        all_batch_sizes = sorted(
            set(bs for bs_data in comparison.values() for bs in bs_data)
        )
        all_p_values = sorted(
            set(p for lr_data in ssvi_p_detail.values()
                for p in lr_data)
        )

        # --- Per-batch-size comparison plots (A-MF vs A-S, x = LR) ---
        for batch_size in all_batch_sizes:
            print(f"\n  Comparison data (batch={batch_size}, highest P only):")
            for model_type in ['nn_gauss', 'ssvi']:
                display = self.MODEL_DISPLAY_NAMES[model_type]
                for lr_str in all_lrs:
                    n = len(comparison.get((model_type, lr_str), {}).get(batch_size, []))
                    print(f"    {display} lr={self._format_lr(lr_str)}: {n} runs")

            print(f"  Generating comparison plots for batch={batch_size}...")
            suffix = f'{experiment_name}_b{batch_size}_comparison'
            self._plot_comparison_train_test(
                comparison, batch_size, all_lrs,
                'train_nlpd', 'test_nlpd',
                ylabel='NLPD',
                title=f'A-MF vs A-S: NLPD (batch={batch_size})',
                save_name=f'{suffix}_nlpd',
            )
            self._plot_comparison_train_test(
                comparison, batch_size, all_lrs,
                'train_rmse', 'test_rmse',
                ylabel='RMSE',
                title=f'A-MF vs A-S: RMSE (batch={batch_size})',
                save_name=f'{suffix}_rmse',
            )
            self._plot_comparison_single(
                comparison, batch_size, all_lrs,
                'final_full_loss',
                ylabel='ELBO',
                title=f'A-MF vs A-S: ELBO (batch={batch_size})',
                save_name=f'{suffix}_elbo',
            )

        # --- Per-batch-size P-effect plots (A-S only, x = LR) ---
        p_batch_sizes = sorted(
            set(bs for lr_data in ssvi_p_detail.values()
                for p_data in lr_data.values() for bs in p_data)
        )
        for batch_size in p_batch_sizes:
            print(f"\n  A-S P-effect data (batch={batch_size}):")
            for p_val in all_p_values:
                for lr_str in all_lrs:
                    n = len(ssvi_p_detail.get(lr_str, {}).get(p_val, {}).get(batch_size, []))
                    print(f"    P={p_val} lr={self._format_lr(lr_str)}: {n} runs")

            print(f"  Generating P-effect plots for batch={batch_size}...")
            suffix = f'{experiment_name}_b{batch_size}_p_effect'
            self._plot_p_effect_train_test(
                ssvi_p_detail, batch_size, all_lrs, all_p_values,
                'train_nlpd', 'test_nlpd',
                ylabel='NLPD',
                title=f'A-S P-Samples Effect (batch={batch_size}): NLPD',
                save_name=f'{suffix}_nlpd',
            )
            self._plot_p_effect_train_test(
                ssvi_p_detail, batch_size, all_lrs, all_p_values,
                'train_rmse', 'test_rmse',
                ylabel='RMSE',
                title=f'A-S P-Samples Effect (batch={batch_size}): RMSE',
                save_name=f'{suffix}_rmse',
            )
            self._plot_p_effect_single(
                ssvi_p_detail, batch_size, all_lrs, all_p_values,
                'final_full_loss',
                ylabel='ELBO',
                title=f'A-S P-Samples Effect (batch={batch_size}): ELBO',
                save_name=f'{suffix}_elbo',
            )

        self.output_dir = original_output_dir


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OIL_FLOW_DIR = os.path.join(BASE_DIR, 'oil_flow_related')

    DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'comparison_results', 'ablation_studies')

    ABLATION_STUDIES = [
        {
            'orig_gplvm_dir': os.path.join(OIL_FLOW_DIR, 'architecture_ablation_v1_orig_gplvm'),
            'ssvi_dir': os.path.join(OIL_FLOW_DIR, 'architecture_ablation_v1_ssvi'),
            'ablation_param': 'arch',
            'param_label': 'Encoder Hidden Dims (total units)',
            'experiment_name': 'oil_flow_architecture_ablation',
        },
        {
            'orig_gplvm_dir': os.path.join(OIL_FLOW_DIR, 'num_induing_ablation_v1_orig_gplvm'),
            'ssvi_dir': os.path.join(OIL_FLOW_DIR, 'num_induing_ablation_v1_ssvi'),
            'ablation_param': 'n_ind',
            'param_label': 'Number of Inducing Points',
            'experiment_name': 'oil_flow_num_inducing_ablation',
        },
    ]

    VARIANCE_ABLATION_STUDIES = [
        {
            'orig_gplvm_dir': os.path.join(OIL_FLOW_DIR, 'variance_ablation_v1_orig_gplvm'),
            'ssvi_dir': os.path.join(OIL_FLOW_DIR, 'variance_ablation_v1_ssvi'),
            'experiment_name': 'oil_flow_variance_ablation',
        },
    ]

    parser = argparse.ArgumentParser(description='Ablation Study Plots for GP-LVM Experiments')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for plots and summaries')
    args = parser.parse_args()

    plotter = AblationStudyPlotter(args.output_dir)

    for study in ABLATION_STUDIES:
        plotter.run_ablation_study(**study)

    for study in VARIANCE_ABLATION_STUDIES:
        plotter.run_variance_ablation_study(**study)

    print(f"\nAll ablation studies complete! Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
