#!/bin/bash
# Slurm wrapper for oil-flow / qPCR GPLVM folder runners (SSVI + original).
#
# Usage (from repo root; mkdir -p slurm_logs first):
#   sbatch ssvi_scripts_experiments/mcml_slurm_run_folder.sbatch \
#       oilflow ssvi oil_flow_related/final_v1_ssvi
#
#   sbatch ssvi_scripts_experiments/mcml_slurm_run_folder.sbatch \
#       oilflow original oil_flow_related/final_v1_orig_gplvm
#
#   sbatch ssvi_scripts_experiments/mcml_slurm_run_folder.sbatch \
#       qpcr ssvi qpcr_related/final_v1_ssvi
#
#   sbatch ssvi_scripts_experiments/mcml_slurm_run_folder.sbatch \
#       qpcr original qpcr_related/final_v1_orig_gplvm
#
#   sbatch ssvi_scripts_experiments/mcml_slurm_run_folder.sbatch \
#       oilflow ssvi oil_flow_related/final_v1_ssvi some_config.yaml
#
# First argument must be: oilflow | qpcr
# Second argument must be: ssvi | original
# Override script path entirely with RUN_FOLDER_SCRIPT (optional).

#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=gplvm_grid
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Slurm copies this script to its spool dir to execute it, so ${BASH_SOURCE[0]}
# is unreliable here. SLURM_SUBMIT_DIR is the directory `sbatch` was invoked in.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPTS_DIR="$PROJECT_ROOT/ssvi_scripts_experiments"

OILFLOW_SSVI_RUN_FOLDER_SCRIPT="$SCRIPTS_DIR/run_ssvi_configs_folder.sh"
OILFLOW_ORIGINAL_RUN_FOLDER_SCRIPT="$SCRIPTS_DIR/run_original_gplvm_configs_folder.sh"
QPCR_SSVI_RUN_FOLDER_SCRIPT="$SCRIPTS_DIR/run_qpcr_ssvi_configs_folder.sh"
QPCR_ORIGINAL_RUN_FOLDER_SCRIPT="$SCRIPTS_DIR/run_qpcr_original_gplvm_configs_folder.sh"

resolve_run_folder_script() {
    local dataset="$1"
    local model_kind="$2"

    if [[ -n "${RUN_FOLDER_SCRIPT:-}" ]]; then
        printf '%s' "$RUN_FOLDER_SCRIPT"
        return
    fi

    case "${dataset}:${model_kind}" in
        oilflow:ssvi) printf '%s' "$OILFLOW_SSVI_RUN_FOLDER_SCRIPT" ;;
        oilflow:original) printf '%s' "$OILFLOW_ORIGINAL_RUN_FOLDER_SCRIPT" ;;
        qpcr:ssvi) printf '%s' "$QPCR_SSVI_RUN_FOLDER_SCRIPT" ;;
        qpcr:original) printf '%s' "$QPCR_ORIGINAL_RUN_FOLDER_SCRIPT" ;;
        *)
            echo "Error: unsupported dataset/model pair '${dataset}' / '${model_kind}'." >&2
            echo "Dataset must be 'oilflow' or 'qpcr'; model must be 'ssvi' or 'original'." >&2
            echo "Example: sbatch $0 qpcr ssvi qpcr_related/final_v1_ssvi" >&2
            exit 1
            ;;
    esac
}

usage() {
    cat <<EOF
Usage: sbatch $0 <oilflow|qpcr> <ssvi|original> <config_subfolder> [config_file] [runner options...]

Examples:
  sbatch $0 oilflow ssvi oil_flow_related/final_v1_ssvi
  sbatch $0 oilflow original oil_flow_related/final_v1_orig_gplvm
  sbatch $0 qpcr ssvi qpcr_related/final_v1_ssvi
  sbatch $0 qpcr original qpcr_related/final_v1_orig_gplvm
  sbatch $0 oilflow ssvi oil_flow_related/final_v1_ssvi some_config.yaml --training-iterations 200
  sbatch $0 qpcr original qpcr_related/final_v1_orig_gplvm --training-steps 200
EOF
}

if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "Error: $PROJECT_ROOT/.venv/bin/activate not found." >&2
    exit 1
fi

DATASET="${1:-}"
if [[ "$DATASET" != oilflow && "$DATASET" != qpcr ]]; then
    usage >&2
    exit 1
fi
shift

MODEL_KIND="${1:-}"
if [[ "$MODEL_KIND" != ssvi && "$MODEL_KIND" != original ]]; then
    usage >&2
    exit 1
fi
shift

CONFIG_SUBFOLDER="${1:?config_subfolder required (e.g. qpcr_related/final_v1_ssvi)}"
shift || true

RUN_FOLDER_SCRIPT="$(resolve_run_folder_script "$DATASET" "$MODEL_KIND")"

if [[ ! -x "$RUN_FOLDER_SCRIPT" ]]; then
    echo "Error: run-folder script not found or not executable: $RUN_FOLDER_SCRIPT" >&2
    exit 1
fi

echo "=========================================="
echo "Job:        ${SLURM_JOB_NAME:-n/a} (${SLURM_JOB_ID:-n/a})"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Python:     $(command -v python)"
echo "GPU(s):     ${CUDA_VISIBLE_DEVICES:-n/a}"
echo "Dataset:    $DATASET"
echo "Model:      $MODEL_KIND"
echo "Runner:     $RUN_FOLDER_SCRIPT"
echo "Folder:     $CONFIG_SUBFOLDER"
echo "Extra:      $*"
echo "=========================================="

cd "$PROJECT_ROOT"
"$RUN_FOLDER_SCRIPT" "$CONFIG_SUBFOLDER" --device cuda "$@"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
