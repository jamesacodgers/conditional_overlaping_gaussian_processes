#!/bin/bash
# Full-size MNIST / FashionMNIST benchmarks: A-MF-GPLVM (*_orig_gplvm) and A-S-GPLVM (*_ssvi).
#
# Usage (from anywhere):
#   GPUS="0 1 2 3" bash ssvi_scripts_experiments/run_full_mnist_benchmarks.sh [config_folder ...]
# Env:
#   PYTHON  interpreter (default: <repo>/.venv/bin/python)
#   GPUS    CUDA device ids, one run per GPU at a time (default: 0)
#
# One worker per GPU pulls the next unclaimed config from a shared queue: the slow SSVI runs
# first, seed 0 before seed 1 within each arm. A finished run leaves <config>.done next to its
# yaml and is skipped on restart; a claimed run leaves <config>.claimed. The scripts cannot
# resume, so after a crash delete the stale *.claimed files and an interrupted run starts again
# from step 0. Several machines sharing this directory (e.g. one launcher per node) also work.
# Logs go to <folder>/logs/<config>.log.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
GPUS="${GPUS:-0}"

cd "$SCRIPT_DIR" || exit 1
# The loaders read ./data/{MNIST,FashionMNIST} relative to this directory
[ -e data ] || ln -s ../data data

if [ $# -gt 0 ]; then
    FOLDERS=("$@")
else
    FOLDERS=(ood_mnist_related/full_mnist_v1_ssvi ood_mnist_related/full_fmnist_v1_ssvi
             ood_mnist_related/full_mnist_v1_orig_gplvm ood_mnist_related/full_fmnist_v1_orig_gplvm)
fi

# Queue order: SSVI folders before MFVI ones, then seed, so the longest runs start first
QUEUE=()
for arm in _ssvi _orig_gplvm; do
    for seed in 0 1 2 3 4; do
        for folder in "${FOLDERS[@]}"; do
            [[ "$folder" == *"$arm" ]] || continue
            for config in "$folder"/*_s_${seed}_*.yaml; do
                [ -e "$config" ] && QUEUE+=("$config")
            done
        done
    done
done
echo "${#QUEUE[@]} configs queued on GPUs: $GPUS"

claim_next() {
    # Print the first config that is neither done nor claimed, and claim it (atomic under flock)
    (
        flock 9
        for config in "${QUEUE[@]}"; do
            base="${config%.yaml}"
            [ -e "$base.done" ] || [ -e "$base.claimed" ] && continue
            echo "$(hostname):$gpu:$(date '+%F %T')" > "$base.claimed"
            echo "$config"
            break
        done
    ) 9> "$SCRIPT_DIR/.full_mnist_queue.lock"
}

worker() {
    local gpu="$1" config folder name script
    while config="$(claim_next)" && [ -n "$config" ]; do
        folder="$(dirname "$config")"
        name="$(basename "$config" .yaml)"
        case "$folder" in
            *_ssvi) script=mnist_ood_GPLVM_SSVI_AM_INF.py ;;
            *_orig_gplvm) script=mnist_ood_GPLVM.py ;;
        esac
        mkdir -p "$folder/logs"
        echo "[$(date '+%F %T')] gpu $gpu start  $config"
        if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$script" --config "$config" --device cuda \
                > "$folder/logs/$name.log" 2>&1; then
            touch "$folder/$name.done"
            rm -f "$folder/$name.claimed"
            echo "[$(date '+%F %T')] gpu $gpu done   $config"
        else
            echo "[$(date '+%F %T')] gpu $gpu FAILED $config (see $folder/logs/$name.log; .claimed left in place)"
        fi
    done
}

for gpu in $GPUS; do
    worker "$gpu" &
done
wait
