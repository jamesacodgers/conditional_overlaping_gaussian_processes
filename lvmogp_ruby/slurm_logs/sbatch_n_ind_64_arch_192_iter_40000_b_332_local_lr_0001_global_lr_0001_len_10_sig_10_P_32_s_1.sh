#!/bin/bash

#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nice=10000

#SBATCH --job-name=toy_data_n_ind_64_arch_192_iter_40000_b_332_local_lr_0001_global_lr_0001_len_10_sig_10_P_32_s_1
#SBATCH --output=/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/lvmogp_ruby/slurm_logs/%x_%j_output.txt
#SBATCH --error=/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/lvmogp_ruby/slurm_logs/%x_%j_error.txt

# Print job information
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Config: n_ind_64_arch_192_iter_40000_b_332_local_lr_0001_global_lr_0001_len_10_sig_10_P_32_s_1.yaml"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "=========================================="
echo ""

# Change to working directory
cd /home/haicu/james.odgers/conditional_overlaping_gaussian_processes/lvmogp_ruby || exit 1

# Build Python command
CMD="python toy_data_LVMOGP_SSVI.py --config \"/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/lvmogp_ruby/configs/inducing/n_ind_64_arch_192_iter_40000_b_332_local_lr_0001_global_lr_0001_len_10_sig_10_P_32_s_1.yaml\" --device cuda"


echo "Executing: $CMD"
echo ""

# Execute command
eval "$CMD"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
