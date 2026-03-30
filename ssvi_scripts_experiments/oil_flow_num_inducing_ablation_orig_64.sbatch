#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#SBATCH --job-name=num_inducing_ablation_orig_64
#SBATCH --output=%x_%j.out

/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/ssvi_scripts_experiments/run_original_gplvm_configs_folder.sh ./oil_flow_related/num_induing_ablation_v2_64_orig_gplvm