#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#SBATCH --job-name=variance_ablation_ssvi
#SBATCH --output=%x_%j.out

/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/ssvi_scripts_experiments/run_ssvi_configs_folder.sh ./oil_flow_related/variance_ablation_v1_ssvi