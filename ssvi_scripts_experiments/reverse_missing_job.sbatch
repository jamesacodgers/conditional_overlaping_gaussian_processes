#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#SBATCH --job-name=reverse_mnist_missing_data_v1_ssvi
#SBATCH --output=%x_%j.out

/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/ssvi_scripts_experiments/run_mnist_ood_ssvi_configs_folder.sh ./ood_mnist_related/reverse_mnist_missing_data_v1_ssvi/