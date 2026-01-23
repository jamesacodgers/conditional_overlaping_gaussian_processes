#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#SBATCH --job-name=fashion_mnist_ood_orig
#SBATCH --output=%x_%j.out

/home/haicu/james.odgers/conditional_overlaping_gaussian_processes/ssvi_scripts_experiments/run_mnist_ood_original_gplvm_configs_folder.sh /ood_mnist_related/fashion_mnist_ood_experiment_final_v2_orig_gplvm/