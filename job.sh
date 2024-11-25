#!/bin/bash
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=10:00:00
#PBS -N test_classification

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate /rds/general/user/jo816/home/conditional_overlaping_gaussian_processes/.conda

export PYTHONPATH=$PYTHONPATH:/rds/general/user/jo816/home/conditional_overlaping_gaussian_processes
echo $PYTHONPATH


# python examples/classification/spectroscopy_mf.py --data_idx 2 --random_seed 6 --num_inducing 8 --num_latent_dims 2

for j in 0 1 2 3 4 5 6 7 8 9  
do 
    for i in 0 1 
    do
        echo "Running dataset $i seed $j..."
        python examples/classification/spectroscopy_mf.py --num_inducing 8 --num_latent_dims 2 --data_idx $i --random_seed $j
    done
done