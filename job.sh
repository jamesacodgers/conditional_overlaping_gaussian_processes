#!/bin/bash
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=01:00:00
#PBS -N test_save_wulfert

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate /rds/general/user/jo816/home/conditional_overlaping_gaussian_processes/.conda

export PYTHONPATH=$PYTHONPATH:/rds/general/user/res219/home/Submission
echo $PYTHONPATH


python examples/struc_variable_signals/Dalby_struc.py --data_idx 2 --random_seed 6 --num_unobs 1 --num_inducing 4 --num_samples 8 --test_size 0.5

# for j in 0 1 2 3 4 5 6 7 8 9  
# do 
#     for i in 0 1 
#     do
#         echo "Running dataset $i seed $j..."
#         python examples/struc_variable_signals/Dalby_struc.py --data_idx $i --random_seed $j --num_unobs 1 --num_inducing 8 --num_samples 64 --test_size 0.5
#     done
# done