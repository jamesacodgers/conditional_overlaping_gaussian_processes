#!/bin/bash
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -N wulfert_p32

cd $PBS_O_WORKDIR

module load anaconda3/personal
source activate /rds/general/user/jo816/home/conditional_overlaping_gaussian_processes/.conda

export PYTHONPATH=$PYTHONPATH:/rds/general/user/jo816/home/conditional_overlaping_gaussian_processes
echo $PYTHONPATH


# python examples/classification/spectroscopy_mf.py --data_idx 2 --random_seed 6 --num_inducing 8 --num_latent_dims 2

for j in 0 1 2 3 4 5 6 7 8 9
do 
    for i in 0 1 2 3 4 5 6 7 8 9
    do
        for ts in 0.1 0.5
        do
            echo "Running dataset $i seed $j..."
            python examples/struc_variable_signals/Wulfert_struc.py --data_idx $i --random_seed $j --num_unobs 0 --num_inducing 8 --num_samples 32 --test_size $ts --add_noise_to_lvs  --init_zero

            echo "Running dataset $i seed $j..."
            python examples/struc_variable_signals/Wulfert_struc.py --data_idx $i --random_seed $j --num_unobs 0 --num_inducing 8 --num_samples 32 --test_size $ts --add_noise_to_lvs

            python examples/struc_variable_signals/Wulfert_struc.py --data_idx $i --random_seed $j --num_unobs 0 --num_inducing 8 --num_samples 32 --test_size $ts --init_zero

            echo "Running dataset $i seed $j..."
            python examples/struc_variable_signals/Wulfert_struc.py --data_idx $i --random_seed $j --num_unobs 0 --num_inducing 8 --num_samples 32 --test_size $ts
        done
    done
done