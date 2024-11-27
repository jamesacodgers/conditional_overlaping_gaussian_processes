
for j in 0 1 2 3 4 5 6 7 8 9
do 
    for i in 0 1 2 3 
    do
        for ts in 0.1 0.5
        do
            echo "Running dataset $i seed $j..."
            python examples/struc_variable_signals/Dalby_struc.py --data_idx $i --random_seed $j --num_unobs 2 --num_inducing 4 --num_samples 8 --num_latent_dims 2 --test_size $ts 
        done
    done
done