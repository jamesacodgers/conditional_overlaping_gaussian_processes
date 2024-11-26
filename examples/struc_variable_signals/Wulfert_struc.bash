
for j in 0 1 2 3 4 5 6 7 8 9
do 
    for i in 0 1 2 3 # 4 5 6 7 8 9
    do
        for ts in 0.1 0.5
        do
            echo "Running dataset $i seed $j..."
            python examples/struc_variable_signals/Wulfert_struc.py --data_idx $i --random_seed $j --num_unobs 0 --num_inducing 8 --num_samples 8 --test_size $ts --add_noise_to_lvs  --init_zero

            echo "Running dataset $i seed $j..."
            python examples/struc_variable_signals/Wulfert_struc.py --data_idx $i --random_seed $j --num_unobs 0 --num_inducing 8 --num_samples 8 --test_size $ts --add_noise_to_lvs

            python examples/struc_variable_signals/Wulfert_struc.py --data_idx $i --random_seed $j --num_unobs 0 --num_inducing 8 --num_samples 8 --test_size $ts --init_zero

            echo "Running dataset $i seed $j..."
            python examples/struc_variable_signals/Wulfert_struc.py --data_idx $i --random_seed $j --num_unobs 0 --num_inducing 8 --num_samples 8 --test_size $ts
        done
    done
done

    
