
for j in 0 
do 
    for i in 0 
    do
        echo "Running dataset $i seed $j..."
        python examples/struc_variable_signals/Dalby_struc.py --data_idx $i --random_seed $j --num_unobs 1 --num_inducing 4 --num_samples 8 --test_size 0.1 --add_noise_to_lvs false
    done
done