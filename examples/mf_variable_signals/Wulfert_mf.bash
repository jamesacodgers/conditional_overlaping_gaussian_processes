
for j in 0 1 2 3 4 5 6 7 8 9
do 
    for i in 0 1 2 3 4 5 6 7 8 9 
    do
        echo "Running dataset $i seed $j..."
        python examples/mf_variable_signals/Wulfert_mf.py --data_idx $i --random_seed $j --test_size 0.1

        python examples/mf_variable_signals/Wulfert_mf.py --data_idx $i --random_seed $j --test_size 0.5
    done
done