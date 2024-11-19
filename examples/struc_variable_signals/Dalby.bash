
for j in 0 1 2 3 4 5 6 7 8 9  
do 
    for i in 0 1 2 3 4 5 6 7 8 9 
    do
        echo "Running dataset $i seed $j..."
        python examples/struc_variable_signals/Dalby_struc.py --data_idx $i --random_seed $j --num_unobs 1 --num_inducing 8 --num_samples 32 --test_size 0.1
    done
done