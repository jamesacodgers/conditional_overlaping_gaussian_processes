


for j in 0 1 2 3 4 5 6 7 8 9  
do 
    for i in 0 1 2 3 4 5 6 7 8 9 
    do
        for l in 8 16
        do
            for t in 0.1 0.5
            do
                echo "Running dataset $i seed $j inducing $l test size $t ..."
                python examples/variable_signals/Dalby_mf.py --data_idx $i --random_seed $j --test_size $t --num_inducing $l
                python examples/variable_signals/Wulfert_mf.py --data_idx $i --random_seed $j --test_size $t --num_inducing $l
            done
        done
    done
done