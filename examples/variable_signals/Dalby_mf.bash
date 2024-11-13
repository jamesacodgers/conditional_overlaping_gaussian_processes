
for j in 0 1 2 3 4 5 6 7 8 9  
do 
    for i in 0 1 2 3 4 5 6 7 8 9 
    do
        echo "Running iteration $i..."
        python examples/variable_signals/Dalby_mf.py --data_idx $i --random_seed $j
    done
done