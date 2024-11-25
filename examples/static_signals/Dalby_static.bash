

for j in 0 1 2 3 4 5 6 7 8 9  
do 
    for i in 0 1 2 3 4 5 6 7 8 9 
    do
        echo "Running dataset $i seed $j..."
        python examples/static_signals/Dalby_static.py --data_idx $i --random_seed $j --num_unobs 1 --test_size 0.5
    done
done