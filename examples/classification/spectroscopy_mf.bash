

for j in 0 1 2 3 4 5 6 7 8 9
do 
    for i in 0 1 2 3 4 5 6 7 8 9
    do
        echo "Running dataset $i seed $j..."
        
        python examples/classification/spectroscopy_mf.py --data_idx $i --random_seed $j --num_inducing 12 --num_latent_dims 3
    done
done
