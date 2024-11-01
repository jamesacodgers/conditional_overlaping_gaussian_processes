
for i in 0 1 2 3 4 5 6 7 8 9 
do
    echo "Running iteration $i..."
    python examples/eiot/Lovastain.py --data_idx $i 
done