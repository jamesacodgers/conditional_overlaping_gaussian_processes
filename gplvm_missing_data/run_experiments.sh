#!/bin/bash

# Run experiments for seeds 1-5
for seed in {1..5}
do
    echo "Running experiments with seed $seed"
    
    # Run GPLVM SSVI for missing data
    echo "Running gplvm_ssvi_for_missing_data..."
    python gplvm_ssvi_for_missing_data_case_1.py --seed $seed

    # Run GPLVM for missing data
    echo "Running gplvm_for_missing_data..."
    python gplvm_for_missing_data_cv_case_1.py --seed $seed
    
    # Run KNN for missing data
    echo "Running knn_for_missing_data..."
    python knn_for_missing_data_cv_case_1.py --seed $seed
done

echo "All experiments completed"