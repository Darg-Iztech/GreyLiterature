#!/bin/bash
for model in "bert" "distilbert"; do
    for data_dir in "data/dp" "data/se"; do
        for crop in 1.0 0.25; do
            for labels in "mean_class" "median_class" "sum_class"; do
                python3 main.py --device='cuda' --model=$model --data_dir=$data_dir --crop=$crop --labels=$labels --epochs=3
            done
        done
    done
done