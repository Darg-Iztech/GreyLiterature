#!/bin/bash
for model in "distilbert" "bert"; do
    for data_dir in "data/dp"; do
        for crop in 0.25 1.0; do
            for labels in "mean_class" "median_class" "sum_class"; do
                python3 -W ignore main.py --device='cuda' --model=$model --data_dir=$data_dir --crop=$crop --labels=$labels --epochs=3 --save_models=False --seed=123
            done
        done
    done
done