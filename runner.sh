#!/bin/bash
for model in "bert" "distilbert"; do
    for crop in 1.0 0.25; do
        for data_dir in "data/se" "data/dp"; do
            for labels in "mean_class" "median_class" "sum_class"; do
                python3 main.py --device='cuda' --model=$model --data_dir=$data_dir --crop=$crop --labels=$labels --epoch=4
            done
        done
    done
done 