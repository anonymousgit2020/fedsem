#!/usr/bin/env bash
python3  -u main.py --dataset=$1 --optimizer=$2  \
            --learning_rate=0.003 --num_rounds=100 --clients_per_round=20 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=5 \
            --model=$3 \
            --mu=$4 \

