#!/bin/bash

for seq_len in 16384 32768 65536 131072
do
for out_len in 256 512 1024 2048
do
echo "=================================================="
echo "T = ${seq_len}    OUT = ${out_len}"
echo "=================================================="

echo "--------------------------------------------------"
echo "                     warmup"
echo "--------------------------------------------------"

python -u scripts/research/bench_serving.py --backend sglang --dataset-name random --num-prompt 100 --random-input-len $seq_len --random-output-len $out_len

echo "--------------------------------------------------"
echo "                     sample"
echo "--------------------------------------------------"

python -u scripts/research/bench_serving.py --backend sglang --dataset-name random --num-prompt 100 --random-input-len $seq_len --random-output-len $out_len

echo "=================================================="
done
done