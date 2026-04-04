#!/bin/bash
# Reproduce all experiments from the paper.
# Usage: bash experiments/run_all.sh

set -e

echo "============================================"
echo "QGNN: Reproducing All Paper Experiments"
echo "============================================"

COMMON="--runs 10 --verbose"

echo ""
echo "=== Standard Benchmarks ==="
echo ""

echo "--- Cora ---"
python train.py --dataset cora --alpha 0.1 --beta 0.1 $COMMON

echo "--- Citeseer ---"
python train.py --dataset citeseer --alpha 0.1 --beta 0.1 $COMMON

echo "--- PPI ---"
python train.py --dataset ppi --alpha 0.1 --beta 0.1 $COMMON

echo ""
echo "=== LRGB Benchmarks ==="
echo ""

echo "--- PascalVOC-SP ---"
python train.py --dataset pascalvoc-sp --alpha 0.1 --beta 0.1 --k 32 --epochs 500 $COMMON

echo "--- COCO-SP ---"
python train.py --dataset coco-sp --alpha 0.1 --beta 0.1 --k 32 --epochs 500 $COMMON

echo "--- Peptides-func ---"
python train.py --dataset peptides-func --alpha 0.1 --beta 0.1 --epochs 500 $COMMON

echo "--- Peptides-struct ---"
python train.py --dataset peptides-struct --alpha 0.15 --beta 0.1 --epochs 500 $COMMON

echo "--- PCQM-Contact ---"
python train.py --dataset pcqm-contact --alpha 0.1 --beta 0.1 --k 32 --epochs 500 $COMMON

echo ""
echo "=== Alpha Ablation Study (Cora) ==="
echo ""

for ALPHA in 0.01 0.05 0.08 0.1 0.12 0.15 0.2 0.3 0.5; do
    echo "--- alpha=$ALPHA ---"
    python train.py --dataset cora --alpha $ALPHA --beta 0.1 --runs 5
done

echo ""
echo "============================================"
echo "All experiments complete. Results in ./results/"
echo "============================================"
