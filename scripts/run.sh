#!/bin/bash
# Compute-node entry point for split evaluation. This scores existing method
# artifacts only; it does not retrain or modify checkpoints.

set -euo pipefail

cd /home/user

if [ ! -f outputs/euclidean_tree/predictions.npz ] \
  || [ ! -f outputs/euclidean_tree_depth5/predictions.npz ] \
  || [ ! -f outputs/finsler_tree/predictions.npz ] \
  || [ ! -f outputs/finsler_tree_depth5/predictions.npz ]; then
  python -m baseline.run_tree_baseline \
    --depth 7 \
    --dim 2 \
    --max-iter 350 \
    --seed 0 \
    --output outputs/euclidean_tree

  python -m baseline.run_tree_baseline \
    --depth 5 \
    --dim 2 \
    --max-iter 300 \
    --seed 1 \
    --output outputs/euclidean_tree_depth5

  python -m method.run_tree_experiment \
    --depth 7 \
    --alpha 0.5 \
    --dim 2 \
    --max-iter 350 \
    --seed 0 \
    --output outputs/finsler_tree

  python -m method.run_tree_experiment \
    --depth 5 \
    --alpha 0.5 \
    --dim 2 \
    --max-iter 300 \
    --seed 1 \
    --output outputs/finsler_tree_depth5
fi

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
