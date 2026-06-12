#!/bin/bash
# Compute-node entry point for split evaluation. The prediction artifacts are
# ignored by git, so recreate the deterministic final artifacts before scoring.

set -euo pipefail

cd /home/user

echo "[run] regenerating deterministic prediction artifacts"
rm -rf \
  outputs/euclidean_tree \
  outputs/euclidean_tree_depth5 \
  outputs/finsler_tree \
  outputs/finsler_tree_depth5

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

find outputs -name predictions.npz -print

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
