#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

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

bash scripts/evaluate.sh
