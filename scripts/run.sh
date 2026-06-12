#!/bin/bash
# Compute-node entry point for split evaluation. This scores existing method
# artifacts only; it does not retrain or modify checkpoints.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
