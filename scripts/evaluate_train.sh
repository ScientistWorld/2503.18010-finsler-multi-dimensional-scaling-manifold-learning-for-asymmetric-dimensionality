#!/bin/bash
# Visible train-slice evaluator. Scores existing method artifacts only; it does
# not train, rerun, or import method code.

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate \
  --output scoring/scores_train.json \
  "$@"
