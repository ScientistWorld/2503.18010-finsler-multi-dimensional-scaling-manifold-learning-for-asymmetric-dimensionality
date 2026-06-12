#!/bin/bash
# Held-out test-slice evaluator. Scores the same method entries as the train
# slice when scoring/scores_train.json is present.

set -e

cd /home/user

mkdir -p scoring

python -m eval.test.evaluate \
  --output scoring/scores_test.json \
  "$@"
