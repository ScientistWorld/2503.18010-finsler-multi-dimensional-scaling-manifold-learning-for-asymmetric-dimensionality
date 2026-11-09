#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# Registers the paper's method, then scores it (and the evaluator's frozen
# Euclidean SMACOF baseline) on the exact paper trees (scoring/scores.json), the
# visible instance family (scoring/scores_train.json) and the hidden instance
# family (scoring/scores_test.json).
#
# Assumes the container is built and the working directory is /home/user.

set -e

cd /home/user

bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
