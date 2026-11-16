#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# Registers the paper's method, then scores it (and the evaluator's frozen
# Euclidean SMACOF baseline) on the paper setting (exact trees + whole citation
# graphs, scoring/scores.json), the visible slice (scoring/scores_train.json)
# and the hidden slice (scoring/scores_test.json).
#
# Assumes the container is built and the working directory is /home/user.

set -e

cd /home/user

# Leftover audit method cards (e.g. from an interrupted audit) would be scored too.
rm -rf outputs/audit

bash scripts/baseline.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
