#!/bin/bash
# Compute-node entry point.
#
# 1. Reproduce the reference end to end, alone: registers the paper's method
#    (outputs/finsler_mds/method.json) and scores it together with the
#    evaluators' frozen Euclidean SMACOF baseline on the paper setting
#    (scoring/scores.json: exact trees + whole citation graphs), the visible
#    slice (scoring/scores_train.json) and the hidden slice
#    (scoring/scores_test.json).
# 2. Hardening audit (this iteration only): replays variants and shortcuts
#    through the unchanged evaluate_train.sh / evaluate_test.sh; it backs up
#    and restores the official score files.
# 3. validate.py.

set -e

cd /home/user

echo "== environment"; python -c "import numpy, scipy, sys; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"; nproc
echo "== reproduce"; time bash scripts/reproduce.sh
mkdir -p redteam/hardening/results/official
cp scoring/scores.json scoring/details.json scoring/scores_train.json scoring/details_train.json \
   scoring/scores_test.json scoring/details_test.json redteam/hardening/results/official/
echo "== audit"; time python -m redteam.hardening.audit run
echo "== validate"; python validate.py
