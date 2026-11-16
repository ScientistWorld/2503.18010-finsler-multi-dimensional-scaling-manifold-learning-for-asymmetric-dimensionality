#!/bin/bash
# Compute-node entry point. The scheduler may restart this job at any time, so
# every step is resumable and skipped when already done for the current code
# (redteam/hardening/audit.py stores a SHA-256 of the evaluators, graphs,
# method, baseline, data and scripts with each finished step).
#
# 1. Reproduce the reference end to end, alone (same as scripts/reproduce.sh,
#    whose extra scripts/baseline.sh pass only re-scores the paper slice):
#      paper -> scoring/scores.json        (exact trees + whole citation graphs)
#      train -> scoring/scores_train.json  (visible slice)
#      test  -> scoring/scores_test.json   (hidden slice)
#    Each finished step is copied to redteam/hardening/results/official/.
# 2. Hardening audit (this iteration only): variants and shortcuts replayed
#    through the unchanged evaluate_train.sh / evaluate_test.sh in chunks; the
#    official score files are restored afterwards.
# 3. validate.py.

set -e

cd /home/user

echo "== environment"; python -c "import numpy, scipy, sys; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"; nproc
echo "== code stamp"; python -m redteam.hardening.audit stamp
echo "== recover"; python -m redteam.hardening.audit recover
for step in paper train test; do
  echo "== reproduce: $step"; time python -m redteam.hardening.audit step "$step"
done
echo "== audit"; time python -m redteam.hardening.audit run
echo "== validate"; python validate.py
