#!/bin/bash
# Compute-node entry point: reproduce the reference end to end.
#
# Registers the paper's method (outputs/finsler_mds/method.json), then scores it
# together with the evaluators' frozen Euclidean SMACOF baseline on the paper's
# exact trees (scoring/scores.json), the visible instance family
# (scoring/scores_train.json) and the hidden instance family
# (scoring/scores_test.json). The hardening audit is replayed with
#   python -m redteam.hardening.audit run
# (it temporarily adds method cards under outputs/audit/ and removes them).

set -e

cd /home/user

bash scripts/reproduce.sh
python validate.py
