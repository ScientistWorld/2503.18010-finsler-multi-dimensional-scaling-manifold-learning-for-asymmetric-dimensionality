#!/bin/bash
# Euclidean SMACOF baseline (symmetric 2D embedding of the directed distances).
#
# Every evaluator scores its own frozen copy of this baseline under the name
# `euclidean_mds`, so it cannot be altered by editing baseline/. baseline/embed.py
# is the public copy and doubles as a template for the method interface.
# Running the paper-setting evaluation therefore runs and scores the baseline.
#
# Usage:
#   bash scripts/baseline.sh

set -e

cd /home/user

bash scripts/evaluate.sh
