#!/bin/bash
# Paper-setting evaluation: runs every registered method (outputs/**/method.json)
# on the exact Appendix D.1.1 directed trees (depth 7 and depth 5) and writes
# scoring/scores.json and scoring/details.json.
#
# This script is SELF-CONTAINED: it only uses code in eval/ and never imports
# method/. A method takes part by writing a method card, see
# briefing/evaluation.md for the interface:
#   outputs/<name>/method.json = {"name": "<name>", "command": ["python", "-m", "..."]}
# The frozen Euclidean SMACOF baseline (euclidean_mds) is always included.
#
# Usage:
#   bash scripts/evaluate.sh

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate \
  --instances paper \
  --output scoring/scores.json \
  --details scoring/details.json
