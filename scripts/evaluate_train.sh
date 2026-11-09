#!/bin/bash
# Visible train-slice evaluator. Runs every registered method
# (outputs/**/method.json) on the visible directed-hierarchy instances and
# writes scoring/scores_train.json (+ scoring/details_train.json). It never
# trains, never imports method/, and has no default method.

set -e

cd /home/user

mkdir -p scoring

python -m eval.train.evaluate \
  --instances family \
  --output scoring/scores_train.json
