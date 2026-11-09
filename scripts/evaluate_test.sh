#!/bin/bash
# Held-out test-slice evaluator. Runs every registered method
# (outputs/**/method.json) on hidden directed-hierarchy instances generated
# inside eval/test/ and writes scoring/scores_test.json
# (+ scoring/details_test.json). Takes no arguments.

set -e

cd /home/user

mkdir -p scoring

python -m eval.test.evaluate \
  --output scoring/scores_test.json
