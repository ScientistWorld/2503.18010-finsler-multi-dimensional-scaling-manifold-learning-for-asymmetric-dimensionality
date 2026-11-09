#!/bin/bash
set -euo pipefail
cd /home/user
case_name="shortcut_symmetric_depth"
python -m redteam.audit prepare "$case_name"
set +e
bash scripts/evaluate_train.sh > "redteam/$case_name/train.log" 2>&1
train_exit=$?
set -e
python -m redteam.audit before-test "$case_name"
set +e
# Fresh interpreter, unmodified evaluator, no arguments.
bash scripts/evaluate_test.sh > "redteam/$case_name/test.log" 2>&1
test_exit=$?
set -e
python -m redteam.audit capture "$case_name" "$train_exit" "$test_exit"
