#!/bin/bash
# Hardening check job: score the reference on the hardened gym, replay the
# genuine variants and shortcuts through the unchanged evaluate_*.sh scripts,
# then regenerate the official score files from the reference method alone.

set -uo pipefail

cd /home/user
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p redteam/hardening/results/official

echo "[run] $(date) clearing generated outputs"
find outputs -mindepth 1 -maxdepth 1 ! -name finsler_mds -exec rm -rf {} + 2>/dev/null || true

echo "[run] $(date) reference: register + paper trees"
bash scripts/method.sh || exit 1
echo "[run] $(date) reference: train slice"
bash scripts/evaluate_train.sh || exit 1
echo "[run] $(date) reference: test slice"
bash scripts/evaluate_test.sh || exit 1
cp scoring/scores.json scoring/details.json scoring/scores_train.json scoring/details_train.json \
   scoring/scores_test.json scoring/details_test.json redteam/hardening/results/official/

echo "[run] $(date) audit batch"
python -m redteam.hardening.audit batch || echo "[run] audit batch failed"
echo "[run] $(date) audit isolated cases"
python -m redteam.hardening.audit isolated || echo "[run] audit isolated failed"
echo "[run] $(date) audit summary"
python -m redteam.hardening.audit summarize || echo "[run] audit summary failed"

echo "[run] $(date) official scoring from the reference method alone"
rm -rf outputs/audit
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
for f in scores.json scores_train.json scores_test.json; do
  cmp -s scoring/$f redteam/hardening/results/official/$f && echo "[run] $f reproducible" || echo "[run] $f DIFFERS from first run"
done
python validate.py > redteam/hardening/results/validate.log 2>&1; echo "[run] validate exit $?"
cat redteam/hardening/results/validate.log
echo "[run] $(date) done"
