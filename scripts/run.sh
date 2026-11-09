#!/bin/bash
set -euo pipefail
cd /home/user
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m redteam.audit all
python -m redteam.summarize
