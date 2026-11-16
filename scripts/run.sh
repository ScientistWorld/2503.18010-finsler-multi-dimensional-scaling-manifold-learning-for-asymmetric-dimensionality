#!/bin/bash
# Calibration job (this iteration only): reference Finsler SMACOF vs Euclidean
# SMACOF on the visible citation instances and the whole graphs, for the
# paper's embedding dimensions 2/5/10 and drifts 0.1-0.5. Writes
# tmp/calib/results.json. The full reproduce + audit run.sh is restored after.
set -e
cd /home/user
echo "== environment"; python -c "import numpy, scipy, sys; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"; nproc
time python tmp/calib/calib_dims.py
